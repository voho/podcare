"""Pipeline orchestration: decode → stages → master → encode."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import audio_io, progress
from .config import Config
from .progress import Reporter
from .session import Session, Track
from .stages import (align, breath, declick, deess, dehum, denoise, dereverb,
                     dropouts, fillers, gate, leveler, master, mixdown,
                     plosives, repair, resonance, silence, tonebalance)

log = logging.getLogger(__name__)


def _atten(cfg: Config) -> str:
    return f"{cfg.df_atten_lim_db():.0f}dB"


@dataclass(frozen=True)
class Stage:
    name: str
    enabled: Callable[[Config], bool]
    level: str  # "track" | "session"
    fn: Callable
    # Returns a one-line summary of the effective parameters this stage will use
    # (strength-derived values resolved), logged before the stage runs.
    describe: Callable[[Config], str] = lambda c: ""


STAGES: list[Stage] = [
    Stage("dropouts", lambda c: c.s > 0 and c.dropouts, "track",
          dropouts.restore_dropouts_track,
          lambda c: f"fill 3-{c.dropout_max_gap_ms():.0f}ms packet-loss gaps "
                    f"(two-sided LPC, speech-gated)"),
    Stage("repair", lambda c: c.s > 0 and (c.declip or c.hpf_hz > 0), "track", repair.repair_track,
          lambda c: f"declick+declip={'on' if c.declip else 'off'} hpf={c.hpf_hz:.0f}Hz"),
    Stage("dehum", lambda c: c.s > 0 and c.dehum, "track", dehum.dehum_track,
          lambda c: f"detect 50/60Hz mains, up to {c.dehum_max_harmonics()} harmonics "
                    f"(margin {c.dehum_margin_db():.0f}dB, Q={dehum._NOTCH_Q:.0f})"),
    Stage("align", lambda c: c.align and not c.nocut, "session", align.align_session,
          lambda c: f"window={c.align_window_s:.0f}s conf_z>={c.align_min_confidence:.0f} "
                    f"(not strength-scaled)"),
    Stage("denoise", lambda c: c.s > 0 and c.denoise, "track", denoise.denoise_track,
          lambda c: f"DeepFilterNet3 atten={_atten(c)}"),
    Stage("dereverb", lambda c: c.s > 0 and c.dereverb, "track", dereverb.dereverb_track,
          lambda c: f"WPE taps={c.wpe_taps()} iters={c.wpe_iterations()} delay={c.wpe_delay} "
                    f"chunk={c.dereverb_chunk_s:.0f}s"),
    Stage("tonebalance", lambda c: c.s > 0 and c.tonebalance, "track",
          tonebalance.tonebalance_track,
          lambda c: f"LTAS->broadcast curve, correction={c.eq_correction():.0%} "
                    f"(clamp +{tonebalance._MAX_BOOST_DB:.0f}/-{tonebalance._MAX_CUT_DB:.0f}dB)"),
    Stage("declick", lambda c: c.s > 0 and c.declick, "track", declick.declick_track,
          lambda c: f"mid-band {declick._LO_HZ:.0f}-{declick._HI_HZ:.0f}Hz transients, "
                    f"crest>{c.declick_crest():.0f}x in quiet gaps"),
    Stage("plosives", lambda c: c.s > 0 and c.plosives, "track", plosives.deplosive_track,
          lambda c: f"band<={c.plosive_max_hz:.0f}Hz burst>{c.plosive_burst_mult():.1f}xmed "
                    f"dom>{c.plosive_dominance():.2f} target={c.plosive_target_mult():.1f}xmed"),
    Stage("deess", lambda c: c.s > 0 and c.deess, "track", deess.deess_track,
          lambda c: f"band={c.deess_lo_hz:.0f}-{c.deess_hi_hz:.0f}Hz ratio>{c.deess_ratio():.2f} "
                    f"max={c.deess_max_db():.1f}dB"),
    Stage("resonance", lambda c: c.s > 0 and c.resonance, "track",
          resonance.resonance_track,
          lambda c: f"dynamic notching 800-9000Hz margin={c.resonance_margin_db():.1f}dB "
                    f"max-cut={c.resonance_max_cut_db():.1f}dB"),
    Stage("gate", lambda c: c.s > 0 and c.gate, "track", gate.gate_track,
          lambda c: f"depth={c.gate_depth_db():.1f}dB level={c.level_target_dbfs:.0f}dBFS"),
    Stage("breath", lambda c: c.s > 0 and c.breath, "track", breath.breath_track,
          lambda c: f"duck unvoiced inhales by {c.breath_depth_db():.0f}dB "
                    f"({breath._MIN_BREATH_S:.2f}-{breath._MAX_BREATH_S:.1f}s)"),
    # Filler cuts run per track BEFORE mixdown so the ASR/aligner sees one clean
    # isolated voice; identical intervals are then removed from every track, so
    # they stay frame-synced going into the sum (the one timeline edit before
    # mixdown that is safe precisely because it is applied to all tracks alike).
    Stage("fillers", lambda c: c.fillers and c.eff_filler_sensitivity() > 0 and not c.nocut,
          "session", fillers.remove_fillers_session,
          lambda c: f"sens={c.eff_filler_sensitivity():.2f} model={c.whisper_model} "
                    f"lang={c.language or 'auto'} pad={c.filler_pad_s * 1000:.0f}ms"),
    Stage("mixdown", lambda c: True, "session", mixdown.mixdown_session,
          lambda c: "sum->mono, headroom=-1dBFS"),
    # Pause tightening is the only timeline edit after mixdown — a single mono
    # track, so there is nothing left to keep in sync.
    Stage("tighten", lambda c: c.s > 0 and c.tighten and not c.nocut, "track",
          silence.tighten_track,
          lambda c: f"max_pause={c.eff_max_pause():.2f}s target={c.eff_target_pause():.2f}s "
                    f"lead/tail={c.lead_trail_s:.1f}s"),
    # Slow loudness ride on the mono bus, last before mastering so the compressor
    # and limiter see already-consistent macro-dynamics.
    Stage("leveler", lambda c: c.s > 0 and c.leveler, "track", leveler.leveler_track,
          lambda c: f"slow ride +/-{c.leveler_range_db():.0f}dB over "
                    f"{leveler._WINDOW_S:.0f}s windows toward median speech loudness"),
]


def load_session(inputs: list[Path], cfg: Config) -> Session:
    tracks: list[Track] = []
    names: set[str] = set()
    for path in inputs:
        name, i = path.stem, 2
        while name in names:
            name, i = f"{path.stem}-{i}", i + 1
        names.add(name)
        log.info("decode: %s", path)
        tracks.append(Track(name, audio_io.decode(path, cfg.sr)))
    return Session(cfg.sr, tracks)


def _dump_stems(session: Session, cfg: Config, idx: int, stage_name: str) -> None:
    assert cfg.keep_stems is not None
    cfg.keep_stems.mkdir(parents=True, exist_ok=True)
    for track in session.tracks:
        out = cfg.keep_stems / f"{idx:02d}-{stage_name}-{track.name}.wav"
        audio_io.write_wav(out, track.audio, session.sr)


def run(inputs: list[Path], out_path: Path, cfg: Config,
        reporter: Reporter | None = None) -> tuple[float, float]:
    """Process input files into out_path; returns (input, output) durations in s.

    `reporter` drives the optional live progress display; it defaults to a silent
    no-op so library/test callers behave exactly as before.
    """
    reporter = reporter or progress.NullReporter()
    audio_io.require_ffmpeg()
    # Fail on an unwritable destination now, not after an hour of processing.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(STAGES) + 1  # +1 for master+encode; used in the [i/n] log tags
    n_enabled = sum(1 for s in STAGES if s.enabled(cfg)) + 1  # the progress bar total
    t_pipeline = time.perf_counter()

    with progress.use(reporter):
        reporter.start(n_enabled)
        session = load_session(inputs, cfg)
        # Decode bookends up front — a corrupt sting must fail now, not after
        # an hour of processing (same fail-fast contract as the out_path check).
        intro = audio_io.decode(cfg.intro_sound, cfg.sr) if cfg.intro_sound else None
        outro = audio_io.decode(cfg.outro_sound, cfg.sr) if cfg.outro_sound else None
        in_duration = session.duration_s()
        done = 0

        for idx, stage in enumerate(STAGES, start=1):
            tag = f"[{idx}/{n_total}] {stage.name}"
            if not stage.enabled(cfg):
                log.info("%s: skipped (disabled)", tag)
                continue
            done += 1
            log.info("%s: start · %s", tag, stage.describe(cfg))
            reporter.begin_stage(done, n_enabled, stage.name, stage.describe(cfg))
            t0 = time.perf_counter()
            if stage.level == "session":
                session = stage.fn(session, cfg)
            else:
                session = Session(session.sr, [stage.fn(t, cfg) for t in session.tracks])
            dt = time.perf_counter() - t0
            log.info("%s: done in %.1fs", tag, dt)
            reporter.end_stage(stage.name, dt)
            if cfg.keep_stems is not None:
                _dump_stems(session, cfg, idx, stage.name)
            # Free the DeepFilterNet model + PyTorch cache before the
            # memory-hungry dereverb/WPE stage that immediately follows denoise,
            # so WPE's large per-chunk transients don't contend for RAM.
            if stage.name == "denoise":
                denoise.unload_deepfilter()

        if len(session.tracks) != 1:
            raise RuntimeError(f"expected one track after mixdown, got {len(session.tracks)}")
        final = session.tracks[0]
        out_duration = final.n_samples / cfg.sr

        tag = f"[{n_total}/{n_total}] master+encode"
        if cfg.master:
            comp = (f"mb-comp(3-band) {cfg.comp_ratio():.1f}:1@{cfg.comp_threshold():.2f} "
                    if cfg.compress and cfg.s > 0 else "comp=off ")
            exc = (f"exciter={cfg.exciter_amount():.1f} "
                   if cfg.exciter and cfg.s > 0 else "exciter=off ")
            params = (f"{comp}{exc}loudnorm I={cfg.lufs:.0f}LUFS TP={cfg.true_peak_db:.1f}dB "
                      f"+TP-limiter -> {cfg.out_sr}Hz")
        else:
            params = f"raw mix, encode -> {cfg.out_sr}Hz"
        log.info("%s: start · %s", tag, params)
        reporter.begin_stage(done + 1, n_enabled, "master+encode", params)
        t0 = time.perf_counter()
        master.master_and_encode(final, cfg, out_path, intro=intro, outro=outro)
        dt = time.perf_counter() - t0
        reporter.end_stage("master+encode", dt)
        log.info("%s: done in %.1fs", tag, dt)

    if cfg.keep_stems is not None:
        # Capture the delivered file too, so the stems dir is a complete A/B set.
        import shutil
        cfg.keep_stems.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_path, cfg.keep_stems / f"{n_total:02d}-master{out_path.suffix.lower()}")

    bit_depth = "/16-bit" if out_path.suffix.lower() in (".wav", ".flac") else ""
    spec = (f"loudnorm I={cfg.lufs:.1f}LUFS TP={cfg.true_peak_db:.1f}dBTP"
            if cfg.master else "raw mix")
    log.info("output: %s — %.1f min, %d Hz%s, %s, %.1f MB", out_path.name,
             out_duration / 60, cfg.out_sr, bit_depth, spec,
             out_path.stat().st_size / 1e6)
    log.info("pipeline: %.1f min in -> %.1f min out in %.1fs total",
             in_duration / 60, out_duration / 60, time.perf_counter() - t_pipeline)
    return in_duration, out_duration
