"""Pipeline orchestration: decode → stages → master → encode."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import audio_io
from .config import Config
from .session import Session, Track
from .stages import (align, deess, denoise, dereverb, fillers, gate, master,
                     mixdown, plosives, repair, silence)

log = logging.getLogger(__name__)


def _atten(cfg: Config) -> str:
    lim = cfg.df_atten_lim_db()
    return "full" if lim is None else f"{lim:.0f}dB"


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
    Stage("repair", lambda c: c.s > 0 and (c.declip or c.hpf_hz > 0), "track", repair.repair_track,
          lambda c: f"declick+declip={'on' if c.declip else 'off'} hpf={c.hpf_hz:.0f}Hz"),
    Stage("align", lambda c: c.align, "session", align.align_session,
          lambda c: f"window={c.align_window_s:.0f}s conf_z>={c.align_min_confidence:.0f} "
                    f"(not strength-scaled)"),
    Stage("denoise", lambda c: c.s > 0 and c.denoise, "track", denoise.denoise_track,
          lambda c: f"backend={c.denoise_backend} prop={c.denoise_prop():.2f} atten={_atten(c)}"),
    Stage("dereverb", lambda c: c.s > 0 and c.dereverb, "track", dereverb.dereverb_track,
          lambda c: f"WPE taps={c.wpe_taps()} iters={c.wpe_iterations()} delay={c.wpe_delay} "
                    f"chunk={c.dereverb_chunk_s:.0f}s"),
    Stage("plosives", lambda c: c.s > 0 and c.plosives, "track", plosives.deplosive_track,
          lambda c: f"band<={c.plosive_max_hz:.0f}Hz burst>{c.plosive_burst_mult():.1f}xmed "
                    f"dom>{c.plosive_dominance():.2f} target={c.plosive_target_mult():.1f}xmed"),
    Stage("deess", lambda c: c.s > 0 and c.deess, "track", deess.deess_track,
          lambda c: f"band={c.deess_lo_hz:.0f}-{c.deess_hi_hz:.0f}Hz ratio>{c.deess_ratio():.2f} "
                    f"max={c.deess_max_db():.1f}dB"),
    Stage("gate", lambda c: c.s > 0 and c.gate, "track", gate.gate_track,
          lambda c: f"depth={c.gate_depth_db():.1f}dB level={c.level_target_dbfs:.0f}dBFS"),
    Stage("mixdown", lambda c: True, "session", mixdown.mixdown_session,
          lambda c: "sum->mono, headroom=-1dBFS"),
    # Timeline edits only after mixdown, so all tracks stay in sync.
    Stage("fillers", lambda c: c.fillers and c.eff_filler_sensitivity() > 0, "track",
          fillers.remove_fillers_track,
          lambda c: f"sens={c.eff_filler_sensitivity():.2f} model={c.whisper_model} "
                    f"pad={c.filler_pad_s * 1000:.0f}ms"),
    Stage("tighten", lambda c: c.s > 0 and c.tighten, "track", silence.tighten_track,
          lambda c: f"max_pause={c.eff_max_pause():.2f}s target={c.eff_target_pause():.2f}s "
                    f"lead/tail={c.lead_trail_s:.1f}s"),
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


def run(inputs: list[Path], out_path: Path, cfg: Config) -> tuple[float, float]:
    """Process input files into out_path; returns (input, output) durations in s."""
    audio_io.require_ffmpeg()
    # Fail on an unwritable destination now, not after an hour of processing.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    session = load_session(inputs, cfg)
    in_duration = session.duration_s()
    n_total = len(STAGES) + 1  # +1 for master+encode
    t_pipeline = time.perf_counter()

    for idx, stage in enumerate(STAGES, start=1):
        tag = f"[{idx}/{n_total}] {stage.name}"
        if not stage.enabled(cfg):
            log.info("%s: skipped (disabled)", tag)
            continue
        log.info("%s: start · %s", tag, stage.describe(cfg))
        t0 = time.perf_counter()
        if stage.level == "session":
            session = stage.fn(session, cfg)
        else:
            session = Session(session.sr, [stage.fn(t, cfg) for t in session.tracks])
        log.info("%s: done in %.1fs", tag, time.perf_counter() - t0)
        if cfg.keep_stems is not None:
            _dump_stems(session, cfg, idx, stage.name)

    if len(session.tracks) != 1:
        raise RuntimeError(f"expected one track after mixdown, got {len(session.tracks)}")
    final = session.tracks[0]
    out_duration = final.n_samples / cfg.sr

    tag = f"[{n_total}/{n_total}] master+encode"
    if cfg.master:
        comp = (f"comp={cfg.comp_ratio():.1f}:1@{cfg.comp_threshold():.2f} "
                if cfg.compress and cfg.s > 0 else "comp=off ")
        params = (f"{comp}loudnorm I={cfg.lufs:.0f}LUFS TP={cfg.true_peak_db:.1f}dB "
                  f"-> {cfg.out_sr}Hz")
    else:
        params = f"raw mix, encode -> {cfg.out_sr}Hz"
    log.info("%s: start · %s", tag, params)
    t0 = time.perf_counter()
    master.master_and_encode(final, cfg, out_path)
    log.info("%s: done in %.1fs", tag, time.perf_counter() - t0)

    log.info("pipeline: %.1f min in -> %.1f min out in %.1fs total",
             in_duration / 60, out_duration / 60, time.perf_counter() - t_pipeline)
    return in_duration, out_duration
