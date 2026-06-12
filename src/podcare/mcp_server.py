"""Model Context Protocol (MCP) server exposing Podcare as tools.

A thin layer over the existing pipeline: every processing stage (and the full
end-to-end pipeline) is published as an MCP tool so an assistant can drive
Podcare programmatically — decode a file, run one stage or the whole chain, and
get a cleaned file back.

Each *stage* tool takes one or more input files, applies just that stage, and
writes the result as 48 kHz float WAV(s) into an output directory — so the
intermediate stays loss-free and can be fed straight into the next stage. The
`master` and `process` tools are the ones that produce a real delivery file
(picking the container from the output extension, resampling once at the end).

Parameters mirror the CLI: one universal ``strength`` knob (0..1) plus a small,
sensible set of per-stage overrides, all with defaults taken from
``podcare.config.Config`` (the single source of truth).

Run it with::

    uv run podcare-mcp            # stdio transport (for desktop/IDE MCP clients)

Requires the optional ``mcp`` dependency: ``uv sync --extra mcp``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import audio_io
from .config import Config
from .pipeline import STAGES, load_session, run
from .session import Session, Track
from .stages.align import align_session
from .stages.breath import breath_track
from .stages.declick import declick_track
from .stages.deess import deess_track
from .stages.dehum import dehum_track
from .stages.denoise import denoise_track
from .stages.dereverb import dereverb_track
from .stages.fillers import remove_fillers_session
from .stages.gate import gate_track
from .stages.leveler import leveler_track
from .stages.master import master_and_encode
from .stages.mixdown import mixdown_session
from .stages.plosives import deplosive_track
from .stages.repair import repair_track
from .stages.silence import tighten_track
from .stages.tonebalance import tonebalance_track

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
    raise SystemExit(
        "the MCP server needs the optional 'mcp' dependency — install it with "
        "`uv sync --extra mcp` (or `pip install 'podcare[mcp]'`)"
    ) from exc

log = logging.getLogger("podcare.mcp")

mcp = FastMCP("podcare")

# Stage name -> the one-line "effective parameters" describer defined in the
# pipeline, so each tool can echo back exactly what the CLI would log.
_DESCRIBE = {s.name: s.describe for s in STAGES}

# Stage names the `process` tool accepts in `disable` (the CLI `--no-*` set).
_TOGGLEABLE = {"declip", "dehum", "align", "denoise", "dereverb", "tonebalance",
               "declick", "plosives", "deess", "gate", "breath", "fillers",
               "tighten", "leveler", "master"}


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _decode(input_paths: list[str], cfg: Config) -> Session:
    audio_io.require_ffmpeg()
    if not input_paths:
        raise ValueError("at least one input file is required")
    return load_session([Path(p) for p in input_paths], cfg)


def _emit(tracks: list[Track], output_dir: str, sr: int) -> list[str]:
    """Write each resulting track as a 48 kHz float WAV; return the paths."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for track in tracks:
        out = out_dir / f"{track.name}.wav"
        audio_io.write_wav(out, track.audio, sr)
        paths.append(str(out))
    return paths


def _track_stage(fn: Callable[[Track, Config], Track], name: str,
                 input_paths: list[str], output_dir: str, cfg: Config) -> dict[str, Any]:
    """Run a per-track stage over every input independently."""
    session = _decode(input_paths, cfg)
    out = [fn(t, cfg) for t in session.tracks]
    summary = _DESCRIBE[name](cfg)
    log.info("%s: %s", name, summary)
    return {"stage": name, "applied": summary,
            "outputs": _emit(out, output_dir, session.sr)}


def _session_stage(fn: Callable[[Session, Config], Session], name: str,
                   input_paths: list[str], output_dir: str, cfg: Config) -> dict[str, Any]:
    """Run a session-level stage that sees all tracks at once."""
    session = fn(_decode(input_paths, cfg), cfg)
    summary = _DESCRIBE[name](cfg)
    log.info("%s: %s", name, summary)
    return {"stage": name, "applied": summary,
            "outputs": _emit(session.tracks, output_dir, session.sr)}


# --------------------------------------------------------------------------- #
# Per-track stage tools — write 48 kHz float WAV(s) so stages chain losslessly.
# --------------------------------------------------------------------------- #
@mcp.tool()
def repair(input_paths: list[str], output_dir: str,
           declip: bool = True, hpf_hz: float = 80.0) -> dict:
    """Restore each track: declick + declip and a rumble high-pass.

    Not strength-scaled — restoration, not a degree of effect.

    Args:
        input_paths: One or more audio files (one per mic/recorder).
        output_dir: Directory to write the cleaned 48 kHz WAV(s) into.
        declip: Run ffmpeg adeclick + adeclip over impulsive clicks and clipping.
        hpf_hz: 2-pole rumble high-pass cutoff in Hz (0 disables it).
    """
    cfg = Config(strength=1.0, declip=declip, hpf_hz=hpf_hz)
    return _track_stage(repair_track, "repair", input_paths, output_dir, cfg)


@mcp.tool()
def dehum(input_paths: list[str], output_dir: str, strength: float = 0.8) -> dict:
    """Remove steady 50/60 Hz mains hum and its harmonics (detection-gated).

    Args:
        input_paths: One or more audio files.
        output_dir: Directory for the output WAV(s).
        strength: 0..1; scales how many harmonics are notched and how readily
            hum is detected (0 is a no-op).
    """
    return _track_stage(dehum_track, "dehum", input_paths, output_dir,
                        Config(strength=strength))


@mcp.tool()
def denoise(input_paths: list[str], output_dir: str, strength: float = 0.8) -> dict:
    """Broadband neural denoise (DeepFilterNet3, 48 kHz full-band).

    Args:
        input_paths: One or more audio files.
        output_dir: Directory for the output WAV(s).
        strength: 0..1; sets the attenuation ceiling (0 → 60 dB).
    """
    return _track_stage(denoise_track, "denoise", input_paths, output_dir,
                        Config(strength=strength))


@mcp.tool()
def dereverb(input_paths: list[str], output_dir: str, strength: float = 0.8,
             chunk_s: float = 30.0, wpe_delay: int = 3) -> dict:
    """WPE dereverberation — strip the late-reverberation room tail.

    Args:
        input_paths: One or more audio files.
        output_dir: Directory for the output WAV(s).
        strength: 0..1; lengthens the prediction filter and adds iterations.
        chunk_s: Processing chunk length in seconds.
        wpe_delay: WPE prediction delay (frames).
    """
    cfg = Config(strength=strength, dereverb_chunk_s=chunk_s, wpe_delay=wpe_delay)
    return _track_stage(dereverb_track, "dereverb", input_paths, output_dir, cfg)


@mcp.tool()
def tonebalance(input_paths: list[str], output_dir: str, strength: float = 0.8) -> dict:
    """Tonal-balance EQ: match each track's LTAS to a broadcast-voice curve.

    Args:
        input_paths: One or more audio files.
        output_dir: Directory for the output WAV(s).
        strength: 0..1; fraction of the measured deviation corrected (strength/3).
    """
    return _track_stage(tonebalance_track, "tonebalance", input_paths, output_dir,
                        Config(strength=strength))


@mcp.tool()
def declick(input_paths: list[str], output_dir: str, strength: float = 0.8) -> dict:
    """Remove mouth clicks, lip smacks and saliva crackle (mid-band transients).

    Args:
        input_paths: One or more audio files.
        output_dir: Directory for the output WAV(s).
        strength: 0..1; lowers the crest-factor threshold to catch subtler clicks.
    """
    return _track_stage(declick_track, "declick", input_paths, output_dir,
                        Config(strength=strength))


@mcp.tool()
def plosives(input_paths: list[str], output_dir: str, strength: float = 0.8,
             max_hz: float = 150.0) -> dict:
    """Duck "p"/"b" plosive pops — abnormal low-frequency bursts.

    Args:
        input_paths: One or more audio files.
        output_dir: Directory for the output WAV(s).
        strength: 0..1; lowers the detection thresholds and deepens the duck.
        max_hz: Plosive band ceiling in Hz.
    """
    cfg = Config(strength=strength, plosive_max_hz=max_hz)
    return _track_stage(deplosive_track, "plosives", input_paths, output_dir, cfg)


@mcp.tool()
def deess(input_paths: list[str], output_dir: str, strength: float = 0.8,
          lo_hz: float = 4500.0, hi_hz: float = 9500.0) -> dict:
    """De-ess — tame harsh "s"/"sh"/"t" sibilance.

    Args:
        input_paths: One or more audio files.
        output_dir: Directory for the output WAV(s).
        strength: 0..1; lowers the trigger ratio and raises the max reduction.
        lo_hz: Low edge of the sibilance band in Hz.
        hi_hz: High edge of the sibilance band in Hz.
    """
    cfg = Config(strength=strength, deess_lo_hz=lo_hz, deess_hi_hz=hi_hz)
    return _track_stage(deess_track, "deess", input_paths, output_dir, cfg)


@mcp.tool()
def gate(input_paths: list[str], output_dir: str, strength: float = 0.8,
         level_target_dbfs: float = -20.0) -> dict:
    """Crosstalk gate + per-track level match (run before mixdown).

    Args:
        input_paths: One or more audio files.
        output_dir: Directory for the output WAV(s).
        strength: 0..1; sets how deep the gate cuts.
        level_target_dbfs: Speech-active RMS target each track is normalized to.
    """
    cfg = Config(strength=strength, level_target_dbfs=level_target_dbfs)
    return _track_stage(gate_track, "gate", input_paths, output_dir, cfg)


@mcp.tool()
def breath(input_paths: list[str], output_dir: str, strength: float = 0.8) -> dict:
    """Duck audible inhale breaths between phrases (never muted).

    Args:
        input_paths: One or more audio files.
        output_dir: Directory for the output WAV(s).
        strength: 0..1; sets the duck depth (capped at 14 dB).
    """
    return _track_stage(breath_track, "breath", input_paths, output_dir,
                        Config(strength=strength))


@mcp.tool()
def leveler(input_paths: list[str], output_dir: str, strength: float = 0.8) -> dict:
    """Slow segment-loudness leveler — a gentle multi-second loudness ride.

    Args:
        input_paths: One or more audio files (typically the mono program).
        output_dir: Directory for the output WAV(s).
        strength: 0..1; sets the maximum ride range (±dB).
    """
    return _track_stage(leveler_track, "leveler", input_paths, output_dir,
                        Config(strength=strength))


@mcp.tool()
def tighten(input_paths: list[str], output_dir: str, strength: float = 0.8,
            max_pause_s: float | None = None, target_pause_s: float | None = None,
            lead_trail_s: float = 0.5) -> dict:
    """Tighten over-long pauses / dead air on the mono program.

    Args:
        input_paths: One or more audio files (typically the mono program).
        output_dir: Directory for the output WAV(s).
        strength: 0..1; shortens both the trigger and the kept beat.
        max_pause_s: Pauses longer than this get shortened (default follows strength).
        target_pause_s: Length an over-long pause is shortened to (default follows strength).
        lead_trail_s: Lead/tail silence trimmed to this length.
    """
    cfg = Config(strength=strength, max_pause_s=max_pause_s,
                 target_pause_s=target_pause_s, lead_trail_s=lead_trail_s)
    return _track_stage(tighten_track, "tighten", input_paths, output_dir, cfg)


# --------------------------------------------------------------------------- #
# Session-level stage tools — they see all tracks at once.
# --------------------------------------------------------------------------- #
@mcp.tool()
def align(input_paths: list[str], output_dir: str,
          window_s: float = 300.0, min_confidence: float = 12.0) -> dict:
    """Align tracks in time and fix inverted polarity (needs ≥ 2 tracks).

    Not strength-scaled — a correctness fix.

    Args:
        input_paths: Two or more audio files, one per mic/recorder.
        output_dir: Directory for the aligned output WAVs (one per input).
        window_s: Seconds of the opening searched for the inter-track offset.
        min_confidence: z-score the GCC-PHAT peak must clear to apply a shift.
    """
    cfg = Config(align_window_s=window_s, align_min_confidence=min_confidence)
    return _session_stage(align_session, "align", input_paths, output_dir, cfg)


@mcp.tool()
def fillers(input_paths: list[str], output_dir: str, strength: float = 0.8,
            sensitivity: float | None = None, whisper_model: str = "large-v3",
            language: str | None = None, pad_s: float = 0.012) -> dict:
    """Remove non-lexical fillers ("um", "uh", "ehm", …) via ASR + forced align.

    Detection is per track; an interval is cut only where every other track is
    silent, and the surviving cuts are removed identically from all tracks so
    they stay frame-synced. Any ASR/alignment failure degrades to a no-op.

    Args:
        input_paths: One or more audio files, one per mic/recorder.
        output_dir: Directory for the trimmed output WAVs (one per input).
        strength: 0..1; effective sensitivity defaults to 0.7 × strength.
        sensitivity: Override 0..1 (0 disables); wins over strength when set.
        whisper_model: faster-whisper model used to transcribe (e.g. large-v3, small).
        language: Force the spoken language (e.g. en, cs, de); None auto-detects.
        pad_s: Padding added around each cut, in seconds.
    """
    cfg = Config(strength=strength, filler_sensitivity=sensitivity,
                 whisper_model=whisper_model, language=language, filler_pad_s=pad_s)
    return _session_stage(remove_fillers_session, "fillers", input_paths, output_dir, cfg)


@mcp.tool()
def mixdown(input_paths: list[str], output_dir: str) -> dict:
    """Sum all cleaned tracks to one mono program with ~1 dB of headroom.

    Args:
        input_paths: One or more audio files, one per mic/recorder.
        output_dir: Directory for the single mono output WAV.
    """
    return _session_stage(mixdown_session, "mixdown", input_paths, output_dir, Config())


# --------------------------------------------------------------------------- #
# Delivery tools — these produce a real, encoded file.
# --------------------------------------------------------------------------- #
@mcp.tool()
def master(input_path: str, output_path: str, strength: float = 0.8,
           compress: bool = True, lufs: float = -16.0, true_peak_db: float = -1.5,
           out_sr: int = 44100, bitrate: str = "192k") -> dict:
    """Master one mono program and encode it: MB-comp → loudnorm → TP-limit → encode.

    The output container is chosen from the ``output_path`` extension
    (.wav/.mp3/.flac/.m4a/.aac); WAV/FLAC are 16-bit with dither.

    Args:
        input_path: A single mono program WAV.
        output_path: Delivery file; its extension selects the container.
        strength: 0..1; firms up the 3-band compression (off at 0). Loudness and
            the true-peak limiter are absolute and always applied.
        compress: Enable the multiband compressor.
        lufs: Integrated-loudness target (EBU R128), validated -40..-5.
        true_peak_db: True-peak ceiling in dBTP.
        out_sr: Output sample rate (single final resample).
        bitrate: MP3/AAC bitrate (ignored for WAV/FLAC).
    """
    audio_io.require_ffmpeg()
    cfg = Config(strength=strength, compress=compress, lufs=lufs,
                 true_peak_db=true_peak_db, out_sr=out_sr, lossy_bitrate=bitrate)
    session = load_session([Path(input_path)], cfg)
    if len(session.tracks) != 1:
        raise ValueError("master expects exactly one input (a mono program)")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    master_and_encode(session.tracks[0], cfg, out)
    return {"stage": "master", "output": str(out), "out_sr": out_sr, "lufs": lufs}


@mcp.tool()
def process(input_paths: list[str], output_path: str, strength: float = 0.8,
            nocut: bool = False, lufs: float = -16.0, out_sr: int = 44100,
            bitrate: str = "192k", whisper_model: str = "large-v3",
            language: str | None = None, filler_sensitivity: float | None = None,
            disable: list[str] | None = None) -> dict:
    """Run the full Podcare pipeline: decode → all stages → master → encode.

    This is the one-shot tool — give it the raw per-mic recordings and it returns
    one polished, broadcast-ready file (the same as the `podcare` CLI).

    Args:
        input_paths: Raw recordings, one per mic/recorder.
        output_path: Delivery file; its extension selects the container.
        strength: 0..1 universal intensity (0 = raw mixdown, 1 = max).
        nocut: Keep the original timeline (skip align/fillers/tighten).
        lufs: Integrated-loudness target (validated -40..-5).
        out_sr: Output sample rate (validated 8000..192000).
        bitrate: MP3/AAC bitrate (ignored for WAV/FLAC).
        whisper_model: faster-whisper model for filler detection.
        language: Force the spoken language; None auto-detects.
        filler_sensitivity: Override filler aggressiveness 0..1 (0 disables).
        disable: Stage names to turn off, e.g. ["dereverb", "tighten"]. Valid:
            declip, dehum, align, denoise, dereverb, tonebalance, declick,
            plosives, deess, gate, breath, fillers, tighten, leveler, master.
    """
    off = set(disable or [])
    unknown = off - _TOGGLEABLE
    if unknown:
        raise ValueError(f"unknown stage name(s) in disable: {sorted(unknown)}; "
                         f"valid: {sorted(_TOGGLEABLE)}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in 0..1")
    if not -40.0 <= lufs <= -5.0:
        raise ValueError("lufs must be between -40 and -5")
    if not 8000 <= out_sr <= 192000:
        raise ValueError("out_sr must be between 8000 and 192000")

    cfg = Config(
        strength=strength, nocut=nocut, lufs=lufs, out_sr=out_sr,
        lossy_bitrate=bitrate, whisper_model=whisper_model, language=language,
        filler_sensitivity=0.0 if "fillers" in off else filler_sensitivity,
        declip="declip" not in off, dehum="dehum" not in off,
        align="align" not in off, denoise="denoise" not in off,
        dereverb="dereverb" not in off, tonebalance="tonebalance" not in off,
        declick="declick" not in off, plosives="plosives" not in off,
        deess="deess" not in off, gate="gate" not in off,
        breath="breath" not in off, fillers="fillers" not in off,
        tighten="tighten" not in off, leveler="leveler" not in off,
        master="master" not in off,
    )
    out = Path(output_path)
    in_dur, out_dur = run([Path(p) for p in input_paths], out, cfg)
    return {"output": str(out), "input_minutes": round(in_dur / 60, 2),
            "output_minutes": round(out_dur / 60, 2)}


def main() -> None:
    """Entry point: serve over stdio for MCP clients."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    mcp.run()


if __name__ == "__main__":
    main()
