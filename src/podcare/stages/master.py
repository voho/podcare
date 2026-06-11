"""Mastering: bus compression + two-pass EBU R128 loudness normalization + encode."""

from __future__ import annotations

import logging
import math
from pathlib import Path

from .. import audio_io
from ..config import Config
from ..dsp import db_to_lin
from ..session import Track

log = logging.getLogger(__name__)


def _compressor(cfg: Config) -> str:
    return (f"acompressor=threshold={cfg.comp_threshold():.3f}:ratio={cfg.comp_ratio():.2f}"
            ":attack=10:release=200:knee=4")


def _limiter(cfg: Config) -> str:
    """Brickwall true-peak safety limiter — the final master node, before the
    single soxr resample. loudnorm's linear gain is not a real lookahead limiter,
    so this catches the inter-sample / codec overs it can leave and lets the
    program sit at the loudness target without clipping consumer DACs. Runs at the
    48 kHz internal rate for ISP headroom; ``level=false`` so it never
    auto-makeup-gains against loudnorm's loudness target. Delivery safety, not an
    intensity — it runs whenever mastering is on, regardless of strength."""
    return (f"alimiter=limit={db_to_lin(cfg.true_peak_db):.6f}"
            ":attack=5:release=50:asc=1:level=false")


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def master_and_encode(track: Track, cfg: Config, out_path: Path) -> None:
    if not cfg.master:
        audio_io.encode(track.audio, cfg.sr, out_path,
                        out_sr=cfg.out_sr, lossy_bitrate=cfg.lossy_bitrate)
        return

    # Loudness normalization always runs (even at strength 0) so the output hits
    # the target level; bus compression is strength-scaled and off at strength 0.
    pre = _compressor(cfg) if (cfg.compress and cfg.s > 0) else None
    measured = audio_io.measure_loudnorm(track.audio, cfg.sr, pre_filters=pre,
                                         lufs=cfg.lufs, true_peak=cfg.true_peak_db)
    # Below loudnorm's -70 LUFS gate (silent / near-silent program) ffmpeg returns
    # -inf/inf, which the second pass rejects ("out of range"). Emit the (silent)
    # mix instead of crashing the whole episode at the final step.
    if not (_finite(measured.get("input_i")) and _finite(measured.get("input_tp"))
            and _finite(measured.get("target_offset"))):
        log.warning("master: program is silent/near-silent (input_i=%s) — skipping loudnorm",
                    measured.get("input_i"))
        audio = audio_io.filter_array(track.audio, cfg.sr, pre) if pre else track.audio
        audio_io.encode(audio, cfg.sr, out_path, out_sr=cfg.out_sr, lossy_bitrate=cfg.lossy_bitrate)
        return
    log.info("master: measured %s LUFS, %s dBTP — normalizing to %.1f LUFS / %.1f dBTP",
             measured.get("input_i"), measured.get("input_tp"), cfg.lufs, cfg.true_peak_db)
    loudnorm = (
        f"loudnorm=I={cfg.lufs}:TP={cfg.true_peak_db}:LRA=11"
        f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )
    # acompressor -> loudnorm -> true-peak limiter, all at the working rate; then
    # a single soxr resample to out_sr in encode().
    graph = ",".join(filter(None, [pre, loudnorm, _limiter(cfg)]))
    mastered = audio_io.filter_array(track.audio, cfg.sr, graph)
    audio_io.encode(mastered, cfg.sr, out_path, out_sr=cfg.out_sr, lossy_bitrate=cfg.lossy_bitrate)
