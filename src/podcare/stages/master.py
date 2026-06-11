"""Mastering: multiband bus compression + two-pass EBU R128 loudness
normalization + true-peak limiting + encode."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from .. import audio_io
from ..config import Config
from ..dsp import db_to_lin
from ..session import Track

log = logging.getLogger(__name__)

# Linkwitz-Riley crossover points (Hz) for the 3-band master compressor.
_SPLIT_LO = 250
_SPLIT_HI = 4000


def _band_comp(threshold: float, ratio: float, attack: int, release: int) -> str:
    return (f"acompressor=threshold={min(threshold, 1.0):.3f}:ratio={ratio:.2f}"
            f":attack={attack}:release={release}:knee=4")


def _multiband(cfg: Config) -> str:
    """3-band (low/mid/high) bus compressor as a filter_complex graph.

    Independent per-band dynamics tame proximity boom and plosive thump (low) and
    harsh upper-mids (high) without ducking the whole program off the loudest
    band — denser, more consistent loudness than a single broadband compressor.
    Low band controls a touch firmer, high band gentler; all derived from the one
    strength knob so 1:1 at strength 0 is the degenerate (no-compression) case.
    """
    r, th = cfg.comp_ratio(), cfg.comp_threshold()
    low = _band_comp(th * 0.9, min(r * 1.1, 20.0), 10, 200)
    mid = _band_comp(th, r, 10, 200)
    high = _band_comp(th * 1.1, max(r * 0.8, 1.0), 5, 120)
    return (f"acrossover=split={_SPLIT_LO} {_SPLIT_HI}:order=4th[lo][mid][hi];"
            f"[lo]{low}[a];[mid]{mid}[b];[hi]{high}[c];"
            "[a][b][c]amix=inputs=3:normalize=0[out]")


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


def _apply_multiband(audio: np.ndarray, cfg: Config) -> np.ndarray:
    out = audio_io.filter_complex_array(audio, cfg.sr, _multiband(cfg))
    # acrossover/amix preserve length, but pin it exactly so downstream timing
    # (reported output duration) can never drift.
    if len(out) >= len(audio):
        return out[: len(audio)]
    return np.pad(out, (0, len(audio) - len(out)))


def master_and_encode(track: Track, cfg: Config, out_path: Path) -> None:
    if not cfg.master:
        audio_io.encode(track.audio, cfg.sr, out_path,
                        out_sr=cfg.out_sr, lossy_bitrate=cfg.lossy_bitrate)
        return

    # Multiband compression (strength-scaled, off at strength 0) is applied first,
    # as its own pass; loudness normalization then always runs on the result so the
    # output hits the target level even at strength 0.
    audio = _apply_multiband(track.audio, cfg) if (cfg.compress and cfg.s > 0) else track.audio

    measured = audio_io.measure_loudnorm(audio, cfg.sr, pre_filters=None,
                                         lufs=cfg.lufs, true_peak=cfg.true_peak_db)
    # Below loudnorm's -70 LUFS gate (silent / near-silent program) ffmpeg returns
    # -inf/inf, which the second pass rejects ("out of range"). Emit the (silent)
    # mix instead of crashing the whole episode at the final step.
    if not (_finite(measured.get("input_i")) and _finite(measured.get("input_tp"))
            and _finite(measured.get("target_offset"))):
        log.warning("master: program is silent/near-silent (input_i=%s) — skipping loudnorm",
                    measured.get("input_i"))
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
    # loudnorm -> true-peak limiter at the working rate; then one soxr resample.
    mastered = audio_io.filter_array(audio, cfg.sr, f"{loudnorm},{_limiter(cfg)}")
    audio_io.encode(mastered, cfg.sr, out_path, out_sr=cfg.out_sr, lossy_bitrate=cfg.lossy_bitrate)
