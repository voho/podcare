"""Breath control: detect and gently duck audible inhale breaths between phrases.

The expander gate only acts below its threshold, and close-mic breaths usually
sit *above* it, so they survive; pause tightening removes dead air, not breaths.
After compression and loudnorm push the quiet detail forward, those inhales become
a prominent amateur tell on earbuds.

Breaths are detected by what they are, not by level alone: short segments that are
**unvoiced** (high zero-crossing rate, no low fundamental), **audible** (above the
noise floor) but **below speech level** (which protects in-word fricatives, that
sit at full speech level), and of breath-like duration. Flagged spans are
**ducked** (never muted) by a capped amount with smooth ramps, so cadence
survives. Runs per track after the gate and before any timeline edit.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfilt

from ..config import Config
from ..dsp import (block_rms, db_to_lin, gain_to_samples, smooth_gain)
from ..session import Track

log = logging.getLogger(__name__)

_BLOCK_S = 0.02
_MIN_BREATH_S = 0.08
_MAX_BREATH_S = 0.7
_VOICED_ZCR = 0.03         # blocks below this are voiced (have a pitch)
_BREATH_MAX_REL_DB = -6.0  # breaths sit below loud speech; fricatives do not
_AUDIBLE_FLOOR_MULT = 2.0  # must clear 2× the noise floor to be worth ducking
_ZCR_MIN = 0.05            # per-segment unvoiced confirmation
_VOICED_MAX = 0.5          # low-band / mid-band energy ratio; breaths are unvoiced


def _block_zcr(audio: np.ndarray, hop: int, n_blocks: int) -> np.ndarray:
    cross = np.abs(np.diff(np.sign(audio))) / 2.0
    cross = np.concatenate([cross, [0.0]])[: n_blocks * hop].reshape(n_blocks, hop)
    return cross.mean(axis=1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a boolean array as (start, end) block indices."""
    padded = np.concatenate([[False], mask, [False]])
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(changes[::2], changes[1::2]))


def _is_breath(seg: np.ndarray, sr: int) -> bool:
    """Spectral confirmation: audible, unvoiced, mid-band-dominated => breath."""
    if len(seg) < sr // 50:
        return False
    x = seg.astype(np.float64)
    zcr = float(np.mean(np.abs(np.diff(np.sign(x)))) / 2.0)
    lo = sosfilt(butter(2, 300, btype="lowpass", fs=sr, output="sos"), x)
    mid = sosfilt(butter(2, [500, 4000], btype="bandpass", fs=sr, output="sos"), x)
    voiced_ratio = (np.mean(lo ** 2) + 1e-12) / (np.mean(mid ** 2) + 1e-12)
    return zcr > _ZCR_MIN and voiced_ratio < _VOICED_MAX


def breath_track(track: Track, cfg: Config) -> Track:
    sr = cfg.sr
    depth = cfg.breath_depth_db()
    hop = int(_BLOCK_S * sr)
    if depth <= 0.0 or len(track.audio) < hop * 4:
        return track
    rms = block_rms(track.audio, hop)
    n_blocks = len(track.audio) // hop
    if n_blocks != len(rms):  # short-audio fast path in block_rms — skip
        return track
    noise_floor = float(np.percentile(rms, 10.0))
    speech_ref = float(np.percentile(rms, 90.0))
    if speech_ref <= 0.0:
        return track

    zcr = _block_zcr(track.audio, hop, n_blocks)
    candidate = ((zcr >= _VOICED_ZCR)                          # unvoiced
                 & (rms > _AUDIBLE_FLOOR_MULT * noise_floor)   # audible
                 & (rms < speech_ref * db_to_lin(_BREATH_MAX_REL_DB)))  # below speech

    gains = np.ones_like(rms)
    n = 0
    duck = db_to_lin(-depth)
    for i0, i1 in _runs(candidate):
        if not (_MIN_BREATH_S <= (i1 - i0) * _BLOCK_S <= _MAX_BREATH_S):
            continue
        if not _is_breath(track.audio[i0 * hop:i1 * hop], sr):
            continue
        gains[i0:i1] = duck
        n += 1
    if n == 0:
        return track

    gains = smooth_gain(gains, attack_blocks=2.0, release_blocks=2.0)
    env = gain_to_samples(gains, hop, len(track.audio))
    log.info("breath: %s — ducked %d breath(s) by %.0f dB", track.name, n, depth)
    return Track(track.name, (track.audio * env).astype(np.float32))
