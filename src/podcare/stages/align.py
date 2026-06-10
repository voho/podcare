"""Inter-track alignment (GCC-PHAT) and polarity correction.

Recorders started at different moments produce offset tracks; mixing them
unaligned smears crosstalk into echo. A miswired or software-inverted mic
produces phase cancellation when mixed. Both are fixed against the first
track as reference, but only when the correlation peak is decisive — remote
(uncorrelated) tracks are left untouched.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import resample_poly

from ..config import Config
from ..session import Session, Track

log = logging.getLogger(__name__)

_COARSE_SR = 8000


def _gcc_phat(a: np.ndarray, b: np.ndarray) -> tuple[int, float, float]:
    """Return (lag, peak_sign, peak_z) maximizing correlation of a and b.

    Positive lag means b is delayed relative to a by `lag` samples.
    """
    n = len(a) + len(b)
    nfft = 1 << int(np.ceil(np.log2(n)))
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b, nfft)
    r = fa * np.conj(fb)
    r /= np.abs(r) + 1e-12
    cc = np.fft.irfft(r, nfft)
    max_lag = min(len(a), len(b)) - 1
    # cc[k] correlates a against b shifted right by k; negative lags wrap around.
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    peak_idx = int(np.argmax(np.abs(cc)))
    abs_cc = np.abs(cc)
    z = float((abs_cc[peak_idx] - abs_cc.mean()) / (abs_cc.std() + 1e-12))
    # irfft(A·conj(B)) peaks at -d when b lags a by d samples; negate so that
    # positive lag = b delayed (verified against synthetic offsets in tests).
    return -int(lags[peak_idx]), float(np.sign(cc[peak_idx])), z


def _refine_lag(ref: np.ndarray, x: np.ndarray, coarse_lag: int, sr: int,
                radius: int) -> tuple[int, float]:
    """Direct cross-correlation around the coarse lag; returns (lag, normalized r)."""
    # Pick the highest-energy 10 s window of the reference for the comparison.
    win = min(10 * sr, len(ref) // 2)
    if win < sr // 10:
        return coarse_lag, 0.0
    hop = sr
    n_pos = max(1, (len(ref) - win) // hop)
    energies = [float(np.sum(ref[i * hop: i * hop + win] ** 2)) for i in range(n_pos)]
    start = int(np.argmax(energies)) * hop
    a = ref[start: start + win].astype(np.float64)

    best_lag, best_r = coarse_lag, 0.0
    a_norm = float(np.sqrt(np.sum(a ** 2)) + 1e-12)
    for lag in range(coarse_lag - radius, coarse_lag + radius + 1):
        # Positive lag = x delayed: x[start+lag:] should line up with ref[start:].
        s = start + lag
        if s < 0 or s + win > len(x):
            continue
        b = x[s: s + win].astype(np.float64)
        r = float(np.dot(a, b) / (a_norm * (np.sqrt(np.sum(b ** 2)) + 1e-12)))
        if abs(r) > abs(best_r):
            best_lag, best_r = lag, r
    return best_lag, best_r


def _apply_offset(track: Track, lag: int) -> Track:
    """Shift a track so it lines up with the reference timeline.

    Positive lag = track is late: drop its first `lag` samples.
    Negative lag = track is early: pad its start.
    """
    if lag > 0:
        audio = track.audio[lag:]
    elif lag < 0:
        audio = np.concatenate([np.zeros(-lag, dtype=np.float32), track.audio])
    else:
        audio = track.audio
    return Track(track.name, audio)


def align_session(session: Session, cfg: Config) -> Session:
    if len(session.tracks) < 2:
        return session
    sr = session.sr
    q = sr // _COARSE_SR
    window = int(cfg.align_window_s * sr)
    ref = session.tracks[0]
    ref_win = ref.audio[:window]
    ref_coarse = resample_poly(ref_win.astype(np.float64), 1, q)

    new_tracks = [ref]
    for track in session.tracks[1:]:
        x_win = track.audio[:window]
        x_coarse = resample_poly(x_win.astype(np.float64), 1, q)
        coarse_lag, _, z = _gcc_phat(ref_coarse, x_coarse)
        if z < cfg.align_min_confidence:
            log.info("align: %s — no confident offset (z=%.1f), leaving as-is", track.name, z)
            new_tracks.append(track)
            continue
        lag, r = _refine_lag(ref.audio, track.audio, coarse_lag * q, sr, radius=2 * q + 8)
        # The PHAT peak can clear the z gate on uncorrelated (remote) tracks; the
        # waveform correlation at the candidate lag is the decisive evidence.
        if abs(r) < 0.08:
            log.info("align: %s — peak z=%.1f not confirmed by waveform correlation "
                     "(r=%+.2f), leaving as-is", track.name, z, r)
            new_tracks.append(track)
            continue
        log.info("align: %s — offset %+0.1f ms (z=%.1f, r=%+.2f)",
                 track.name, 1000 * lag / sr, z, r)
        shifted = _apply_offset(track, lag)
        if r < -0.05:
            log.info("align: %s — inverted polarity detected, flipping", track.name)
            shifted = Track(shifted.name, -shifted.audio)
        new_tracks.append(shifted)

    # Equalize lengths so session-level stages see one timeline.
    max_len = max(t.n_samples for t in new_tracks)
    padded = [
        Track(t.name, np.pad(t.audio, (0, max_len - t.n_samples))) if t.n_samples < max_len else t
        for t in new_tracks
    ]
    return Session(sr, padded)
