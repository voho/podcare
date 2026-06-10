"""Plosive ("p-pop") ducking: attenuate low-frequency burst frames in the STFT."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import istft, stft

from ..config import Config
from ..session import Track

log = logging.getLogger(__name__)

_NPERSEG = 1024
_NOVERLAP = 768


def deplosive_track(track: Track, cfg: Config) -> Track:
    sr = cfg.sr
    if len(track.audio) < _NPERSEG * 4:
        return track
    freqs, _, z = stft(track.audio, fs=sr, nperseg=_NPERSEG, noverlap=_NOVERLAP)
    lf = freqs <= cfg.plosive_max_hz
    if not lf.any():
        return track

    lf_energy = (np.abs(z[lf]) ** 2).sum(axis=0)
    total_energy = (np.abs(z) ** 2).sum(axis=0) + 1e-20
    med = float(np.median(lf_energy)) + 1e-20

    # A plosive frame has an abnormal LF burst that also dominates the spectrum.
    burst = (lf_energy > cfg.plosive_burst_mult() * med) & \
        (lf_energy / total_energy > cfg.plosive_dominance())
    if not burst.any():
        return track

    # Duck flagged LF bins back toward the typical LF level; spread one frame
    # each side so the gain ramps instead of stepping.
    target = cfg.plosive_target_mult() * med
    gain = np.ones(z.shape[1])
    gain[burst] = np.sqrt(target / lf_energy[burst])
    # Feather ±1 frame, padding edges with the identity gain so a first/last-frame
    # burst doesn't wrap its duck onto the opposite edge of the track.
    g_prev = np.concatenate([[1.0], gain[:-1]])
    g_next = np.concatenate([gain[1:], [1.0]])
    gain = np.minimum.reduce([g_prev, gain, g_next])
    z[lf] *= gain[np.newaxis, :]

    _, out = istft(z, fs=sr, nperseg=_NPERSEG, noverlap=_NOVERLAP)
    out = out.astype(np.float32)
    if len(out) < len(track.audio):
        out = np.pad(out, (0, len(track.audio) - len(out)))
    n_burst = int(burst.sum())
    log.info("plosives: %s — ducked %d frames", track.name, n_burst)
    return Track(track.name, out[: len(track.audio)])
