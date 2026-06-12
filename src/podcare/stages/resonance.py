"""Dynamic resonance / harshness suppression ("Soothe-lite").

The static tonal-balance EQ cannot catch resonant peaks that come and go with
the voice — ringy room modes, nasal honk, 2-5 kHz harshness spikes — a major
cause of earbud fatigue. Per STFT frame, a median filter across frequency
estimates the broad spectral envelope; any bin that pokes more than a margin
above its own envelope is pulled back down by the excess (capped), with
attack/release smoothing across time so notches fade in and out musically.
Cut-only, narrow-band, bounded — at strength 0 the cap is 0 dB, a bitwise
no-op (and the stage is skipped anyway).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import istft, stft

from ..config import Config
from ..dsp import process_chunked
from ..session import Track

log = logging.getLogger(__name__)

_NFFT = 1024
_HOP = 256
_LO_HZ = 800.0      # leave the voice fundamental / warmth region alone
_HI_HZ = 9000.0     # above this is air, handled by de-ess / tonal balance
_ENV_BINS = 9       # ~420 Hz median window @48k/1024: bridges a resonant peak
_ATTACK_S = 0.005   # cut engages fast enough to catch a transient ring
_RELEASE_S = 0.080  # ...and lets go gently so notches never flutter


def _suppress_chunk(audio: np.ndarray, cfg: Config) -> np.ndarray:
    margin = cfg.resonance_margin_db()
    max_cut = cfg.resonance_max_cut_db()
    if max_cut <= 0.0:
        return audio
    if len(audio) < _NFFT:
        # Shorter than one analysis frame: scipy would auto-shrink nperseg and
        # clash with the fixed noverlap — and no resonance is resolvable anyway.
        return audio
    f, _, z = stft(audio.astype(np.float64), fs=cfg.sr, nperseg=_NFFT,
                   noverlap=_NFFT - _HOP, padded=True)
    mag_db = 20.0 * np.log10(np.abs(z) + 1e-12)
    env_db = median_filter(mag_db, size=(_ENV_BINS, 1), mode="nearest")
    cut = np.clip(mag_db - env_db - margin, 0.0, max_cut)
    cut[~((f >= _LO_HZ) & (f <= _HI_HZ)), :] = 0.0
    # Asymmetric one-pole smoothing along time (attack = cut rising).
    frame_s = _HOP / cfg.sr
    a_att = float(np.exp(-frame_s / _ATTACK_S))
    a_rel = float(np.exp(-frame_s / _RELEASE_S))
    smoothed = np.empty_like(cut)
    prev = np.zeros(cut.shape[0])
    for j in range(cut.shape[1]):
        coef = np.where(cut[:, j] > prev, a_att, a_rel)
        prev = coef * prev + (1.0 - coef) * cut[:, j]
        smoothed[:, j] = prev
    z *= 10.0 ** (-smoothed / 20.0)
    _, out = istft(z, fs=cfg.sr, nperseg=_NFFT, noverlap=_NFFT - _HOP)
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    return out[: len(audio)].astype(np.float32)


def resonance_track(track: Track, cfg: Config) -> Track:
    if cfg.resonance_max_cut_db() <= 0.0:
        return track
    audio = process_chunked(track.audio, cfg.sr, lambda c: _suppress_chunk(c, cfg),
                            chunk_s=30.0, label=f"resonance · {track.name}")
    before = float(np.mean(track.audio.astype(np.float64) ** 2)) + 1e-20
    after = float(np.mean(audio.astype(np.float64) ** 2)) + 1e-20
    log.info("resonance: %s — %+.2f dB energy change",
             track.name, 10 * np.log10(after / before))
    return Track(track.name, audio)
