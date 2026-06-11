"""De-hum / de-buzz: detect mains hum and surgically notch its harmonics.

Steady 50/60 Hz mains hum and its harmonics (100/120, 150/180, …) are one of the
most instantly-noticeable amateur tells, and the 2-pole 80 Hz high-pass in repair
only dents the fundamental while the neural denoiser (trained on broadband noise)
does not reliably kill a stationary tonal comb. This stage finds the mains
fundamental from the long-term spectrum, then applies narrow zero-phase notches
(`iirnotch` via `sosfiltfilt`) at each harmonic that actually protrudes above the
local spectral floor — so a clean track gets zero notches and the voice between
harmonics is untouched.

Runs early (after repair, before align) so the stationary tones don't bias the
GCC-PHAT alignment, get smeared by WPE, or feed the denoiser's noise estimate.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import iirnotch, sosfiltfilt, tf2sos, welch

from ..config import Config
from ..session import Track

log = logging.getLogger(__name__)

_NOTCH_Q = 30.0          # narrow — a few Hz wide, so voice harmonics survive
_NPERSEG = 1 << 15       # ~1.5 Hz bins at 48 kHz, enough to resolve 50 vs 60
_MAINS = (50.0, 60.0)
_MAX_HARMONIC_HZ = 1500.0  # low-order harmonics dominate; above this, leave it


def _local_floor_db(p_db: np.ndarray, idx: int, half: int = 16) -> float:
    """Median level of the bins around idx, excluding the ±1-bin peak itself."""
    lo, hi = max(0, idx - half), min(len(p_db), idx + half + 1)
    neighbourhood = np.concatenate([p_db[lo:max(lo, idx - 1)], p_db[idx + 2:hi]])
    return float(np.median(neighbourhood)) if len(neighbourhood) else float(p_db[idx])


def detect_mains(f: np.ndarray, p_db: np.ndarray, margin_db: float) -> float | None:
    """Return 50.0 or 60.0 if a sharp tone in 45–65 Hz clears the floor, else None."""
    band = np.where((f >= 45.0) & (f <= 65.0))[0]
    if not len(band):
        return None
    pk = band[int(np.argmax(p_db[band]))]
    if p_db[pk] - _local_floor_db(p_db, pk) < margin_db:
        return None
    return min(_MAINS, key=lambda m: abs(f[pk] - m))


def find_hum_harmonics(f: np.ndarray, p_db: np.ndarray, f0: float, sr: int,
                       max_harmonics: int, margin_db: float) -> list[float]:
    """Harmonic frequencies (k·f0) whose bin protrudes above the local floor."""
    nyq = sr / 2.0
    freqs: list[float] = []
    for k in range(1, max_harmonics + 1):
        fc = k * f0
        if fc > _MAX_HARMONIC_HZ or fc >= nyq * 0.95:
            break
        idx = int(np.argmin(np.abs(f - fc)))
        # The fundamental already cleared detection; gate the rest on protrusion.
        if k == 1 or p_db[idx] - _local_floor_db(p_db, idx) >= margin_db:
            freqs.append(fc)
    return freqs


def dehum_track(track: Track, cfg: Config) -> Track:
    sr = cfg.sr
    max_h = cfg.dehum_max_harmonics()
    if max_h < 1 or len(track.audio) < _NPERSEG:
        return track
    margin = cfg.dehum_margin_db()
    f, psd = welch(track.audio.astype(np.float64), fs=sr,
                   nperseg=min(_NPERSEG, len(track.audio)))
    p_db = 10.0 * np.log10(psd + 1e-20)

    f0 = detect_mains(f, p_db, margin)
    if f0 is None:
        log.info("dehum: %s — no mains hum detected", track.name)
        return track
    freqs = find_hum_harmonics(f, p_db, f0, sr, max_h, margin)
    if not freqs:
        return track

    sos = np.vstack([tf2sos(*iirnotch(fc, _NOTCH_Q, fs=sr)) for fc in freqs])
    out = sosfiltfilt(sos.astype(np.float64),
                      track.audio.astype(np.float64)).astype(np.float32)
    log.info("dehum: %s — %.0f Hz mains, notched %d harmonic(s) up to %.0f Hz",
             track.name, f0, len(freqs), freqs[-1])
    return Track(track.name, out)
