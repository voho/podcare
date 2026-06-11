"""Tonal-balance EQ: nudge each mic's long-term speech spectrum toward a
balanced broadcast-voice curve.

The chain has no spectral shaping otherwise, so a dull lavalier and a bright
condenser stay timbrally mismatched after all the cleanup. This stage measures
each track's long-term average spectrum (LTAS) over *speech-active* frames only,
compares it to a fixed produced-voice target curve, and applies a gentle,
**broad** correction — a low/high shelf plus a low-mid and a presence bell — to
pull every mic toward one curve. Only the shape is matched (both curves are
normalized to the speech body first), so it never changes overall loudness.

Deliberately conservative: corrections are broad (4 wide filters, not surgical
notches), boosts are clamped tighter than cuts (a noisy band is never lifted),
and only a fraction of the measured deviation is applied — `strength / 3`, so it
stays subtle even at full strength. Realized as minimal-phase RBJ biquads via
``sosfilt`` (length-preserving, zero added latency).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import sosfilt

from ..config import Config
from ..dsp import block_rms, speech_threshold
from ..session import Track

log = logging.getLogger(__name__)

# Standard 1/3-octave analysis centers (Hz).
_GRID = np.array([50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
                  1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                  10000, 12500, 16000], dtype=np.float64)

# Produced broadcast-voice target LTAS (relative dB vs frequency). Derived from
# the international long-term-average speech spectrum with a mild "produced"
# tweak — a touch less low-mid boom and a gentle 2–4 kHz presence lift for
# intelligibility. Absolute level is irrelevant; only the shape is matched.
_TARGET_HZ = np.array([60, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000,
                       6300, 10000, 16000], dtype=np.float64)
_TARGET_DB = np.array([-18, -7, 2, 5, 4, 2, 0, -3, -5, -9, -15, -21, -28],
                      dtype=np.float64)

# Body band used to normalize both curves (the stable core of voice energy).
_ANCHOR = (_GRID >= 400) & (_GRID <= 1500)

# Broad correction filters: (name, kind, f0, Q, averaging band in Hz).
_BANDS = [
    ("low", "lowshelf", 120.0, None, (0.0, 150.0)),
    ("low-mid", "peak", 300.0, 1.0, (200.0, 450.0)),
    ("presence", "peak", 3000.0, 0.9, (2200.0, 4500.0)),
    ("air", "highshelf", 7000.0, None, (6000.0, 24000.0)),
]

_MAX_BOOST_DB = 3.0   # boosts clamped tighter than cuts — never lift a noisy band
_MAX_CUT_DB = 6.0
_MIN_APPLY_DB = 0.3   # skip a filter (and the whole stage) below this
_MIN_SPEECH_S = 1.0   # need this much speech to trust the LTAS


def _peaking(f0: float, q: float, gain_db: float, fs: int) -> list[float]:
    """RBJ peaking-EQ biquad as an SOS row."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * q)
    cw = np.cos(w0)
    b0, b1, b2 = 1 + alpha * a, -2 * cw, 1 - alpha * a
    a0, a1, a2 = 1 + alpha / a, -2 * cw, 1 - alpha / a
    return [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]


def _shelf(f0: float, gain_db: float, fs: int, *, high: bool) -> list[float]:
    """RBJ low/high-shelf biquad (slope S=1) as an SOS row."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2.0 * np.sqrt(2.0)
    sa = np.sqrt(a)
    if high:
        b0 = a * ((a + 1) + (a - 1) * cw + 2 * sa * alpha)
        b1 = -2 * a * ((a - 1) + (a + 1) * cw)
        b2 = a * ((a + 1) + (a - 1) * cw - 2 * sa * alpha)
        a0 = (a + 1) - (a - 1) * cw + 2 * sa * alpha
        a1 = 2 * ((a - 1) - (a + 1) * cw)
        a2 = (a + 1) - (a - 1) * cw - 2 * sa * alpha
    else:
        b0 = a * ((a + 1) - (a - 1) * cw + 2 * sa * alpha)
        b1 = 2 * a * ((a - 1) - (a + 1) * cw)
        b2 = a * ((a + 1) - (a - 1) * cw - 2 * sa * alpha)
        a0 = (a + 1) + (a - 1) * cw + 2 * sa * alpha
        a1 = -2 * ((a - 1) + (a + 1) * cw)
        a2 = (a + 1) + (a - 1) * cw - 2 * sa * alpha
    return [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]


def _target_on_grid() -> np.ndarray:
    return np.interp(np.log(_GRID), np.log(_TARGET_HZ), _TARGET_DB)


def tonal_correction(meas_db: np.ndarray, fraction: float
                     ) -> list[tuple[str, str, float, float | None, float]]:
    """Reduce a measured LTAS (dB on _GRID) to broad filter gains.

    Pure function: returns (name, kind, f0, Q, gain_db) per band. Both target and
    measured curves are normalized to the speech body, so only spectral *shape*
    drives the correction; `fraction` scales the applied deviation and gains are
    clamped (boosts tighter than cuts).
    """
    tgt = _target_on_grid()
    dev = (tgt - tgt[_ANCHOR].mean()) - (meas_db - meas_db[_ANCHOR].mean())
    specs: list[tuple[str, str, float, float | None, float]] = []
    for name, kind, f0, q, (lo, hi) in _BANDS:
        region = (_GRID >= lo) & (_GRID <= hi)
        if not region.any():
            continue
        g = float(np.mean(dev[region])) * fraction
        g = float(np.clip(g, -_MAX_CUT_DB, _MAX_BOOST_DB))
        specs.append((name, kind, f0, q, g))
    return specs


def _build_sos(specs: list[tuple[str, str, float, float | None, float]],
               fs: int) -> np.ndarray:
    rows = []
    for _name, kind, f0, q, g in specs:
        if kind == "peak":
            rows.append(_peaking(f0, float(q), g, fs))
        else:
            rows.append(_shelf(f0, g, fs, high=(kind == "highshelf")))
    return np.array(rows, dtype=np.float64)


def _measure_ltas(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Long-term average spectrum (dB on _GRID) over speech-active frames only."""
    from scipy.signal import welch

    hop = max(1, int(0.02 * sr))
    rms = block_rms(audio, hop)
    speech = rms >= speech_threshold(rms)
    sample_mask = np.repeat(speech, hop)
    if len(sample_mask) < len(audio):
        sample_mask = np.concatenate(
            [sample_mask, np.zeros(len(audio) - len(sample_mask), dtype=bool)])
    else:
        sample_mask = sample_mask[: len(audio)]
    speech_samples = audio[sample_mask]
    if len(speech_samples) < int(_MIN_SPEECH_S * sr):
        return None
    nperseg = int(min(4096, len(speech_samples)))
    f, psd = welch(speech_samples.astype(np.float64), fs=sr, nperseg=nperseg)
    return np.interp(_GRID, f, 10.0 * np.log10(psd + 1e-12))


def tonebalance_track(track: Track, cfg: Config) -> Track:
    meas = _measure_ltas(track.audio, cfg.sr)
    if meas is None:
        log.info("tonebalance: %s — not enough speech to measure, leaving as-is", track.name)
        return track
    specs = tonal_correction(meas, cfg.eq_correction())
    active = [s for s in specs if abs(s[4]) >= _MIN_APPLY_DB]
    if not active:
        log.info("tonebalance: %s — already balanced, no correction", track.name)
        return track
    out = sosfilt(_build_sos(active, cfg.sr), track.audio).astype(np.float32)
    summary = ", ".join(f"{name} {g:+.1f}dB" for name, _k, _f, _q, g in active)
    log.info("tonebalance: %s — %s (frac %.2f)", track.name, summary, cfg.eq_correction())
    return Track(track.name, out)
