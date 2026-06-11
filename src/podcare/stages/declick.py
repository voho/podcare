"""Mouth-click / de-crackle: remove short mid-band transients (lip smacks,
tongue clicks, saliva crackle) that ffmpeg `adeclick` (vinyl/digital impulses)
and the neural denoiser leave intact.

The cleaner the rest of the pipeline gets, the more audible these become, because
compression and loudnorm lift the quiet inter-word detail. A mouth click is a
very short (~1–3 STFT frames) mid-band energy spike with a high crest factor over
a quiet local background, and — crucially — it sits in a *quiet* neighbourhood
(between or inside words), not under loud speech where it would be masked anyway.

Detection is therefore gated three ways to protect real consonants (t/k/p, s/sh):
a high crest over a robust local median, a requirement that the frame be a local
energy peak, and a requirement that the broadband level there be well below the
speech reference. Flagged frames have their ≥1.5 kHz bins ducked toward the local
baseline, feathered ±1 frame. Chunked like the other spectral stages.
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

_NPERSEG = 256
_NOVERLAP = 192          # hop 64 samples ≈ 1.3 ms at 48 kHz — fine time resolution
_CHUNK_S = 60.0
_LO_HZ = 1500.0          # clicks are mid-band weighted; protect the voice fundamental
_HI_HZ = 6000.0
_BASE_FRAMES = 201       # ~250 ms local-median window for the baseline
_TARGET_MULT = 2.0       # duck a click down to ~2× the local baseline
_QUIET_FRAC = 0.35       # only de-click where broadband level < 35% of speech ref


def _declick_chunk(audio: np.ndarray, cfg: Config, sr: int) -> tuple[np.ndarray, int]:
    if len(audio) < _NPERSEG * 4:
        return audio, 0
    f, _, z = stft(audio, fs=sr, nperseg=_NPERSEG, noverlap=_NOVERLAP)
    mid = (f >= _LO_HZ) & (f <= _HI_HZ)
    if not mid.any():
        return audio, 0

    mag2 = np.abs(z) ** 2
    e = mag2[mid].sum(axis=0)                 # mid-band energy per frame
    total = mag2.sum(axis=0) + 1e-20          # broadband energy per frame
    base = median_filter(e, size=_BASE_FRAMES) + 1e-20
    crest = e / base
    speech_ref = float(np.percentile(total, 90.0)) + 1e-20
    # Local broadband baseline (median is robust to the 1-frame click spike), so
    # "quiet neighbourhood" means the surrounding region — not the click itself —
    # is below the speech reference. A click is a loud spike *in* a quiet gap.
    base_total = median_filter(total, size=_BASE_FRAMES)

    is_peak = np.empty_like(e, dtype=bool)
    is_peak[1:-1] = (e[1:-1] > e[:-2]) & (e[1:-1] > e[2:])
    is_peak[0] = is_peak[-1] = False
    quiet = base_total < _QUIET_FRAC * speech_ref  # not inside loud speech
    click = (crest > cfg.declick_crest()) & is_peak & quiet
    if not click.any():
        return audio, 0

    target = base * _TARGET_MULT
    gain = np.ones(z.shape[1])
    gain[click] = np.sqrt(np.clip(target[click] / e[click], 0.0, 1.0))
    g_prev = np.concatenate([[1.0], gain[:-1]])
    g_next = np.concatenate([gain[1:], [1.0]])
    gain = np.minimum.reduce([g_prev, gain, g_next])

    z[f >= _LO_HZ] *= gain[np.newaxis, :]
    _, out = istft(z, fs=sr, nperseg=_NPERSEG, noverlap=_NOVERLAP)
    out = out.astype(np.float32)
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    return out[: len(audio)], int(click.sum())


def declick_track(track: Track, cfg: Config) -> Track:
    sr = cfg.sr
    counts: list[int] = []

    def run_chunk(chunk: np.ndarray) -> np.ndarray:
        out, n = _declick_chunk(chunk, cfg, sr)
        counts.append(n)
        return out

    audio = process_chunked(track.audio, sr, run_chunk, chunk_s=_CHUNK_S)
    n = sum(counts)
    if n:
        log.info("declick: %s — removed %d click frame(s)", track.name, n)
    return Track(track.name, audio)
