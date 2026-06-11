"""Plosive ("p-pop") ducking: attenuate low-frequency burst frames in the STFT."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import istft, stft

from ..config import Config
from ..dsp import process_chunked
from ..session import Track

log = logging.getLogger(__name__)

_NPERSEG = 1024
_NOVERLAP = 768
# Process in chunks so the full-track STFT (a complex spectrogram that grows
# linearly with length) can never OOM a multi-hour render. Detection is
# frame-local — burst flagging is relative to each chunk's own LF-energy median,
# which over a 60 s speech chunk is statistically the same as the whole-track
# median — and the crossfade in process_chunked hides any seam.
_CHUNK_S = 60.0


def _deplosive_chunk(audio: np.ndarray, cfg: Config, sr: int) -> tuple[np.ndarray, int]:
    """Duck plosive bursts in one chunk; returns (same-length audio, frames ducked)."""
    if len(audio) < _NPERSEG * 4:
        return audio, 0
    freqs, _, z = stft(audio, fs=sr, nperseg=_NPERSEG, noverlap=_NOVERLAP)
    lf = freqs <= cfg.plosive_max_hz
    if not lf.any():
        return audio, 0

    lf_energy = (np.abs(z[lf]) ** 2).sum(axis=0)
    total_energy = (np.abs(z) ** 2).sum(axis=0) + 1e-20
    med = float(np.median(lf_energy)) + 1e-20

    # A plosive frame has an abnormal LF burst that also dominates the spectrum.
    burst = (lf_energy > cfg.plosive_burst_mult() * med) & \
        (lf_energy / total_energy > cfg.plosive_dominance())
    if not burst.any():
        return audio, 0

    # Duck flagged LF bins back toward the typical LF level; spread one frame
    # each side so the gain ramps instead of stepping.
    target = cfg.plosive_target_mult() * med
    gain = np.ones(z.shape[1])
    gain[burst] = np.sqrt(target / lf_energy[burst])
    # Feather ±1 frame, padding edges with the identity gain so a first/last-frame
    # burst doesn't wrap its duck onto the opposite edge of the chunk.
    g_prev = np.concatenate([[1.0], gain[:-1]])
    g_next = np.concatenate([gain[1:], [1.0]])
    gain = np.minimum.reduce([g_prev, gain, g_next])
    z[lf] *= gain[np.newaxis, :]

    _, out = istft(z, fs=sr, nperseg=_NPERSEG, noverlap=_NOVERLAP)
    out = out.astype(np.float32)
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    return out[: len(audio)], int(burst.sum())


def deplosive_track(track: Track, cfg: Config) -> Track:
    sr = cfg.sr
    counts: list[int] = []

    def run_chunk(chunk: np.ndarray) -> np.ndarray:
        out, n = _deplosive_chunk(chunk, cfg, sr)
        counts.append(n)
        return out

    audio = process_chunked(track.audio, sr, run_chunk, chunk_s=_CHUNK_S)
    n_burst = sum(counts)
    if n_burst:
        log.info("plosives: %s — ducked %d frames", track.name, n_burst)
    return Track(track.name, audio)
