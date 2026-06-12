"""Shared DSP helpers: block envelopes, gain smoothing, crossfaded edits."""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from .progress import active

log = logging.getLogger(__name__)


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _peak_rss_mb() -> float:
    """Process peak resident set size in MB (high-watermark, no dependency).

    ru_maxrss is bytes on macOS/BSD but kibibytes on Linux; normalise both.
    """
    import resource
    import sys

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return rss * scale / (1024.0 * 1024.0)


def block_rms(audio: np.ndarray, hop: int) -> np.ndarray:
    """RMS per non-overlapping block of `hop` samples (tail block ignored)."""
    if len(audio) == 0:
        # Empty input would make np.mean return NaN and poison every downstream
        # threshold; report a single silent block instead.
        return np.full(1, float(np.sqrt(1e-20)), dtype=np.float32)
    n_blocks = len(audio) // hop
    if n_blocks == 0:
        return np.array([np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-20)],
                        dtype=np.float32)
    x = audio[: n_blocks * hop].astype(np.float64).reshape(n_blocks, hop)
    return np.sqrt((x ** 2).mean(axis=1) + 1e-20).astype(np.float32)


def smooth_gain(gains: np.ndarray, attack_blocks: float, release_blocks: float) -> np.ndarray:
    """Asymmetric one-pole smoothing of a block gain curve.

    Attack (gain falling) reacts in ~attack_blocks, release (gain recovering)
    in ~release_blocks, so attenuation is fast and recovery is gradual.
    """
    a_att = float(np.exp(-1.0 / max(attack_blocks, 1e-6)))
    a_rel = float(np.exp(-1.0 / max(release_blocks, 1e-6)))
    out = np.empty_like(gains, dtype=np.float64)
    prev = float(gains[0])
    for i, g in enumerate(gains.astype(np.float64)):
        coef = a_att if g < prev else a_rel
        prev = coef * prev + (1.0 - coef) * g
        out[i] = prev
    return out.astype(np.float32)


def gain_to_samples(gains: np.ndarray, hop: int, n_samples: int) -> np.ndarray:
    """Interpolate block gains (block centers) to a per-sample envelope."""
    if len(gains) == 1:
        return np.full(n_samples, gains[0], dtype=np.float32)
    centers = np.arange(len(gains), dtype=np.float64) * hop + hop / 2.0
    pos = np.arange(n_samples, dtype=np.float64)
    return np.interp(pos, centers, gains.astype(np.float64)).astype(np.float32)


def merge_intervals(intervals: list[tuple[float, float]],
                    min_gap: float = 0.0) -> list[tuple[float, float]]:
    """Sort and merge overlapping/near-touching (start, end) second intervals."""
    if not intervals:
        return []
    ordered = sorted((max(0.0, s), e) for s, e in intervals if e > s)
    merged: list[tuple[float, float]] = []
    for s, e in ordered:
        if merged and s <= merged[-1][1] + min_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def crossfade_concat(segments: list[np.ndarray], xf: int) -> np.ndarray:
    """Join segments with equal-power crossfades of up to `xf` samples.

    Each boundary blends the tail of the running output with the head of the
    next segment over min(xf, len(out), len(seg)) samples using cos/sin
    curves, so correlated material keeps ~constant power across the join.
    Boundaries with fewer than 2 overlap samples are butt-joined. Preserves
    the dtype of segments[0]; the caller keeps segments dtype-consistent.
    """
    segments = [s for s in segments if len(s) > 0]
    if not segments:
        return np.zeros(0, dtype=np.float32)
    out = segments[0].copy()
    for seg in segments[1:]:
        n = min(xf, len(out), len(seg))
        if n >= 2:
            t = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
            out[-n:] = out[-n:] * np.cos(t) + seg[:n] * np.sin(t)
            out = np.concatenate([out, seg[n:]])
        else:
            out = np.concatenate([out, seg])
    return out


def remove_intervals(audio: np.ndarray, sr: int, intervals: list[tuple[float, float]],
                     xfade_s: float = 0.015) -> np.ndarray:
    """Cut (start, end) second intervals out of the audio with equal-power crossfades.

    Intervals must be merged/sorted (see merge_intervals).
    """
    if not intervals:
        return audio
    xf = max(2, int(xfade_s * sr))
    segments: list[np.ndarray] = []
    pos = 0
    for s, e in intervals:
        s_i = max(0, min(len(audio), int(round(s * sr))))
        e_i = max(s_i, min(len(audio), int(round(e * sr))))
        if e_i <= s_i:
            # Interval rounds to under one sample — not a real cut. Skipping it
            # avoids splitting the audio here and spuriously crossfading away
            # ~xfade_s of good signal across a zero-width edit.
            continue
        if s_i > pos:
            segments.append(audio[pos:s_i])
        pos = e_i
    if pos < len(audio):
        segments.append(audio[pos:])
    segments = [s for s in segments if len(s) > 0]
    if not segments:
        return audio[:0]
    return crossfade_concat(segments, xf)


def process_chunked(audio: np.ndarray, sr: int, fn, *, chunk_s: float,
                    overlap_s: float = 1.0, label: str = "") -> np.ndarray:
    """Apply fn(chunk)->chunk over the signal in chunks with crossfaded overlap.

    Keeps memory bounded for whole-episode processing; the linear crossfade in
    the overlap region hides any boundary discontinuity between chunks. `label`
    titles the per-chunk progress sub-bar (no-op unless a live reporter is set).
    """
    chunk = int(chunk_s * sr)
    overlap = int(overlap_s * sr)
    if len(audio) <= chunk + overlap:
        out = fn(audio)
        if len(out) != len(audio):
            raise ValueError(f"chunk fn changed length: {len(audio)} -> {len(out)}")
        return out

    # Chunk count is exact for the loop below; end_sub() clamps the bar to full
    # in case rounding ever leaves it a fraction short.
    n_chunks = math.ceil((len(audio) - chunk - overlap) / chunk) + 1
    reporter = active()
    reporter.begin_sub(n_chunks, "chunk", label)
    try:
        out = np.zeros(len(audio), dtype=np.float64)
        weight = np.zeros(len(audio), dtype=np.float64)
        start = 0
        n_done = 0
        slowest = 0.0
        while start < len(audio):
            end = min(len(audio), start + chunk + overlap)
            t_chunk = time.perf_counter()
            piece = fn(audio[start:end]).astype(np.float64)
            dt_chunk = time.perf_counter() - t_chunk
            if len(piece) != end - start:
                raise ValueError(f"chunk fn changed length: {end - start} -> {len(piece)}")
            n_done += 1
            slowest = max(slowest, dt_chunk)
            # Per-chunk timing + peak RSS: distinguishes "slow because compute"
            # from "slow because memory pressure" when a stage drags (DEBUG so it
            # only surfaces when explicitly wanted; the summary below is at INFO).
            log.debug("%s chunk %d/%d: %.2fs, peak RSS %.0f MB",
                      label or "chunk", n_done, n_chunks, dt_chunk, _peak_rss_mb())
            w = np.ones(len(piece))
            if start > 0:
                n = min(overlap, len(piece))
                w[:n] = np.linspace(0.0, 1.0, n)
            if end < len(audio):
                n = min(overlap, len(piece))
                w[-n:] = np.minimum(w[-n:], np.linspace(1.0, 0.0, n))
            out[start:end] += piece * w
            weight[start:end] += w
            reporter.advance_sub()
            if end == len(audio):
                break
            start += chunk
        if label:
            log.info("%s — %d chunks, slowest %.2fs, peak RSS %.0f MB",
                     label, n_done, slowest, _peak_rss_mb())
        return (out / np.maximum(weight, 1e-9)).astype(np.float32)
    finally:
        reporter.end_sub()


def speech_threshold(rms: np.ndarray, *, floor_percentile: float = 10.0,
                     floor_factor: float = 3.0, min_dbfs: float = -55.0,
                     speech_rel_db: float = -28.0) -> float:
    """Adaptive speech/noise RMS threshold for a block-RMS envelope.

    Takes the strictest of: a multiple of the noise floor, a level relative to
    loud speech (so low-level noise between phrases still counts as silence),
    and an absolute minimum.
    """
    floor = float(np.percentile(rms, floor_percentile))
    speech_ref = float(np.percentile(rms, 90.0))
    return max(floor * floor_factor,
               speech_ref * db_to_lin(speech_rel_db),
               db_to_lin(min_dbfs))
