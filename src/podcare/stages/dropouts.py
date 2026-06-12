"""Dropout / short-gap restoration: fill brief packet-loss holes via two-sided LPC.

Remote-guest recordings (VoIP and double-enders over flaky links) arrive with
3-50 ms holes where packets were lost — audible as tiny stutters. Each hole is
refilled by linear prediction: extrapolate the speech forward from the audio
before the gap and backward from the audio after it, then equal-power
crossfade the two estimates across the gap. Strict caps keep it honest: only
short gaps (strength-scaled, <= 50 ms), only gaps whose surroundings are
speech-active (a quiet moment inside a real pause is not a dropout), and no
more than ~1.2 s filled per minute — beyond that the track is corrupt, not
packet-lossy, and fabricating more would do harm.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..dsp import block_rms, db_to_lin
from ..session import Track

log = logging.getLogger(__name__)

_BLOCK_S = 0.003           # detection resolution: 3 ms RMS blocks
_DROP_DB = 25.0            # a dropout sits >= this far below its local context
_CONTEXT_S = 0.3           # local window for the context reference level
_LPC_ORDER = 96            # deliberately above the design-doc ~32: higher order gives
                           # better zero-crossing phase resolution for clean context fits
_LPC_CONTEXT_S = 0.08      # deliberately above the design-doc ~30 ms: gives a full clean
                           # window after _trim_clean removes any trailing zeros at the seam
_EDGE_XF_S = 0.002         # fade the fill into the real audio at the seams
_MAX_FILL_PER_MIN_S = 1.2  # honesty cap: ~2% of any minute


def _find_dropouts(audio: np.ndarray, sr: int, max_gap_s: float) -> list[tuple[int, int]]:
    """(start, end) sample intervals of fillable dropouts, earliest first."""
    hop = max(1, int(_BLOCK_S * sr))
    rms = block_rms(audio, hop)
    if len(rms) < 8:
        return []
    db = 20.0 * np.log10(rms.astype(np.float64) + 1e-12)
    ctx = max(1, int(_CONTEXT_S / _BLOCK_S))
    # Speech-activity threshold for the pre/post context check: a block is
    # "speech-active" if its RMS is above this level (so a dip inside a real
    # pause — where context is also quiet — is rejected). We use the relative-
    # to-loud-speech and absolute-minimum components of speech_threshold, but
    # NOT the floor*floor_factor component: on uniform-level signals (e.g. a
    # pure tone) the 10th-percentile floor can land mid-speech and the 3×
    # multiplier then exceeds the actual speech level, causing false misses.
    speech_rms = max(float(np.percentile(rms, 90.0)) * db_to_lin(-28.0),
                     db_to_lin(-55.0))
    max_blocks = max(0, int(round(max_gap_s / _BLOCK_S)))
    out: list[tuple[int, int]] = []

    # Vectorized pre-filter: only visit blocks that are plausibly dropout-like.
    # Two cheap criteria, unioned: (a) below a global speech-level reference
    # (p90) minus the drop margin — p90 rather than median so long silences
    # can't drag the reference down; (b) below an absolute near-silence level,
    # which catches every true packet dropout (zeros decode to ~-200 dBFS)
    # even on sparse recordings where <10% speech leaves p90 in the quiet zone.
    # The per-candidate local-context check below remains the authoritative
    # test; this union only has to be a superset of what it would flag.
    global_ref = float(np.percentile(db, 90.0))
    candidates = np.flatnonzero((db <= global_ref - _DROP_DB) | (db <= -65.0))

    k = 0  # index into candidates array
    while k < len(candidates):
        i = int(candidates[k])
        lo, hi = max(0, i - ctx), min(len(db), i + ctx)
        context = np.concatenate([db[lo:i], db[i + 1: hi]])
        ref = float(np.median(context)) if len(context) else -120.0
        if db[i] > ref - _DROP_DB:
            # Global filter passed but local context says no — skip.
            k += 1
            continue
        # Expand the run of contiguous low-level blocks (same semantics as the
        # original linear scan: use THIS block's local ref for the expansion).
        j = i
        while j < len(db) and db[j] <= ref - _DROP_DB:
            j += 1
        # Advance k past all candidates that fall inside [i, j).
        while k < len(candidates) and candidates[k] < j:
            k += 1
        # Fill only if: short enough, and speech-active on BOTH sides (so the
        # level genuinely collapses and recovers — a dip inside a real pause
        # has quiet context and is excluded).
        pre = rms[max(0, i - ctx): i]
        post = rms[j: j + ctx]
        if (1 <= (j - i) <= max_blocks
                and len(pre) and len(post)
                and float(np.median(pre)) > speech_rms
                and float(np.median(post)) > speech_rms):
            out.append((i * hop, min(len(audio), j * hop)))
    return out


def _lpc_coeffs(x: np.ndarray, order: int) -> np.ndarray:
    """Levinson-Durbin forward predictor: x[n] ~= sum(a[k] * x[n-1-k])."""
    x = x.astype(np.float64)
    n = len(x)
    if n <= order + 1:
        return np.zeros(order)
    r = np.correlate(x, x, mode="full")[n - 1: n + order]
    if r[0] <= 1e-12:
        return np.zeros(order)
    a = np.zeros(order)
    e = float(r[0])
    for k in range(order):
        acc = r[k + 1] - float(np.dot(a[:k], r[1: k + 1][::-1]))
        ref = acc / e
        new_a = a.copy()
        new_a[k] = ref
        new_a[:k] = a[:k] - ref * a[:k][::-1]
        a = new_a
        e *= (1.0 - ref * ref)
        if e <= 1e-12:
            break
    return a


def _extrapolate(context: np.ndarray, n_out: int, order: int) -> np.ndarray:
    """Continue `context` for n_out samples with a one-step LPC predictor."""
    a = _lpc_coeffs(context, order)
    if not np.any(a):
        return np.zeros(n_out)
    hist = list(context.astype(np.float64)[-order:])
    out = np.empty(n_out)
    for i in range(n_out):
        nxt = float(np.dot(a, hist[::-1]))
        nxt = float(np.clip(nxt, -4.0, 4.0))  # bound a marginally unstable filter
        out[i] = nxt
        hist.pop(0)
        hist.append(nxt)
    return out


def _trim_clean(arr: np.ndarray, *, from_end: bool = True,
                threshold: float = 1e-7) -> np.ndarray:
    """Drop near-zero samples from the tail (or head) of arr.

    Detection boundaries often land a block *inside* the zeroed region, so the
    last few samples of `pre` (or first few of `post`) are from the hole itself.
    Trimming them prevents the LPC history from being seeded with zeros, which
    would otherwise produce a near-silent extrapolation even when the predictor
    has good frequency-resolution coefficients.

    Note: aggressive trimming can shrink one side below the usable minimum
    (``_LPC_ORDER * 2`` samples), which silently disables that side's
    extrapolation; the other side still covers the gap via its own estimate.
    """
    nz = np.flatnonzero(np.abs(arr) > threshold)
    if not len(nz):
        return arr
    return arr[:nz[-1] + 1] if from_end else arr[nz[0]:]


def _fill_gap(audio: np.ndarray, sr: int, start: int, end: int) -> None:
    """Replace audio[start:end] in place with a two-sided LPC estimate."""
    n = end - start
    ctx = int(_LPC_CONTEXT_S * sr)
    pre = _trim_clean(audio[max(0, start - ctx): start], from_end=True)
    post = _trim_clean(audio[end: end + ctx], from_end=False)
    fwd = _extrapolate(pre, n, _LPC_ORDER) if len(pre) > _LPC_ORDER * 2 else np.zeros(n)
    bwd = (_extrapolate(post[::-1], n, _LPC_ORDER)[::-1]
           if len(post) > _LPC_ORDER * 2 else np.zeros(n))
    t = np.linspace(0.0, np.pi / 2.0, n)
    fill = fwd * np.cos(t) + bwd * np.sin(t)
    xf = min(max(2, int(_EDGE_XF_S * sr)), n // 2)
    if xf >= 2:  # feather the seams so the splice never clicks
        w = np.linspace(0.0, 1.0, xf)
        fill[:xf] = fill[:xf] * w + audio[start: start + xf].astype(np.float64) * (1.0 - w)
        fill[-xf:] = fill[-xf:] * (1.0 - w) + audio[end - xf: end].astype(np.float64) * w
    audio[start:end] = fill.astype(np.float32)


def restore_dropouts_track(track: Track, cfg: Config) -> Track:
    max_gap_s = cfg.dropout_max_gap_ms() / 1000.0
    if max_gap_s <= 0:
        return track
    gaps = _find_dropouts(track.audio, cfg.sr, max_gap_s)
    if not gaps:
        log.info("dropouts: %s — none detected", track.name)
        return track
    audio = track.audio.copy()
    budget_s = _MAX_FILL_PER_MIN_S * (len(audio) / cfg.sr / 60.0)
    filled_s, n_filled = 0.0, 0
    for start, end in gaps:
        dur = (end - start) / cfg.sr
        if filled_s + dur > budget_s:
            log.warning("dropouts: %s — fill budget (%.1fs) reached, %d gap(s) left "
                        "unfilled (track may be corrupt rather than packet-lossy)",
                        track.name, budget_s, len(gaps) - n_filled)
            break
        _fill_gap(audio, cfg.sr, start, end)
        filled_s += dur
        n_filled += 1
    if n_filled == 0:
        return track
    log.info("dropouts: %s — filled %d gap(s), %.0f ms total",
             track.name, n_filled, filled_s * 1000)
    return Track(track.name, audio)
