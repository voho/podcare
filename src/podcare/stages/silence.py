"""Pause tightening: shorten dead air, trim lead/tail silence."""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..dsp import block_rms, merge_intervals, remove_intervals, speech_threshold
from ..session import Track

log = logging.getLogger(__name__)

_BLOCK_S = 0.010


def find_pause_cuts(speech_mask: np.ndarray, block_s: float, *, max_pause_s: float,
                    target_pause_s: float, lead_trail_s: float) -> list[tuple[float, float]]:
    """Compute cut intervals (seconds) from a per-block speech mask."""
    if not speech_mask.any():
        return []
    # A one-sided override (only --max-pause or only --target-pause, the other
    # following strength) could leave target > max and invert the cut interval.
    target_pause_s = min(target_pause_s, max_pause_s)
    cuts: list[tuple[float, float]] = []
    speech_idx = np.flatnonzero(speech_mask)
    first_s, last_s = speech_idx[0] * block_s, (speech_idx[-1] + 1) * block_s
    total_s = len(speech_mask) * block_s

    if first_s > lead_trail_s:
        cuts.append((0.0, first_s - lead_trail_s))
    if total_s - last_s > lead_trail_s:
        cuts.append((last_s + lead_trail_s, total_s))

    # Interior pauses: runs of silence between first and last speech block.
    padded = np.concatenate([[True], speech_mask[speech_idx[0]: speech_idx[-1] + 1], [True]])
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    for run_start, run_end in zip(changes[::2], changes[1::2]):
        start_s = (speech_idx[0] + run_start) * block_s
        dur = (run_end - run_start) * block_s
        if dur > max_pause_s:
            cuts.append((start_s + target_pause_s / 2.0,
                         start_s + dur - target_pause_s / 2.0))
    return merge_intervals(cuts)


def tighten_track(track: Track, cfg: Config) -> Track:
    sr = cfg.sr
    hop = int(_BLOCK_S * sr)
    rms = block_rms(track.audio, hop)
    # Conservative: the gate has already pulled quiet speech down ~15 dB, so the
    # pause threshold must sit well below it — only true dead air gets cut.
    thresh = speech_threshold(rms, floor_percentile=5.0, floor_factor=3.0,
                              min_dbfs=-60.0, speech_rel_db=-45.0)
    speech = rms >= thresh
    cuts = find_pause_cuts(speech, _BLOCK_S, max_pause_s=cfg.eff_max_pause(),
                           target_pause_s=cfg.eff_target_pause(), lead_trail_s=cfg.lead_trail_s)
    if not cuts:
        return track
    cut_s = sum(e - s for s, e in cuts)
    log.info("tighten: removing %.1f s of dead air across %d pauses", cut_s, len(cuts))
    return Track(track.name, remove_intervals(track.audio, sr, cuts, xfade_s=0.03))
