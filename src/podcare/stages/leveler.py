"""Segment loudness leveler: a slow, gating-aware ride that evens out the
minutes-scale loudness drift the rest of the chain ignores.

Integrated EBU R128 loudnorm fixes only the whole-file average, and the master
compressor reacts far too fast; neither addresses a guest who fades over a
segment, a host who leans back and gets quiet, or the energy gap between an intro
and a tired late take. That slow drift is the main reason listeners keep riding
the volume on phones.

This stage computes a slow short-term loudness envelope over ~3 s windows,
**counting only speech blocks** (so it never amplifies room tone in the pauses),
pulls each window toward the program's median speech loudness with a tightly
clamped gain, and applies it heavily smoothed at multi-second time constants — so
it is inaudible as processing but very audible in the result. Runs on the mono
bus, before the master compressor/limiter, so those see consistent macro-dynamics.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..dsp import block_rms, db_to_lin, gain_to_samples, smooth_gain, speech_threshold
from ..session import Track

log = logging.getLogger(__name__)

_BLOCK_S = 0.1
_WINDOW_S = 3.0
_MIN_SPEECH_BLOCKS = 20  # need a few seconds of speech to have a stable target


def leveler_track(track: Track, cfg: Config) -> Track:
    sr = cfg.sr
    max_db = cfg.leveler_range_db()
    hop = int(_BLOCK_S * sr)
    if max_db <= 0.0 or len(track.audio) < hop * (_MIN_SPEECH_BLOCKS + 1):
        return track
    rms = block_rms(track.audio, hop)
    if len(rms) < _MIN_SPEECH_BLOCKS:
        return track
    speech = rms >= speech_threshold(rms)
    if int(speech.sum()) < _MIN_SPEECH_BLOCKS:
        return track

    # Speech-weighted short-term loudness: a moving average of block RMS that only
    # counts speech blocks, so pauses neither pull the level down nor get boosted.
    win = max(1, int(_WINDOW_S / _BLOCK_S))
    w = np.ones(win)
    num = np.convolve(rms * speech, w, mode="same")
    den = np.convolve(speech.astype(np.float64), w, mode="same")
    valid = den > 0.0
    st = np.where(valid, num / np.maximum(den, 1e-9), np.nan)
    target = float(np.median(st[valid & speech]))
    if not np.isfinite(target) or target <= 0.0:
        return track

    # Pull each window toward the program median, clamped to the ride range; where
    # no speech is nearby, gain is 1 (hold).
    st_filled = np.where(valid, st, target)
    lo, hi = db_to_lin(-max_db), db_to_lin(max_db)
    raw = np.clip(target / np.maximum(st_filled, 1e-9), lo, hi)
    gains = smooth_gain(raw.astype(np.float32), attack_blocks=win, release_blocks=win)
    env = gain_to_samples(gains, hop, len(track.audio))

    log.info("leveler: %s — ride %+.1f..%+.1f dB toward median speech loudness",
             track.name, 20 * np.log10(float(gains.min()) + 1e-12),
             20 * np.log10(float(gains.max()) + 1e-12))
    return Track(track.name, (track.audio * env).astype(np.float32))
