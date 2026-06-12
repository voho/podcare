"""Downward expander (crosstalk/room gate) + per-track speech level matching."""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..dsp import block_rms, db_to_lin, gain_to_samples, smooth_gain, speech_threshold
from ..session import Track

log = logging.getLogger(__name__)

_HOP = 256


def gate_track(track: Track, cfg: Config) -> Track:
    sr = cfg.sr
    rms = block_rms(track.audio, _HOP)
    thresh = speech_threshold(rms)
    depth = db_to_lin(-cfg.gate_depth_db())

    # 2:1 downward expansion below threshold, floored at -gate_depth_db.
    gains = np.ones_like(rms)
    below = rms < thresh
    gains[below] = np.maximum(depth, rms[below] / thresh)

    # Standard gate timing: OPEN fast (~10 ms) so a word arriving after a
    # ducked pause never loses its onset, CLOSE slow (~160 ms hold) so the
    # gate doesn't chatter at phrase ends. (In smooth_gain terms attack =
    # gain falling = closing; release = gain rising = opening. The previous
    # 5 ms-close/160 ms-open inversion swallowed the first ~150 ms of every
    # quiet word that followed a pause.)
    gains = smooth_gain(gains, attack_blocks=30.0, release_blocks=2.0)
    env = gain_to_samples(gains, _HOP, len(track.audio))
    gated = track.audio * env

    # Match speech-active loudness across tracks so the mix is balanced.
    active = rms[rms >= thresh]
    if len(active):
        speech_rms = float(np.sqrt(np.mean(active.astype(np.float64) ** 2)))
        gain = db_to_lin(cfg.level_target_dbfs) / max(speech_rms, 1e-9)
        gain = float(np.clip(gain, db_to_lin(-12.0), db_to_lin(24.0)))
        log.info("gate: %s — level match %+.1f dB", track.name, 20 * np.log10(gain))
        gated = gated * gain
    return Track(track.name, gated)
