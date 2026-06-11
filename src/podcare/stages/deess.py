"""De-esser: dynamic gain reduction of the sibilance band (split-band design).

The sibilant band is extracted zero-phase so `full = band + rest` holds exactly;
gain reduction is applied to the band only and the signal recombined.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ..config import Config
from ..dsp import block_rms, db_to_lin, gain_to_samples, smooth_gain
from ..session import Track

log = logging.getLogger(__name__)

_HOP = 128


def deess_track(track: Track, cfg: Config) -> Track:
    sr = cfg.sr
    if len(track.audio) < sr // 10:
        return track
    hi = min(cfg.deess_hi_hz, sr / 2 * 0.98)
    # float32 sos + float32 signal keeps the whole zero-phase filter in single
    # precision, avoiding a full-length float64 copy of the track (multi-GB on
    # hour-long episodes) for no audible loss on a 4th-order speech-band filter.
    sos = butter(4, [cfg.deess_lo_hz, hi], btype="bandpass", fs=sr,
                 output="sos").astype(np.float32)
    band = sosfiltfilt(sos, track.audio).astype(np.float32)
    rest = track.audio - band

    band_rms = block_rms(band, _HOP)
    full_rms = block_rms(track.audio, _HOP)
    dominance = band_rms / (full_rms + 1e-12)

    ratio, max_db = cfg.deess_ratio(), cfg.deess_max_db()
    min_gain = db_to_lin(-max_db)
    # Audibility gate relative to the track's own speech level (with an absolute
    # floor). De-essing runs before the gate's per-track level match, so an
    # absolute-only threshold would never engage on a quiet mic; pegging it
    # ~30 dB below the speech reference tracks the recording's actual level.
    speech_ref = float(np.percentile(full_rms, 90.0))
    audible = full_rms > max(db_to_lin(-55.0), speech_ref * db_to_lin(-30.0))
    sibilant = (dominance > ratio) & audible
    gains = np.ones_like(band_rms)
    # Pull the band back to the dominance threshold, capped at deess_max_db.
    gains[sibilant] = np.maximum(min_gain,
                                 ratio * full_rms[sibilant] / band_rms[sibilant])

    # ~3 ms attack, ~30 ms release at hop 128 @ 48 kHz.
    gains = smooth_gain(gains, attack_blocks=1.0, release_blocks=11.0)
    env = gain_to_samples(gains, _HOP, len(track.audio))
    n_sib = int(sibilant.sum())
    if n_sib:
        log.info("deess: %s — tamed %.1f s of sibilance", track.name, n_sib * _HOP / sr)
    return Track(track.name, rest + band * env)
