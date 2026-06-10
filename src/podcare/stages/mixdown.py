"""Sum all tracks to one mono mix with clipping headroom."""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..session import Session, Track

log = logging.getLogger(__name__)


def mixdown_session(session: Session, cfg: Config) -> Session:
    if len(session.tracks) == 1:
        return session
    max_len = max(t.n_samples for t in session.tracks)
    mix = np.zeros(max_len, dtype=np.float64)
    for t in session.tracks:
        mix[: t.n_samples] += t.audio
    peak = float(np.max(np.abs(mix))) if max_len else 0.0
    headroom = 10 ** (-1.0 / 20.0)  # -1 dBFS
    if peak > headroom:
        log.info("mixdown: peak %.2f dBFS — scaling by %.1f dB",
                 20 * np.log10(peak + 1e-12), 20 * np.log10(headroom / peak))
        mix *= headroom / peak
    return Session(session.sr, [Track("mix", mix.astype(np.float32))])
