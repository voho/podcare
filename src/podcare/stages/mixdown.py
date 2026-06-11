"""Sum all tracks to one mono mix with clipping headroom."""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..session import Session, Track

log = logging.getLogger(__name__)


_HEADROOM = 10 ** (-1.0 / 20.0)  # -1 dBFS


def _apply_headroom(mix: np.ndarray, label: str) -> np.ndarray:
    """Scale down to -1 dBFS if the program would otherwise clip."""
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > _HEADROOM:
        log.info("mixdown: %s peak %.2f dBFS — scaling by %.1f dB", label,
                 20 * np.log10(peak + 1e-12), 20 * np.log10(_HEADROOM / peak))
        mix = mix * (_HEADROOM / peak)
    return mix.astype(np.float32)


def mixdown_session(session: Session, cfg: Config) -> Session:
    # A single track still gets the headroom guarantee so mastering always sees
    # a signal with ~1 dB of headroom regardless of track count.
    if len(session.tracks) == 1:
        t = session.tracks[0]
        return Session(session.sr, [Track(t.name, _apply_headroom(t.audio, t.name))])
    max_len = max(t.n_samples for t in session.tracks)
    mix = np.zeros(max_len, dtype=np.float64)
    for t in session.tracks:
        mix[: t.n_samples] += t.audio
    return Session(session.sr, [Track("mix", _apply_headroom(mix, "mix"))])
