"""Audio session model: one Track per microphone, one Session per episode."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track:
    """One microphone/recorder track as float32 mono samples."""

    name: str
    audio: np.ndarray  # float32, shape (n_samples,)

    def __post_init__(self) -> None:
        self.audio = np.ascontiguousarray(self.audio, dtype=np.float32)
        if self.audio.ndim != 1:
            raise ValueError(f"Track {self.name!r} must be mono, got shape {self.audio.shape}")

    @property
    def n_samples(self) -> int:
        return len(self.audio)


@dataclass
class Session:
    """All tracks of an episode on a shared timeline and sample rate."""

    sr: int
    tracks: list[Track] = field(default_factory=list)

    def duration_s(self) -> float:
        if not self.tracks:
            return 0.0
        return max(t.n_samples for t in self.tracks) / self.sr
