"""Synthetic audio fixtures with known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

SR = 48000


def speech_like(duration_s: float, *, seed: int = 0, level: float = 0.25,
                syllable_hz: float = 3.0, sr: int = SR) -> np.ndarray:
    """Voiced-speech stand-in: harmonics + noise under a syllabic on/off envelope."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    voiced = (
        0.6 * np.sin(2 * np.pi * 160 * t)
        + 0.3 * np.sin(2 * np.pi * 320 * t)
        + 0.15 * np.sin(2 * np.pi * 480 * t)
        + 0.1 * rng.standard_normal(n)
    )
    # Random syllabic gating, smoothed to avoid clicks.
    n_syl = max(2, int(duration_s * syllable_hz))
    gate = (rng.random(n_syl) > 0.35).astype(np.float64)
    env = np.interp(np.linspace(0, n_syl - 1, n), np.arange(n_syl), gate)
    return (level * voiced * env).astype(np.float32)


@pytest.fixture
def sr() -> int:
    return SR
