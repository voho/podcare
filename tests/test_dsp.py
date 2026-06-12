import numpy as np
import pytest

from podcare.dsp import (block_rms, crossfade_concat, gain_to_samples,
                         merge_intervals, process_chunked, remove_intervals,
                         smooth_gain)

SR = 48000


def test_block_rms_empty_is_finite():
    # Empty input must not poison downstream thresholds with NaN.
    rms = block_rms(np.zeros(0, dtype=np.float32), hop=480)
    assert rms.shape == (1,)
    assert np.isfinite(rms).all()


def test_remove_intervals_subsample_is_noop():
    # An interval that rounds to under one sample must not split/crossfade audio.
    audio = np.ones(SR * 2, dtype=np.float32)
    out = remove_intervals(audio, SR, [(1.0, 1.0 + 1e-7)], xfade_s=0.015)
    assert len(out) == len(audio)
    assert np.allclose(out, 1.0)


def test_process_chunked_rejects_length_change_on_short_input():
    # The single-chunk fast path enforces the same length invariant as the
    # multi-chunk path, so a misbehaving fn can't silently drift the timeline.
    audio = np.zeros(SR, dtype=np.float32)  # 1 s < chunk+overlap -> fast path
    with pytest.raises(ValueError):
        process_chunked(audio, SR, lambda c: c[:-10], chunk_s=60.0, overlap_s=1.0)


def test_block_rms_constant_signal():
    rms = block_rms(np.full(SR, 0.5, dtype=np.float32), hop=480)
    assert rms.shape == (100,)
    assert np.allclose(rms, 0.5, atol=1e-4)


def test_merge_intervals_overlap_and_order():
    merged = merge_intervals([(3.0, 4.0), (0.5, 1.0), (0.9, 2.0)])
    assert merged == [(0.5, 2.0), (3.0, 4.0)]


def test_remove_intervals_duration_and_content():
    audio = np.ones(SR * 10, dtype=np.float32)
    out = remove_intervals(audio, SR, [(2.0, 5.0)], xfade_s=0.015)
    # 3 s removed, minus one crossfade worth of overlap.
    assert abs(len(out) / SR - 7.0) < 0.05
    # Equal-power crossfade of identical material stays near 1, never above sqrt(2).
    assert np.max(out) < 1.45
    assert np.min(out) > 0.5


def test_remove_intervals_at_edges():
    audio = np.ones(SR * 4, dtype=np.float32)
    out = remove_intervals(audio, SR, [(0.0, 1.0), (3.0, 4.0)])
    assert abs(len(out) / SR - 2.0) < 0.05


def test_smooth_gain_attack_faster_than_release():
    gains = np.ones(100, dtype=np.float32)
    gains[40:60] = 0.1
    smoothed = smooth_gain(gains, attack_blocks=1.0, release_blocks=20.0)
    # Fast attack: nearly fully attenuated a few blocks into the dip.
    assert smoothed[45] < 0.2
    # Slow release: still well below 1 a few blocks after the dip ends.
    assert smoothed[65] < 0.8


def test_gain_to_samples_interpolates():
    env = gain_to_samples(np.array([1.0, 0.0], dtype=np.float32), hop=100, n_samples=200)
    assert env.shape == (200,)
    assert env[50] == 1.0  # first block center
    assert abs(env[100] - 0.5) < 0.02


def test_process_chunked_identity_roundtrip():
    rng = np.random.default_rng(1)
    audio = rng.standard_normal(SR * 5).astype(np.float32)
    out = process_chunked(audio, SR, lambda c: c, chunk_s=1.0, overlap_s=0.25)
    assert out.shape == audio.shape
    assert np.allclose(out, audio, atol=1e-5)


def test_crossfade_concat_blends_smoothly():
    a = np.ones(1000, dtype=np.float32)
    b = np.full(1000, 0.5, dtype=np.float32)
    out = crossfade_concat([a, b], xf=200)
    assert len(out) == 1800  # 1000 + 1000 - 200 overlap
    blend = out[800:1000]
    assert blend[0] == pytest.approx(1.0, abs=0.05)    # starts at a's level
    assert blend[-1] == pytest.approx(0.5, abs=0.05)   # ends at b's level
    # equal-power curves: correlated material may peak up to sqrt(2), never more
    assert float(np.max(blend)) < np.sqrt(2.0) + 0.01
    assert np.isfinite(out).all()


def test_crossfade_concat_single_and_empty():
    a = np.ones(100, dtype=np.float32)
    assert np.array_equal(crossfade_concat([a], xf=50), a)
    assert len(crossfade_concat([], xf=50)) == 0
    # segments shorter than 2 overlap samples are butt-joined, not dropped
    out = crossfade_concat([a, np.ones(1, dtype=np.float32)], xf=50)
    assert len(out) == 101
