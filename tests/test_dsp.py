import numpy as np

from podcare.dsp import (block_rms, gain_to_samples, merge_intervals,
                         process_chunked, remove_intervals, smooth_gain)

SR = 48000


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
