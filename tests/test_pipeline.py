"""End-to-end: synthetic two-mic session → polished file."""

import numpy as np
import pytest
import soundfile as sf

from podcare import audio_io, pipeline
from podcare.config import Config

from conftest import SR, speech_like


@pytest.fixture
def two_mic_session(tmp_path):
    """Two correlated mic tracks with offset, noise, clipping and a long pause."""
    rng = np.random.default_rng(42)
    host = speech_like(6, seed=1, level=0.4)
    guest = speech_like(6, seed=2, level=0.35)
    pause = np.zeros(SR * 5, dtype=np.float32)
    program_a = np.concatenate([host, pause, 0.15 * guest])          # host mic
    program_b = np.concatenate([0.15 * host, pause, guest])          # guest mic w/ bleed
    noise_a = (0.01 * rng.standard_normal(len(program_a))).astype(np.float32)
    noise_b = (0.02 * rng.standard_normal(len(program_b))).astype(np.float32)
    a = np.clip((program_a + noise_a) * 1.4, -1, 1)                  # mildly clipped
    b = np.concatenate([np.zeros(int(0.8 * SR), dtype=np.float32),   # recorder started late
                        program_b + noise_b])
    path_a, path_b = tmp_path / "host.wav", tmp_path / "guest.wav"
    sf.write(path_a, a, SR)
    sf.write(path_b, b, SR)
    return path_a, path_b


def _measured_lufs(path, sr=SR) -> float:
    audio = audio_io.decode(path, sr)
    measured = audio_io.measure_loudnorm(audio, sr, pre_filters=None,
                                         lufs=-16, true_peak=-1.5)
    return float(measured["input_i"])


def test_end_to_end(two_mic_session, tmp_path):
    out = tmp_path / "episode.mp3"
    cfg = Config(
        filler_sensitivity=0.0,        # no Whisper download in CI
        denoise_backend="spectral",    # keep the test fast; DFN covered separately
        align_window_s=20.0,
    )
    in_dur, out_dur = pipeline.run(list(two_mic_session), out, cfg)

    assert out.exists() and out.stat().st_size > 10_000
    assert abs(in_dur - 17.8) < 0.2
    # The 5 s pause must shrink to ~0.7 s; nothing else should vanish.
    assert 11.0 < out_dur < 15.5, f"unexpected output duration {out_dur:.1f} s"
    encoded_dur = len(audio_io.decode(out, SR)) / SR
    assert abs(encoded_dur - out_dur) < 0.3
    # Loudness within ±1.5 LU of the -16 LUFS target.
    lufs = _measured_lufs(out)
    assert abs(lufs - (-16.0)) < 1.5, f"loudness {lufs:.1f} LUFS"


def test_single_track_wav_out(tmp_path):
    rng = np.random.default_rng(5)
    voice = speech_like(8, seed=5, level=0.35)
    noisy = voice + (0.01 * rng.standard_normal(len(voice))).astype(np.float32)
    src = tmp_path / "solo.wav"
    sf.write(src, noisy, SR)
    out = tmp_path / "solo_clean.wav"
    cfg = Config(filler_sensitivity=0.0, denoise_backend="spectral")
    in_dur, out_dur = pipeline.run([src], out, cfg)
    assert out.exists()
    assert 0 < out_dur <= in_dur + 0.1
    info = sf.info(str(out))
    assert info.samplerate == 44100
    assert info.subtype == "PCM_16"
