import numpy as np
import pytest

from podcare.config import Config
from podcare.session import Session, Track
from podcare.stages.align import align_session
from podcare.stages.deess import deess_track
from podcare.stages.dereverb import dereverb_track
from podcare.stages.fillers import find_filler_intervals
from podcare.stages.gate import gate_track
from podcare.stages.mixdown import mixdown_session
from podcare.stages.plosives import deplosive_track
from podcare.stages.repair import repair_track
from podcare.stages.silence import find_pause_cuts, tighten_track

from conftest import SR, speech_like

CFG = Config()


def _xcorr_peak(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class TestAlign:
    def test_recovers_known_offset(self):
        voice = speech_like(20, seed=3)
        offset = int(1.337 * SR)
        late = np.concatenate([np.zeros(offset, dtype=np.float32),
                               0.8 * voice + speech_like(20, seed=9, level=0.02)[: len(voice)]])
        session = Session(SR, [Track("ref", voice), Track("late", late)])
        aligned = align_session(session, CFG)
        r = _xcorr_peak(aligned.tracks[0].audio, aligned.tracks[1].audio)
        assert r > 0.85, f"tracks not aligned (r={r:.2f})"

    def test_fixes_inverted_polarity(self):
        voice = speech_like(20, seed=4)
        flipped = -0.9 * voice
        session = Session(SR, [Track("ref", voice), Track("flip", flipped)])
        aligned = align_session(session, CFG)
        r = _xcorr_peak(aligned.tracks[0].audio, aligned.tracks[1].audio)
        assert r > 0.85, f"polarity not corrected (r={r:.2f})"

    def test_leaves_uncorrelated_tracks_alone(self):
        a = speech_like(20, seed=5)
        b = speech_like(20, seed=6)
        session = Session(SR, [Track("a", a), Track("b", b)])
        aligned = align_session(session, CFG)
        assert np.array_equal(aligned.tracks[1].audio[: len(b)], b)


class TestRepair:
    def test_declip_reduces_clipped_samples(self):
        t = np.arange(SR * 3) / SR
        clean = 1.6 * np.sin(2 * np.pi * 220 * t)
        clipped = np.clip(clean, -1.0, 1.0).astype(np.float32)
        before = np.mean(np.abs(clipped) > 0.98)
        out = repair_track(Track("x", clipped), CFG)
        after = np.mean(np.abs(out.audio) > 0.98)
        assert after < before * 0.7, f"clipping not reduced ({before:.3f} -> {after:.3f})"


class TestDeess:
    def test_reduces_sibilance_band_only(self):
        rng = np.random.default_rng(7)
        t = np.arange(SR * 2) / SR
        voice = 0.3 * np.sin(2 * np.pi * 200 * t)
        sib = np.zeros_like(voice)
        burst = rng.standard_normal(SR // 4)
        from scipy.signal import butter, sosfilt
        sos = butter(4, [5000, 9000], btype="bandpass", fs=SR, output="sos")
        sib[SR : SR + SR // 4] = 0.5 * sosfilt(sos, burst)
        audio = (voice + sib).astype(np.float32)

        out = deess_track(Track("x", audio), CFG).audio
        sib_region = slice(SR, SR + SR // 4)
        band_before = sosfilt(sos, audio.astype(np.float64))[sib_region]
        band_after = sosfilt(sos, out.astype(np.float64))[sib_region]
        reduction_db = 20 * np.log10(np.std(band_after) / np.std(band_before) + 1e-12)
        assert reduction_db < -3, f"sibilance only reduced by {reduction_db:.1f} dB"
        # The 200 Hz voice outside the burst is untouched.
        lead = slice(0, SR // 2)
        assert np.allclose(out[lead], audio[lead], atol=0.02)


class TestGate:
    def test_attenuates_noise_between_speech(self):
        rng = np.random.default_rng(8)
        voice = speech_like(4, seed=8, level=0.3)
        program = np.concatenate([voice, np.zeros(SR * 2, dtype=np.float32), voice])
        # Constant mic noise floor across the whole take, as in real recordings.
        audio = (program + 0.01 * rng.standard_normal(len(program))).astype(np.float32)
        out = gate_track(Track("x", audio), CFG).audio
        gap = slice(int(4.5 * SR), int(5.5 * SR))
        speech_gain = np.std(out[: SR * 4]) / (np.std(audio[: SR * 4]) + 1e-12)
        gap_gain = np.std(out[gap]) / (np.std(audio[gap]) + 1e-12)
        assert gap_gain < speech_gain * 0.5, (
            f"gap not gated (speech x{speech_gain:.2f}, gap x{gap_gain:.2f})")


class TestDereverb:
    def test_runs_and_preserves_shape(self):
        audio = speech_like(3, seed=10)
        out = dereverb_track(Track("x", audio), CFG)
        assert out.audio.shape == audio.shape
        assert np.isfinite(out.audio).all()

    def test_reduces_reverb_tail_energy(self):
        from scipy.signal import fftconvolve
        rng = np.random.default_rng(11)
        dry = speech_like(8, seed=11, level=0.3)
        # Synthetic room: exponentially decaying noise tail, ~400 ms RT.
        ir = rng.standard_normal(int(0.4 * SR)) * np.exp(-np.arange(int(0.4 * SR)) / (0.08 * SR))
        ir[0] = 1.0
        wet = fftconvolve(dry, ir)[: len(dry)].astype(np.float32)
        wet /= np.max(np.abs(wet))
        out = dereverb_track(Track("x", wet), CFG).audio
        # Energy in the gaps between syllables (where only reverb tail lives) drops.
        envelope = np.abs(dry) > 0.01
        gaps = ~envelope
        before = float(np.mean(wet[gaps] ** 2))
        after = float(np.mean(out[gaps] ** 2))
        assert after < before * 0.7, f"reverb tail not reduced ({before:.2e} -> {after:.2e})"


class TestPlosives:
    def test_edge_burst_does_not_wrap(self):
        # A plosive at the very start must not duck the very end (no np.roll wrap).
        rng = np.random.default_rng(21)
        sr = SR
        voice = 0.2 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr).astype(np.float32)
        pop = np.zeros(sr, dtype=np.float32)
        pop[:2000] = (3.0 * np.sin(2 * np.pi * 80 * np.arange(2000) / sr)).astype(np.float32)
        audio = (voice + pop).astype(np.float32)
        out = deplosive_track(Track("x", audio), CFG).audio
        # Final 50 ms (no pop there) should be essentially untouched.
        tail = slice(len(audio) - sr // 20, len(audio))
        assert np.allclose(out[tail], audio[tail], atol=2e-3)


class TestMixdown:
    def test_sums_to_single_track_with_headroom(self):
        a = np.full(SR, 0.8, dtype=np.float32)
        b = np.full(SR, 0.8, dtype=np.float32)
        out = mixdown_session(Session(SR, [Track("a", a), Track("b", b)]), CFG)
        assert len(out.tracks) == 1
        assert np.max(np.abs(out.tracks[0].audio)) <= 10 ** (-1.0 / 20.0) + 1e-3


class TestMasterSilence:
    def test_silent_mix_does_not_crash(self, tmp_path):
        from podcare.stages.master import master_and_encode
        out = tmp_path / "silent.wav"
        master_and_encode(Track("mix", np.zeros(SR * 2, dtype=np.float32)), CFG, out)
        assert out.exists() and out.stat().st_size > 0


class TestFillerDecisions:
    WORDS = [
        ("Hello", 0.0, 0.30, 0.99),
        (" um,", 0.50, 0.80, 0.90),
        (" nice", 1.00, 1.30, 0.95),
        (" ehm", 1.40, 1.46, 0.40),   # short + low confidence
        (" world.", 1.60, 2.00, 0.99),
    ]

    def test_zero_sensitivity_cuts_nothing(self):
        assert find_filler_intervals(self.WORDS, 0.0, 0.01) == []

    def test_default_sensitivity_cuts_confident_filler_only(self):
        cuts = find_filler_intervals(self.WORDS, 0.5, 0.01)
        assert len(cuts) == 1
        s, e = cuts[0]
        assert abs(s - 0.49) < 0.02 and abs(e - 0.81) < 0.02

    def test_high_sensitivity_cuts_borderline_filler_too(self):
        cuts = find_filler_intervals(self.WORDS, 1.0, 0.01)
        assert len(cuts) == 2

    def test_never_cuts_real_words(self):
        cuts = find_filler_intervals(self.WORDS, 1.0, 0.01)
        for s, e in cuts:
            assert not (s < 0.15 < e) and not (s < 1.8 < e)


class TestSilence:
    def test_pause_cut_math(self):
        block_s = 0.01
        mask = np.ones(1000, dtype=bool)   # 10 s
        mask[300:800] = False              # 5 s pause
        cuts = find_pause_cuts(mask, block_s, max_pause_s=2.0,
                               target_pause_s=0.7, lead_trail_s=0.5)
        assert len(cuts) == 1
        s, e = cuts[0]
        assert abs((e - s) - (5.0 - 0.7)) < 0.05

    def test_tighten_track_shrinks_long_pause(self):
        voice = speech_like(2, seed=12, level=0.3)
        rng = np.random.default_rng(12)
        silence = (1e-4 * rng.standard_normal(SR * 6)).astype(np.float32)
        audio = np.concatenate([voice, silence, voice])
        out = tighten_track(Track("x", audio), CFG)
        out_s = len(out.audio) / SR
        # 10 s in, ~5.3 s of the pause removed.
        assert out_s < 5.5, f"pause not tightened (out {out_s:.1f} s)"
        assert out_s > 3.5, f"speech was cut (out {out_s:.1f} s)"
