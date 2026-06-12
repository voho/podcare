import numpy as np
import pytest
import soundfile as sf

from podcare.config import Config
from podcare.session import Session, Track
from podcare.stages.master import master_and_encode
from podcare.stages.align import align_session
from podcare.stages.deess import deess_track
from podcare.stages.dereverb import dereverb_track
from podcare.stages.dropouts import restore_dropouts_track
from podcare.stages.fillers import find_filler_intervals
from podcare.stages.gate import gate_track
from podcare.stages.mixdown import mixdown_session
from podcare.stages.plosives import deplosive_track
from podcare.stages.repair import repair_track
from podcare.stages.resonance import resonance_track
from podcare.stages.silence import find_pause_cuts, tighten_track
from podcare.stages.tonebalance import tonebalance_track

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


class TestDeHum:
    def test_detects_and_notches_mains(self):
        from podcare.stages.dehum import dehum_track
        from scipy.signal import butter, sosfilt
        voice = speech_like(8, seed=20, level=0.3)
        t = np.arange(len(voice)) / SR
        hum = sum(0.02 / k * np.sin(2 * np.pi * 50 * k * t)
                  for k in range(1, 6)).astype(np.float32)
        audio = (voice + hum).astype(np.float32)
        out = dehum_track(Track("x", audio), CFG).audio
        assert out.shape == audio.shape and np.isfinite(out).all()

        def band(x, f0):
            sos = butter(4, [f0 - 3, f0 + 3], btype="bandpass", fs=SR, output="sos")
            return np.std(sosfilt(sos, x.astype(np.float64)))
        assert 20 * np.log10(band(out, 50) / band(audio, 50) + 1e-12) < -12

    def test_clean_track_untouched(self):
        from podcare.stages.dehum import dehum_track
        clean = speech_like(8, seed=21, level=0.3)
        out = dehum_track(Track("c", clean), CFG).audio
        assert np.array_equal(out, clean)


class TestDeclick:
    def test_removes_click_preserves_speech(self):
        from podcare.stages.declick import declick_track
        rng = np.random.default_rng(31)
        voice = speech_like(6, seed=31, level=0.3)
        voice[int(2.0 * SR):int(2.5 * SR)] *= 0.02  # quiet gap
        ca = int(2.25 * SR)
        click = np.zeros_like(voice)
        click[ca:ca + 60] = (0.5 * rng.standard_normal(60)).astype(np.float32)
        audio = (voice + click).astype(np.float32)
        out = declick_track(Track("x", audio), CFG).audio
        assert out.shape == audio.shape and np.isfinite(out).all()

        def rms(x, a, b):
            return float(np.sqrt(np.mean(x[a:b].astype(np.float64) ** 2)))
        red = 20 * np.log10(rms(out, ca - 100, ca + 160) / rms(audio, ca - 100, ca + 160) + 1e-12)
        assert red < -5, f"click only reduced {red:.1f} dB"

    def test_clean_speech_untouched(self):
        from podcare.stages.declick import declick_track
        clean = speech_like(6, seed=33, level=0.3)
        out = declick_track(Track("c", clean), CFG).audio
        assert float(np.corrcoef(clean, out)[0, 1]) > 0.999


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

    def test_reduces_sibilance_on_quiet_recording(self):
        # Same sibilant burst at -26 dB: the adaptive audibility gate (relative to
        # the track's own speech level) must still engage, where the old absolute
        # -45 dBFS gate would mis-calibrate on a low-level mic.
        rng = np.random.default_rng(7)
        t = np.arange(SR * 2) / SR
        voice = 0.3 * np.sin(2 * np.pi * 200 * t)
        sib = np.zeros_like(voice)
        burst = rng.standard_normal(SR // 4)
        from scipy.signal import butter, sosfilt
        sos = butter(4, [5000, 9000], btype="bandpass", fs=SR, output="sos")
        sib[SR : SR + SR // 4] = 0.5 * sosfilt(sos, burst)
        audio = (0.05 * (voice + sib)).astype(np.float32)  # quiet take

        out = deess_track(Track("x", audio), CFG).audio
        sib_region = slice(SR, SR + SR // 4)
        band_before = sosfilt(sos, audio.astype(np.float64))[sib_region]
        band_after = sosfilt(sos, out.astype(np.float64))[sib_region]
        reduction_db = 20 * np.log10(np.std(band_after) / np.std(band_before) + 1e-12)
        assert reduction_db < -3, f"quiet sibilance only reduced by {reduction_db:.1f} dB"


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

    def test_quiet_word_after_pause_not_swallowed(self):
        # A quiet word ABOVE the expander threshold, arriving right after a
        # ducked pause, must not lose its onset to a slow-opening gate.
        rng = np.random.default_rng(9)
        loud = speech_like(3, seed=9, level=0.3)
        pause = (0.002 * rng.standard_normal(int(0.6 * SR))).astype(np.float32)
        t = np.arange(int(0.15 * SR)) / SR
        word = (0.03 * (0.6 * np.sin(2 * np.pi * 160 * t)
                        + 0.3 * np.sin(2 * np.pi * 320 * t))).astype(np.float32)
        audio = np.concatenate([loud, pause, word, pause, loud])
        out = gate_track(Track("x", audio), CFG).audio
        a = len(loud) + len(pause)
        b = a + len(word)

        def db(x):
            return 10 * np.log10(float(np.mean(x.astype(np.float64) ** 2)) + 1e-20)

        d_word = db(out[a:b]) - db(audio[a:b])
        d_loud = db(out[: len(loud)]) - db(audio[: len(loud)])
        assert d_word - d_loud > -2.0, (
            f"quiet word swallowed: {d_word - d_loud:+.1f} dB vs loud speech")


class TestDenoise:
    def test_noise_suppression_bounded_by_dry_floor(self):
        # Ambience preservation: DFN may clean, but the dry-mix floor bounds the
        # worst-case removal near the configured dry level (-15 dB default) so
        # marginal quiet words and room tone are never erased outright.
        from podcare.stages.denoise import denoise_track
        rng = np.random.default_rng(14)
        noise = (0.01 * rng.standard_normal(SR * 4)).astype(np.float32)
        out = denoise_track(Track("x", noise), CFG).audio
        delta = (10 * np.log10(float(np.mean(out.astype(np.float64) ** 2)) + 1e-20)
                 - 10 * np.log10(float(np.mean(noise.astype(np.float64) ** 2)) + 1e-20))
        assert -17.0 < delta < -12.0, (
            f"dry floor should bound suppression near -15 dB (got {delta:+.1f} dB)")


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
    def test_long_track_chunked_preserves_shape(self):
        # A track longer than the 60 s chunk runs through process_chunked; it must
        # keep its length and stay finite (the multi-hour OOM fix).
        n = int(62 * SR)
        t = np.arange(n) / SR
        audio = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        out = deplosive_track(Track("x", audio), CFG).audio
        assert out.shape == audio.shape
        assert np.isfinite(out).all()

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


class TestToneBalance:
    def test_balanced_track_no_correction(self):
        from podcare.stages.tonebalance import _target_on_grid, tonal_correction
        specs = tonal_correction(_target_on_grid(), fraction=0.27)
        assert all(abs(g) < 0.1 for *_, g in specs)

    def test_boomy_track_cuts_low(self):
        from podcare.stages.tonebalance import _GRID, _target_on_grid, tonal_correction
        meas = _target_on_grid().copy()
        meas[_GRID <= 200] += 10.0  # excess low end
        gains = {n: g for n, _k, _f, _q, g in tonal_correction(meas, 0.5)}
        assert gains["low"] < -0.5

    def test_dull_track_lifts_presence_clamped(self):
        from podcare.stages.tonebalance import (_GRID, _MAX_BOOST_DB,
                                                _target_on_grid, tonal_correction)
        meas = _target_on_grid().copy()
        meas[(_GRID >= 2200) & (_GRID <= 4500)] -= 12.0  # deficit in presence
        gains = {n: g for n, _k, _f, _q, g in tonal_correction(meas, 1.0)}
        assert gains["presence"] > 0.5
        assert gains["presence"] <= _MAX_BOOST_DB + 1e-9  # clamped

    def test_zero_fraction_is_identity(self):
        from podcare.stages.tonebalance import _GRID, _target_on_grid, tonal_correction
        meas = _target_on_grid().copy()
        meas[_GRID <= 200] += 10.0
        assert all(g == 0.0 for *_, g in tonal_correction(meas, 0.0))

    def test_track_runs_and_reduces_boom(self):
        from scipy.signal import butter, sosfilt
        voice = speech_like(6, seed=15, level=0.3)
        boom_sos = butter(2, [120, 320], btype="bandpass", fs=SR, output="sos")
        boom = sosfilt(boom_sos, voice.astype(np.float64)).astype(np.float32) * 1.5
        audio = (voice + boom).astype(np.float32)
        out = tonebalance_track(Track("x", audio), CFG).audio
        assert out.shape == audio.shape and np.isfinite(out).all()
        meas_sos = butter(4, [150, 300], btype="bandpass", fs=SR, output="sos")
        before = np.std(sosfilt(meas_sos, audio.astype(np.float64)))
        after = np.std(sosfilt(meas_sos, out.astype(np.float64)))
        assert after < before, f"boom not reduced ({before:.4f} -> {after:.4f})"


class TestMixdown:
    def test_sums_to_single_track_with_headroom(self):
        a = np.full(SR, 0.8, dtype=np.float32)
        b = np.full(SR, 0.8, dtype=np.float32)
        out = mixdown_session(Session(SR, [Track("a", a), Track("b", b)]), CFG)
        assert len(out.tracks) == 1
        assert np.max(np.abs(out.tracks[0].audio)) <= 10 ** (-1.0 / 20.0) + 1e-3

    def test_single_track_gets_headroom(self):
        # A lone hot track must still be brought under -1 dBFS before mastering.
        loud = np.full(SR, 0.97, dtype=np.float32)
        out = mixdown_session(Session(SR, [Track("solo", loud)]), CFG)
        assert len(out.tracks) == 1
        assert np.max(np.abs(out.tracks[0].audio)) <= 10 ** (-1.0 / 20.0) + 1e-3


class TestMasterSilence:
    def test_silent_mix_does_not_crash(self, tmp_path):
        from podcare.stages.master import master_and_encode
        out = tmp_path / "silent.wav"
        master_and_encode(Track("mix", np.zeros(SR * 2, dtype=np.float32)), CFG, out)
        assert out.exists() and out.stat().st_size > 0


class TestMasterLimiter:
    def test_limiter_caps_hot_peaks_without_makeup(self):
        from podcare import audio_io
        from podcare.dsp import db_to_lin
        from podcare.stages.master import _limiter
        t = np.arange(SR) / SR
        hot = (1.26 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)  # +2 dBFS
        out = audio_io.filter_array(hot, SR, _limiter(Config(true_peak_db=-1.5)))
        assert len(out) == len(hot)
        peak = float(np.max(np.abs(out[SR // 4:])))  # skip the attack ramp
        # Capped at the -1.5 dBFS limit; sits near 0 dBFS if level=false were wrong.
        assert peak <= db_to_lin(-1.5) * 1.02
        assert peak > db_to_lin(-3.0)  # limited, not crushed

    def test_multiband_reduces_dynamics(self):
        from podcare.stages.master import _apply_multiband
        t = np.arange(SR * 3) / SR
        sig = (0.1 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        loud = []
        for c in range(5):  # sustained loud bursts over a quiet bed
            i = int((0.4 + 0.5 * c) * SR)
            sig[i:i + 4000] += 0.5
            loud.append((i, i + 4000))
        out = _apply_multiband(sig.astype(np.float32), Config())
        assert len(out) == len(sig) and np.isfinite(out).all()

        def rms(x, a, b):
            return float(np.sqrt(np.mean(x[a:b].astype(np.float64) ** 2)) + 1e-12)
        quiet = (int(0.05 * SR), int(0.35 * SR))
        before = rms(sig, *loud[2]) / rms(sig, *quiet)
        after = rms(out, *loud[2]) / rms(out, *quiet)
        assert after < before, f"loud/quiet ratio not reduced ({before:.1f} -> {after:.1f})"

    def test_master_holds_true_peak_ceiling(self, tmp_path):
        from podcare import audio_io
        from podcare.stages.master import master_and_encode
        rng = np.random.default_rng(3)
        t = np.arange(SR * 4) / SR
        sig = (0.18 * np.sin(2 * np.pi * 180 * t)
               + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
        for c in range(6):  # loud isolated transients -> high crest factor
            i = int((0.3 + 0.6 * c) * SR)
            sig[i:i + 40] += 0.95
        out = tmp_path / "m.wav"
        master_and_encode(Track("mix", sig.astype(np.float32)),
                          Config(true_peak_db=-1.5), out)
        a = audio_io.decode(out, SR)
        m = audio_io.measure_loudnorm(a, SR, pre_filters=None, lufs=-16, true_peak=-1.5)
        assert float(m["input_tp"]) <= -1.0, f"true peak {m['input_tp']} exceeds ceiling"
        assert abs(float(m["input_i"]) - (-16.0)) < 1.5  # loudness target still hit


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


class TestBreath:
    def test_ducks_breath_preserves_speech(self):
        from scipy.signal import butter, sosfilt
        from podcare.stages.breath import breath_track
        rng = np.random.default_rng(40)
        v1 = speech_like(3, seed=40, level=0.35)
        v2 = speech_like(3, seed=41, level=0.35)
        gap = np.zeros(int(0.45 * SR), dtype=np.float32)
        br = sosfilt(butter(2, [500, 3500], btype="bandpass", fs=SR, output="sos"),
                     rng.standard_normal(len(gap)))
        br = (0.025 * br / np.max(np.abs(br))).astype(np.float32)
        seg = (gap + br).astype(np.float32)
        audio = np.concatenate([v1, seg, v2]).astype(np.float32)
        out = breath_track(Track("x", audio), CFG).audio
        assert out.shape == audio.shape and np.isfinite(out).all()

        def rms(x, a, b):
            return float(np.sqrt(np.mean(x[a:b].astype(np.float64) ** 2)))
        b0, b1 = len(v1), len(v1) + len(seg)
        assert 20 * np.log10(rms(out, b0, b1) / rms(audio, b0, b1) + 1e-12) < -4
        assert float(np.corrcoef(audio[:b0], out[:b0])[0, 1]) > 0.999  # speech kept

    def test_clean_speech_untouched(self):
        from podcare.stages.breath import breath_track
        clean = speech_like(5, seed=44, level=0.35)
        out = breath_track(Track("c", clean), CFG).audio
        assert float(np.corrcoef(clean, out)[0, 1]) > 0.999

    @staticmethod
    def _quiet_word(core_voiced: bool) -> tuple[np.ndarray, int, int]:
        """Loud phrase, pause, quiet word (unvoiced flanks ± voiced core), pause, loud."""
        from scipy.signal import butter, sosfilt
        rng = np.random.default_rng(12)
        loud = speech_like(3, seed=12, level=0.3)
        pause = (0.002 * rng.standard_normal(int(0.5 * SR))).astype(np.float32)
        sos = butter(2, [500, 4000], btype="bandpass", fs=SR, output="sos")

        def fric(dur: float) -> np.ndarray:
            n = sosfilt(sos, rng.standard_normal(int(dur * SR)))
            return (0.045 * n / (np.std(n) + 1e-9)).astype(np.float32)

        if core_voiced:
            t = np.arange(int(0.10 * SR)) / SR
            core = (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
            word = np.concatenate([fric(0.10), core, fric(0.12)])
        else:
            word = fric(0.25)
        audio = np.concatenate([loud, pause, word, pause, loud]).astype(np.float32)
        a = len(loud) + len(pause)
        return audio, a, a + len(word)

    @staticmethod
    def _delta_db(out, audio, a, b):
        def db(x):
            return 10 * np.log10(float(np.mean(x.astype(np.float64) ** 2)) + 1e-20)
        return db(out[a:b]) - db(audio[a:b])

    def test_quiet_word_with_voiced_core_untouched(self):
        # "six"/"sest" said softly: unvoiced flanks around a voiced core. The
        # flanks are NOT breaths — they belong to a word — and must survive.
        from podcare.stages.breath import breath_track
        audio, a, b = self._quiet_word(core_voiced=True)
        out = breath_track(Track("x", audio), CFG).audio
        assert self._delta_db(out, audio, a, b) > -1.5, "quiet word breath-ducked"

    def test_isolated_breath_still_ducked(self):
        # A fully-unvoiced isolated burst between phrases IS a breath: duck it.
        from podcare.stages.breath import breath_track
        audio, a, b = self._quiet_word(core_voiced=False)
        out = breath_track(Track("x", audio), CFG).audio
        assert self._delta_db(out, audio, a, b) < -5.0, "real breath no longer ducked"


class TestFillerCrossTrack:
    def test_is_silent_during(self):
        from podcare.stages.fillers import _is_silent_during
        mask = np.zeros(100, dtype=bool)
        mask[50:60] = True  # speech in blocks 50..60 (1.00–1.20 s @ 0.02 s blocks)
        assert _is_silent_during(mask, 0.0, 0.5, 0.02) is True
        assert _is_silent_during(mask, 1.0, 1.2, 0.02) is False

    def test_single_track_keeps_all_fillers(self):
        from podcare.stages.fillers import select_cross_track_safe
        ivs = [[(1.0, 1.3), (2.0, 2.2)]]
        masks = [np.ones(400, dtype=bool)]
        assert select_cross_track_safe(ivs, masks, 0.02) == [(1.0, 1.3), (2.0, 2.2)]

    def test_drops_filler_when_another_speaker_talks_under_it(self):
        from podcare.stages.fillers import select_cross_track_safe
        n = 200  # 4 s @ 0.02 s blocks
        ivs = [[(1.0, 1.3), (2.0, 2.3)], []]
        other = np.zeros(n, dtype=bool)
        other[100:116] = True  # track 1 talks during 2.0–2.3 s, silent at 1.0–1.3 s
        masks = [np.ones(n, dtype=bool), other]
        # Only the isolated filler survives; the co-occurring one is kept (not cut).
        assert select_cross_track_safe(ivs, masks, 0.02) == [(1.0, 1.3)]


class TestLeveler:
    def test_evens_out_loudness_drift(self):
        from podcare.stages.leveler import leveler_track
        a = speech_like(6, seed=50, level=0.30)
        b = speech_like(6, seed=51, level=0.11)  # quieter second segment
        audio = np.concatenate([a, b]).astype(np.float32)
        out = leveler_track(Track("x", audio), CFG).audio
        assert out.shape == audio.shape and np.isfinite(out).all()

        def rms(x):
            return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        h = len(a)
        before = abs(20 * np.log10(rms(audio[h - 2 * SR:h]) / rms(audio[-2 * SR:])))
        after = abs(20 * np.log10(rms(out[h - 2 * SR:h]) / rms(out[-2 * SR:])))
        assert after < before - 2.0, f"drift not reduced ({before:.1f} -> {after:.1f} dB)"

    def test_steady_level_barely_touched(self):
        from podcare.stages.leveler import leveler_track
        steady = speech_like(8, seed=52, level=0.25)
        out = leveler_track(Track("c", steady), CFG).audio
        assert float(np.corrcoef(steady, out)[0, 1]) > 0.99


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


class TestDropouts:
    @staticmethod
    def _signal_with_holes(holes: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        """6 s tone with a quiet lead-in (so speech_threshold has a noise floor),
        with (at_s, dur_s) holes zeroed out. Returns (clean, damaged)."""
        n = 6 * SR
        t = np.arange(n) / SR
        clean = (0.3 * (np.sin(2 * np.pi * 160 * t) + 0.4 * np.sin(2 * np.pi * 320 * t)))
        clean[: SR // 2] *= 0.003  # near-silent lead-in establishes the noise floor
        clean = clean.astype(np.float32)
        damaged = clean.copy()
        for at, dur in holes:
            damaged[int(at * SR): int((at + dur) * SR)] = 0.0
        return clean, damaged

    def test_fills_short_gaps(self):
        clean, damaged = self._signal_with_holes([(2.0, 0.020), (4.0, 0.035)])
        out = restore_dropouts_track(Track("x", damaged), CFG).audio
        assert out.shape == damaged.shape and np.isfinite(out).all()
        for at, dur in [(2.0, 0.020), (4.0, 0.035)]:
            seg = out[int(at * SR): int((at + dur) * SR)].astype(np.float64)
            ref = clean[int((at - 0.05) * SR): int(at * SR)].astype(np.float64)
            seg_rms = np.sqrt(np.mean(seg ** 2))
            ref_rms = np.sqrt(np.mean(ref ** 2))
            assert seg_rms > 0.5 * ref_rms, f"gap at {at}s not filled ({seg_rms:.4f} vs {ref_rms:.4f})"

    def test_leaves_long_gaps_alone(self):
        _, damaged = self._signal_with_holes([(2.0, 0.080)])  # 80 ms > 40 ms cap at s=0.8
        out = restore_dropouts_track(Track("x", damaged), CFG).audio
        seg = out[int(2.0 * SR): int(2.08 * SR)]
        assert np.max(np.abs(seg)) < 1e-6, "80 ms gap must not be fabricated"

    def test_leaves_real_pauses_alone(self):
        # a 20 ms quiet dip inside a 400 ms pause is a pause, not a dropout
        n = 6 * SR
        t = np.arange(n) / SR
        sig = (0.3 * np.sin(2 * np.pi * 160 * t)).astype(np.float32)
        sig[: SR // 2] *= 0.003
        sig[int(2.8 * SR): int(3.2 * SR)] *= 0.003          # real pause
        before = sig.copy()
        out = restore_dropouts_track(Track("x", sig), CFG).audio
        pause = slice(int(2.8 * SR), int(3.2 * SR))
        assert np.array_equal(out[pause], before[pause])

    def test_strength_zero_is_identity(self):
        _, damaged = self._signal_with_holes([(2.0, 0.020)])
        out = restore_dropouts_track(Track("x", damaged), Config(strength=0.0)).audio
        assert np.array_equal(out, damaged)

    def test_fill_has_no_seam_clicks_on_broadband(self):
        voice = speech_like(6, seed=31, level=0.3)
        # ensure speech is active around the hole (envelope gating is random)
        t = np.arange(len(voice)) / SR
        voice = (voice + 0.2 * np.sin(2 * np.pi * 200 * t).astype(np.float32)).astype(np.float32)
        start, end = int(2.0 * SR), int(2.02 * SR)
        damaged = voice.copy()
        damaged[start:end] = 0.0
        out = restore_dropouts_track(Track("x", damaged), CFG).audio
        filled = out[start:end]
        assert float(np.sqrt(np.mean(filled.astype(np.float64) ** 2))) > 1e-4, "hole not filled"
        # seam continuity: the step across each seam must be comparable to the
        # CLEAN side's own steps (one-sided window so the fill can't inflate
        # the bound and mask an unfeathered splice)
        for seam, clean in ((start, slice(start - 480, start)),
                            (end, slice(end, end + 480))):
            local = np.abs(np.diff(out[clean].astype(np.float64)))
            step = abs(float(out[seam]) - float(out[seam - 1]))
            assert step < 10.0 * float(np.percentile(local, 95) + 1e-6), f"click at seam {seam}"


class TestResonance:
    @staticmethod
    def _band_rms(x: np.ndarray, lo: float, hi: float, half: str = "second") -> float:
        from scipy.signal import butter, sosfilt
        sos = butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
        y = sosfilt(sos, x.astype(np.float64))
        y = y[len(y) // 2:] if half == "second" else y[: len(y) // 2]
        return float(np.sqrt(np.mean(y ** 2)))

    def test_tames_injected_ring(self):
        voice = speech_like(6, seed=7, level=0.25)
        t = np.arange(len(voice)) / SR
        ring = (0.15 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
        ring[: len(ring) // 2] = 0.0          # transient: rings only in 2nd half
        audio = (voice + ring).astype(np.float32)
        out = resonance_track(Track("x", audio), CFG).audio
        assert out.shape == audio.shape and np.isfinite(out).all()
        # ring band cut by >= 3 dB in the ringing half...
        assert (self._band_rms(out, 2900, 3100)
                < self._band_rms(audio, 2900, 3100) * 0.71)
        # ...while the non-ringing half is essentially untouched broadband
        in_rms = float(np.std(audio[: len(audio) // 2])) + 1e-9
        out_rms = float(np.std(out[: len(out) // 2])) + 1e-9
        assert abs(20 * np.log10(out_rms / in_rms)) < 0.5

    def test_strength_zero_is_identity(self):
        audio = speech_like(4, seed=8)
        out = resonance_track(Track("x", audio), Config(strength=0.0)).audio
        assert np.array_equal(out, audio)


class TestExciter:
    def test_adds_a_tasteful_amount_of_air(self, tmp_path):
        # Realistic source material: voice plus broadband 4-8 kHz "consonant"
        # energy for aexciter to derive harmonics from. The synthesized air
        # lands above the source band, so measure >9 kHz. The default is a
        # *touch* of air (not the old harsh setting), so assert it is added but
        # bounded — the upper bound guards against re-introducing an extreme
        # default.
        from scipy.signal import butter, sosfilt
        rng = np.random.default_rng(3)
        voice = speech_like(4, seed=3, level=0.3)
        src = sosfilt(butter(4, [4000, 8000], btype="bandpass", fs=SR, output="sos"),
                      rng.standard_normal(len(voice)))
        sig = (voice + 0.05 * src / (np.std(src) + 1e-9)).astype(np.float32)
        on_path, off_path = tmp_path / "on.wav", tmp_path / "off.wav"
        master_and_encode(Track("x", sig), Config(out_sr=SR), on_path)
        master_and_encode(Track("x", sig), Config(out_sr=SR, exciter=False), off_path)

        def air(p):
            a, _ = sf.read(p)
            sos = butter(4, 9000, btype="highpass", fs=SR, output="sos")
            return float(np.std(sosfilt(sos, a)))

        ratio = air(on_path) / (air(off_path) + 1e-12)
        assert 1.1 < ratio < 2.0, f"exciter air {ratio:.2f}x — want present but not extreme"

    def test_amount_override_drives_more_air(self, tmp_path):
        from scipy.signal import butter, sosfilt
        rng = np.random.default_rng(3)
        voice = speech_like(4, seed=3, level=0.3)
        src = sosfilt(butter(4, [4000, 8000], btype="bandpass", fs=SR, output="sos"),
                      rng.standard_normal(len(voice)))
        sig = (voice + 0.05 * src / (np.std(src) + 1e-9)).astype(np.float32)

        def air(cfg):
            p = tmp_path / "x.wav"
            master_and_encode(Track("x", sig), cfg, p)
            a, _ = sf.read(p)
            return float(np.std(sosfilt(butter(4, 9000, btype="highpass", fs=SR,
                                                output="sos"), a)))

        gentle = air(Config(out_sr=SR, exciter_amount=0.4))
        hot = air(Config(out_sr=SR, exciter_amount=2.0, exciter_drive=8.5))
        assert hot > gentle, "exciter_amount override must scale the air"

    def test_strength_zero_is_noop(self, tmp_path):
        t = np.arange(SR * 2) / SR
        sig = (0.2 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        p_s0, p_off = tmp_path / "s0.wav", tmp_path / "off.wav"
        master_and_encode(Track("x", sig), Config(out_sr=SR, strength=0.0), p_s0)
        master_and_encode(Track("x", sig), Config(out_sr=SR, strength=0.0, exciter=False), p_off)
        a, _ = sf.read(p_s0)
        b, _ = sf.read(p_off)
        assert np.array_equal(a, b), "strength=0 exciter must be a no-op"


class TestBookends:
    def test_assembles_with_crossfades(self, tmp_path):
        prog = speech_like(4, seed=2, level=0.3)
        t = np.arange(SR) / SR  # 1 s sting
        sting = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out_path = tmp_path / "o.wav"
        master_and_encode(Track("x", prog), Config(out_sr=SR), out_path,
                          intro=sting, outro=sting)
        a, file_sr = sf.read(out_path)
        assert file_sr == SR
        expected = len(sting) + len(prog) + len(sting) - 2 * int(0.1 * SR)
        assert abs(len(a) - expected) < int(0.02 * SR), \
            f"length {len(a)} != intro+prog+outro-2*xfade ({expected})"
        # the joined program must still be true-peak safe
        assert float(np.max(np.abs(a))) <= 1.0

    def test_no_bookends_is_unchanged_behavior(self, tmp_path):
        prog = speech_like(2, seed=2, level=0.3)
        p1, p2 = tmp_path / "a.wav", tmp_path / "b.wav"
        master_and_encode(Track("x", prog), Config(out_sr=SR), p1)
        master_and_encode(Track("x", prog), Config(out_sr=SR), p2, intro=None, outro=None)
        a, _ = sf.read(p1)
        b, _ = sf.read(p2)
        assert np.array_equal(a, b)
