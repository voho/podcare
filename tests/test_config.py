"""The universal strength knob maps monotonically and overrides win."""

import pytest

from podcare.config import Config
from podcare.pipeline import STAGES


def _enabled(strength):
    return {s.name for s in STAGES if s.enabled(Config(strength=strength))}


def test_strength_zero_skips_enhancement_stages():
    # strength 0 = no enhancement: only align + mixdown run in the stage loop.
    # (master still runs afterward to loudness-normalize — it is not in STAGES.)
    assert _enabled(0.0) == {"align", "mixdown"}


def test_strength_above_zero_enables_processing():
    enabled = _enabled(0.8)
    for name in ("repair", "denoise", "dereverb", "tonebalance", "plosives",
                 "deess", "gate", "fillers", "tighten", "mixdown", "align"):
        assert name in enabled, f"{name} should run at strength 0.8"


def test_noop_endpoints_at_zero():
    c = Config(strength=0.0)
    assert c.df_atten_lim_db() == 0.0
    assert c.deess_max_db() == 0.0
    assert c.gate_depth_db() == 0.0
    assert c.eff_filler_sensitivity() == 0.0
    assert c.comp_ratio() == 1.0  # 1:1 = no compression


def test_strength_default():
    assert Config().strength == 0.8


def test_df_atten_is_finite_and_continuous():
    # The denoise ceiling is always a finite dB value (no jump to "unlimited"),
    # keeping the single knob continuous.
    assert Config(strength=1.0).df_atten_lim_db() == pytest.approx(60.0)
    a = Config(strength=0.94).df_atten_lim_db()
    b = Config(strength=0.96).df_atten_lim_db()
    assert abs(b - a) < 2.0  # no discontinuity across the old 0.95 break point


@pytest.mark.parametrize("getter", [
    lambda c: c.df_atten_lim_db() or 0.0,
    lambda c: c.wpe_taps(),
    lambda c: c.wpe_iterations(),
    lambda c: c.deess_max_db(),
    lambda c: c.gate_depth_db(),
    lambda c: c.eff_filler_sensitivity(),
    lambda c: c.comp_ratio(),
    lambda c: c.dropout_max_gap_ms(),
    lambda c: c.resonance_max_cut_db(),
    lambda c: c.exciter_amount(),
])
def test_intensity_rises_with_strength(getter):
    low, high = getter(Config(strength=0.1)), getter(Config(strength=0.9))
    assert high > low


@pytest.mark.parametrize("getter", [
    lambda c: c.deess_ratio(),          # lower ratio = more aggressive
    lambda c: c.eff_max_pause(),        # shorter trigger = more cutting
    lambda c: c.eff_target_pause(),
    lambda c: c.comp_threshold(),
    lambda c: c.resonance_margin_db(),   # lower margin = more sensitive
])
def test_aggressiveness_inverse_params_fall_with_strength(getter):
    low, high = getter(Config(strength=0.1)), getter(Config(strength=0.9))
    assert high < low


def test_strength_clamped():
    assert Config(strength=5.0).s == 1.0
    assert Config(strength=-1.0).s == 0.0


def test_explicit_overrides_beat_strength():
    c = Config(strength=0.8, filler_sensitivity=0.0, max_pause_s=3.0, target_pause_s=1.1)
    assert c.eff_filler_sensitivity() == 0.0
    assert c.eff_max_pause() == 3.0
    assert c.eff_target_pause() == 1.1


def test_filler_sensitivity_follows_strength_when_unset():
    assert Config(strength=1.0).eff_filler_sensitivity() == pytest.approx(0.7)
    assert Config(strength=0.0).eff_filler_sensitivity() == 0.0


def test_new_stage_helpers_are_identity_at_zero_strength():
    c = Config(strength=0.0)
    assert c.dropout_max_gap_ms() == 0.0
    # max_cut=0 zeroes all reduction; margin_db is the sensitivity, not the gate
    assert c.resonance_max_cut_db() == 0.0
    assert c.exciter_amount() == 0.0


def test_plosive_target_below_burst_threshold():
    # Flagged frames must be ducked, so the target multiplier stays under the
    # burst-detection multiplier at every strength.
    for s in (0.0, 0.5, 0.8, 1.0):
        c = Config(strength=s)
        assert c.plosive_target_mult() < c.plosive_burst_mult()
