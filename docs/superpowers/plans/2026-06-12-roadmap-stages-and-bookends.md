# Roadmap Stages 1–3 + Intro/Outro Bookends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the four approved features from `docs/superpowers/specs/2026-06-12-roadmap-stages-and-bookends-design.md`: dropout restoration, dynamic resonance suppression, harmonic exciter, and intro/outro bookends.

**Architecture:** Two new per-track DSP stages (`dropouts.py` before repair, `resonance.py` after de-ess) following the existing `(Track, Config) -> Track` pattern; the exciter is an ffmpeg filter node inside the existing master chain; bookends are final assembly in `master_and_encode` using a `crossfade_concat` helper extracted from `remove_intervals`. All strength-scaled features have a documented no-op-at-0 endpoint.

**Tech Stack:** Python 3.11, numpy/scipy (no new dependencies), ffmpeg via existing `audio_io` helpers, pytest with synthetic-signal tests (`conftest.speech_like`, `SR = 48000`).

**Conventions used throughout:**
- Run tests with `uv run pytest <path> -v` from the repo root (`/Users/vojta/Dev/podcare`).
- Every stage's strength helpers live on `Config` and use `_lerp(identity_value, max_value, self.s)`.
- Per-track stages log one `log.info("<stage>: %s — ...", track.name, ...)` line.
- Tests live in classes (`class TestX:`) in `tests/test_stages.py`, using `from conftest import SR, speech_like` and `CFG = Config()` (strength 0.8).

---

### Task 1: Extract `crossfade_concat` from `remove_intervals`

The equal-power segment join currently embedded in `remove_intervals` is needed by the bookends feature. Extract it; existing tests must stay green.

**Files:**
- Modify: `src/podcare/dsp.py` (function `remove_intervals`, lines ~88-125)
- Test: `tests/test_dsp.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dsp.py` (it already imports from `podcare.dsp` at the top — add `crossfade_concat` to that import list):

```python
def test_crossfade_concat_joins_with_equal_power_blend():
    a = np.ones(1000, dtype=np.float32)
    b = np.full(1000, 0.5, dtype=np.float32)
    out = crossfade_concat([a, b], xf=200)
    assert len(out) == 1800  # 1000 + 1000 - 200 overlap
    blend = out[800:1000]
    assert blend[0] == pytest.approx(1.0, abs=0.05)    # starts at a's level
    assert blend[-1] == pytest.approx(0.5, abs=0.05)   # ends at b's level
    assert np.isfinite(out).all()


def test_crossfade_concat_single_and_empty():
    a = np.ones(100, dtype=np.float32)
    assert np.array_equal(crossfade_concat([a], xf=50), a)
    assert len(crossfade_concat([], xf=50)) == 0
    # segments shorter than 2 overlap samples are butt-joined, not dropped
    out = crossfade_concat([a, np.ones(1, dtype=np.float32)], xf=50)
    assert len(out) == 101
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dsp.py -v -k crossfade_concat`
Expected: FAIL / ERROR with `ImportError: cannot import name 'crossfade_concat'`

- [ ] **Step 3: Implement — add the helper and refactor `remove_intervals` to use it**

In `src/podcare/dsp.py`, add directly **above** `remove_intervals`:

```python
def crossfade_concat(segments: list[np.ndarray], xf: int) -> np.ndarray:
    """Join segments with equal-power crossfades of up to `xf` samples.

    Each boundary blends the tail of the running output with the head of the
    next segment over min(xf, len(out), len(seg)) samples using cos/sin
    curves, so correlated material keeps ~constant power across the join.
    Boundaries with fewer than 2 overlap samples are butt-joined.
    """
    segments = [s for s in segments if len(s) > 0]
    if not segments:
        return np.zeros(0, dtype=np.float32)
    out = segments[0].copy()
    for seg in segments[1:]:
        n = min(xf, len(out), len(seg))
        if n >= 2:
            t = np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)
            out[-n:] = out[-n:] * np.cos(t) + seg[:n] * np.sin(t)
            out = np.concatenate([out, seg[n:]])
        else:
            out = np.concatenate([out, seg])
    return out
```

Then replace the tail of `remove_intervals` (everything from `segments = [s for s in segments if len(s) > 0]` to the end of the function) with:

```python
    segments = [s for s in segments if len(s) > 0]
    if not segments:
        return audio[:0]
    return crossfade_concat(segments, xf)
```

- [ ] **Step 4: Run the dsp test file to verify all pass (including pre-existing `remove_intervals` tests)**

Run: `uv run pytest tests/test_dsp.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/podcare/dsp.py tests/test_dsp.py
git commit -m "refactor(dsp): extract crossfade_concat from remove_intervals"
```

---

### Task 2: Config — toggles, paths, strength helpers

**Files:**
- Modify: `src/podcare/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py`:

(a) Add three entries to the existing `test_intensity_rises_with_strength` parametrize list:

```python
    lambda c: c.dropout_max_gap_ms(),
    lambda c: c.resonance_max_cut_db(),
    lambda c: c.exciter_amount(),
```

(b) Add one entry to the existing `test_aggressiveness_inverse_params_fall_with_strength` parametrize list:

```python
    lambda c: c.resonance_margin_db(),   # lower margin = more sensitive
```

(c) Add a new test function at module level:

```python
def test_new_stage_helpers_are_identity_at_zero_strength():
    c = Config(strength=0.0)
    assert c.dropout_max_gap_ms() == 0.0
    assert c.resonance_max_cut_db() == 0.0
    assert c.exciter_amount() == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'dropout_max_gap_ms'`

- [ ] **Step 3: Implement the Config additions**

In `src/podcare/config.py`, add fields. After the `declip`/`hpf_hz` block (Repair) — i.e. just **before** the `# De-hum` comment — insert:

```python
    # Dropout / short-gap restoration (fills brief packet-loss holes via LPC;
    # strength scales the longest gap it may fill — 0 ms at strength 0 = no-op)
    dropouts: bool = True
```

After the `deess` block (`deess_hi_hz` line), insert:

```python
    # Dynamic resonance / harshness suppression ("Soothe-lite" STFT notching)
    resonance: bool = True
```

After the `master`/`compress`/`lufs`/`true_peak_db`/`lossy_bitrate` block, insert:

```python
    # Harmonic presence exciter (inside master; synthesizes "air" above ~7.4 kHz)
    exciter: bool = True

    # Intro/outro bookends — optional sounds assembled around the finished
    # program (loudness-matched, 100 ms equal-power crossfades). Assembly, not
    # processing: used whenever set, not strength-scaled. The CLI ignores them
    # under --nocut because an intro shifts the whole timeline.
    intro_sound: Path | None = None
    outro_sound: Path | None = None
```

Then add the strength helpers in the helper section (e.g. directly after the `wpe_iterations` helper):

```python
    # Dropouts: longest fillable gap. 0 ms at strength 0 (nothing qualifies ->
    # identity); 50 ms at full strength — the README contract's honesty cap.
    def dropout_max_gap_ms(self) -> float:
        return _lerp(0.0, 50.0, self.s)

    # Resonance: a bin must poke margin dB above its own spectral envelope to
    # count as a resonance (lower = more sensitive); the cut is capped at
    # max_cut. Cap 0 at strength 0 makes the math itself the identity.
    def resonance_margin_db(self) -> float:
        return _lerp(18.0, 6.0, self.s)

    def resonance_max_cut_db(self) -> float:
        return _lerp(0.0, 10.0, self.s)

    # Exciter: ffmpeg aexciter `amount`. 0 adds no harmonics (identity);
    # 2.5 ceiling is deliberately conservative ("easy to overdo").
    def exciter_amount(self) -> float:
        return _lerp(0.0, 2.5, self.s)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/podcare/config.py tests/test_config.py
git commit -m "feat(config): toggles + strength helpers for dropouts, resonance, exciter, bookends"
```

---

### Task 3: Dropout restoration stage

**Files:**
- Create: `src/podcare/stages/dropouts.py`
- Test: `tests/test_stages.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stages.py` (and add `from podcare.stages.dropouts import restore_dropouts_track` to the imports at the top):

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_stages.py -v -k Dropouts`
Expected: ERROR at collection — `ModuleNotFoundError: No module named 'podcare.stages.dropouts'`

- [ ] **Step 3: Implement the stage**

Create `src/podcare/stages/dropouts.py`:

```python
"""Dropout / short-gap restoration: fill brief packet-loss holes via two-sided LPC.

Remote-guest recordings (VoIP and double-enders over flaky links) arrive with
3-50 ms holes where packets were lost — audible as tiny stutters. Each hole is
refilled by linear prediction: extrapolate the speech forward from the audio
before the gap and backward from the audio after it, then equal-power
crossfade the two estimates across the gap. Strict caps keep it honest: only
short gaps (strength-scaled, <= 50 ms), only gaps whose surroundings are
speech-active (a quiet moment inside a real pause is not a dropout), and no
more than ~1.2 s filled per minute — beyond that the track is corrupt, not
packet-lossy, and fabricating more would do harm.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..dsp import block_rms, speech_threshold
from ..session import Track

log = logging.getLogger(__name__)

_BLOCK_S = 0.003           # detection resolution: 3 ms RMS blocks
_DROP_DB = 25.0            # a dropout sits >= this far below its local context
_CONTEXT_S = 0.3           # local window for the context reference level
_LPC_ORDER = 32
_LPC_CONTEXT_S = 0.03      # clean audio used to fit each one-sided predictor
_EDGE_XF_S = 0.002         # fade the fill into the real audio at the seams
_MAX_FILL_PER_MIN_S = 1.2  # honesty cap: ~2% of any minute


def _find_dropouts(audio: np.ndarray, sr: int, max_gap_s: float) -> list[tuple[int, int]]:
    """(start, end) sample intervals of fillable dropouts, earliest first."""
    hop = max(1, int(_BLOCK_S * sr))
    rms = block_rms(audio, hop)
    if len(rms) < 8:
        return []
    db = 20.0 * np.log10(rms.astype(np.float64) + 1e-12)
    ctx = max(1, int(_CONTEXT_S / _BLOCK_S))
    speech_rms = speech_threshold(rms)
    max_blocks = max(0, int(round(max_gap_s / _BLOCK_S)))
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(db):
        lo, hi = max(0, i - ctx), min(len(db), i + ctx)
        context = np.concatenate([db[lo:i], db[i + 1: hi]])
        ref = float(np.median(context)) if len(context) else -120.0
        if db[i] > ref - _DROP_DB:
            i += 1
            continue
        j = i
        while j < len(db) and db[j] <= ref - _DROP_DB:
            j += 1
        # Fill only if: short enough, and speech-active on BOTH sides (so the
        # level genuinely collapses and recovers — a dip inside a real pause
        # has quiet context and is excluded).
        pre = rms[max(0, i - ctx): i]
        post = rms[j: j + ctx]
        if (1 <= (j - i) <= max_blocks
                and len(pre) and len(post)
                and float(np.median(pre)) > speech_rms
                and float(np.median(post)) > speech_rms):
            out.append((i * hop, min(len(audio), j * hop)))
        i = j
    return out


def _lpc_coeffs(x: np.ndarray, order: int) -> np.ndarray:
    """Levinson-Durbin forward predictor: x[n] ~= sum(a[k] * x[n-1-k])."""
    x = x.astype(np.float64)
    n = len(x)
    if n <= order + 1:
        return np.zeros(order)
    r = np.correlate(x, x, mode="full")[n - 1: n + order]
    if r[0] <= 1e-12:
        return np.zeros(order)
    a = np.zeros(order)
    e = float(r[0])
    for k in range(order):
        acc = r[k + 1] - float(np.dot(a[:k], r[1: k + 1][::-1]))
        ref = acc / e
        new_a = a.copy()
        new_a[k] = ref
        new_a[:k] = a[:k] - ref * a[:k][::-1]
        a = new_a
        e *= (1.0 - ref * ref)
        if e <= 1e-12:
            break
    return a


def _extrapolate(context: np.ndarray, n_out: int, order: int) -> np.ndarray:
    """Continue `context` for n_out samples with a one-step LPC predictor."""
    a = _lpc_coeffs(context, order)
    if not np.any(a):
        return np.zeros(n_out)
    hist = list(context.astype(np.float64)[-order:])
    out = np.empty(n_out)
    for i in range(n_out):
        nxt = float(np.dot(a, hist[::-1]))
        nxt = float(np.clip(nxt, -4.0, 4.0))  # bound a marginally unstable filter
        out[i] = nxt
        hist.pop(0)
        hist.append(nxt)
    return out


def _fill_gap(audio: np.ndarray, sr: int, start: int, end: int) -> None:
    """Replace audio[start:end] in place with a two-sided LPC estimate."""
    n = end - start
    ctx = int(_LPC_CONTEXT_S * sr)
    pre = audio[max(0, start - ctx): start]
    post = audio[end: end + ctx]
    fwd = _extrapolate(pre, n, _LPC_ORDER) if len(pre) > _LPC_ORDER * 2 else np.zeros(n)
    bwd = (_extrapolate(post[::-1], n, _LPC_ORDER)[::-1]
           if len(post) > _LPC_ORDER * 2 else np.zeros(n))
    t = np.linspace(0.0, np.pi / 2.0, n)
    fill = fwd * np.cos(t) + bwd * np.sin(t)
    xf = min(max(2, int(_EDGE_XF_S * sr)), n // 2)
    if xf >= 2:  # feather the seams so the splice never clicks
        w = np.linspace(0.0, 1.0, xf)
        fill[:xf] = fill[:xf] * w + audio[start: start + xf].astype(np.float64) * (1.0 - w)
        fill[-xf:] = fill[-xf:] * (1.0 - w) + audio[end - xf: end].astype(np.float64) * w
    audio[start:end] = fill.astype(np.float32)


def restore_dropouts_track(track: Track, cfg: Config) -> Track:
    max_gap_s = cfg.dropout_max_gap_ms() / 1000.0
    if max_gap_s <= 0:
        return track
    gaps = _find_dropouts(track.audio, cfg.sr, max_gap_s)
    if not gaps:
        log.info("dropouts: %s — none detected", track.name)
        return track
    audio = track.audio.copy()
    budget_s = _MAX_FILL_PER_MIN_S * (len(audio) / cfg.sr / 60.0)
    filled_s, n_filled = 0.0, 0
    for start, end in gaps:
        dur = (end - start) / cfg.sr
        if filled_s + dur > budget_s:
            log.warning("dropouts: %s — fill budget (%.1fs) reached, %d gap(s) left "
                        "unfilled (track may be corrupt rather than packet-lossy)",
                        track.name, budget_s, len(gaps) - n_filled)
            break
        _fill_gap(audio, cfg.sr, start, end)
        filled_s += dur
        n_filled += 1
    log.info("dropouts: %s — filled %d gap(s), %.0f ms total",
             track.name, n_filled, filled_s * 1000)
    return Track(track.name, audio)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_stages.py -v -k Dropouts`
Expected: 4 PASS. If `test_fills_short_gaps` fails on the RMS ratio, debug the detection first (`_find_dropouts` on the test signal must return exactly the two punched holes) before touching thresholds.

- [ ] **Step 5: Run the full stage suite for regressions**

Run: `uv run pytest tests/test_stages.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/podcare/stages/dropouts.py tests/test_stages.py
git commit -m "feat: dropout/short-gap restoration stage (two-sided LPC fill)"
```

---

### Task 4: Dynamic resonance suppression stage

**Files:**
- Create: `src/podcare/stages/resonance.py`
- Test: `tests/test_stages.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stages.py` (add `from podcare.stages.resonance import resonance_track` to imports):

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_stages.py -v -k Resonance`
Expected: ERROR — `ModuleNotFoundError: No module named 'podcare.stages.resonance'`

- [ ] **Step 3: Implement the stage**

Create `src/podcare/stages/resonance.py`:

```python
"""Dynamic resonance / harshness suppression ("Soothe-lite").

The static tonal-balance EQ cannot catch resonant peaks that come and go with
the voice — ringy room modes, nasal honk, 2-5 kHz harshness spikes — a major
cause of earbud fatigue. Per STFT frame, a median filter across frequency
estimates the broad spectral envelope; any bin that pokes more than a margin
above its own envelope is pulled back down by the excess (capped), with
attack/release smoothing across time so notches fade in and out musically.
Cut-only, narrow-band, bounded — at strength 0 the cap is 0 dB, a bitwise
no-op (and the stage is skipped anyway).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import istft, stft

from ..config import Config
from ..dsp import process_chunked
from ..session import Track

log = logging.getLogger(__name__)

_NFFT = 1024
_HOP = 256
_LO_HZ = 800.0      # leave the voice fundamental / warmth region alone
_HI_HZ = 9000.0     # above this is air, handled by de-ess / tonal balance
_ENV_BINS = 9       # ~420 Hz median window @48k/1024: bridges a resonant peak
_ATTACK_S = 0.005   # cut engages fast enough to catch a transient ring
_RELEASE_S = 0.080  # ...and lets go gently so notches never flutter


def _suppress_chunk(audio: np.ndarray, cfg: Config) -> np.ndarray:
    margin = cfg.resonance_margin_db()
    max_cut = cfg.resonance_max_cut_db()
    if max_cut <= 0.0:
        return audio
    f, _, z = stft(audio.astype(np.float64), fs=cfg.sr, nperseg=_NFFT,
                   noverlap=_NFFT - _HOP, padded=True)
    mag_db = 20.0 * np.log10(np.abs(z) + 1e-12)
    env_db = median_filter(mag_db, size=(_ENV_BINS, 1), mode="nearest")
    cut = np.clip(mag_db - env_db - margin, 0.0, max_cut)
    cut[~((f >= _LO_HZ) & (f <= _HI_HZ)), :] = 0.0
    # Asymmetric one-pole smoothing along time (attack = cut rising).
    frame_s = _HOP / cfg.sr
    a_att = float(np.exp(-frame_s / _ATTACK_S))
    a_rel = float(np.exp(-frame_s / _RELEASE_S))
    smoothed = np.empty_like(cut)
    prev = np.zeros(cut.shape[0])
    for j in range(cut.shape[1]):
        coef = np.where(cut[:, j] > prev, a_att, a_rel)
        prev = coef * prev + (1.0 - coef) * cut[:, j]
        smoothed[:, j] = prev
    z *= 10.0 ** (-smoothed / 20.0)
    _, out = istft(z, fs=cfg.sr, nperseg=_NFFT, noverlap=_NFFT - _HOP)
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    return out[: len(audio)].astype(np.float32)


def resonance_track(track: Track, cfg: Config) -> Track:
    if cfg.resonance_max_cut_db() <= 0.0:
        return track
    audio = process_chunked(track.audio, cfg.sr, lambda c: _suppress_chunk(c, cfg),
                            chunk_s=30.0, label=f"resonance · {track.name}")
    before = float(np.mean(track.audio.astype(np.float64) ** 2)) + 1e-20
    after = float(np.mean(audio.astype(np.float64) ** 2)) + 1e-20
    log.info("resonance: %s — %+.2f dB energy change",
             track.name, 10 * np.log10(after / before))
    return Track(track.name, audio)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_stages.py -v -k Resonance`
Expected: 2 PASS. If the ring isn't cut ≥ 3 dB, check that scipy's `stft` returns shape `(freq, time)` (it does) and that the 3 kHz bin actually exceeds envelope + margin (print `cut.max()` — it should reach `max_cut` ≈ 8.0 at strength 0.8).

- [ ] **Step 5: Commit**

```bash
git add src/podcare/stages/resonance.py tests/test_stages.py
git commit -m "feat: dynamic resonance/harshness suppression stage (Soothe-lite STFT notching)"
```

---

### Task 5: Harmonic exciter in the master chain

**Files:**
- Modify: `src/podcare/stages/master.py`
- Test: `tests/test_stages.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stages.py` (add `from podcare.stages.master import master_and_encode` and `import soundfile as sf` to imports):

```python
class TestExciter:
    def test_adds_high_band_energy(self, tmp_path):
        from scipy.signal import butter, sosfilt
        t = np.arange(SR * 4) / SR
        dull = (0.3 * (np.sin(2 * np.pi * 400 * t)
                       + 0.5 * np.sin(2 * np.pi * 2000 * t)
                       + 0.25 * np.sin(2 * np.pi * 5000 * t))).astype(np.float32)
        on_path, off_path = tmp_path / "on.wav", tmp_path / "off.wav"
        master_and_encode(Track("x", dull), Config(out_sr=SR), on_path)
        master_and_encode(Track("x", dull), Config(out_sr=SR, exciter=False), off_path)

        def hf_rms(p):
            a, _ = sf.read(p)
            sos = butter(4, 7000, btype="highpass", fs=SR, output="sos")
            return float(np.std(sosfilt(sos, a)))

        assert hf_rms(on_path) > hf_rms(off_path) * 1.5, \
            "exciter must add energy above 7 kHz"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_stages.py -v -k Exciter`
Expected: FAIL on the assertion (both files identical — no exciter exists yet)

- [ ] **Step 3: Implement**

In `src/podcare/stages/master.py`, add after `_limiter`:

```python
def _exciter(cfg: Config) -> str:
    """Harmonic presence exciter — synthesizes new harmonics above ~7.4 kHz to
    restore the "air" heavy denoise/dereverb removes, rather than boosting
    (possibly noisy) existing highs. Runs before the loudnorm measurement so
    the added energy is included in the loudness math. Conservative ceiling:
    amount tops out at 2.5 (easy to overdo — roadmap note)."""
    return f"aexciter=amount={cfg.exciter_amount():.2f}:freq=7400"
```

In `master_and_encode`, directly after the multiband line
(`audio = _apply_multiband(track.audio, cfg) if (cfg.compress and cfg.s > 0) else track.audio`), insert:

```python
    if cfg.exciter and cfg.s > 0:
        excited = audio_io.filter_array(audio, cfg.sr, _exciter(cfg))
        # aexciter preserves length; pin it exactly like the multiband pass.
        audio = (excited[: len(audio)] if len(excited) >= len(audio)
                 else np.pad(excited, (0, len(audio) - len(excited))))
        log.info("master: exciter amount=%.2f", cfg.exciter_amount())
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_stages.py -v -k Exciter`
Expected: PASS. If ffmpeg errors on the filter string, run `ffmpeg -h filter=aexciter` and adjust only the parameter syntax (keep `amount` strength-scaled).

- [ ] **Step 5: Commit**

```bash
git add src/podcare/stages/master.py tests/test_stages.py
git commit -m "feat(master): harmonic presence exciter before loudness normalization"
```

---

### Task 6: Intro/outro bookends — assembly in master, plumbing in pipeline

**Files:**
- Modify: `src/podcare/stages/master.py`
- Modify: `src/podcare/pipeline.py` (the `run()` function, around lines 129-131 and 170)
- Test: `tests/test_stages.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stages.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_stages.py -v -k Bookends`
Expected: FAIL — `master_and_encode() got an unexpected keyword argument 'intro'`

- [ ] **Step 3: Implement in master.py**

Add to the imports in `src/podcare/stages/master.py`:

```python
from ..dsp import crossfade_concat, db_to_lin
```

(`db_to_lin` is already imported — just add `crossfade_concat` to that line.)

Add module constant near the other constants:

```python
_BOOKEND_XF_S = 0.1  # equal-power crossfade at each bookend join
```

Add two helpers after `_exciter`:

```python
def _prepare_bookend(audio: np.ndarray, cfg: Config, name: str) -> np.ndarray:
    """Loudness-align an intro/outro to the program target so a music sting
    cannot blast ears relative to speech. A near-silent bookend (below
    loudnorm's -70 LUFS gate) is used as-is, mirroring the silent-program
    handling in master_and_encode."""
    measured = audio_io.measure_loudnorm(audio, cfg.sr, pre_filters=None,
                                         lufs=cfg.lufs, true_peak=cfg.true_peak_db)
    if not (_finite(measured.get("input_i")) and _finite(measured.get("input_tp"))
            and _finite(measured.get("target_offset"))):
        log.warning("master: %s sound is silent/near-silent — using it as-is", name)
        return audio
    loudnorm = (
        f"loudnorm=I={cfg.lufs}:TP={cfg.true_peak_db}:LRA=11"
        f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )
    return audio_io.filter_array(audio, cfg.sr, loudnorm)


def _assemble_bookends(program: np.ndarray, cfg: Config,
                       intro: np.ndarray | None,
                       outro: np.ndarray | None) -> tuple[np.ndarray, bool]:
    """intro ⤳ program ⤳ outro with equal-power crossfades. Returns
    (audio, joined); joined=False means no bookends were given."""
    if intro is None and outro is None:
        return program, False
    parts: list[np.ndarray] = []
    if intro is not None:
        parts.append(_prepare_bookend(intro, cfg, "intro"))
    parts.append(program)
    if outro is not None:
        parts.append(_prepare_bookend(outro, cfg, "outro"))
    # Crossfade 100 ms, clamped to half the shortest bookend so a tiny sting
    # is never consumed whole by its own fade.
    xf = int(_BOOKEND_XF_S * cfg.sr)
    for p in parts:
        if p is not program:
            xf = min(xf, max(2, len(p) // 2))
    joined = crossfade_concat(parts, xf)
    log.info("master: bookends — intro=%s outro=%s xfade=%dms",
             intro is not None, outro is not None, int(1000 * xf / cfg.sr))
    return joined, True
```

Change the `master_and_encode` signature to:

```python
def master_and_encode(track: Track, cfg: Config, out_path: Path, *,
                      intro: np.ndarray | None = None,
                      outro: np.ndarray | None = None) -> None:
```

Apply assembly in all three exit paths:

(a) the `if not cfg.master:` early path becomes:

```python
    if not cfg.master:
        assembled, _ = _assemble_bookends(track.audio, cfg, intro, outro)
        audio_io.encode(assembled, cfg.sr, out_path,
                        out_sr=cfg.out_sr, lossy_bitrate=cfg.lossy_bitrate)
        return
```

(b) the silent-program warning path: replace its `audio_io.encode(audio, ...)` with:

```python
        assembled, _ = _assemble_bookends(audio, cfg, intro, outro)
        audio_io.encode(assembled, cfg.sr, out_path, out_sr=cfg.out_sr,
                        lossy_bitrate=cfg.lossy_bitrate)
        return
```

(c) the normal path: replace the final two lines
(`mastered = ...` and `audio_io.encode(mastered, ...)`) with:

```python
    mastered = audio_io.filter_array(audio, cfg.sr, f"{loudnorm},{_limiter(cfg)}")
    assembled, joined = _assemble_bookends(mastered, cfg, intro, outro)
    if joined:
        # crossfade overlaps of two TP-limited signals can momentarily sum
        # above the ceiling — one more limiter pass over the joined program.
        assembled = audio_io.filter_array(assembled, cfg.sr, _limiter(cfg))
    audio_io.encode(assembled, cfg.sr, out_path, out_sr=cfg.out_sr,
                    lossy_bitrate=cfg.lossy_bitrate)
```

- [ ] **Step 4: Plumb decode-early through pipeline.py**

In `src/podcare/pipeline.py` `run()`, directly after `session = load_session(inputs, cfg)` add:

```python
        # Decode bookends up front — a corrupt sting must fail now, not after
        # an hour of processing (same fail-fast contract as the out_path check).
        intro = audio_io.decode(cfg.intro_sound, cfg.sr) if cfg.intro_sound else None
        outro = audio_io.decode(cfg.outro_sound, cfg.sr) if cfg.outro_sound else None
```

And change the master call (`master.master_and_encode(final, cfg, out_path)`) to:

```python
        master.master_and_encode(final, cfg, out_path, intro=intro, outro=outro)
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_stages.py -v -k Bookends`
Expected: 2 PASS

- [ ] **Step 6: Run the whole stage file**

Run: `uv run pytest tests/test_stages.py -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/podcare/stages/master.py src/podcare/pipeline.py tests/test_stages.py
git commit -m "feat: intro/outro bookends — loudness-matched assembly with crossfades"
```

---

### Task 7: Wire new stages into pipeline.STAGES and the CLI

**Files:**
- Modify: `src/podcare/pipeline.py` (imports + `STAGES`)
- Modify: `src/podcare/cli.py` (flags, `_validate`, nocut policy, `Config(...)`)
- Test: `tests/test_cli.py` (create if it does not exist)

- [ ] **Step 1: Write the failing tests**

Create (or append to) `tests/test_cli.py`:

```python
import logging
from pathlib import Path

import pytest

from podcare import cli
from podcare.pipeline import STAGES


def _args(argv):
    return cli.build_parser().parse_args(argv)


def test_new_stages_registered_in_order():
    names = [s.name for s in STAGES]
    assert names.index("dropouts") < names.index("repair")
    assert names.index("deess") < names.index("resonance") < names.index("gate")


def test_bookend_flags_parse(tmp_path):
    intro = tmp_path / "i.wav"
    intro.write_bytes(b"")
    args = _args(["in.wav", "-o", "out.mp3", "--intro-sound", str(intro)])
    assert args.intro_sound == intro and args.outro_sound is None


def test_nocut_ignores_bookends(caplog):
    args = _args(["in.wav", "-o", "out.mp3", "--nocut",
                  "--intro-sound", "i.wav", "--outro-sound", "o.wav"])
    with caplog.at_level(logging.WARNING):
        intro, outro = cli._effective_bookends(args)
    assert intro is None and outro is None
    assert "--intro-sound" in caplog.text and "--nocut" in caplog.text


def test_bookends_kept_without_nocut():
    args = _args(["in.wav", "-o", "out.mp3", "--intro-sound", "i.wav"])
    intro, outro = cli._effective_bookends(args)
    assert intro == Path("i.wav") and outro is None


def test_validate_rejects_missing_bookend(tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(b"")
    args = _args([str(src), "-o", str(tmp_path / "out.mp3"),
                  "--intro-sound", str(tmp_path / "missing.wav")])
    with pytest.raises(SystemExit, match="intro-sound"):
        cli._validate(args)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `'dropouts' is not in list`, `unrecognized arguments: --intro-sound`, `AttributeError: ... '_effective_bookends'`

- [ ] **Step 3: Register the stages in pipeline.py**

In `src/podcare/pipeline.py`, extend the stages import to include `dropouts` and `resonance` (the `from .stages import (...)` list). Then:

(a) Insert as the FIRST entry of `STAGES` (before `Stage("repair", ...)`):

```python
    Stage("dropouts", lambda c: c.s > 0 and c.dropouts, "track",
          dropouts.restore_dropouts_track,
          lambda c: f"fill 3-{c.dropout_max_gap_ms():.0f}ms packet-loss gaps "
                    f"(two-sided LPC, speech-gated)"),
```

(b) Insert directly AFTER the `Stage("deess", ...)` entry:

```python
    Stage("resonance", lambda c: c.s > 0 and c.resonance, "track",
          resonance.resonance_track,
          lambda c: f"dynamic notching 800-9000Hz margin={c.resonance_margin_db():.1f}dB "
                    f"max-cut={c.resonance_max_cut_db():.1f}dB"),
```

- [ ] **Step 4: Add CLI flags, validation, nocut policy, Config wiring**

In `src/podcare/cli.py`:

(a) In `build_parser()`, after the `--keep-stems` argument add:

```python
    p.add_argument("--intro-sound", type=Path, default=None, metavar="AUDIO",
                   help="optional sound placed before the program (loudness-matched to "
                        "the output target, joined with a 100 ms crossfade); ignored "
                        "with --nocut")
    p.add_argument("--outro-sound", type=Path, default=None, metavar="AUDIO",
                   help="optional sound placed after the program (loudness-matched, "
                        "100 ms crossfade); ignored with --nocut")
```

(b) In the `toggles` for-loop list in `build_parser()`, insert three tuples at
these exact positions:

- `("dropouts", "dropout / short-gap restoration"),` as the FIRST list entry
  (before `("declip", ...)`)
- `("resonance", "dynamic resonance / harshness suppression"),` directly after
  `("deess", "de-essing")`
- `("exciter", "harmonic presence exciter"),` directly after
  `("leveler", "slow segment-loudness leveling")`

(c) In `_validate(args)`, add at the end:

```python
    for flag, path in (("--intro-sound", args.intro_sound),
                       ("--outro-sound", args.outro_sound)):
        if path is not None and not path.exists():
            raise SystemExit(f"error: {flag} file not found: {path}")
```

(d) Add a helper above `main()`:

```python
def _effective_bookends(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    """Bookends after the --nocut policy. An intro shifts the entire timeline,
    defeating --nocut's sample-alignment purpose, so both are ignored there."""
    if not args.nocut:
        return args.intro_sound, args.outro_sound
    ignored = [name for name, given in (("--intro-sound", args.intro_sound),
                                        ("--outro-sound", args.outro_sound)) if given]
    if ignored:
        log.warning("%s ignored because --nocut keeps the original timeline",
                    ", ".join(ignored))
    return None, None
```

(e) In `main()`, before constructing `Config`, add:

```python
    intro_sound, outro_sound = _effective_bookends(args)
```

and add to the `Config(...)` construction:

```python
        dropouts=not args.no_dropouts,
        resonance=not args.no_resonance,
        exciter=not args.no_exciter,
        intro_sound=intro_sound,
        outro_sound=outro_sound,
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all PASS

- [ ] **Step 6: Run the FULL suite**

Run: `uv run pytest -q`
Expected: all PASS (note: any test asserting stage counts/log tags `[i/16]` would need the new total `[i/18]` — `n_total` derives from `len(STAGES)` so code is automatic; fix any literal-count assertions the suite surfaces)

- [ ] **Step 7: Commit**

```bash
git add src/podcare/pipeline.py src/podcare/cli.py tests/test_cli.py
git commit -m "feat: register dropouts/resonance stages; CLI flags for new stages + bookends"
```

---

### Task 8: README documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the two new pipeline stage sections**

Insert a new section between `### 0. Decode` and `### 1. Repair`:

```markdown
### 1. Dropout restoration — `--no-dropouts` *(per track)*

Remote-guest tracks (VoIP, double-enders over flaky links) arrive with brief
holes — 3–50 ms of missing signal where packets were lost, audible as tiny
stutters. Each hole is refilled by linear prediction: the speech is
extrapolated forward from the audio before the gap and backward from the audio
after it, and the two estimates are crossfaded across the gap.

Strict caps keep it honest: only gaps up to a strength-scaled maximum
(0 ms at strength 0 → 50 ms at 1) are filled, only when the surroundings are
speech-active (a quiet moment inside a real pause is not a dropout), and never
more than ~1.2 s per minute — beyond that the track is corrupt, not
packet-lossy.

- **Strength mapping:** longest fillable gap `0 → 50 ms`.
- **Runs first** so every later stage (including declick) sees gap-free audio.
```

Insert a new section between the De-ess section and the Gate section:

```markdown
### 11. Resonance suppression — `--no-resonance` *(per track)*

A "Soothe-lite" dynamic resonance tamer. Static EQ can't catch resonant peaks
that come and go with the voice — ringy room modes, nasal honk, 2–5 kHz
harshness spikes — a major cause of earbud fatigue. Per STFT frame, a median
filter across frequency estimates the broad spectral envelope; any bin poking
more than a margin above its own envelope is pulled back down by the excess
(capped), with attack/release smoothing so notches fade in and out musically.
Cut-only, and only between 800 Hz and 9 kHz.

- **Strength mapping:** detection margin `18 → 6 dB`; max cut `0 → 10 dB`
  (a true no-op at strength 0).
```

- [ ] **Step 2: Renumber the stage headings**

Explicit mapping (old → new): Decode stays `0`; Repair `1→2`, De-hum `2→3`,
Align `3→4`, Denoise `4→5`, Dereverb `5→6`, Tonal balance `6→7`,
Mouth-click `7→8`, Plosives `8→9`, De-ess `9→10` — then the new Resonance
section is `11` — Gate `10→12`, Breath `11→13`, Fillers `12→14`,
Mixdown `13→15`, Pause tightening `14→16`, Leveler `15→17`, Master `16→18`,
Encode `17→19`. Update any intra-doc anchors that reference the numbers.

- [ ] **Step 3: Document the exciter inside the Master section**

Add to the Master section's description list:

```markdown
- **Harmonic exciter** (`--no-exciter`): synthesizes a touch of new harmonics
  above ~7.4 kHz (ffmpeg `aexciter`) to restore the "air" heavy
  denoise/dereverb removes and to cut through tiny speakers — rather than
  boosting (possibly noisy) existing highs. Runs before the loudness
  measurement so the added energy is included in the loudness math.
  Strength maps `amount 0 → 2.5`; deliberately conservative.
```

- [ ] **Step 4: Document the bookends**

(a) Add to the `### General` CLI options table/list:

```markdown
- `--intro-sound AUDIO` / `--outro-sound AUDIO` — optional sounds placed
  before/after the finished program. Each is loudness-matched to the output
  target and joined with a 100 ms equal-power crossfade. Ignored with
  `--nocut` (an intro would shift the whole timeline).
```

(b) Add a short subsection after the Encode section:

```markdown
## Intro / outro

`--intro-sound` and `--outro-sound` append a sting or theme around the
finished program **after all processing**: each file (anything ffmpeg reads;
downmixed to mono) is loudness-normalized to the same target as the program —
so a hot music sting can't blast ears relative to speech — then joined with a
100 ms equal-power crossfade and passed through one final true-peak limiter.
Not available with `--nocut`, which promises a sample-aligned timeline.
```

(c) Add the three new toggles to the `### Stage toggles` list:
`--no-dropouts`, `--no-resonance`, `--no-exciter` with one-line descriptions
matching the CLI help strings.

- [ ] **Step 5: Update the roadmap section**

In `## Roadmap — remaining proposed stages`: move items 1–3 into the
"✅ Shipped from the original roadmap" note (append "dynamic resonance
suppression, harmonic presence exciter, and dropout / short-gap restoration")
and delete their table rows, leaving only item 4 (music-bed ducking + stereo
delivery) in the table.

- [ ] **Step 6: Verify and commit**

Run: `uv run pytest -q` (docs change — suite must still be green)

```bash
git add README.md
git commit -m "docs: document dropouts, resonance, exciter stages + intro/outro bookends"
```

---

### Task 9: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q`
Expected: all PASS

- [ ] **Step 2: Full pipeline on real audio with stems**

Run:
```bash
uv run podcare tests/test-input-audio/hodinky/hodinky.mp3 --nocut --strength 0.8 \
  --keep-stems "$TMPDIR/stems" -o "$TMPDIR/hodinky_e2e.mp3"
```
Expected: `[1/18] dropouts` and the resonance stage appear in the log with
their describe lines; exciter logged inside master; pipeline completes;
stems for the new stages exist in `$TMPDIR/stems`.

- [ ] **Step 3: Bookends smoke test**

Generate a sting and run with bookends (no `--nocut`):
```bash
ffmpeg -y -f lavfi -i "sine=frequency=440:duration=2" -ar 48000 "$TMPDIR/sting.wav"
uv run podcare tests/test-input-audio/hodinky/hodinky.mp3 --strength 0.8 \
  --intro-sound "$TMPDIR/sting.wav" --outro-sound "$TMPDIR/sting.wav" \
  -o "$TMPDIR/hodinky_bookends.mp3"
```
Expected: log line `master: bookends — intro=True outro=True xfade=100ms`;
output duration ≈ program + 4 s − 0.2 s.

Also verify the nocut warning:
```bash
uv run podcare tests/test-input-audio/hodinky/hodinky.mp3 --nocut \
  --intro-sound "$TMPDIR/sting.wav" -o "$TMPDIR/hodinky_nocut.mp3" 2>&1 | head -5
```
Expected: `--intro-sound ignored because --nocut keeps the original timeline`.

- [ ] **Step 4: Strength sweep**

Run strength 0 and confirm the new stages are skipped:
```bash
uv run podcare tests/test-input-audio/hodinky/hodinky.mp3 --strength 0 \
  -o "$TMPDIR/hodinky_s0.mp3" 2>&1 | grep -E "dropouts|resonance|exciter|skipped"
```
Expected: dropouts/resonance lines show `skipped (disabled)`; no exciter log line.

- [ ] **Step 5: Listen** (user step — flag for the user)

A/B `$TMPDIR/stems` pairs around the new stages and both bookend joins.
