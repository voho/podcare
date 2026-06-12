# Roadmap stages 1–3 + intro/outro bookends — design

**Date:** 2026-06-12 · **Status:** implemented (2026-06-12)

> **Implementation deviations** (each empirically validated during review, with
> rationale in code comments): dropouts uses LPC order **96** / **80 ms**
> context (not ~32/~30 ms — zero-crossing phase resolution + clean context
> after boundary trimming) and an inline speech-level gate (not
> `speech_threshold`, whose noise-floor term mis-fires on uniform-level
> signals); the exciter uses **freq=4000, ceil=16000** (not ~7.4 kHz — aexciter
> derives harmonics FROM the band above `freq`, so 4–8 kHz speech content
> lands at 8–16 kHz); bookend crossfades clamp **per join**, and joined output
> gets a safety limiter pass in the no-master path too.

## Context

The README roadmap (`## Roadmap — remaining proposed stages`) lists four design-reviewed
proposals. The user asked to implement items 1–3 (item 4, music-bed ducking + stereo
delivery, stays on the roadmap — its I/O-contract change is out of scope). During design
the user added one new feature: optional intro/outro sounds appended to the finished
program. Goals: keep speech natural, balanced, and intelligible without losing the
original's energy. Constraints (from the roadmap contract): every stage obeys the
`--strength` knob with a true no-op at strength 0, slots into the existing stage order,
and adds no heavyweight dependency.

## 1. Dynamic resonance / harshness suppression

- **Module:** `src/podcare/stages/resonance.py`, per-track stage.
- **Slot:** immediately after `deess` in `pipeline.STAGES` (de-ess handles its tuned
  sibilance band first; this catches remaining transient resonant peaks anywhere in the
  mid band).
- **Toggle:** `resonance: bool = True` in Config; `--no-resonance` in CLI.
- **Algorithm ("Soothe-lite"):** scipy STFT, size 1024 / hop 256 at the 48 kHz working
  rate. Per frame: spectral envelope = median smoothing of log-magnitude across
  frequency (window ~9 bins ≈ 420 Hz — wide enough to bridge a resonant peak, narrow
  enough to track formant structure). Excess = bin dB − envelope dB − margin. Gain
  reduction per bin = min(excess, max_cut), applied only inside ~800 Hz–9 kHz.
  Attack/release smoothing of the per-bin gain across frames (attack ~5 ms, release
  ~80 ms, via the existing `smooth_gain` pattern adapted to 2-D) so notches fade
  musically instead of gating. iSTFT back; trim/pad to input length.
- **Strength mapping:** `resonance_margin_db()` lerps 18 → 6 dB; `resonance_max_cut_db()`
  lerps 0 → 10 dB. Max-cut 0 at strength 0 makes the math itself a no-op (and the stage
  is skipped at s = 0 like every stage).
- **Memory:** runs through `process_chunked` (30 s chunks are fine; the stage is cheap).

## 2. Harmonic presence exciter

- **Module:** extends `src/podcare/stages/master.py` (it is an ffmpeg filter node, not a
  per-track stage).
- **Slot:** in the master chain **after** multiband compression and **before** the
  loudnorm measurement pass, so the synthesized harmonics are included in the loudness
  math and cannot push the program over target. Chain becomes:
  multiband → aexciter → measure → loudnorm → limiter.
- **Toggle:** `exciter: bool = True`; `--no-exciter`.
- **Filter:** ffmpeg `aexciter`, onset ~7.4 kHz (default region), conservative ceiling:
  `exciter_amount()` lerps 0 → 2.5 with strength (aexciter `amount`; 0 = no-op),
  mild fixed `drive`/`blend` defaults. Reuses `audio_io.filter_array`.
- Applied only when `cfg.master and cfg.exciter and cfg.s > 0` (same guard style as
  `compress`).

## 3. Dropout / short-gap restoration

- **Module:** `src/podcare/stages/dropouts.py`, per-track stage.
- **Slot:** **before** `repair` (first processing stage) — restores continuity so
  declick never rings on hard gap edges and every later stage sees gap-free audio.
- **Toggle:** `dropouts: bool = True`; `--no-dropouts`.
- **Detection:** block RMS (existing `block_rms`, ~3 ms hop). A dropout is a run of
  blocks ≥ ~25 dB below the local context median (±300 ms), lasting 3–50 ms, with sharp
  edges (level recovers on both sides). Gaps bordering real silence are excluded: the
  surrounding context must be speech-active per the existing `speech_threshold` helper —
  this is the "never fabricate content in a pause" guard.
- **Fill:** two-sided LPC extrapolation. Levinson–Durbin (numpy-only, ~order 32) on
  ~30 ms of clean context each side; extrapolate forward from the pre-gap context and
  backward from the post-gap context; equal-power crossfade the two estimates across the
  gap; short (~2 ms) crossfades into the surrounding real audio.
- **Honesty caps:** max fillable gap `dropout_max_gap_ms()` lerps 0 → 50 ms with
  strength (no-op at 0); total filled time capped at ~1.2 s per minute of audio (2 %) —
  beyond that the track is likely corrupt, not packet-lossy, and we stop filling and log.
- **Logging:** count + total ms filled per track (mirrors other stages' info lines).

## 4. Intro/outro bookends

- **CLI:** `--intro-sound PATH`, `--outro-sound PATH` (both optional, independent).
  Config: `intro_sound: Path | None`, `outro_sound: Path | None`.
- **Fail fast:** path existence checked in `_validate`; files are decoded at pipeline
  start (alongside `load_session`), not at the end — a corrupt sting must not kill a
  multi-hour render at the last step. Decoded via existing `audio_io.decode` (any ffmpeg
  format, downmixed to mono, resampled to the working rate). Held in memory (stings are
  seconds long) and passed from `pipeline.run` into `master_and_encode` as explicit
  arguments.
- **Assembly (in `master_and_encode`, after program loudnorm+limit):**
  1. Each bookend is loudness-normalized to the same `cfg.lufs` target as the program
     (two-pass loudnorm reusing `measure_loudnorm`; a music sting must not blast ears
     relative to speech). A silent/near-silent bookend (loudnorm gate −70) is used as-is
     with a warning, mirroring the master's silent-program handling.
  2. Join: intro ⤳ program ⤳ outro with **100 ms equal-power crossfades**. The
     segment-join logic currently embedded in `remove_intervals` (dsp.py) is extracted
     into a shared `crossfade_concat(segments, xf)` helper and reused. If a bookend is
     shorter than 200 ms, the crossfade clamps to half its length.
  3. One final true-peak limiter pass (`_limiter`) over the joined audio — crossfade
     overlaps can sum above the TP target — then the single encode.
- **`--nocut` interaction:** an intro shifts the whole timeline, defeating `--nocut`'s
  sample-alignment purpose. Under `--nocut`, both flags are ignored with a warning
  (extends the existing ignored-flags warning block in `cli.main`).
- **Not strength-scaled** — this is assembly, not processing; it runs whenever paths are
  given (like `--out-sr`).

## Cross-cutting

- **Config:** 3 new bool toggles + 2 path fields + strength helpers
  (`resonance_margin_db`, `resonance_max_cut_db`, `exciter_amount`,
  `dropout_max_gap_ms`) following the `_lerp` pattern with documented no-op-at-0
  endpoints.
- **Pipeline:** `dropouts` and `resonance` registered in `STAGES` with `describe`
  lambdas (effective params logged at start, consistent with existing stages). Exciter
  is described inside the master log line.
- **README:** new pipeline sections for Dropout restoration (before Repair) and
  Resonance suppression (after De-ess); exciter documented inside the Master section;
  sections renumbered; bookends documented in CLI options + a new "Intro / outro"
  subsection; roadmap table updated — items 1–3 move to the "shipped" note, item 4
  remains.
- **CLI `--help`:** all five new flags with one-line help text.

## Testing

Synthetic tests in `tests/test_stages.py` following existing patterns (e.g. the
dereverb RIR test):

1. **Resonance:** speech-shaped noise + injected 3 kHz ringing burst → burst band energy
   reduced ≥ 3 dB while frames without the burst change < 0.5 dB; at strength 0 the
   stage is skipped (output is the input, unchanged); output finite, length preserved.
2. **Exciter:** band-limited (≤ 6 kHz) signal → master output gains energy above 7 kHz
   with exciter on vs `--no-exciter`; level stays at LUFS target.
3. **Dropouts:** speech-like signal with punched 20 ms holes → holes refilled
   (energy in gap ≥ ~50 % of neighbors, no discontinuity clicks); 80 ms holes →
   untouched; holes inside real pauses → untouched; strength 0 → identity.
4. **Bookends:** output length = intro + program + outro − 2 × crossfade; crossfade
   regions blend monotonically; `--nocut` + bookends → warning, bookends absent;
   missing file → clean `_validate` error. `crossfade_concat` extraction covered by
   existing `remove_intervals` tests staying green.

## Verification (end-to-end)

1. `uv run pytest` — full suite green.
2. Full run on `tests/test-input-audio/hodinky/hodinky.mp3` at strength 0.8 with
   `--keep-stems` — A/B the new stems (dropouts, resonance) and the exciter on/off
   master; confirm per-stage log lines and timings look sane.
3. Run with `--intro-sound`/`--outro-sound` on a short music sting — listen to both
   joins; confirm loudness match and no clicks.
4. Strength sweep 0 / 0.4 / 0.8 / 1.0 — confirm graceful scaling, and strength 0 output
   identical to before this change.
