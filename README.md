![logo](https://github.com/voho/podcare/blob/main/logo.png?raw=true)

# Podcare

Turn raw podcast mic recordings into a polished, broadcast-ready episode with one command.

```bash
podcare host.wav guest.wav -o episode.mp3
```

**Input:** one or more WAV/MP3/FLAC files (one per mic/recorder; anything ffmpeg reads).
**Output:** one WAV/MP3/FLAC/M4A — aligned, declipped, denoised (DeepFilterNet3
neural enhancement), dereverbed, de-plosived, de-essed, filler-words removed,
pauses tightened, compressed, and loudness-normalized to podcast standard
(−16 LUFS / −1.5 dBTP). Output is 44.1 kHz (16-bit + dither for WAV/FLAC);
processing runs internally at 48 kHz float32. Defaults favor quality over speed.

---

## Install

Requires [ffmpeg](https://ffmpeg.org) on PATH and Python ≥ 3.11.

```bash
uv sync            # installs torch, DeepFilterNet, faster-whisper, nara-wpe, …
uv run podcare --help
```

Everything runs on CPU. The first run that uses filler removal downloads the
Whisper model (`large-v3` is ~3 GB; pick a smaller `--whisper-model` to skip
that) plus the wav2vec2 forced-alignment model (~360 MB), both cached afterward.
DeepFilterNet3 weights ship inside the `deepfilternet` package, so denoise works
offline from the start.

---

## Usage

```bash
# Two mics, full polish, MP3 out
uv run podcare host.wav guest.wav -o episode.mp3

# Gentler overall treatment
uv run podcare host.wav guest.wav -o episode.mp3 --strength 0.4

# Single track to 16-bit/44.1k WAV, force-aggressive filler removal
uv run podcare interview.mp3 -o clean.wav --filler-sensitivity 0.9

# Pin the language so filler detection never mis-detects (and never crashes
# on an unsupported one — see the Filler section)
uv run podcare cz-show.flac -o out.mp3 --language cs

# Turn off the stages you don't want
uv run podcare raw.wav -o out.wav --no-dereverb --no-tighten

# Faster preview pass (smaller Whisper, lighter dereverb)
uv run podcare a.flac b.flac -o draft.mp3 --whisper-model small --strength 0.5

# Debug: write every stage's intermediate audio (and the final master) so you
# can A/B them
uv run podcare a.wav b.wav -o out.wav --keep-stems stems/
```

Each input file is treated as **one speaker's mic**. Give Podcare the separate
recorder/mic tracks, not a pre-mixed file, so it can align them, gate crosstalk,
balance levels, and detect each speaker's fillers on a clean isolated voice
before summing. A single pre-mixed file works too — the multi-track-only stages
(align, mixdown) simply no-op, and filler detection runs on the one track.

---

## The `--strength` knob

Podcare has **one universal intensity dial**, `--strength` (0–1, default
**0.8**). Every stage maps it to its own notion of "how hard to work" — more
noise removed, deeper de-essing, tighter pauses, firmer compression, and so on.
At `--strength 0` every enhancement stage is a true no-op and is skipped
entirely: the pipeline becomes just **align → mixdown → loudness-normalize →
encode** (so the output still hits your `--lufs` target, but its tone and
dynamics are untouched — useful as an A/B baseline). `1` is the most aggressive.
The exact per-stage mapping is listed in each [pipeline section](#the-pipeline)
below, and every value is derived from
[src/podcare/config.py](src/podcare/config.py) (the single source of truth).

Two stages ignore strength on purpose: **repair** and **align** are *correctness*
operations (fix clipping, fix timing/polarity), not matters of degree.

You can still override individual stages — `--filler-sensitivity`, `--max-pause`,
`--target-pause`, `--lufs`, `--language` — and an explicit value always wins over
what `--strength` would have chosen.

---

## Command-line options

### General

| Flag | Default | Description |
|---|---|---|
| `AUDIO…` (positional) | — | One or more input files, one per mic/recorder. Any format ffmpeg can decode. All are resampled to 48 kHz mono float internally. |
| `-o, --output PATH` | *required* | Output file. The extension picks the container: `.wav`, `.mp3`, `.flac`, `.m4a`/`.aac`. Parent dirs are created automatically. |
| `-v, --verbose` | off | Debug-level logging (per-stage decisions, offsets, gains, frame counts). |
| `--version` | — | Print version and exit. |

### Tuning

| Flag | Default | Description |
|---|---|---|
| `--strength 0..1` | `0.8` | Universal processing intensity (see [above](#the-strength-knob)). Scales every strength-driven parameter in the pipeline. |
| `--filler-sensitivity 0..1` | follows `--strength` | Override how aggressively non-lexical fillers ("um", "uh", "ehm", "hmm", …) are cut. `0` disables the stage. When unset, defaults to `0.7 × strength` (deliberately conservative to protect real speech). |
| `--whisper-model NAME` | `large-v3` | faster-whisper model used to transcribe before forced alignment. `large-v3` is most accurate (and slowest); `medium`/`small`/`base`/`tiny` trade accuracy for speed and download size. |
| `--language CODE` | auto | Force the spoken language for filler detection (e.g. `en`, `cs`, `de`). When unset, the language is auto-detected per track. Pin it for bilingual shows or to avoid a mis-detection. |
| `--max-pause SECONDS` | follows `--strength` | Override: silences longer than this get shortened. When unset, `lerp(4.0 → 1.0)` over strength. Must exceed `--target-pause`. |
| `--target-pause SECONDS` | follows `--strength` | Override: the length an over-long pause is shortened *to*. When unset, `lerp(1.2 → 0.4)` over strength. |
| `--lufs DB` | `-16` | Output integrated-loudness target (EBU R128). Validated to `-40 … -5`. `-16` is the podcast/streaming norm; `-14` louder, `-19`/`-23` broadcast-quieter. |
| `--out-sr HZ` | `44100` | Output sample rate (validated `8000 … 192000`). Resampling happens exactly once, at the end. WAV/FLAC are always 16-bit + dither; MP3/M4A carry no bit depth. |
| `--bitrate RATE` | `192k` | Bitrate for lossy outputs (MP3/AAC), e.g. `256k`, `320k`. Ignored for WAV/FLAC. |
| `--keep-stems DIR` | off | Write each stage's intermediate audio **and the final master** into `DIR` as numbered files — great for hearing what each stage did. |

### Stage toggles

Each switch disables one stage; everything else still runs.

| Flag | Disables |
|---|---|
| `--no-declip` | Distortion repair (declick + declip + rumble high-pass) |
| `--no-align` | Inter-track time-offset and polarity correction |
| `--no-denoise` | Noise reduction (DeepFilterNet3) |
| `--no-dereverb` | WPE dereverberation |
| `--no-plosives` | Plosive ("p-pop") ducking |
| `--no-deess` | De-essing (sibilance control) |
| `--no-gate` | Crosstalk/room gate **and** per-track level matching |
| `--no-fillers` | Filler-word removal |
| `--no-tighten` | Pause tightening / dead-air trimming |
| `--no-master` | Bus compression + loudness normalization |

---

## The pipeline

Stages run in this fixed order. **Track-level** stages process each mic
independently; **session-level** stages see all tracks at once. The ordering is
deliberate: repair before anything reads the signal, all per-mic cleanup
(including filler detection) before the tracks are summed, and the only post-sum
edit is on the single mono program — so the tracks can never drift out of sync.

```
            ┌──────────────────────── per track ─────────────────────────┐
decode ──▶ repair ─ align ─ denoise ─ dereverb ─ plosives ─ deess ─ gate ─ fillers ─┐
 (load     (declick (offset  (DFN3)    (WPE)     (LF burst (sibilance (expander (per-mic   │
  48k)      declip   + pol.            tail)      ducking)  control)  + level)  ASR cuts)   │
            HPF)     fix)                                                                   ▼
                                                                                       mixdown
   encode ◀──── master ◀──── tighten ◀──────────────────────────────────────────────  (sum→mono)
  (resample   (compress     (shorten
   44.1k/16    loudnorm      dead air)
   + dither)   +TP-limit)
```

Every stage section below follows the **same layout**:

- **Fixes** — the podcast defect it removes.
- **How it works** — the algorithm, in plain terms.
- **Strength** — how `--strength` drives it (or why it ignores strength).
- **Parameters** — every knob and how it's controlled: `Strength` (scaled by
  `--strength`), a CLI flag, a `Config` field (advanced, edit
  [src/podcare/config.py](src/podcare/config.py)), or `Hardcoded`. All
  strength-derived numbers show the `0 → 1` range and the value at the default
  `0.8`.

---

### 0. Decode

**Fixes.** Format/rate/channel-count differences between source files.

**How it works.** Every input is decoded through ffmpeg to **48 kHz mono
float32**. 48 kHz is DeepFilterNet3's native rate, so the entire chain runs at
one rate and resamples just once at the very end. Stereo inputs are downmixed to
mono (podcast voice is one channel per mic). A file that decodes to zero samples
is rejected immediately with a clear error.

**Strength.** Not applicable — this is I/O.

| Parameter | Value | Controlled by |
|---|---|---|
| Internal sample rate | 48000 Hz | `Config.sr` (do not change — DFN3 requires 48 k) |
| Channels | mono | Hardcoded |

---

### 1. Repair — `--no-declip` *(per track)*

**Fixes.** Clicks, glitches, hard clipping (recorded too hot), and subsonic
rumble/thumps.

**How it works.** Three ffmpeg filters run in series: **`adeclick`** interpolates
over impulsive clicks; **`adeclip`** reconstructs samples driven past full-scale,
restoring the rounded peaks clipping flattened; and a **2-pole high-pass**
removes rumble, desk thumps, and HVAC roar below the voice fundamental.
Distortion repaired here can't be denoised away later, which is why it's first.

**Strength.** Not strength-scaled — this is restoration, not a degree of effect.

| Parameter | Value | Controlled by |
|---|---|---|
| Declick + declip enabled | on | CLI `--no-declip` |
| High-pass cutoff | 80 Hz | `Config.hpf_hz` |
| High-pass slope | 2-pole (−12 dB/oct) | Hardcoded |

---

### 2. Align + polarity — `--no-align` *(session-level, ≥ 2 tracks)*

**Fixes.** Recorders that started at different instants (offset tracks smear
crosstalk into echo) and miswired/inverted mics (phase cancellation when summed).

**How it works.** Both are fixed against the first track as reference. A coarse
**GCC-PHAT** cross-correlation (on an 8 kHz downsample of the opening window)
estimates the time offset — GCC-PHAT whitens the cross-spectrum so the peak stays
sharp even when two mics sound very different. The peak must clear a **confidence
gate** (z-score) **and** be confirmed by a direct sample-domain waveform
correlation (|r| ≥ 0.08) before any shift is applied — this two-key check stops
genuinely **uncorrelated remote recordings** (different rooms, no bleed) from
being shifted on a spurious peak. If that confirming correlation is negative, the
track's **polarity is inverted** so it adds to rather than cancels the reference.
Tracks are then zero-padded to equal length.

**Strength.** Not strength-scaled — it's a correctness fix.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-align` |
| Offset search window | first 300 s | `Config.align_window_s` |
| Peak confidence gate | z ≥ 12 | `Config.align_min_confidence` |
| Waveform confirm threshold | \|r\| ≥ 0.08 | Hardcoded |
| Polarity-flip threshold | r < −0.05 | Hardcoded |
| Coarse correlation rate | 8 kHz | Hardcoded |

---

### 3. Denoise — `--no-denoise` *(per track)*

**Fixes.** Broadband noise — room tone, hiss, fans, hum, distant traffic, breath
noise.

**How it works.** **DeepFilterNet3**, a full-band 48 kHz neural
speech-enhancement model that separates voice from noise far more cleanly than
classical methods and also tames light reverb. Its weights ship inside the
`deepfilternet` package (no download). Processed in **60 s chunks with a 1 s
crossfade** so memory stays bounded on hour-long files.

**Strength.** Sets the **attenuation ceiling** — how far the model is allowed to
push noise down. The ceiling is a continuous, finite dB value (no jump to
"unlimited"); 60 dB at the top is already effectively full suppression for
speech.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-denoise` |
| Model | DeepFilterNet3 (48 kHz full-band) | Hardcoded |
| Attenuation ceiling | `0 → 60 dB` (48 dB @ 0.8) | **Strength** |
| Chunk / crossfade | 60 s / 1 s | Hardcoded |

> Note: DeepFilterNet 0.5.x imports `torchaudio.backend.common`, which newer
> torchaudio removed. Podcare installs a tiny compatibility shim
> ([src/podcare/_compat.py](src/podcare/_compat.py)) at import time so current
> torch/torchaudio still work — no action needed.

---

### 4. Dereverb — `--no-dereverb` *(per track, WPE)*

**Fixes.** The room — the late-reverberation "tail" that makes voices sound
distant or boxy.

**How it works.** **WPE** (Weighted Prediction Error), the standard
linear-prediction dereverb: it estimates a per-frequency filter that predicts the
reverb tail from the recent past and subtracts it. This is complementary to the
neural denoiser (which targets *noise*, not room response). Run in **30 s chunks
with crossfade**.

**Strength.** Lengthens the prediction filter and adds iterations — more reverb
removed, at higher CPU cost.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-dereverb` |
| WPE taps (filter length) | `6 → 16` (14 @ 0.8) | **Strength** |
| WPE iterations | `1 → 7` (6 @ 0.8) | **Strength** |
| WPE prediction delay | 3 | `Config.wpe_delay` |
| Chunk length | 30 s | `Config.dereverb_chunk_s` |
| STFT size / shift | 1024 / 256 | Hardcoded |

---

### 5. Plosive ducking — `--no-plosives` *(per track)*

**Fixes.** "P"/"B" pops — the burst of low-frequency energy a plosive blasts into
the mic.

**How it works.** Per-track STFT (processed in **60 s chunks** so the spectrogram
can never OOM a multi-hour render). Frames are flagged where energy below the
plosive ceiling is both abnormally high (≫ the chunk's own median) **and**
dominates the frame's full spectrum; just those low-frequency bins are ducked
back toward the typical level. The attenuation is feathered one frame each side
so the gain ramps rather than steps — the voice's pitch and body stay intact.

**Strength.** Lowers the detection thresholds (catch more pops) and deepens the
duck.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-plosives` |
| Plosive band ceiling | 150 Hz | `Config.plosive_max_hz` |
| Burst threshold (×median) | `24 → 4` (8.0 @ 0.8) | **Strength** |
| Spectral-dominance threshold | `0.80 → 0.40` (0.48 @ 0.8) | **Strength** |
| Duck target (×median) | `8 → 3` (4.0 @ 0.8) | **Strength** |
| STFT size / overlap | 1024 / 768 | Hardcoded |
| Chunk / gain spread | 60 s / ±1 frame | Hardcoded |

---

### 6. De-ess — `--no-deess` *(per track)*

**Fixes.** Harsh "S"/"SH"/"T" sibilance.

**How it works.** A zero-phase **split-band** design guarantees
`full = sibilant_band + rest` exactly: the sibilance band is extracted (in single
precision, so a long episode never spawns a multi-GB float64 copy), and whenever
its short-time energy exceeds a fraction of the full-band energy (a sibilant is
sounding), the band is dynamically attenuated and recombined with the untouched
rest. Fast attack / slow release (~3 ms / ~30 ms) keeps it transparent — only the
sibilants duck, not the whole top end. The audibility gate is **relative to the
track's own speech level** (≈30 dB below it, floored at −55 dBFS), so de-essing
engages correctly even on a quiet mic (this stage runs before the gate's level
match).

**Strength.** Lowers the trigger ratio (catch more) and raises the maximum
reduction.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-deess` |
| Sibilance band | 4500–9500 Hz | `Config.deess_lo_hz` / `deess_hi_hz` |
| Trigger ratio (band/full RMS) | `0.90 → 0.30` (0.42 @ 0.8) | **Strength** |
| Max reduction | `0 → 14 dB` (11.2 dB @ 0.8) | **Strength** |
| Attack / release | ~3 ms / ~30 ms | Hardcoded |
| Band filter order | 4th-order Butterworth | Hardcoded |
| Audibility gate | speech-relative (−30 dB, floor −55 dBFS) | Hardcoded |

---

### 7. Gate + level match — `--no-gate` *(per track, before mixdown)*

**Fixes.** Crosstalk/bleed and room tone between phrases; mismatched speaker
levels.

**How it works.** Two jobs. The **downward expander (gate):** when a track's
short-time level falls below an adaptive speech/noise threshold, it's pushed down
(2:1 expansion, floored at the gate depth), suppressing the other speaker's bleed
plus room tone; a slow release protects word tails. **Level matching:** each
track's *speech-active* RMS (ignoring the gated gaps) is normalized toward a
target so a quiet guest and a loud host arrive at the mix balanced.

**Strength.** Sets how deep the gate cuts.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-gate` |
| Gate depth (max attenuation) | `0 → 24 dB` (19.2 dB @ 0.8) | **Strength** |
| Speech-level target | −20 dBFS | `Config.level_target_dbfs` |
| Expansion ratio | 2:1 | Hardcoded |
| Attack / release | ~5 ms / ~160 ms | Hardcoded |
| Level-match clamp | −12 … +24 dB | Hardcoded |
| Threshold | adaptive (noise-floor + speech-relative) | Hardcoded |

---

### 8. Filler-word removal — `--no-fillers`, `--filler-sensitivity`, `--whisper-model`, `--language` *(session-level, per-track detection, before mixdown)*

**Fixes.** Non-lexical fillers — "um", "uh", "ehm", "er", "hmm", "mm", … —
located by transcription, not by listening for a sound.

**How it works.** This runs **before mixdown, per track**, so the ASR and aligner
always see one **clean isolated voice** rather than the summed two-speaker mix.
For each mic: faster-whisper transcribes it, then **WhisperX force-aligns** that
transcript with a wav2vec2 CTC model to get **phoneme-tight word boundaries**
(much more precise than Whisper's own word timings — the difference between a
clean cut and clipping the next word). Words whose normalized text is in the
filler lexicon become cut candidates. Because Whisper tends to *skip*
disfluencies, transcription is biased toward verbatim output (a filler-laden
initial prompt, `condition_on_previous_text=False`).

To keep every track frame-aligned, a candidate is only cut when **every other
track is silent** during it (so cutting a host's "um" can never clip a word the
guest was saying underneath); the surviving intervals are then removed
**identically from all tracks**, which preserves sync going into the sum. The
Whisper and alignment models are loaded once and reused across tracks. If
transcription or alignment is unavailable for a track — unsupported language, a
bad `--whisper-model`, a download/TLS failure — that track is **left unedited
with a logged warning** rather than aborting the render (use `--language` to pin
a supported language).

**Strength.** Sets three gates a candidate must clear: an **alignment-score
floor** `0.9 − 0.6·sens`, a **minimum duration** `0.24 − 0.2·sens` s, and (below
0.7) an **isolation** requirement that the filler be flanked by a small silence.
Effective sensitivity follows strength conservatively (`0.7 × strength`) and is
overridden by an explicit `--filler-sensitivity`.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-fillers` |
| Sensitivity | `0.7 × strength` (0.56 @ 0.8) unless overridden | **Strength** / CLI `--filler-sensitivity` |
| Alignment-score floor | `0.9 − 0.6 × sensitivity` | Derived from sensitivity |
| Minimum duration | `0.24 − 0.2 × sensitivity` s | Derived from sensitivity |
| Isolation required | when sensitivity < 0.7 | Derived from sensitivity |
| Cross-track safety | cut only where all other tracks are silent | Hardcoded |
| Transcription model | large-v3 (faster-whisper) | CLI `--whisper-model` |
| Language | auto-detect per track | CLI `--language` |
| Forced aligner | wav2vec2 CTC (WhisperX) | Hardcoded |
| Cut edge padding | 12 ms | `Config.filler_pad_s` |
| Filler lexicon | um/uh/ehm/er/hmm/mm/… (affirmative "mhm" excluded) | Hardcoded |

---

### 9. Mixdown *(session-level)*

**Fixes.** Many cleaned mics → one program; sum-clipping.

**How it works.** All cleaned, gated, level-matched, filler-trimmed tracks are
**summed to one mono program**. If the sum would clip, it's scaled back to leave
~1 dB of headroom (true loudness is set later by mastering). A **single track is
held to the same −1 dBFS headroom guarantee**, so mastering always sees a signal
with ~1 dB of headroom regardless of track count. From here on there is exactly
one timeline.

**Strength.** Not applicable — it's a sum with a clip guard.

| Parameter | Value | Controlled by |
|---|---|---|
| Post-sum headroom | −1 dBFS | Hardcoded |

---

### 10. Pause tightening — `--no-tighten`, `--max-pause`, `--target-pause` *(on the mono program)*

**Fixes.** Dead air — pacing.

**How it works.** Block-RMS energy detection finds silent runs on the mixed
program; a run longer than the max-pause is shortened to the target-pause with a
crossfade, and lead/trail silence is trimmed. The threshold sits well below
speech level (and below the post-gate noise floor) so breaths, beats, and quiet
reactions survive — only true dead air is cut. This is the **only timeline edit
after mixdown**; it operates on a single mono track, so there is nothing left to
keep in sync.

**Strength.** Shortens both the trigger and the kept beat.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-tighten` |
| Max pause (trigger) | `4.0 → 1.0 s` (1.60 s @ 0.8) unless overridden | **Strength** / CLI `--max-pause` |
| Target pause (kept) | `1.2 → 0.4 s` (0.56 s @ 0.8) unless overridden | **Strength** / CLI `--target-pause` |
| Lead/tail trim | 0.5 s | `Config.lead_trail_s` |
| Detection block | 10 ms | Hardcoded |
| Silence threshold | adaptive (well below speech) | Hardcoded |
| Edit crossfade | 30 ms | Hardcoded |

---

### 11. Master — `--no-master`, `--lufs`, `--bitrate`

**Fixes.** Inconsistent dynamics and loudness — the finishing chain.

**How it works.** Applied as one ffmpeg graph. **Bus compression**
(`acompressor`, soft knee) evens out the remaining swings between a loud laugh
and a soft aside (off at `--strength 0`). Then **two-pass loudness normalization**
(`loudnorm`, EBU R128 — always on, even at strength 0, so the output hits its
target level): the first pass *measures* integrated loudness, range, and true
peak; the second applies a **linear** gain to hit exactly the target LUFS with
true peak under the ceiling. Two-pass + linear is what makes the result accurate
and transparent rather than pumping. Finally, a **brickwall true-peak limiter**
(`alimiter`, `level=false` so it never auto-makeup-gains against the loudness
target) is the last node before the resample — loudnorm's single linear gain is
not a real lookahead limiter, so this catches the inter-sample / codec overs it
can leave and guarantees the delivered file never clips a consumer DAC. It runs
at the 48 kHz internal rate for ISP headroom. A silent/near-silent program (below
loudnorm's −70 LUFS gate) skips normalization and is emitted as-is rather than
erroring.

**Strength.** Firms up the compression (higher ratio, lower threshold); loudness,
true-peak targets, and the limiter are absolute delivery settings, not
strength-scaled (the limiter runs at every strength, even 0).

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-master` |
| Compressor enabled | on (off at strength 0) | `Config.compress` |
| Compression ratio | `1.0 → 3.5` (3.0 @ 0.8) | **Strength** |
| Compression threshold | `0.30 → 0.10` amplitude (0.14 @ 0.8) | **Strength** |
| Compressor attack/release/knee | 10 ms / 200 ms / 4 | Hardcoded |
| Integrated loudness target | −16 LUFS | CLI `--lufs` |
| True-peak ceiling | −1.5 dBTP | `Config.true_peak_db` |
| Loudness range (LRA) | 11 | Hardcoded |
| Normalization | two-pass, linear | Hardcoded |
| True-peak limiter | brickwall at the ceiling (`alimiter`, no makeup) | Hardcoded |

---

### 12. Encode + resample

**Fixes.** Delivery format and the single, clean rate conversion.

**How it works.** A **single** resample to the output rate closes the chain —
done here once with **soxr** (very-high-quality band-limited resampling), rather
than repeatedly mid-pipeline. WAV and FLAC are written **16-bit with
triangular-HP dither** (the correct way to reduce bit depth without quantization
distortion); MP3/AAC are encoded from float at the chosen bitrate. The container
is chosen from the output file's extension. The final line of the log confirms
the delivered file's duration, rate, bit depth, loudness target, and size.

**Strength.** Not applicable — delivery.

| Parameter | Value | Controlled by |
|---|---|---|
| Output sample rate | 44100 Hz | CLI `--out-sr` |
| WAV/FLAC bit depth | 16-bit + triangular-HP dither | Hardcoded |
| Resampler | soxr VHQ (very-high-quality) | Hardcoded |
| Lossy bitrate | 192k | CLI `--bitrate` |

---

## Roadmap — proposed stages for maximum quality

The current chain removes defects and normalizes loudness well. The biggest
remaining quality gains — distilled from a multi-engineer design review — are
listed below in priority order. Each is designed to slot cleanly into the
existing order and to obey the same `--strength` contract (a true no-op at
strength 0). All are achievable with the current stack (numpy/scipy/ffmpeg/torch);
none need a new heavyweight dependency.

> ✅ **Shipped:** the **true-peak limiter** (formerly the #2 must-have) is now part
> of the [Master stage](#11-master----no-master---lufs---bitrate) — a brickwall
> `alimiter` after loudnorm that guarantees the delivered file never
> inter-sample-clips. The remaining proposals are below.

| # | Proposed stage | Tier | Where | What it adds |
|---|---|---|---|---|
| 1 | **Tonal-balance / LTAS EQ** | must-have | per track, after dereverb | Measures each mic's long-term spectrum (Welch PSD over speech-active frames) and gently shapes it toward a broadcast voice curve (pink-ish tilt + presence) via ffmpeg `firequalizer`. Makes a dull lavalier and a bright condenser sit together and translates better on phone speakers. The single highest-impact lever after loudness. |
| 2 | **Segment loudness leveler** | must-have | on the mono bus, before master | A slow (seconds-scale), gating-aware short-term-LUFS ride that evens out the minutes-scale loudness drift loudnorm's single global number ignores (a guest fading over a segment, a tired late take). Reuses the existing block-gain machinery at a long time constant. |
| 3 | **De-hum / de-buzz** | high-value | per track, after repair | Detection-gated harmonic notch comb (`scipy.iirnotch`, zero-phase) that auto-detects 50/60 Hz mains and surgically removes its harmonics — the most instantly-noticeable amateur tell, which the 80 Hz HPF only dents and the neural denoiser doesn't reliably kill. Notches nothing on a clean track. |
| 4 | **Multiband bus compressor** | high-value | inside master, replacing the single comp | Splits the bus into 3 phase-coherent bands (ffmpeg `acrossover`) and compresses each independently, so a boomy low-mid or a sibilant peak no longer ducks the whole program. Denser, more consistent loudness without pumping. |
| 5 | **Dynamic resonance/harshness suppression** | high-value | per track, around de-ess | "Soothe-style" adaptive STFT notching of transient resonant peaks (ringy room modes, nasal honk, 2–5 kHz spikes) that static EQ can't catch because they come and go — a major cause of earbud fatigue. |
| 6 | **Breath control** | high-value | per track, after gate | Detect-and-**duck** (not cut) audible inhales between phrases — the gate misses them because they sit above its threshold. Classifies non-speech islands as breath vs. silence by spectral shape and attenuates ~6–16 dB, preserving natural cadence. |
| 7 | **Mouth-click / de-crackle** | nice-to-have | per track, after denoise | STFT transient removal of wet mouth clicks and lip smacks that `adeclick` (vinyl/digital impulses) and the neural denoiser leave intact — and which the cleaner the rest of the chain gets, the *more* audible they become. |
| 8 | **Harmonic presence exciter** | nice-to-have | inside master, late | A touch of high-band saturation (ffmpeg `aexciter`) to restore "air" lost to heavy denoise/dereverb and to cut through tiny speakers — synthesizes new harmonics rather than boosting (possibly noisy) existing highs. Easy to overdo; conservative ceiling. |
| 9 | **Dropout / short-gap restoration** | nice-to-have | per track, early | LPC/interpolation fill of brief (<~50 ms) signal dropouts from remote-guest packet loss, so remote guests sound locally recorded. Strict gap caps so it never fabricates real content. |
| 10 | **Music-bed ducking + stereo delivery** | skip (unless requested) | I/O contract change | Sidechain-duck an optional intro/outro music bed under speech, and offer stereo (artifact-free dual-mono) output. Format/feature work, not a voice-fidelity fix — it would expand the "mics-in, one-mono-file-out" contract, so it's deferred. |

**Suggested target order** once these land:
`repair → de-hum → dropout-fix → align → denoise → dereverb → tonal-balance →
mouth-declick → plosives → de-resonance → de-ess → gate → breath → fillers →
mixdown → segment-leveler → tighten → multiband-comp → exciter → loudnorm →
true-peak-limiter → resample/encode`.

---

## How it's built

- **Python 3.11 + uv.** One linear pipeline over a `Session` (a list of `Track`s
  + sample rate). Each stage is a small function `(Session|Track, Config) →
  Session|Track`; the stage list lives in
  [src/podcare/pipeline.py](src/podcare/pipeline.py), every tunable (and the
  strength→stage mapping) in [src/podcare/config.py](src/podcare/config.py).
- **ffmpeg** for decode/encode and the repair + master filters; **numpy/scipy**
  for the hand-written DSP (align, plosives, de-ess, gate, tighten);
  **DeepFilterNet/torch**, **nara-wpe**, and **faster-whisper + WhisperX** for the
  ML/heavy stages; **soxr** for the single final resample.
- **Robustness by design.** Heavy spectral stages (denoise, dereverb, plosives)
  are chunked so memory stays bounded on multi-hour episodes; the optional ML
  filler pass degrades to a no-op (with a warning) rather than aborting a render;
  silent programs and bad CLI inputs fail fast and cleanly.
- Tests use synthetic fixtures with known ground truth (a known offset alignment
  must recover, injected sibilance must reduce, a long pause must shrink, final
  loudness within ±1.5 LU of target) plus the strength-mapping invariants, the
  cross-track filler-safety logic, and CLI validation: `uv run pytest`.
