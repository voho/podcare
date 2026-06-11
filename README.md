![logo](https://github.com/voho/podcare/blob/main/logo.png?raw=true)

# Podcare

Turn raw podcast mic recordings into a polished, broadcast-ready episode with one command.

```bash
podcare host.wav guest.wav -o episode.mp3
```

**Input:** one or more WAV/MP3/FLAC files (one per mic/recorder; anything ffmpeg reads).
**Output:** one WAV/MP3/FLAC/M4A — aligned, declipped, de-hummed, denoised
(DeepFilterNet3 neural enhancement), dereverbed, tonally balanced, de-clicked,
de-plosived, de-essed, breath-controlled, filler-words removed, pauses tightened,
loudness-leveled, multiband-compressed, loudness-normalized and true-peak limited
to podcast standard (−16 LUFS / −1.5 dBTP). Output is 44.1 kHz (16-bit + dither
for WAV/FLAC); processing runs internally at 48 kHz float32. Defaults favor
quality over speed.

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
true-peak limit → encode** (so the output still hits your `--lufs` target and is
clip-safe, but its tone and dynamics are untouched — useful as an A/B baseline).
`1` is the most aggressive. The exact per-stage mapping is listed in each
[pipeline section](#the-pipeline) below, and every value is derived from
[src/podcare/config.py](src/podcare/config.py) (the single source of truth).

Two stages ignore strength on purpose: **repair** and **align** are *correctness*
operations (fix clipping, fix timing/polarity), not matters of degree. Loudness
normalization and the true-peak limiter are absolute delivery settings and also
run regardless of strength.

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
| `--filler-sensitivity 0..1` | follows `--strength` | Override how aggressively non-lexical fillers ("um", "uh", "ehm", "hmm", …) are cut. `0` disables the stage. When unset, defaults to `0.7 × strength`. |
| `--whisper-model NAME` | `large-v3` | faster-whisper model used to transcribe before forced alignment. Smaller models (`medium`/`small`/`base`/`tiny`) trade accuracy for speed and download size. |
| `--language CODE` | auto | Force the spoken language for filler detection (e.g. `en`, `cs`, `de`). When unset, the language is auto-detected per track. |
| `--max-pause SECONDS` | follows `--strength` | Override: silences longer than this get shortened. When unset, `lerp(4.0 → 1.0)` over strength. Must exceed `--target-pause`. |
| `--target-pause SECONDS` | follows `--strength` | Override: the length an over-long pause is shortened *to*. When unset, `lerp(1.2 → 0.4)` over strength. |
| `--lufs DB` | `-16` | Output integrated-loudness target (EBU R128). Validated to `-40 … -5`. |
| `--out-sr HZ` | `44100` | Output sample rate (validated `8000 … 192000`). Resampling happens exactly once, at the end. |
| `--bitrate RATE` | `192k` | Bitrate for lossy outputs (MP3/AAC). Ignored for WAV/FLAC. |
| `--keep-stems DIR` | off | Write each stage's intermediate audio **and the final master** into `DIR` as numbered files. |

### Stage toggles

Each switch disables one stage; everything else still runs.

| Flag | Disables |
|---|---|
| `--no-declip` | Distortion repair (declick + declip + rumble high-pass) |
| `--no-dehum` | Mains-hum (50/60 Hz) harmonic removal |
| `--no-align` | Inter-track time-offset and polarity correction |
| `--no-denoise` | Noise reduction (DeepFilterNet3) |
| `--no-dereverb` | WPE dereverberation |
| `--no-tonebalance` | Tonal-balance / LTAS corrective EQ |
| `--no-declick` | Mouth-click / de-crackle removal |
| `--no-plosives` | Plosive ("p-pop") ducking |
| `--no-deess` | De-essing (sibilance control) |
| `--no-gate` | Crosstalk/room gate **and** per-track level matching |
| `--no-breath` | Breath ducking |
| `--no-fillers` | Filler-word removal |
| `--no-tighten` | Pause tightening / dead-air trimming |
| `--no-leveler` | Slow segment-loudness leveling |
| `--no-master` | Multiband compression + loudness normalization + limiting |

---

## The pipeline

Stages run in this fixed order. **Track-level** stages process each mic
independently; **session-level** stages see all tracks at once. The ordering is
deliberate: restoration (repair, de-hum) first, then enhancement, then all per-mic
cleanup — *including filler detection* — before the tracks are summed; the only
post-sum edits are on the single mono program, so the tracks can never drift out
of sync.

```
            ┌────────────────────────────── per track ───────────────────────────────┐
decode ─▶ repair ─ dehum ─ align ─ denoise ─ dereverb ─ tonebalance ─ declick ─ plosives ─┐
 (load    (declick (mains  (offset (DFN3)    (WPE       (LTAS->voice  (mouth   (p-pop      │
  48k)     declip   hum     + pol.            tail)      curve EQ)     clicks)  ducking)    │
          HPF)      notch)  fix)                                                            ▼
        ┌──────────────────────────────────────────────────────── deess ─ gate ─ breath ─ fillers
        │                                                         (sibilance (level (duck   (per-mic
        ▼                                                          control)  match) inhale) ASR cuts)
     mixdown ─▶ tighten ─▶ leveler ─▶ master ──────────────────────▶ encode
    (sum→mono) (shorten   (slow      (mb-comp + loudnorm            (44.1k/16
                dead air)  loudness   + TP-limiter)                  + dither)
                          ride)
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
mono. A file that decodes to zero samples is rejected immediately.

**Strength.** Not applicable — this is I/O.

| Parameter | Value | Controlled by |
|---|---|---|
| Internal sample rate | 48000 Hz | `Config.sr` (do not change — DFN3 requires 48 k) |
| Channels | mono | Hardcoded |

---

### 1. Repair — `--no-declip` *(per track)*

**Fixes.** Clicks, glitches, hard clipping (recorded too hot), subsonic rumble.

**How it works.** Three ffmpeg filters in series: **`adeclick`** interpolates over
impulsive clicks; **`adeclip`** reconstructs samples driven past full-scale; and a
**2-pole high-pass** removes rumble, desk thumps, and HVAC roar below the voice
fundamental. Distortion repaired here can't be denoised away later, so it's first.

**Strength.** Not strength-scaled — restoration, not a degree of effect.

| Parameter | Value | Controlled by |
|---|---|---|
| Declick + declip enabled | on | CLI `--no-declip` |
| High-pass cutoff | 80 Hz | `Config.hpf_hz` |
| High-pass slope | 2-pole (−12 dB/oct) | Hardcoded |

---

### 2. De-hum — `--no-dehum` *(per track)*

**Fixes.** Steady 50/60 Hz mains hum and its harmonics (100/120, 150/180, …) and
ground-loop/USB buzz — one of the most instantly-noticeable amateur tells, which
the 80 Hz high-pass only dents and the broadband neural denoiser doesn't reliably
kill as a tonal comb.

**How it works.** The mains fundamental is found from the long-term spectrum
(Welch PSD): the sharpest peak in 45–65 Hz that clears the local spectral floor,
snapped to 50 or 60 Hz. Narrow zero-phase notches (`iirnotch` via `sosfiltfilt`,
Q ≈ 30) are then placed at each harmonic that **actually protrudes** above the
local floor — so a clean track gets zero notches and the voice between harmonics
is untouched. Runs early so the stationary tones don't bias GCC-PHAT alignment,
get smeared by WPE, or feed the denoiser's noise estimate.

**Strength.** Scales how many harmonics are removed and how readily hum is
detected. Two no-op guards at strength 0: zero harmonics and an unreachable
detection margin (plus the stage is skipped).

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-dehum` |
| Max harmonics | `0 → 12` (10 @ 0.8) | **Strength** |
| Detection margin over floor | `20 → 6 dB` (≈9 dB @ 0.8) | **Strength** |
| Notch Q | 30 (a few Hz wide) | Hardcoded |
| Highest harmonic | 1500 Hz | Hardcoded |

---

### 3. Align + polarity — `--no-align` *(session-level, ≥ 2 tracks)*

**Fixes.** Recorders started at different instants (offset tracks smear crosstalk
into echo) and miswired/inverted mics (phase cancellation when summed).

**How it works.** Both are fixed against the first track as reference. A coarse
**GCC-PHAT** cross-correlation (on an 8 kHz downsample of the opening window)
estimates the offset, then a direct sample-domain waveform correlation confirms
it (|r| ≥ 0.08) — so genuinely **uncorrelated remote recordings** are left
untouched. A negative confirming correlation flips the track's **polarity**.
Tracks are then zero-padded to equal length.

**Strength.** Not strength-scaled — a correctness fix.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-align` |
| Offset search window | first 300 s | `Config.align_window_s` |
| Peak confidence gate | z ≥ 12 | `Config.align_min_confidence` |
| Waveform confirm threshold | \|r\| ≥ 0.08 | Hardcoded |
| Polarity-flip threshold | r < −0.05 | Hardcoded |

---

### 4. Denoise — `--no-denoise` *(per track)*

**Fixes.** Broadband noise — room tone, hiss, fans, distant traffic, breath noise.

**How it works.** **DeepFilterNet3**, a full-band 48 kHz neural speech-enhancement
model that separates voice from noise far more cleanly than classical methods.
Weights ship inside the package (no download). Processed in **60 s chunks with a
1 s crossfade** so memory stays bounded on hour-long files.

**Strength.** Sets the **attenuation ceiling** — a continuous, finite dB value;
60 dB at the top is already effectively full suppression for speech.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-denoise` |
| Model | DeepFilterNet3 (48 kHz full-band) | Hardcoded |
| Attenuation ceiling | `0 → 60 dB` (48 dB @ 0.8) | **Strength** |
| Chunk / crossfade | 60 s / 1 s | Hardcoded |

> Note: DeepFilterNet 0.5.x imports `torchaudio.backend.common`, which newer
> torchaudio removed. A tiny compatibility shim
> ([src/podcare/_compat.py](src/podcare/_compat.py)) handles it — no action needed.

---

### 5. Dereverb — `--no-dereverb` *(per track, WPE)*

**Fixes.** The room — the late-reverberation "tail" that makes voices sound
distant or boxy.

**How it works.** **WPE** (Weighted Prediction Error) linear-prediction dereverb:
it estimates a per-frequency filter that predicts the reverb tail from the recent
past and subtracts it. Complementary to the neural denoiser. Run in **30 s chunks
with crossfade**.

**Strength.** Lengthens the prediction filter and adds iterations.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-dereverb` |
| WPE taps (filter length) | `6 → 16` (14 @ 0.8) | **Strength** |
| WPE iterations | `1 → 7` (6 @ 0.8) | **Strength** |
| WPE prediction delay | 3 | `Config.wpe_delay` |
| Chunk length | 30 s | `Config.dereverb_chunk_s` |

---

### 6. Tonal balance — `--no-tonebalance` *(per track)*

**Fixes.** The chain has no spectral shaping otherwise, so a dull lavalier and a
bright condenser stay timbrally mismatched after all the cleanup; gross tilt,
proximity-effect low-mid boom, and dull/harsh tops go uncorrected.

**How it works.** Each track's long-term average spectrum (LTAS) is measured over
**speech-active frames only** (Welch PSD), normalized to the speech body, and
compared to a fixed **produced broadcast-voice target curve**. The deviation is
realized as four **broad** minimal-phase RBJ biquads (low/high shelf + low-mid and
presence bells) via `sosfilt` — length-preserving, zero added latency. Only the
spectral *shape* is matched, so it never changes overall loudness. Deliberately
gentle: broad filters, boosts clamped tighter than cuts (a noisy band is never
lifted), and only a fraction of the deviation applied (`strength / 3`), so it stays
subtle even at full strength.

**Strength.** Scales the fraction of the measured deviation applied — `strength / 3`
(0 at strength 0).

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-tonebalance` |
| Correction fraction | `strength / 3` (0.27 @ 0.8) | **Strength** |
| Boost clamp | +3 dB | Hardcoded |
| Cut clamp | −6 dB | Hardcoded |
| Filters | low/high shelf + low-mid (300 Hz) + presence (3 kHz) bells | Hardcoded |
| Target curve | produced broadcast-voice LTAS | Hardcoded |

---

### 7. Mouth-click / de-crackle — `--no-declick` *(per track)*

**Fixes.** Wet mouth clicks, lip smacks, tongue clicks and saliva crackle that
`adeclick` (vinyl/digital impulses) and the neural denoiser leave intact — and
which become *more* audible as compression and loudnorm lift the quiet inter-word
detail.

**How it works.** STFT transient detection (≈1.3 ms hop). A click is a short
mid-band (1.5–6 kHz) energy spike with a high crest factor over a robust local
median that is **also a local peak** and sits in a **quiet neighbourhood** (the
median-filtered broadband level there is well below the speech reference). Those
three gates protect real consonants (t/k/p, s/sh), which sit at full speech level.
Flagged frames have their ≥1.5 kHz bins ducked toward the baseline, feathered
±1 frame. Chunked like the other spectral stages.

**Strength.** Lowers the crest threshold (catch progressively subtler clicks).
Huge at strength 0 (nothing triggers).

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-declick` |
| Crest threshold | `40 → 8×` (14× @ 0.8) | **Strength** |
| Detection band | 1500–6000 Hz | Hardcoded |
| Quiet-neighbourhood gate | < 35% of speech reference | Hardcoded |
| Local-median window | ~250 ms | Hardcoded |

---

### 8. Plosive ducking — `--no-plosives` *(per track)*

**Fixes.** "P"/"B" pops — the burst of low-frequency energy a plosive blasts into
the mic.

**How it works.** Per-track STFT (in **60 s chunks** so the spectrogram can never
OOM a multi-hour render). Frames are flagged where energy below the plosive
ceiling is both abnormally high (≫ the chunk's own median) **and** dominates the
frame's spectrum; just those LF bins are ducked back toward the typical level,
feathered ±1 frame.

**Strength.** Lowers the detection thresholds and deepens the duck.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-plosives` |
| Plosive band ceiling | 150 Hz | `Config.plosive_max_hz` |
| Burst threshold (×median) | `24 → 4` (8.0 @ 0.8) | **Strength** |
| Spectral-dominance threshold | `0.80 → 0.40` (0.48 @ 0.8) | **Strength** |
| Duck target (×median) | `8 → 3` (4.0 @ 0.8) | **Strength** |

---

### 9. De-ess — `--no-deess` *(per track)*

**Fixes.** Harsh "S"/"SH"/"T" sibilance.

**How it works.** A zero-phase **split-band** design (in single precision, so a
long episode never spawns a multi-GB float64 copy) guarantees
`full = sibilant_band + rest` exactly: whenever the band's short-time energy
exceeds a fraction of the full-band energy, the band is attenuated and recombined.
Fast attack / slow release keeps it transparent. The audibility gate is **relative
to the track's own speech level** (≈30 dB below it, floored at −55 dBFS), so it
engages correctly on a quiet mic.

**Strength.** Lowers the trigger ratio and raises the maximum reduction.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-deess` |
| Sibilance band | 4500–9500 Hz | `Config.deess_lo_hz` / `deess_hi_hz` |
| Trigger ratio (band/full RMS) | `0.90 → 0.30` (0.42 @ 0.8) | **Strength** |
| Max reduction | `0 → 14 dB` (11.2 dB @ 0.8) | **Strength** |
| Attack / release | ~3 ms / ~30 ms | Hardcoded |

---

### 10. Gate + level match — `--no-gate` *(per track, before mixdown)*

**Fixes.** Crosstalk/bleed and room tone between phrases; mismatched speaker
levels.

**How it works.** A **downward expander (gate):** below an adaptive speech/noise
threshold the track is pushed down (2:1, floored at the gate depth), suppressing
the other speaker's bleed; a slow release protects word tails. **Level matching:**
each track's *speech-active* RMS is normalized toward a target so a quiet guest and
a loud host arrive balanced.

**Strength.** Sets how deep the gate cuts.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-gate` |
| Gate depth (max attenuation) | `0 → 24 dB` (19.2 dB @ 0.8) | **Strength** |
| Speech-level target | −20 dBFS | `Config.level_target_dbfs` |
| Expansion ratio | 2:1 | Hardcoded |

---

### 11. Breath control — `--no-breath` *(per track)*

**Fixes.** Audible inhale breaths between phrases. The gate only acts below its
threshold and breaths usually sit *above* it, so they survive; after compression
and loudnorm push the quiet detail forward, they become a prominent earbud tell.

**How it works.** Breaths are detected by *what they are*, not level alone: short
segments that are **unvoiced** (high zero-crossing rate, no low fundamental),
**audible** (above the noise floor) but **below speech level** (which protects
in-word fricatives, that sit at full speech level), of breath-like duration, and
confirmed by spectral shape (mid-band, unvoiced). Flagged spans are **ducked**
(never muted) by a capped amount with smooth ramps, preserving cadence.

**Strength.** Sets the duck depth — capped at 14 dB (a duck, never a mute).

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-breath` |
| Duck depth | `0 → 14 dB` (11.2 dB @ 0.8) | **Strength** |
| Breath duration window | 0.08–0.7 s | Hardcoded |
| Voiced/unvoiced split | zero-crossing rate | Hardcoded |
| Level gate | below speech, above 2× noise floor | Hardcoded |

---

### 12. Filler-word removal — `--no-fillers`, `--filler-sensitivity`, `--whisper-model`, `--language` *(session-level, per-track detection, before mixdown)*

**Fixes.** Non-lexical fillers — "um", "uh", "ehm", "er", "hmm", "mm", … — located
by transcription, not by listening for a sound.

**How it works.** Runs **before mixdown, per track**, so the ASR and aligner always
see one **clean isolated voice** rather than the summed mix. For each mic:
faster-whisper transcribes it (biased verbatim with a filler-laden prompt), then
**WhisperX force-aligns** the transcript with a wav2vec2 CTC model for
**phoneme-tight word boundaries**. To keep every track frame-aligned, a candidate
is cut only when **every other track is silent** during it; the surviving intervals
are removed **identically from all tracks**. Models are loaded once and reused; any
ASR/alignment failure (unsupported language, bad model, download error) **degrades
to a logged no-op** rather than aborting the render.

**Strength.** Sets the alignment-score floor `0.9 − 0.6·sens`, minimum duration
`0.24 − 0.2·sens` s, and (below 0.7) an isolation requirement. Effective
sensitivity follows strength conservatively (`0.7 × strength`).

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-fillers` |
| Sensitivity | `0.7 × strength` (0.56 @ 0.8) | **Strength** / CLI `--filler-sensitivity` |
| Cross-track safety | cut only where all other tracks are silent | Hardcoded |
| Transcription model | large-v3 (faster-whisper) | CLI `--whisper-model` |
| Language | auto-detect per track | CLI `--language` |
| Forced aligner | wav2vec2 CTC (WhisperX) | Hardcoded |
| Filler lexicon | um/uh/ehm/er/hmm/mm/… (affirmative "mhm" excluded) | Hardcoded |

---

### 13. Mixdown *(session-level)*

**Fixes.** Many cleaned mics → one program; sum-clipping.

**How it works.** All cleaned, gated, level-matched, filler-trimmed tracks are
**summed to one mono program**, scaled back to leave ~1 dB of headroom if the sum
would clip. A **single track** gets the same −1 dBFS guarantee. From here on there
is exactly one timeline.

**Strength.** Not applicable.

| Parameter | Value | Controlled by |
|---|---|---|
| Post-sum headroom | −1 dBFS | Hardcoded |

---

### 14. Pause tightening — `--no-tighten`, `--max-pause`, `--target-pause` *(mono program)*

**Fixes.** Dead air — pacing.

**How it works.** Block-RMS detection finds silent runs on the mixed program; a run
longer than the max-pause is shortened to the target-pause with a crossfade, and
lead/trail silence is trimmed. The threshold sits well below speech (and below the
post-gate noise floor) so breaths, beats and quiet reactions survive.

**Strength.** Shortens both the trigger and the kept beat.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-tighten` |
| Max pause (trigger) | `4.0 → 1.0 s` (1.60 s @ 0.8) | **Strength** / CLI `--max-pause` |
| Target pause (kept) | `1.2 → 0.4 s` (0.56 s @ 0.8) | **Strength** / CLI `--target-pause` |
| Lead/tail trim | 0.5 s | `Config.lead_trail_s` |
| Edit crossfade | 30 ms | Hardcoded |

---

### 15. Segment loudness leveler — `--no-leveler` *(mono program)*

**Fixes.** Minutes-scale loudness drift that the rest of the chain ignores: a guest
who fades over a segment, a host who leans back, the gap between an intro and a
tired late take. Integrated loudnorm fixes only the whole-file average and the
master compressor reacts far too fast.

**How it works.** A slow short-term loudness envelope is computed over ~3 s windows
**counting only speech blocks** (so pauses are neither pulled down nor boosted),
each window is pulled toward the program's median speech loudness with a tightly
clamped gain, and the result is applied heavily smoothed at multi-second time
constants — inaudible as processing but very audible in the result. Runs on the
mono bus before the master compressor so it sees consistent macro-dynamics.

**Strength.** Sets the maximum ride range (±dB); 0 at strength 0.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-leveler` |
| Ride range | `±0 → ±8 dB` (±6.4 dB @ 0.8) | **Strength** |
| Short-term window | 3 s | Hardcoded |
| Target | program median speech loudness | Hardcoded |
| Gating | speech-only (pauses held) | Hardcoded |

---

### 16. Master — `--no-master`, `--lufs`, `--bitrate`

**Fixes.** Band-specific dynamics, inconsistent loudness, and inter-sample peaks —
the finishing chain.

**How it works.** Three steps. **Multiband compression** splits the bus into three
phase-coherent bands (`acrossover`, LR4 at 250 Hz / 4 kHz) and compresses each
independently (`acompressor` per band) so a boomy low-mid or a sibilant peak no
longer ducks the whole program — denser, more consistent loudness than a single
broadband compressor (off at `--strength 0`, where every band is 1:1). Then
**two-pass EBU R128 loudness normalization** (`loudnorm`, always on): the first
pass measures, the second applies a **linear** gain to hit exactly the target LUFS.
Finally a **brickwall true-peak limiter** (`alimiter`, `level=false` so it never
fights the loudness target) catches the inter-sample / codec overs loudnorm's
single linear gain can leave, guaranteeing the delivered file never clips a
consumer DAC. A silent/near-silent program (below loudnorm's −70 LUFS gate) skips
normalization rather than erroring.

**Strength.** Firms up the per-band compression (higher ratios, lower thresholds);
loudness, true-peak target, and the limiter are absolute delivery settings, not
strength-scaled (the limiter runs at every strength, even 0).

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-master` |
| Multiband compression | 3-band LR4 @ 250 Hz / 4 kHz (off at strength 0) | `Config.compress` |
| Mid-band ratio | `1.0 → 3.5` (3.0 @ 0.8); low ×1.1, high ×0.8 | **Strength** |
| Mid-band threshold | `0.30 → 0.10` amplitude (0.14 @ 0.8) | **Strength** |
| Integrated loudness target | −16 LUFS | CLI `--lufs` |
| True-peak ceiling | −1.5 dBTP | `Config.true_peak_db` |
| Normalization | two-pass, linear | Hardcoded |
| True-peak limiter | brickwall at the ceiling (`alimiter`, no makeup) | Hardcoded |

---

### 17. Encode + resample

**Fixes.** Delivery format and the single, clean rate conversion.

**How it works.** A **single** resample to the output rate with **soxr** (VHQ)
closes the chain. WAV and FLAC are written **16-bit with triangular-HP dither**;
MP3/AAC are encoded from float. The container is chosen from the file extension,
and the final log line confirms the delivered duration, rate, bit depth, loudness
target, and size.

**Strength.** Not applicable — delivery.

| Parameter | Value | Controlled by |
|---|---|---|
| Output sample rate | 44100 Hz | CLI `--out-sr` |
| WAV/FLAC bit depth | 16-bit + triangular-HP dither | Hardcoded |
| Resampler | soxr VHQ | Hardcoded |
| Lossy bitrate | 192k | CLI `--bitrate` |

---

## Roadmap — remaining proposed stages

The chain below already covers the full restoration → enhancement → master path.
The remaining design-review proposals, in priority order. Each is designed to slot
cleanly into the existing order and obey the same `--strength` contract (a true
no-op at strength 0); none need a new heavyweight dependency.

> ✅ **Shipped from the original roadmap:** tonal-balance LTAS EQ, true-peak
> limiter, segment loudness leveler, de-hum / de-buzz, multiband bus compressor,
> breath control, and mouth-click / de-crackle removal are all implemented above.

| # | Proposed stage | Tier | Where | What it adds |
|---|---|---|---|---|
| 1 | **Dynamic resonance / harshness suppression** | high-value | per track, around de-ess | "Soothe-style" adaptive STFT notching of transient resonant peaks (ringy room modes, nasal honk, 2–5 kHz spikes) that the static tonal-balance EQ can't catch because they come and go with the voice — a major cause of earbud fatigue. The trickiest to tune transparently. |
| 2 | **Harmonic presence exciter** | nice-to-have | inside master, late | A touch of high-band saturation (ffmpeg `aexciter`) to restore "air" lost to heavy denoise/dereverb and cut through tiny speakers — synthesizes new harmonics rather than boosting (possibly noisy) existing highs. Easy to overdo; conservative ceiling. |
| 3 | **Dropout / short-gap restoration** | nice-to-have | per track, early | LPC/interpolation fill of brief (<~50 ms) signal dropouts from remote-guest packet loss, so remote guests sound locally recorded. Strict gap caps so it never fabricates real content. |
| 4 | **Music-bed ducking + stereo delivery** | skip (unless requested) | I/O contract change | Sidechain-duck an optional intro/outro music bed under speech, and offer stereo (artifact-free dual-mono) output. Format/feature work, not a voice-fidelity fix — it would expand the "mics-in, one-mono-file-out" contract. |

---

## How it's built

- **Python 3.11 + uv.** One linear pipeline over a `Session` (a list of `Track`s
  + sample rate). Each stage is a small function `(Session|Track, Config) →
  Session|Track`; the stage list lives in
  [src/podcare/pipeline.py](src/podcare/pipeline.py), every tunable (and the
  strength→stage mapping) in [src/podcare/config.py](src/podcare/config.py).
- **ffmpeg** for decode/encode and the repair + master filters (incl. the
  multiband compressor and true-peak limiter); **numpy/scipy** for the hand-written
  DSP (de-hum, tonal-balance EQ, de-click, plosives, de-ess, gate, breath,
  leveler, tighten, align); **DeepFilterNet/torch**, **nara-wpe**, and
  **faster-whisper + WhisperX** for the ML/heavy stages; **soxr** for the single
  final resample.
- **Robustness by design.** Heavy spectral stages (denoise, dereverb, plosives,
  de-click) are chunked so memory stays bounded on multi-hour episodes; the
  optional ML filler pass degrades to a no-op (with a warning) rather than aborting
  a render; silent programs and bad CLI inputs fail fast and cleanly.
- Tests use synthetic fixtures with known ground truth (recover a known offset,
  reduce injected sibilance/hum/clicks, even out a loudness drift, hold the
  true-peak ceiling, final loudness within ±1.5 LU of target) plus the
  strength-mapping invariants, the cross-track filler-safety logic, and CLI
  validation: `uv run pytest`.
