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
that). DeepFilterNet3 weights ship inside the `deepfilternet` package, so
denoise works offline from the start.

---

## Usage

```bash
# Two mics, full polish, MP3 out
uv run podcare host.wav guest.wav -o episode.mp3

# Gentler overall treatment
uv run podcare host.wav guest.wav -o episode.mp3 --strength 0.4

# Single track to 16-bit/44.1k WAV, force-aggressive filler removal
uv run podcare interview.mp3 -o clean.wav --filler-sensitivity 0.9

# Turn off the stages you don't want
uv run podcare raw.wav -o out.wav --no-dereverb --no-tighten

# Faster preview pass (classical denoiser, small Whisper)
uv run podcare a.flac b.flac -o draft.mp3 --denoise-backend spectral --whisper-model small

# Debug: write every stage's intermediate audio so you can A/B them
uv run podcare a.wav b.wav -o out.wav --keep-stems stems/
```

Each input file is treated as **one speaker's mic**. Give Podcare the separate
recorder/mic tracks, not a pre-mixed file, so it can align them, gate crosstalk,
and balance levels before summing. A single pre-mixed file works too — the
multi-track-only stages (align, mixdown) simply no-op.

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
below.

Two stages ignore strength on purpose: **repair** and **align** are *correctness*
operations (fix clipping, fix timing/polarity), not matters of degree.

You can still override individual stages — `--filler-sensitivity`, `--max-pause`,
`--target-pause`, `--lufs` — and an explicit value always wins over what
`--strength` would have chosen.

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
| `--whisper-model NAME` | `large-v3` | faster-whisper model for filler detection. `large-v3` is most accurate (and slowest); `medium`/`small`/`base`/`tiny` trade accuracy for speed and download size. Only word timestamps/probabilities are used — never the transcript text — so this affects detection accuracy, not output wording. |
| `--denoise-backend auto\|deepfilter\|spectral` | `auto` | Noise-reduction engine. `deepfilter` = DeepFilterNet3 neural (best). `spectral` = classical gating (`noisereduce`, fast, no torch). `auto` uses `deepfilter` when it imports, else `spectral`. |
| `--max-pause SECONDS` | follows `--strength` | Override: silences longer than this get shortened. When unset, `lerp(3.5 → 1.0)` over strength. Must exceed `--target-pause`. |
| `--target-pause SECONDS` | follows `--strength` | Override: the length an over-long pause is shortened *to*. When unset, `lerp(1.0 → 0.4)` over strength. |
| `--lufs DB` | `-16` | Output integrated-loudness target (EBU R128). `-16` is the podcast/streaming norm; `-14` louder, `-19`/`-23` broadcast-quieter. |
| `--out-sr HZ` | `44100` | Output sample rate. Resampling happens exactly once, at the end. WAV/FLAC are always 16-bit + dither; MP3/M4A carry no bit depth. |
| `--bitrate RATE` | `192k` | Bitrate for lossy outputs (MP3/AAC). Ignored for WAV/FLAC. e.g. `256k`, `320k`. |
| `--keep-stems DIR` | off | Write each stage's intermediate audio into `DIR` as numbered WAVs — great for hearing what each stage did. |

### Stage toggles

Each switch disables one stage; everything else still runs.

| Flag | Disables |
|---|---|
| `--no-declip` | Distortion repair (declick + declip + rumble high-pass) |
| `--no-align` | Inter-track time-offset and polarity correction |
| `--no-denoise` | Noise reduction (neural or spectral) |
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
deliberate: repair before anything reads the signal, all per-mic cleanup before
the tracks are summed, and every timeline edit (cut) only *after* mixdown — so a
single timeline is edited and the tracks can never drift out of sync.

```
decode ─ repair ─ align ─ denoise ─ dereverb ─ plosives ─ deess ─ gate ─┐
 (load   (declick  (offset  (DFN3 /   (WPE)     (LF burst  (sibilance (expander
  48k)    declip    + pol.   spectral)           ducking)   control)   + level)
          HPF)      fix)                                                     │
                                                                            ▼
   encode ◀─ master ◀─ tighten ◀─ fillers ◀─────────────────────────── mixdown
  (resample  (compress  (shorten   (Whisper                             (sum to
   44.1k/16  + 2-pass    dead air)  word cuts)                           mono)
   + dither)  loudnorm)
```

In each section below: **How it works** describes the algorithm, then
**Parameters** lists every knob and how it's controlled — `Strength` (scaled by
`--strength`), a CLI flag, a `Config` field (advanced, edit
[src/podcare/config.py](src/podcare/config.py)), or `Hardcoded`.

---

### 0. Decode

**How it works.** Every input is decoded through ffmpeg to **48 kHz mono
float32**. 48 kHz is DeepFilterNet3's native rate, so the entire chain runs at
one rate and resamples just once at the very end. Stereo inputs are downmixed to
mono (podcast voice is one channel per mic). A file that decodes to zero samples
is rejected immediately with a clear error.

| Parameter | Value | Controlled by |
|---|---|---|
| Internal sample rate | 48000 Hz | `Config.sr` (do not change — DFN3 requires 48 k) |
| Channels | mono | Hardcoded |

---

### 1. Repair — `--no-declip`

**How it works.** Cleanup leads so every later stage sees an undistorted signal.
Three ffmpeg filters run in series: **`adeclick`** interpolates over impulsive
clicks/glitches; **`adeclip`** reconstructs samples driven past full-scale
(recording too hot), restoring the rounded peaks clipping flattened; and a
**2-pole high-pass** removes subsonic rumble, desk thumps, and HVAC roar below
the voice fundamental. Distortion repaired here can't be denoised away later,
which is why it's first. *Not strength-scaled — this is restoration, not a
degree of effect.*

| Parameter | Value | Controlled by |
|---|---|---|
| Declick + declip enabled | on | CLI `--no-declip` |
| High-pass cutoff | 80 Hz | `Config.hpf_hz` |
| High-pass slope | 2-pole (−12 dB/oct) | Hardcoded |

---

### 2. Align + polarity — `--no-align` *(session-level, ≥ 2 tracks)*

**How it works.** Independent recorders rarely start at the same instant, and a
miswired/inverted mic causes **phase cancellation** when summed. Both are fixed
against the first track as reference. A coarse **GCC-PHAT** cross-correlation
(on an 8 kHz downsample of the opening window) estimates the time offset —
GCC-PHAT whitens the cross-spectrum so the peak stays sharp even when two mics
sound very different. The peak must clear a **confidence gate** (z-score) **and**
be confirmed by a direct sample-domain waveform correlation (|r| ≥ 0.08) before
any shift is applied — this two-key check stops genuinely **uncorrelated remote
recordings** (different rooms, no bleed) from being shifted on a spurious peak.
If that confirming correlation is negative, the track's **polarity is inverted**
so it adds to rather than cancels the reference. Tracks are then zero-padded to
equal length. With one input, this stage no-ops. *Not strength-scaled — it's a
correctness fix.*

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-align` |
| Offset search window | first 300 s | `Config.align_window_s` |
| Peak confidence gate | z ≥ 12 | `Config.align_min_confidence` |
| Waveform confirm threshold | \|r\| ≥ 0.08 | Hardcoded |
| Polarity-flip threshold | r < 0 | Hardcoded |
| Coarse correlation rate | 8 kHz | Hardcoded |

---

### 3. Denoise — `--no-denoise`, `--denoise-backend`

**How it works.** Broadband noise reduction — room tone, hiss, fans, hum,
distant traffic. Default backend is **DeepFilterNet3**, a full-band 48 kHz neural
speech-enhancement model that separates voice from noise far more cleanly than
classical methods and also tames light reverb and breath noise; it's processed
in 60 s chunks with a 1 s crossfade to bound memory on long files. The
**spectral** backend is the classical `noisereduce` algorithm — much faster, no
torch. Strength sets how much noise is removed: for DeepFilterNet it's an
attenuation ceiling (unlimited at the top), for spectral it's the
`prop_decrease`.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-denoise` |
| Backend | auto → deepfilter | CLI `--denoise-backend` |
| Spectral `prop_decrease` | `0.4 → 1.0` (0.88 @ 0.8) | **Strength** |
| DFN attenuation ceiling | `6 → 60 dB`, none ≥ 0.95 (≈49 dB @ 0.8) | **Strength** |
| Chunk / crossfade | 60 s / 1 s | Hardcoded |
| Spectral FFT size | 2048 | Hardcoded |

> Note: DeepFilterNet 0.5.x imports `torchaudio.backend.common`, which newer
> torchaudio removed. Podcare installs a tiny compatibility shim at import time
> so current torch/torchaudio still work — no action needed.

---

### 4. Dereverb — `--no-dereverb` *(WPE)*

**How it works.** Removes the room — the late-reverberation "tail" that makes
voices sound distant or boxy. Uses **WPE** (Weighted Prediction Error), the
standard linear-prediction dereverb: it estimates a per-frequency filter that
predicts the reverb tail from the recent past and subtracts it. This is
complementary to the neural denoiser (which targets *noise*, not room response).
Run in 30 s chunks with crossfade. Strength lengthens the filter and adds
prediction iterations — more reverb removed, at higher CPU cost.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-dereverb` |
| WPE taps (filter length) | `6 → 16` (14 @ 0.8) | **Strength** |
| WPE iterations | `1 → 7` (6 @ 0.8) | **Strength** |
| WPE prediction delay | 3 | `Config.wpe_delay` |
| Chunk length | 30 s | `Config.dereverb_chunk_s` |
| STFT size / shift | 1024 / 256 | Hardcoded |

---

### 5. Plosive ducking — `--no-plosives`

**How it works.** "P"/"B" sounds blast a burst of low-frequency energy into the
mic ("p-pop"). Podcare takes each track's STFT and flags frames where energy
below the plosive ceiling is both abnormally high (≫ the track's own median)
**and** dominates the frame's full spectrum, then ducks just those low-frequency
bins back toward the typical level. The attenuation is feathered one frame each
side so the gain ramps rather than steps — the voice's pitch and body are left
intact. Strength lowers the detection thresholds (catch more pops) and deepens
the duck.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-plosives` |
| Plosive band ceiling | 150 Hz | `Config.plosive_max_hz` |
| Burst threshold (×median) | `14 → 4` (6.0 @ 0.8) | **Strength** |
| Spectral-dominance threshold | `0.6 → 0.4` (0.36 @ 0.8) | **Strength** |
| Duck target (×median) | `6 → 3` (3.6 @ 0.8) | **Strength** |
| STFT size / overlap | 1024 / 768 | Hardcoded |
| Gain spread | ±1 frame | Hardcoded |

---

### 6. De-ess — `--no-deess`

**How it works.** Tames harsh "S"/"SH"/"T" sibilance. A zero-phase **split-band**
design guarantees `full = sibilant_band + rest` exactly: the sibilance band is
extracted, and whenever its short-time energy exceeds a fraction of the
full-band energy (i.e. a sibilant is sounding), the band is dynamically
attenuated and recombined with the untouched rest. Fast attack / slow release
(~3 ms / ~30 ms) keeps it transparent — only the sibilants duck, not the whole
top end. Strength lowers the trigger ratio (catch more) and raises the maximum
reduction.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-deess` |
| Sibilance band | 4500–9500 Hz | `Config.deess_lo_hz` / `deess_hi_hz` |
| Trigger ratio (band/full RMS) | `0.6 → 0.3` (0.36 @ 0.8) | **Strength** |
| Max reduction | `4 → 14 dB` (12 dB @ 0.8) | **Strength** |
| Attack / release | ~3 ms / ~30 ms | Hardcoded |
| Band filter order | 4th-order Butterworth | Hardcoded |
| Audibility gate | −45 dBFS | Hardcoded |

---

### 7. Gate + level match — `--no-gate` *(per track, before mixdown)*

**How it works.** Two jobs, both about making multiple mics sit together. The
**downward expander (gate):** when a track's short-time level falls below an
adaptive speech/noise threshold, it's pushed down (2:1 expansion, floored at the
gate depth), suppressing the other speaker's **crosstalk/bleed** plus room tone
between phrases; a slow release protects word tails. **Level matching:** each
track's *speech-active* RMS (ignoring the gated gaps) is normalized toward a
target so a quiet guest and a loud host arrive at the mix balanced. Strength sets
how deep the gate cuts.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-gate` |
| Gate depth (max attenuation) | `6 → 24 dB` (20.4 dB @ 0.8) | **Strength** |
| Speech-level target | −20 dBFS | `Config.level_target_dbfs` |
| Expansion ratio | 2:1 | Hardcoded |
| Attack / release | ~5 ms / ~160 ms | Hardcoded |
| Level-match clamp | −12 … +24 dB | Hardcoded |
| Threshold | adaptive (noise-floor + speech-relative) | Hardcoded |

---

### 8. Mixdown *(session-level)*

**How it works.** All cleaned, gated, level-matched tracks are **summed to one
mono program**. If the sum would clip, it's scaled back to leave ~1 dB of
headroom (true loudness is set later by mastering). With a single input there's
nothing to sum and the track passes straight through. From here on there is
exactly one timeline.

| Parameter | Value | Controlled by |
|---|---|---|
| Post-sum headroom | −1 dBFS | Hardcoded |

---

### 9. Filler-word removal — `--no-fillers`, `--filler-sensitivity`, `--whisper-model`

**How it works.** Cuts non-lexical fillers — "um", "uh", "ehm", "er", "hmm",
"mm", … — detected from **Whisper word timestamps**, not by listening for a
sound. faster-whisper transcribes the mix with word-level timing and per-word
probabilities; words whose normalized text is in the filler lexicon become cut
candidates. Because Whisper tends to *skip* disfluencies, transcription is biased
toward verbatim output (a filler-laden initial prompt,
`condition_on_previous_text=False`). Sensitivity sets three gates a candidate
must clear: a **probability floor** `0.9 − 0.6·sens`, a **minimum duration**
`0.24 − 0.2·sens` s, and (below 0.7) an **isolation** requirement that the filler
be flanked by a small silence. Each accepted filler is removed with a short
equal-power crossfade. Only timestamps/probabilities are used — the transcript
text is never written anywhere. If faster-whisper isn't installed, the stage logs
a warning and is skipped rather than failing.

The effective sensitivity follows strength conservatively (`0.7 × strength`, so
0.56 at the default) to protect real speech, and is overridden by an explicit
`--filler-sensitivity`.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-fillers` |
| Sensitivity | `0.7 × strength` (0.56 @ 0.8) unless overridden | **Strength** / CLI `--filler-sensitivity` |
| Probability floor | `0.9 − 0.6 × sensitivity` | Derived from sensitivity |
| Minimum duration | `0.24 − 0.2 × sensitivity` s | Derived from sensitivity |
| Isolation required | when sensitivity < 0.7 | Derived from sensitivity |
| Whisper model | large-v3 | CLI `--whisper-model` |
| Cut edge padding | 12 ms | `Config.filler_pad_s` |
| Filler lexicon | um/uh/ehm/er/hmm/… | Hardcoded |

---

### 10. Pause tightening — `--no-tighten`, `--max-pause`, `--target-pause`

**How it works.** Removes dead air for pacing. Block-RMS energy detection finds
silent runs on the mixed program; a run longer than the max-pause is shortened to
the target-pause with a crossfade, and lead/trail silence is trimmed. The
threshold sits well below speech level (and below the post-gate noise floor) so
breaths, beats, and quiet reactions survive — only true dead air is cut. Strength
shortens both the trigger and the kept beat.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-tighten` |
| Max pause (trigger) | `3.5 → 1.0 s` (1.5 s @ 0.8) unless overridden | **Strength** / CLI `--max-pause` |
| Target pause (kept) | `1.0 → 0.4 s` (0.52 s @ 0.8) unless overridden | **Strength** / CLI `--target-pause` |
| Lead/tail trim | 0.5 s | `Config.lead_trail_s` |
| Detection block | 10 ms | Hardcoded |
| Silence threshold | adaptive (well below speech) | Hardcoded |
| Edit crossfade | 30 ms | Hardcoded |

---

### 11. Master — `--no-master`, `--lufs`, `--bitrate`

**How it works.** The finishing chain, applied as one ffmpeg graph. **Bus
compression** (`acompressor`, soft knee) evens out the remaining swings between a
loud laugh and a soft aside, giving the episode consistent weight (compression
is off at `--strength 0`). Then **two-pass loudness normalization** (`loudnorm`,
EBU R128 — always on, even at strength 0, so the output hits its target level):
the first pass *measures* integrated loudness, range, and true peak; the second
applies a **linear** gain to hit exactly the target LUFS with true peak under the
ceiling. Two-pass + linear is what makes the result accurate and transparent
rather than pumping. A silent/near-silent program (below loudnorm's −70 LUFS
gate) skips normalization and is emitted as-is rather than erroring. Strength
firms up the compression (higher ratio, lower threshold);
loudness and true-peak targets are absolute, not strength-scaled.

| Parameter | Value | Controlled by |
|---|---|---|
| Stage enabled | on | CLI `--no-master` |
| Compressor enabled | on | `Config.compress` |
| Compression ratio | `1.5 → 3.5` (3.1 @ 0.8) | **Strength** |
| Compression threshold | `0.20 → 0.10` amplitude (0.12 @ 0.8) | **Strength** |
| Compressor attack/release/knee | 10 ms / 200 ms / 4 | Hardcoded |
| Integrated loudness target | −16 LUFS | CLI `--lufs` |
| True-peak ceiling | −1.5 dBTP | `Config.true_peak_db` |
| Loudness range (LRA) | 11 | Hardcoded |
| Normalization | two-pass, linear | Hardcoded |

---

### 12. Encode + resample

**How it works.** A **single** resample to the output rate closes the chain —
done here once, with an enlarged resampling kernel, rather than repeatedly
mid-pipeline. WAV and FLAC are written **16-bit with triangular-HP dither** (the
correct way to reduce bit depth without quantization distortion); MP3/AAC are
encoded from float at the chosen bitrate. The container is chosen from the output
file's extension.

| Parameter | Value | Controlled by |
|---|---|---|
| Output sample rate | 44100 Hz | CLI `--out-sr` |
| WAV/FLAC bit depth | 16-bit + triangular-HP dither | Hardcoded |
| Resampler kernel size | 64 | Hardcoded |
| Lossy bitrate | 192k | CLI `--bitrate` |

---

## How it's built

- **Python 3.11 + uv.** One linear pipeline over a `Session` (a list of `Track`s
  + sample rate). Each stage is a small function `(Session|Track, Config) →
  Session|Track`; the stage list lives in
  [src/podcare/pipeline.py](src/podcare/pipeline.py), every tunable (and the
  strength→stage mapping) in [src/podcare/config.py](src/podcare/config.py).
- **ffmpeg** for decode/encode and the repair + master filters; **numpy/scipy**
  for the hand-written DSP (align, plosives, de-ess, gate, tighten);
  **DeepFilterNet/torch**, **nara-wpe**, and **faster-whisper** for the ML/heavy
  stages.
- Tests use synthetic fixtures with known ground truth (a known offset alignment
  must recover, injected sibilance de-ess must reduce, a long pause tightening
  must shrink, final loudness within ±1.5 LU of target) plus the strength-mapping
  invariants: `uv run pytest`.
