"""All pipeline tunables in one place.

One universal `strength` knob (0..1) drives how hard every stage works; its
meaning differs per stage (see the `*_from_strength` helpers below). A handful
of fields are absolute *targets* or *formats* (loudness, sample rate, de-ess
band edges) rather than intensities, so they are not strength-scaled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


@dataclass(frozen=True)
class Config:
    # Global — processing runs at 48 kHz (DeepFilterNet's native rate); the
    # final encode resamples once to out_sr (16-bit + dither for WAV/FLAC).
    sr: int = 48000
    out_sr: int = 44100
    keep_stems: Path | None = None

    # Timeline lock. When True, every stage that would change the audio's length
    # or timing is skipped — inter-track alignment (shifts/pads), filler-word cuts
    # and pause/silence tightening — so the output stays sample-for-sample on the
    # input timeline. For cleaning audio that must drop straight back onto a video
    # edit; all non-destructive stages (denoise, EQ, de-ess, gate, leveler, master,
    # …) still run. (Set --out-sr to the source rate to also keep the sample count.)
    nocut: bool = False

    # Universal processing strength (0..1). The single intensity knob, calibrated
    # so that strength=0 is a true no-op (every strength-driven stage is skipped;
    # the pipeline becomes just align -> mixdown -> encode) and strength=1 is the
    # most aggressive treatment. Each stage maps it to its own working parameters
    # via the helpers at the bottom of this class, whose strength=0 endpoint is
    # the identity value for that stage.
    strength: float = 0.8

    # Repair (restorative; not strength-scaled)
    declip: bool = True
    hpf_hz: float = 80.0

    # Dropout / short-gap restoration (fills brief packet-loss holes via LPC;
    # strength scales the longest gap it may fill — 0 ms at strength 0 = no-op)
    dropouts: bool = True

    # De-hum (mains-hum harmonic notching; detection-gated, strength-scaled)
    dehum: bool = True

    # Alignment / polarity (correctness; not strength-scaled)
    align: bool = True
    align_window_s: float = 300.0  # search the first N seconds for the offset
    align_min_confidence: float = 12.0  # z-score of the GCC-PHAT peak required to apply a shift

    # Denoise — DeepFilterNet3 (neural, full-band 48 kHz). Strength sets the
    # attenuation ceiling (see df_atten_lim_db).
    denoise: bool = True

    # Dereverb (WPE linear prediction; complements DeepFilterNet)
    dereverb: bool = True
    # 15s chunks keep WPE's per-chunk complex128 transients (~hundreds of MB)
    # small so they don't contend for RAM right after the neural denoiser; the
    # 1s crossfade overlap hides chunk seams and a static room needs no more
    # context than this.
    dereverb_chunk_s: float = 15.0
    wpe_delay: int = 3

    # Tonal-balance EQ (per-track LTAS match to a broadcast voice curve)
    tonebalance: bool = True

    # Mouth-click / de-crackle (short mid-band transient removal)
    declick: bool = True

    # Plosive ducking
    plosives: bool = True
    plosive_max_hz: float = 150.0

    # De-esser (band edges are fixed; depth/threshold follow strength)
    deess: bool = True
    deess_lo_hz: float = 4500.0
    deess_hi_hz: float = 9500.0

    # Dynamic resonance / harshness suppression ("Soothe-lite" STFT notching)
    resonance: bool = True

    # Gate / level match (level target is absolute; gate depth follows strength)
    gate: bool = True
    level_target_dbfs: float = -20.0  # speech-active RMS target per track before mixdown

    # Breath control (detect + duck inhales between phrases)
    breath: bool = True

    # Filler-word removal ("um", "ehm", ...). filler_sensitivity=None follows
    # strength; set it explicitly (0..1, 0 disables) to override.
    fillers: bool = True
    filler_sensitivity: float | None = None
    whisper_model: str = "large-v3"
    language: str | None = None  # force ASR/alignment language; None = auto-detect
    filler_pad_s: float = 0.012

    # Segment loudness leveler (slow gating-aware ride on the mono bus)
    leveler: bool = True

    # Pause tightening. max/target pause = None follow strength; set to override.
    tighten: bool = True
    max_pause_s: float | None = None
    target_pause_s: float | None = None
    lead_trail_s: float = 0.5

    # Mastering
    master: bool = True
    compress: bool = True
    lufs: float = -16.0
    true_peak_db: float = -1.5
    lossy_bitrate: str = "192k"  # MP3/AAC bitrate; ignored for WAV/FLAC

    # Harmonic presence exciter (inside master; synthesizes "air" above ~8 kHz)
    exciter: bool = True

    # Intro/outro bookends — optional sounds assembled around the finished
    # program (loudness-matched, 100 ms equal-power crossfades). Assembly, not
    # processing: used whenever set, not strength-scaled. The CLI ignores them
    # under --nocut because an intro shifts the whole timeline.
    intro_sound: Path | None = None
    outro_sound: Path | None = None

    # ------------------------------------------------------------------ #
    # Strength → per-stage intensity. Defaults below are anchored so that
    # strength=0.8 lands on strong-but-safe processing.
    # ------------------------------------------------------------------ #
    @property
    def s(self) -> float:
        return min(1.0, max(0.0, self.strength))

    # Each helper's strength=0 endpoint is the identity (no-op) value for its
    # stage; strength=1 is the most aggressive. Stages are also skipped outright
    # at strength=0 (see pipeline.STAGES), so these endpoints mainly shape the
    # smooth ramp just above 0.

    # De-hum: strength scales how many harmonics are removed and how readily hum
    # is detected. At strength 0 there are 0 harmonics and an unreachable
    # detection margin, so it is a true no-op (and the stage is skipped anyway).
    def dehum_max_harmonics(self) -> int:
        return int(round(_lerp(0.0, 12.0, self.s)))

    def dehum_margin_db(self) -> float:
        return _lerp(20.0, 6.0, self.s)

    # Denoise: 0 removes nothing (0 dB ceiling), 1 removes the most.
    def df_atten_lim_db(self) -> float:
        # DeepFilterNet attenuation ceiling in dB (0 = no attenuation). Capped at
        # a finite 60 dB at the top — already effectively full suppression for
        # speech — so the single knob stays continuous (no jump to "unlimited").
        return _lerp(0.0, 60.0, self.s)

    # Tonal balance: fraction of the measured LTAS deviation that is corrected.
    # Deliberately gentle (strength / 3) so the EQ stays subtle even at full
    # strength; 0 at strength 0 (identity).
    def eq_correction(self) -> float:
        return self.s / 3.0

    # Mouth-click: crest-factor threshold a frame must clear to be flagged. Huge
    # at strength 0 (nothing triggers -> identity), easing to 8× at full strength.
    def declick_crest(self) -> float:
        return _lerp(40.0, 8.0, self.s)

    # Dereverb: longer filter + more iterations remove more reverb at higher cost.
    def wpe_taps(self) -> int:
        return int(round(_lerp(6, 16, self.s)))

    def wpe_iterations(self) -> int:
        # WPE's EM converges within ~3 iterations for speech (nara-wpe's own
        # examples use 2-5); a higher ceiling mostly burns time and allocates
        # more large transients for negligible extra reverb suppression.
        return int(round(_lerp(1, 3, self.s)))

    # Dropouts: longest fillable gap. 0 ms at strength 0 (nothing qualifies ->
    # identity); 50 ms at full strength — the README contract's honesty cap.
    def dropout_max_gap_ms(self) -> float:
        return _lerp(0.0, 50.0, self.s)

    # Resonance: a bin must poke margin dB above its own spectral envelope to
    # count as a resonance (lower = more sensitive); the cut is capped at
    # max_cut.
    def resonance_margin_db(self) -> float:
        return _lerp(18.0, 6.0, self.s)

    # Cap 0 at strength 0 → no reduction applied, making the math a no-op.
    def resonance_max_cut_db(self) -> float:
        return _lerp(0.0, 10.0, self.s)

    # Exciter: ffmpeg aexciter `amount`. 0 adds no harmonics (identity);
    # 2.5 ceiling is deliberately conservative ("easy to overdo").
    def exciter_amount(self) -> float:
        return _lerp(0.0, 2.5, self.s)

    # Plosives: 0 flags ~nothing (very high burst threshold, shallow duck); 1
    # catches the most and ducks hardest. Target stays below burst at all s.
    def plosive_burst_mult(self) -> float:
        return _lerp(24.0, 4.0, self.s)

    def plosive_dominance(self) -> float:
        return _lerp(0.8, 0.4, self.s)

    def plosive_target_mult(self) -> float:
        return _lerp(8.0, 3.0, self.s)

    # De-ess: 0 never triggers / 0 dB reduction; 1 triggers easily and deeply.
    def deess_ratio(self) -> float:
        return _lerp(0.9, 0.3, self.s)

    def deess_max_db(self) -> float:
        return _lerp(0.0, 14.0, self.s)

    # Gate: 0 dB expansion (no gating) at 0, deepest crosstalk suppression at 1.
    def gate_depth_db(self) -> float:
        return _lerp(0.0, 24.0, self.s)

    # Breath: how far detected inhales are ducked. 0 dB at strength 0 (identity),
    # capped at 14 dB at full strength — a duck, never a mute.
    def breath_depth_db(self) -> float:
        return _lerp(0.0, 14.0, self.s)

    # Fillers: 0 at strength 0 (off); deliberately conservative ceiling (protects
    # real speech). Explicit --filler-sensitivity override wins.
    def eff_filler_sensitivity(self) -> float:
        if self.filler_sensitivity is not None:
            return float(min(1.0, max(0.0, self.filler_sensitivity)))
        return 0.7 * self.s

    # Tighten: 0 keeps long pauses (no cutting); 1 tightens hardest.
    def eff_max_pause(self) -> float:
        return self.max_pause_s if self.max_pause_s is not None else _lerp(4.0, 1.0, self.s)

    def eff_target_pause(self) -> float:
        return self.target_pause_s if self.target_pause_s is not None else _lerp(1.2, 0.4, self.s)

    # Leveler: maximum slow ride range (±dB). 0 at strength 0 (identity).
    def leveler_range_db(self) -> float:
        return _lerp(0.0, 8.0, self.s)

    # Master: 1:1 (no compression) at 0, firmest at 1.
    def comp_ratio(self) -> float:
        return _lerp(1.0, 3.5, self.s)

    def comp_threshold(self) -> float:
        return _lerp(0.30, 0.10, self.s)  # linear amplitude; lower = more compression
