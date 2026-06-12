"""Mastering: multiband bus compression + two-pass EBU R128 loudness
normalization + true-peak limiting + encode."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from .. import audio_io
from ..config import Config
from ..dsp import crossfade_concat, db_to_lin
from ..session import Track

log = logging.getLogger(__name__)

# Linkwitz-Riley crossover points (Hz) for the 3-band master compressor.
_SPLIT_LO = 250
_SPLIT_HI = 4000
_BOOKEND_XF_S = 0.1  # equal-power crossfade at each bookend join


def _band_comp(threshold: float, ratio: float, attack: int, release: int) -> str:
    return (f"acompressor=threshold={min(threshold, 1.0):.3f}:ratio={ratio:.2f}"
            f":attack={attack}:release={release}:knee=4")


def _multiband(cfg: Config) -> str:
    """3-band (low/mid/high) bus compressor as a filter_complex graph.

    Independent per-band dynamics tame proximity boom and plosive thump (low) and
    harsh upper-mids (high) without ducking the whole program off the loudest
    band — denser, more consistent loudness than a single broadband compressor.
    Low band controls a touch firmer, high band gentler; all derived from the one
    strength knob so 1:1 at strength 0 is the degenerate (no-compression) case.
    """
    r, th = cfg.comp_ratio(), cfg.comp_threshold()
    low = _band_comp(th * 0.9, min(r * 1.1, 20.0), 10, 200)
    mid = _band_comp(th, r, 10, 200)
    high = _band_comp(th * 1.1, max(r * 0.8, 1.0), 5, 120)
    return (f"acrossover=split={_SPLIT_LO} {_SPLIT_HI}:order=4th[lo][mid][hi];"
            f"[lo]{low}[a];[mid]{mid}[b];[hi]{high}[c];"
            "[a][b][c]amix=inputs=3:normalize=0[out]")


def _limiter(cfg: Config) -> str:
    """Brickwall true-peak safety limiter — the final master node, before the
    single soxr resample. loudnorm's linear gain is not a real lookahead limiter,
    so this catches the inter-sample / codec overs it can leave and lets the
    program sit at the loudness target without clipping consumer DACs. Runs at the
    48 kHz internal rate for ISP headroom; ``level=false`` so it never
    auto-makeup-gains against loudnorm's loudness target. Delivery safety, not an
    intensity — it runs whenever mastering is on, regardless of strength."""
    return (f"alimiter=limit={db_to_lin(cfg.true_peak_db):.6f}"
            ":attack=5:release=50:asc=1:level=false")


def _exciter(cfg: Config) -> str:
    """Harmonic presence exciter — synthesizes new harmonics to restore the
    "air" that heavy denoise/dereverb removes, rather than boosting (possibly
    noisy) existing highs. Runs before the loudnorm measurement so the added
    energy is included in the loudness math.

    Source-band choice: ``freq=4000`` tells aexciter to derive harmonics from
    content above 4 kHz. Harmonics land at integer multiples of the source
    partials — a 5 kHz component generates harmonics at 10 kHz, 15 kHz, …
    (the "air" band, 8–16 kHz). For speech, consonants and sibilance sit in
    the 4–8 kHz band, so their harmonics land exactly in the desired air
    region. Using ``freq=7400`` (the previous value) placed the source band
    above the highest test-signal component (5 kHz), so the filter had nothing
    to excite and only leaked distortion.

    Amount: ``cfg.eff_exciter_amount()`` maps 0→1.0 with strength (overridable).
    Deliberately gentle: amount 0.8 at ``drive`` 4 lifts the >6 kHz presence
    band ~+3 dB — a touch of air, not a sheen. A hotter amount/drive both reads
    as harsh and excites residual HF noise on the voice. ``drive`` (smaller =
    smoother harmonics) is ``cfg.exciter_drive``. ``ceil=16000`` lifts
    aexciter's default ~10 kHz harmonic ceiling so the synthesized air reaches
    the 12–16 kHz region it is meant to restore. Output peak stays well within
    the safe range for loudnorm; no chained limiter is needed."""
    return (f"aexciter=amount={cfg.eff_exciter_amount():.2f}"
            f":drive={cfg.exciter_drive:.1f}:freq=4000:ceil=16000")


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


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
    """intro -> program -> outro with equal-power crossfades. Returns
    (audio, joined); joined=False means no bookends were given.

    Each join's crossfade is 100 ms clamped to half of THAT bookend, so a tiny
    intro sting is never consumed whole by its own fade — and doesn't shorten
    the outro's fade either."""
    if intro is None and outro is None:
        return program, False

    def xf_for(bookend: np.ndarray) -> int:
        return min(int(_BOOKEND_XF_S * cfg.sr), max(2, len(bookend) // 2))

    out = program
    xf_in = xf_out = 0
    if intro is not None:
        prepared = _prepare_bookend(intro, cfg, "intro")
        xf_in = xf_for(prepared)
        out = crossfade_concat([prepared, out], xf_in)
    if outro is not None:
        prepared = _prepare_bookend(outro, cfg, "outro")
        xf_out = xf_for(prepared)
        out = crossfade_concat([out, prepared], xf_out)
    log.info("master: bookends — intro=%s outro=%s xfade=%d/%dms",
             intro is not None, outro is not None,
             int(1000 * xf_in / cfg.sr), int(1000 * xf_out / cfg.sr))
    return out, True


def _apply_multiband(audio: np.ndarray, cfg: Config) -> np.ndarray:
    out = audio_io.filter_complex_array(audio, cfg.sr, _multiband(cfg))
    # acrossover/amix preserve length, but pin it exactly so downstream timing
    # (reported output duration) can never drift.
    if len(out) >= len(audio):
        return out[: len(audio)]
    return np.pad(out, (0, len(audio) - len(out)))


def master_and_encode(track: Track, cfg: Config, out_path: Path, *,
                      intro: np.ndarray | None = None,
                      outro: np.ndarray | None = None) -> None:
    if not cfg.master:
        assembled, joined = _assemble_bookends(track.audio, cfg, intro, outro)
        if joined:
            # Equal-power crossfades of same-sign material can peak ~1.27x; the
            # raw mix was never limited, and encode hard-clips at int16. One
            # delivery-safety limiter pass when (and only when) bookends were
            # joined — the no-bookends raw path stays bit-identical.
            assembled = audio_io.filter_array(assembled, cfg.sr, _limiter(cfg))
        audio_io.encode(assembled, cfg.sr, out_path,
                        out_sr=cfg.out_sr, lossy_bitrate=cfg.lossy_bitrate)
        return

    # Multiband compression (strength-scaled, off at strength 0) is applied first,
    # as its own pass; loudness normalization then always runs on the result so the
    # output hits the target level even at strength 0.
    audio = _apply_multiband(track.audio, cfg) if (cfg.compress and cfg.s > 0) else track.audio

    if cfg.exciter and cfg.s > 0:
        excited = audio_io.filter_array(audio, cfg.sr, _exciter(cfg))
        # aexciter preserves length; pin it exactly like the multiband pass.
        audio = (excited[: len(audio)] if len(excited) >= len(audio)
                 else np.pad(excited, (0, len(audio) - len(excited))))
        log.info("master: exciter amount=%.2f drive=%.1f",
                 cfg.eff_exciter_amount(), cfg.exciter_drive)

    measured = audio_io.measure_loudnorm(audio, cfg.sr, pre_filters=None,
                                         lufs=cfg.lufs, true_peak=cfg.true_peak_db)
    # Below loudnorm's -70 LUFS gate (silent / near-silent program) ffmpeg returns
    # -inf/inf, which the second pass rejects ("out of range"). Emit the (silent)
    # mix instead of crashing the whole episode at the final step.
    if not (_finite(measured.get("input_i")) and _finite(measured.get("input_tp"))
            and _finite(measured.get("target_offset"))):
        log.warning("master: program is silent/near-silent (input_i=%s) — skipping loudnorm",
                    measured.get("input_i"))
        assembled, _ = _assemble_bookends(audio, cfg, intro, outro)
        audio_io.encode(assembled, cfg.sr, out_path, out_sr=cfg.out_sr, lossy_bitrate=cfg.lossy_bitrate)
        return
    log.info("master: measured %s LUFS, %s dBTP — normalizing to %.1f LUFS / %.1f dBTP",
             measured.get("input_i"), measured.get("input_tp"), cfg.lufs, cfg.true_peak_db)
    loudnorm = (
        f"loudnorm=I={cfg.lufs}:TP={cfg.true_peak_db}:LRA=11"
        f":measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )
    # loudnorm -> true-peak limiter at the working rate; then one soxr resample.
    mastered = audio_io.filter_array(audio, cfg.sr, f"{loudnorm},{_limiter(cfg)}")
    assembled, joined = _assemble_bookends(mastered, cfg, intro, outro)
    if joined:
        # crossfade overlaps of two TP-limited signals can momentarily sum
        # above the ceiling — one more limiter pass over the joined program.
        assembled = audio_io.filter_array(assembled, cfg.sr, _limiter(cfg))
    audio_io.encode(assembled, cfg.sr, out_path, out_sr=cfg.out_sr,
                    lossy_bitrate=cfg.lossy_bitrate)
