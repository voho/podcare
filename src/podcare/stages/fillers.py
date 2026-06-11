"""Filler-word removal ("um", "ehm", …) via Whisper transcription + WhisperX
wav2vec2 forced alignment.

faster-whisper transcribes each mic (biased toward verbatim output with a
filler-laden initial prompt, since Whisper tends to skip disfluencies); WhisperX
then force-aligns that transcript with a wav2vec2 CTC model to get phoneme-tight
word boundaries — far more precise cut points than Whisper's own word timings.

Detection runs **per track, before mixdown**, so the ASR/aligner always sees one
clean isolated voice rather than the summed two-speaker mix. To keep every track
frame-aligned, a filler is only cut when **every other track is silent** during
it; the surviving intervals are then removed identically from all tracks, so they
stay in sync. Sensitivity (0..1) maps to the per-word score / duration / isolation
thresholds a candidate must clear; cuts use short equal-power crossfades.
"""

from __future__ import annotations

import logging
import string

import numpy as np
from scipy.signal import resample_poly

from ..config import Config
from ..dsp import (block_rms, merge_intervals, remove_intervals,
                   speech_threshold)
from ..progress import active
from ..session import Session, Track

log = logging.getLogger(__name__)

_FILLERS = {
    "um", "umm", "ummm", "uh", "uhh", "uhm", "uhmm",
    "hm", "hmm", "hmmm", "mm", "mmm",
    "em", "ehm", "ehmm", "eh", "ehh", "er", "erm", "err",
    "ah", "ahh", "aah", "ahm",
    "ee", "eee", "é", "éé", "ééé", "ehe",
}
# Note: affirmative back-channels ("mhm" = "yes") are deliberately excluded —
# they carry meaning and must not be cut.
_STRIP = string.punctuation + "…—–‐'’  "

_VERBATIM_PROMPT = (
    "Umm, so, uh, yeah... ehm, I was thinking, hmm, like, you know... "
    "Ehm, no, ééé, vlastně, hmm."
)

_WHISPER_SR = 16000
_GUARD_BLOCK_S = 0.02  # resolution of the cross-track "is anyone else talking?" mask

# Heavy models are reused across tracks (the pipeline calls this once per mic).
_asr_cache: dict = {}
_align_cache: dict = {}


def _normalize(word: str) -> str:
    return word.lower().strip(_STRIP)


def find_filler_intervals(words: list[tuple[str, float, float, float]],
                          sensitivity: float, pad_s: float) -> list[tuple[float, float]]:
    """Pick cut intervals from (text, start, end, probability) words.

    Pure function so the decision logic is testable without a Whisper model.
    """
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    if sensitivity == 0.0:
        return []
    p_min = 0.9 - 0.6 * sensitivity
    dur_min = 0.24 - 0.2 * sensitivity
    require_isolation = sensitivity < 0.7

    intervals: list[tuple[float, float]] = []
    for i, (text, start, end, prob) in enumerate(words):
        if _normalize(text) not in _FILLERS:
            continue
        if prob < p_min or (end - start) < dur_min:
            continue
        if require_isolation:
            gap_before = start - words[i - 1][2] if i > 0 else 1.0
            gap_after = words[i + 1][1] - end if i + 1 < len(words) else 1.0
            if min(gap_before, gap_after) < 0.03:
                continue
        intervals.append((start - pad_s, end + pad_s))
    return merge_intervals(intervals, min_gap=0.05)


def _is_silent_during(mask: np.ndarray, start_s: float, end_s: float, block_s: float) -> bool:
    """True if no speech block of `mask` overlaps the [start, end] second window."""
    i0 = max(0, int(np.floor(start_s / block_s)))
    i1 = min(len(mask), int(np.ceil(end_s / block_s)))
    if i0 >= i1:
        return True
    return not bool(mask[i0:i1].any())


def select_cross_track_safe(intervals_per_track: list[list[tuple[float, float]]],
                            masks: list[np.ndarray],
                            block_s: float) -> list[tuple[float, float]]:
    """Keep each track's filler intervals only where every OTHER track is silent.

    The same interval list is cut from every track to preserve sync, so a filler
    is only safe to remove when no other speaker is talking underneath it.
    Pure function — testable with synthetic masks. With a single track there are
    no "other" tracks, so all candidates are kept.
    """
    n = len(intervals_per_track)
    safe: list[tuple[float, float]] = []
    for ti, ivs in enumerate(intervals_per_track):
        for s, e in ivs:
            if all(_is_silent_during(masks[oj], s, e, block_s)
                   for oj in range(n) if oj != ti):
                safe.append((s, e))
    return merge_intervals(safe, min_gap=0.05)


def _get_asr(model_name: str):
    asr = _asr_cache.get(model_name)
    if asr is None:
        from faster_whisper import WhisperModel
        asr = WhisperModel(model_name, device="cpu", compute_type="int8")
        _asr_cache[model_name] = asr
    return asr


def _get_align(language: str):
    cached = _align_cache.get(language)
    if cached is None:
        import whisperx
        cached = whisperx.load_align_model(language_code=language, device="cpu")
        _align_cache[language] = cached
    return cached


def _transcribe_and_align(audio: np.ndarray, sr: int, model_name: str,
                          language: str | None = None, label: str = ""
                          ) -> list[tuple[str, float, float, float]]:
    """Transcribe with faster-whisper, then force-align with WhisperX.

    Returns (word, start, end, score) with wav2vec2-tight boundaries (seconds on
    the original timeline). Any failure — unsupported language, bad model name,
    download/TLS error — degrades to [] so the optional filler pass never aborts
    a multi-hour render; the cause is logged.
    """
    from .._compat import ensure_ml_compat

    ensure_ml_compat()
    audio16 = np.ascontiguousarray(
        resample_poly(audio.astype(np.float64), _WHISPER_SR, sr), dtype=np.float32)
    reporter = active()
    try:
        import whisperx

        asr = _get_asr(model_name)
        # transcribe() returns a lazy segment generator + info up front; drive the
        # progress sub-bar by transcribed audio-seconds (info.duration) as we pull it.
        segments, info = asr.transcribe(
            audio16,
            language=language,
            condition_on_previous_text=False,
            initial_prompt=_VERBATIM_PROMPT,
            vad_filter=False,
            beam_size=5,
        )
        reporter.begin_sub(info.duration, "s", f"transcribe · {label}")
        segs, last = [], 0.0
        for s in segments:
            segs.append({"start": float(s.start), "end": float(s.end), "text": s.text})
            reporter.advance_sub(max(0.0, float(s.end) - last))
            last = float(s.end)
        reporter.end_sub()
        lang = language or info.language
        log.info("fillers: transcribed %d segments (language %s)", len(segs), lang)
        if not segs:
            return []
        reporter.begin_sub(0, "", f"align · {label}")  # opaque step → spinner
        align_model, meta = _get_align(lang)
        aligned = whisperx.align(segs, align_model, meta, audio16, "cpu",
                                 return_char_alignments=False)
    except Exception as exc:  # noqa: BLE001 — optional stage must degrade, not crash
        log.warning("fillers: transcription/alignment unavailable (%s) — "
                    "leaving this track unedited", exc)
        return []
    finally:
        reporter.end_sub()

    words: list[tuple[str, float, float, float]] = []
    for seg in aligned["segments"]:
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append((w["word"], float(w["start"]), float(w["end"]),
                              float(w.get("score", 1.0))))
    return words


def _speech_mask(audio: np.ndarray, sr: int) -> np.ndarray:
    hop = max(1, int(_GUARD_BLOCK_S * sr))
    rms = block_rms(audio, hop)
    return rms >= speech_threshold(rms)


def remove_fillers_session(session: Session, cfg: Config) -> Session:
    """Detect fillers per track on the shared post-align timeline, then remove the
    cross-track-safe intervals identically from every track (keeping them synced)."""
    sens = cfg.eff_filler_sensitivity()
    if sens <= 0.0 or not session.tracks:
        return session

    masks = [_speech_mask(t.audio, session.sr) for t in session.tracks]
    n = len(session.tracks)
    per_track = []
    for i, t in enumerate(session.tracks, start=1):
        label = f"{t.name} ({i}/{n})" if n > 1 else t.name
        words = _transcribe_and_align(t.audio, session.sr, cfg.whisper_model,
                                      cfg.language, label=label)
        per_track.append(find_filler_intervals(words, sens, cfg.filler_pad_s))
    n_found = sum(len(p) for p in per_track)
    cuts = select_cross_track_safe(per_track, masks, _GUARD_BLOCK_S)
    if not cuts:
        log.info("fillers: nothing to cut (%d candidate(s), none cross-track safe)", n_found)
        return session

    cut_s = sum(e - s for s, e in cuts)
    log.info("fillers: cutting %d fillers (%.1f s) from %d track(s); %d candidate(s) found",
             len(cuts), cut_s, len(session.tracks), n_found)
    new_tracks = [Track(t.name, remove_intervals(t.audio, session.sr, cuts))
                  for t in session.tracks]
    return Session(session.sr, new_tracks)
