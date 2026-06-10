"""Filler-word removal ("um", "ehm", …) driven by Whisper word timestamps.

Whisper tends to skip disfluencies, so transcription is biased toward verbatim
output with a filler-laden initial prompt. Sensitivity (0..1) maps to the word
probability / duration / isolation thresholds a candidate must clear before
it is cut; cuts use short equal-power crossfades.
"""

from __future__ import annotations

import logging
import string

import numpy as np
from scipy.signal import resample_poly

from ..config import Config
from ..dsp import merge_intervals, remove_intervals
from ..session import Track

log = logging.getLogger(__name__)

_FILLERS = {
    "um", "umm", "ummm", "uh", "uhh", "uhm", "uhmm",
    "hm", "hmm", "hmmm", "mm", "mmm", "mhm",
    "em", "ehm", "ehmm", "eh", "ehh", "er", "erm", "err",
    "ah", "ahh", "aah", "ahm",
    "ee", "eee", "é", "éé", "ééé", "ehe",
}
_STRIP = string.punctuation + "…—–‐'’  "

_VERBATIM_PROMPT = (
    "Umm, so, uh, yeah... ehm, I was thinking, hmm, like, you know... "
    "Ehm, no, ééé, vlastně, hmm."
)

_WHISPER_SR = 16000


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


def _transcribe_words(audio: np.ndarray, sr: int,
                      model_name: str) -> list[tuple[str, float, float, float]]:
    from faster_whisper import WhisperModel

    audio16 = resample_poly(audio.astype(np.float64), _WHISPER_SR, sr).astype(np.float32)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        audio16,
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=_VERBATIM_PROMPT,
        vad_filter=False,
        beam_size=5,
    )
    log.info("fillers: transcribing (detected language %s, p=%.2f)",
             info.language, info.language_probability)
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append((w.word, float(w.start), float(w.end), float(w.probability)))
    return words


def remove_fillers_track(track: Track, cfg: Config) -> Track:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        log.warning("fillers: faster-whisper not installed — skipping filler removal")
        return track
    words = _transcribe_words(track.audio, cfg.sr, cfg.whisper_model)
    intervals = find_filler_intervals(words, cfg.eff_filler_sensitivity(), cfg.filler_pad_s)
    if not intervals:
        log.info("fillers: nothing to cut")
        return track
    cut_s = sum(e - s for s, e in intervals)
    log.info("fillers: cutting %d fillers (%.1f s total)", len(intervals), cut_s)
    return Track(track.name, remove_intervals(track.audio, cfg.sr, intervals))
