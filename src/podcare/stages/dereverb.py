"""Dereverberation via single-channel WPE (nara-wpe), processed in chunks.

WPE estimates a per-frequency linear filter that predicts (and removes) late
reverberation — complementary to the neural denoiser, which targets noise and
only mildly reduces reverb. Room acoustics are static, so chunking with a
crossfade keeps memory bounded on hour-long tracks without audible seams.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..dsp import process_chunked
from ..session import Track

log = logging.getLogger(__name__)

_STFT_SIZE = 1024
_STFT_SHIFT = 256


def _wpe_chunk(audio: np.ndarray, cfg: Config) -> np.ndarray:
    from nara_wpe.utils import istft, stft
    from nara_wpe.wpe import wpe

    y = stft(audio[np.newaxis, :].astype(np.float64), size=_STFT_SIZE, shift=_STFT_SHIFT)
    # stft -> (D, T, F); wpe wants (F, D, T)
    y = y.transpose(2, 0, 1)
    z = wpe(y, taps=cfg.wpe_taps(), delay=cfg.wpe_delay, iterations=cfg.wpe_iterations(),
            statistics_mode="full")
    out = istft(z.transpose(1, 2, 0), size=_STFT_SIZE, shift=_STFT_SHIFT)[0]
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    return out[: len(audio)].astype(np.float32)


def dereverb_track(track: Track, cfg: Config) -> Track:
    audio = process_chunked(track.audio, cfg.sr, lambda chunk: _wpe_chunk(chunk, cfg),
                            chunk_s=cfg.dereverb_chunk_s)
    before = float(np.mean(track.audio.astype(np.float64) ** 2)) + 1e-20
    after = float(np.mean(audio.astype(np.float64) ** 2)) + 1e-20
    log.info("dereverb: %s — %+.1f dB energy change", track.name, 10 * np.log10(after / before))
    return Track(track.name, audio)
