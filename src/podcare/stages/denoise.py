"""Noise reduction.

Backends:
- "deepfilter": DeepFilterNet3, a 48 kHz full-band neural speech-enhancement
  model — the quality default. Also tames residual reverb and breath noise.
- "spectral": noisereduce spectral gating — dependency-light fallback.
- "auto": deepfilter when importable, else spectral.
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import Config
from ..dsp import process_chunked
from ..session import Track

log = logging.getLogger(__name__)

_DF_CHUNK_S = 60.0
_df_runtime: tuple | None = None


def _shim_torchaudio_backend() -> None:
    """DeepFilterNet 0.5.x imports torchaudio.backend.common, removed in
    torchaudio >= 2.2; alias it so the import succeeds."""
    import sys
    import types

    import torchaudio

    if "torchaudio.backend.common" in sys.modules:
        return
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")
    common.AudioMetaData = getattr(torchaudio, "AudioMetaData", object)
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common


def _load_deepfilter() -> tuple:
    global _df_runtime
    if _df_runtime is None:
        _shim_torchaudio_backend()
        from df.enhance import enhance, init_df

        model, df_state, _ = init_df(log_level="WARNING", log_file=None)
        _df_runtime = (enhance, model, df_state)
    return _df_runtime


def _resolve_backend(cfg: Config) -> str:
    if cfg.denoise_backend in ("deepfilter", "spectral"):
        return cfg.denoise_backend
    try:
        _load_deepfilter()
        return "deepfilter"
    except Exception as exc:  # ImportError, model load failure, ...
        log.warning("denoise: DeepFilterNet unavailable (%s) — using spectral gating", exc)
        return "spectral"


def _deepfilter_denoise(track: Track, cfg: Config) -> Track:
    import torch

    enhance, model, df_state = _load_deepfilter()
    if df_state.sr() != cfg.sr:
        raise RuntimeError(f"DeepFilterNet expects {df_state.sr()} Hz, pipeline is {cfg.sr} Hz")
    atten = cfg.df_atten_lim_db()

    def run_chunk(chunk: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))[None, :]
            out = enhance(model, df_state, t, atten_lim_db=atten)
        return out[0].cpu().numpy().astype(np.float32)

    audio = process_chunked(track.audio, cfg.sr, run_chunk, chunk_s=_DF_CHUNK_S)
    return Track(track.name, audio)


def _spectral_denoise(track: Track, cfg: Config) -> Track:
    import noisereduce  # deferred: pulls in matplotlib

    audio = noisereduce.reduce_noise(
        y=track.audio,
        sr=cfg.sr,
        stationary=False,
        prop_decrease=cfg.denoise_prop(),
        n_fft=2048,
    )
    return Track(track.name, audio.astype(np.float32))


def denoise_track(track: Track, cfg: Config) -> Track:
    backend = _resolve_backend(cfg)
    out = _deepfilter_denoise(track, cfg) if backend == "deepfilter" else _spectral_denoise(track, cfg)
    before = float(np.mean(track.audio.astype(np.float64) ** 2)) + 1e-20
    after = float(np.mean(out.audio.astype(np.float64) ** 2)) + 1e-20
    log.info("denoise: %s — backend %s, %+.1f dB energy change",
             track.name, backend, 10 * np.log10(after / before))
    return out
