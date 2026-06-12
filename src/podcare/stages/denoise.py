"""Neural noise reduction with DeepFilterNet3.

DeepFilterNet3 is a full-band 48 kHz neural speech-enhancement model — it
separates voice from noise far more cleanly than classical methods and also
tames light reverb and breath noise. Weights ship inside the `deepfilternet`
package, so it runs offline. Processed in chunks with a crossfade so memory
stays bounded on hour-long tracks. Strength sets the attenuation ceiling.
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


def _load_deepfilter() -> tuple:
    global _df_runtime
    if _df_runtime is None:
        from .._compat import ensure_ml_compat

        ensure_ml_compat()
        from df.enhance import enhance, init_df

        model, df_state, _ = init_df(log_level="WARNING", log_file=None)
        _df_runtime = (enhance, model, df_state)
    return _df_runtime


def unload_deepfilter() -> None:
    """Release the cached DeepFilterNet model and PyTorch's allocator cache.

    The model is loaded once and memoized in `_df_runtime`; nothing after the
    denoise stage uses it, yet it (plus PyTorch's cached allocations) would
    otherwise stay resident through the memory-hungry dereverb/WPE stage. The
    pipeline calls this at the denoise->dereverb boundary to free that RAM
    before WPE allocates its large per-chunk transients.
    """
    global _df_runtime
    if _df_runtime is None:
        return
    _df_runtime = None  # drop the only reference to (enhance, model, df_state)

    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    # Return PyTorch's cached blocks to the OS. Guard each backend: the call
    # raises if that backend isn't built/available on this platform.
    for backend in (getattr(torch, "mps", None), getattr(torch, "cuda", None)):
        empty = getattr(backend, "empty_cache", None)
        if empty is None:
            continue
        try:
            empty()
        except Exception:  # noqa: BLE001 — best-effort cache release
            pass


def denoise_track(track: Track, cfg: Config) -> Track:
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

    audio = process_chunked(track.audio, cfg.sr, run_chunk, chunk_s=_DF_CHUNK_S,
                            label=f"denoise · {track.name}")
    before = float(np.mean(track.audio.astype(np.float64) ** 2)) + 1e-20
    after = float(np.mean(audio.astype(np.float64) ** 2)) + 1e-20
    log.info("denoise: %s — DeepFilterNet3, atten=%.0fdB, %+.1f dB energy change",
             track.name, atten, 10 * np.log10(after / before))
    return Track(track.name, audio)
