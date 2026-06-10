"""Compatibility shims for the ML stack, applied once before importing the
heavy models (DeepFilterNet, WhisperX/pyannote).

- torchaudio >= 2.2 removed several legacy symbols (`AudioMetaData`,
  `list_audio_backends`, the `torchaudio.backend.common` module) that
  DeepFilterNet 0.5.x and pyannote.audio 3.x still import. We alias them.
- python.org CPython builds ship without a CA bundle, so `torch.hub`'s urllib
  download of the wav2vec2 alignment model fails TLS verification. Point it at
  certifi's bundle.
"""

from __future__ import annotations

import os
import sys
import types

_applied = False


def ensure_ml_compat() -> None:
    global _applied
    if _applied:
        return

    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = type("AudioMetaData", (), {})
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["ffmpeg", "soundfile"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "ffmpeg"
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda _backend: None
    if "torchaudio.backend.common" not in sys.modules:
        backend = types.ModuleType("torchaudio.backend")
        common = types.ModuleType("torchaudio.backend.common")
        common.AudioMetaData = torchaudio.AudioMetaData
        sys.modules["torchaudio.backend"] = backend
        sys.modules["torchaudio.backend.common"] = common

    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

    _applied = True
