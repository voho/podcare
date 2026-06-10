"""ffmpeg-backed decode/encode and filter helpers."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


class FfmpegError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise FfmpegError("ffmpeg not found on PATH — install it (e.g. `brew install ffmpeg`)")


def _run(cmd: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")
        raise FfmpegError(f"ffmpeg failed ({' '.join(cmd[:6])} …):\n{stderr[-2000:]}")
    return proc


def decode(path: Path, sr: int) -> np.ndarray:
    """Decode any audio file to float32 mono at the given sample rate."""
    if not path.exists():
        raise FileNotFoundError(path)
    proc = _run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(path),
         "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    )
    audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if len(audio) == 0:
        raise FfmpegError(f"{path}: decoded to zero samples — is this an audio file?")
    return audio


def filter_array(audio: np.ndarray, sr: int, filters: str) -> np.ndarray:
    """Pipe a float32 mono array through an ffmpeg audio filtergraph."""
    proc = _run(
        ["ffmpeg", "-hide_banner", "-nostdin",
         "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
         "-af", filters, "-f", "f32le", "-ar", str(sr), "-ac", "1", "-"],
        input_bytes=np.ascontiguousarray(audio, dtype=np.float32).tobytes(),
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    sf.write(str(path), audio, sr, subtype="FLOAT")


def _codec_args(out_path: Path, mp3_bitrate: str) -> tuple[list[str], bool]:
    """Codec args for the output container and whether it stores 16-bit PCM."""
    suffix = out_path.suffix.lower()
    if suffix == ".mp3":
        return ["-c:a", "libmp3lame", "-b:a", mp3_bitrate], False
    if suffix in (".m4a", ".aac"):
        return ["-c:a", "aac", "-b:a", mp3_bitrate], False
    if suffix == ".flac":
        return ["-c:a", "flac", "-sample_fmt", "s16"], True
    if suffix == ".wav":
        return ["-c:a", "pcm_s16le"], True
    raise FfmpegError(f"Unsupported output format {suffix!r} — use .wav, .mp3, .flac or .m4a")


def encode(audio: np.ndarray, sr: int, out_path: Path, *, mp3_bitrate: str = "192k",
           filters: str | None = None, out_ar: int | None = None) -> None:
    """Encode a float32 mono array to the output file, optionally through a filtergraph.

    out_ar resamples at the end of the chain (also needed after loudnorm, which
    internally upsamples to 192 kHz). For 16-bit PCM targets the same aresample
    applies triangular-HP dither; lossy codecs are fed float directly.
    """
    codec, wants_s16 = _codec_args(out_path, mp3_bitrate)
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y",
           "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-"]
    chain = [filters] if filters else []
    if out_ar:
        resample = f"aresample=out_sample_rate={out_ar}:filter_size=64"
        if wants_s16:
            resample += ":out_sample_fmt=s16:dither_method=triangular_hp"
        chain.append(resample)
    if chain:
        cmd += ["-af", ",".join(chain)]
    cmd += codec + [str(out_path)]
    _run(cmd, input_bytes=np.ascontiguousarray(audio, dtype=np.float32).tobytes())


def measure_loudnorm(audio: np.ndarray, sr: int, *, pre_filters: str | None,
                     lufs: float, true_peak: float) -> dict:
    """First loudnorm pass: measure integrated loudness stats, return the JSON dict."""
    graph = (f"{pre_filters}," if pre_filters else "") + \
        f"loudnorm=I={lufs}:TP={true_peak}:LRA=11:print_format=json"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "measure.wav"
        write_wav(tmp, audio, sr)
        proc = _run(["ffmpeg", "-hide_banner", "-nostdin", "-i", str(tmp),
                     "-af", graph, "-f", "null", "-"])
    stderr = proc.stderr.decode(errors="replace")
    start = stderr.rfind("{")
    if start == -1:
        raise FfmpegError(f"loudnorm measurement produced no JSON:\n{stderr[-1000:]}")
    # ffmpeg appends progress lines after the JSON block; parse just the object.
    obj, _ = json.JSONDecoder().raw_decode(stderr[start:])
    return obj
