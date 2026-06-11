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
import soxr

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


def filter_complex_array(audio: np.ndarray, sr: int, graph: str, *,
                         out_label: str = "out") -> np.ndarray:
    """Pipe a float32 mono array through an ffmpeg -filter_complex graph.

    Unlike filter_array (simple, single in/out -af), this supports multi-pad
    graphs (split/compress-per-band/mix) by mapping the named [out_label] pad.
    """
    proc = _run(
        ["ffmpeg", "-hide_banner", "-nostdin",
         "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
         "-filter_complex", graph, "-map", f"[{out_label}]",
         "-f", "f32le", "-ar", str(sr), "-ac", "1", "-"],
        input_bytes=np.ascontiguousarray(audio, dtype=np.float32).tobytes(),
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    sf.write(str(path), audio, sr, subtype="FLOAT")


def _codec_args(out_path: Path, lossy_bitrate: str) -> tuple[list[str], bool]:
    """Codec args for the output container and whether it stores 16-bit PCM."""
    suffix = out_path.suffix.lower()
    if suffix == ".mp3":
        return ["-c:a", "libmp3lame", "-b:a", lossy_bitrate], False
    if suffix in (".m4a", ".aac"):
        return ["-c:a", "aac", "-b:a", lossy_bitrate], False
    if suffix == ".flac":
        return ["-c:a", "flac", "-sample_fmt", "s16"], True
    if suffix == ".wav":
        return ["-c:a", "pcm_s16le"], True
    raise FfmpegError(f"Unsupported output format {suffix!r} — use .wav, .mp3, .flac or .m4a")


def encode(audio: np.ndarray, sr: int, out_path: Path, *, out_sr: int | None = None,
           lossy_bitrate: str = "192k") -> None:
    """Write a float32 mono array to out_path, resampling to out_sr with soxr (VHQ).

    Resampling is the single final step of the chain, done with soxr's
    very-high-quality band-limited resampler. 16-bit PCM targets (WAV/FLAC) get
    triangular-HP dither at quantization; lossy codecs are fed float directly.
    """
    codec, wants_s16 = _codec_args(out_path, lossy_bitrate)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    target_sr = out_sr or sr
    if target_sr != sr:
        audio = np.ascontiguousarray(
            soxr.resample(audio, sr, target_sr, quality="VHQ"), dtype=np.float32)
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y",
           "-f", "f32le", "-ar", str(target_sr), "-ac", "1", "-i", "-"]
    if wants_s16:
        cmd += ["-af", "aresample=out_sample_fmt=s16:dither_method=triangular_hp"]
    cmd += codec + [str(out_path)]
    _run(cmd, input_bytes=audio.tobytes())


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
