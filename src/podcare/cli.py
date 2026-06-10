"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__, pipeline
from .audio_io import FfmpegError
from .config import Config

log = logging.getLogger("podcare")

_SUPPORTED_OUT = {".wav", ".mp3", ".flac", ".m4a", ".aac"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="podcare",
        description="Polish raw podcast mic recordings into one broadcast-ready file.",
    )
    p.add_argument("inputs", nargs="+", type=Path, metavar="AUDIO",
                   help="input WAV/MP3 files, one per mic/recorder")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="output file (.wav, .mp3, .flac, .m4a)")
    p.add_argument("--version", action="version", version=f"podcare {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    tune = p.add_argument_group("tuning")
    tune.add_argument("--strength", type=float, default=0.8, metavar="0..1",
                      help="universal processing intensity; every stage scales off this "
                           "(0 = no change / raw mixdown, 1 = max, default 0.8)")
    tune.add_argument("--filler-sensitivity", type=float, default=None, metavar="0..1",
                      help="override filler-cut aggressiveness (0 disables; default follows "
                           "--strength)")
    tune.add_argument("--whisper-model", default="large-v3",
                      help="faster-whisper model for filler detection (default: large-v3)")
    tune.add_argument("--denoise-backend", choices=["auto", "deepfilter", "spectral"],
                      default="auto", help="neural (deepfilter) or classical denoiser")
    tune.add_argument("--max-pause", type=float, default=None, metavar="SECONDS",
                      help="override: pauses longer than this get shortened (default follows "
                           "--strength)")
    tune.add_argument("--target-pause", type=float, default=None, metavar="SECONDS",
                      help="override: length pauses are shortened to (default follows --strength)")
    tune.add_argument("--lufs", type=float, default=-16.0,
                      help="output integrated loudness target (default -16)")
    tune.add_argument("--out-sr", type=int, default=44100, metavar="HZ",
                      help="output sample rate (default 44100; WAV/FLAC are 16-bit)")
    tune.add_argument("--bitrate", default="192k", help="mp3/aac bitrate (default 192k)")
    tune.add_argument("--keep-stems", type=Path, metavar="DIR",
                      help="write per-stage intermediate WAVs into DIR")

    toggles = p.add_argument_group("stage toggles")
    for flag, help_text in [
        ("declip", "distortion repair (declick/declip)"),
        ("align", "inter-track offset/polarity correction"),
        ("denoise", "noise reduction"),
        ("dereverb", "WPE dereverberation"),
        ("plosives", "plosive ducking"),
        ("deess", "de-essing"),
        ("gate", "crosstalk gate + level match"),
        ("fillers", "filler-word removal"),
        ("tighten", "pause tightening"),
        ("master", "compression + loudness normalization"),
    ]:
        toggles.add_argument(f"--no-{flag}", action="store_true", help=f"disable {help_text}")
    return p


def _validate(args: argparse.Namespace) -> None:
    for path in args.inputs:
        if not path.exists():
            raise SystemExit(f"error: input not found: {path}")
    if args.output.suffix.lower() not in _SUPPORTED_OUT:
        raise SystemExit(f"error: unsupported output format {args.output.suffix!r} "
                         f"(use one of {', '.join(sorted(_SUPPORTED_OUT))})")
    if not 0.0 <= args.strength <= 1.0:
        raise SystemExit("error: --strength must be in 0..1")
    if args.filler_sensitivity is not None and not 0.0 <= args.filler_sensitivity <= 1.0:
        raise SystemExit("error: --filler-sensitivity must be in 0..1")
    if (args.max_pause is not None and args.target_pause is not None
            and args.target_pause >= args.max_pause):
        raise SystemExit("error: --target-pause must be smaller than --max-pause")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )
    _validate(args)

    cfg = Config(
        keep_stems=args.keep_stems,
        strength=args.strength,
        declip=not args.no_declip,
        align=not args.no_align,
        denoise=not args.no_denoise,
        denoise_backend=args.denoise_backend,
        dereverb=not args.no_dereverb,
        plosives=not args.no_plosives,
        deess=not args.no_deess,
        gate=not args.no_gate,
        fillers=not args.no_fillers,
        filler_sensitivity=0.0 if args.no_fillers else args.filler_sensitivity,
        whisper_model=args.whisper_model,
        tighten=not args.no_tighten,
        max_pause_s=args.max_pause,
        target_pause_s=args.target_pause,
        master=not args.no_master,
        lufs=args.lufs,
        out_sr=args.out_sr,
        mp3_bitrate=args.bitrate,
    )
    try:
        in_dur, out_dur = pipeline.run(args.inputs, args.output, cfg)
    except (FfmpegError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 2
    print(f"✓ {args.output}  ({in_dur / 60:.1f} min in → {out_dur / 60:.1f} min out)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
