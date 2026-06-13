"""Command-line interface."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from . import __version__, pipeline, progress
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
                   help="input audio files (WAV/MP3/FLAC/M4A/… anything ffmpeg reads), "
                        "one per mic/recorder")
    p.add_argument("-o", "--output", type=Path, required=True,
                   help="output file (.wav, .mp3, .flac, .m4a, .aac)")
    p.add_argument("--version", action="version", version=f"podcare {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--progress", choices=("auto", "rich", "plain", "none"), default="auto",
                   help="progress display: auto (live bars on a terminal, plain logs when "
                        "piped or under -v), rich (force live bars), plain (log lines only), "
                        "none (warnings + final summary only)")
    p.add_argument("--nocut", action="store_true",
                   help="keep the original timeline: skip every length-changing stage "
                        "(alignment, filler-word cuts, pause tightening) so the output stays "
                        "sample-aligned with the input — for re-importing onto a video edit")

    tune = p.add_argument_group("tuning")
    tune.add_argument("--strength", type=float, default=0.8, metavar="0..1",
                      help="universal processing intensity; every stage scales off this "
                           "(0 = no change / raw mixdown, 1 = max, default 0.8)")
    tune.add_argument("--filler-sensitivity", type=float, default=None, metavar="0..1",
                      help="override filler-cut aggressiveness (0 disables; default follows "
                           "--strength)")
    tune.add_argument("--whisper-model", default="large-v3",
                      help="faster-whisper model for filler detection (default: large-v3)")
    tune.add_argument("--language", default=None, metavar="CODE",
                      help="force spoken language for filler detection (e.g. en, cs, de); "
                           "default auto-detects per track")
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
    tune.add_argument("--intro-sound", type=Path, default=None, metavar="AUDIO",
                      help="optional sound placed before the program (loudness-matched to "
                           "the output target, joined with a 100 ms crossfade); ignored "
                           "with --nocut")
    tune.add_argument("--outro-sound", type=Path, default=None, metavar="AUDIO",
                      help="optional sound placed after the program (loudness-matched, "
                           "100 ms crossfade); ignored with --nocut")

    toggles = p.add_argument_group("stage toggles")
    for flag, help_text in [
        ("dropouts", "dropout / short-gap restoration"),
        ("declip", "distortion repair (declick/declip + rumble high-pass)"),
        ("dehum", "mains-hum (50/60 Hz) harmonic removal"),
        ("align", "inter-track offset/polarity correction"),
        ("denoise", "noise reduction"),
        ("dereverb", "WPE dereverberation"),
        ("tonebalance", "tonal-balance / LTAS corrective EQ"),
        ("declick", "mouth-click / de-crackle removal"),
        ("plosives", "plosive ducking"),
        ("deess", "de-essing"),
        ("resonance", "dynamic resonance / harshness suppression"),
        ("gate", "crosstalk gate + level match"),
        ("breath", "breath ducking"),
        ("fillers", "filler-word removal"),
        ("tighten", "pause tightening"),
        ("leveler", "slow segment-loudness leveling"),
        ("exciter", "harmonic presence exciter"),
        ("master", "multiband compression + loudness normalization + limiting"),
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
    # Catch nonsense up front rather than after a multi-hour render fails deep in
    # soxr/ffmpeg or the late filler stage.
    if not 8000 <= args.out_sr <= 192000:
        raise SystemExit("error: --out-sr must be between 8000 and 192000 Hz")
    if not -40.0 <= args.lufs <= -5.0:
        raise SystemExit("error: --lufs must be between -40 and -5 (e.g. -16 podcast, -14 streaming)")
    if not re.fullmatch(r"\d+k?", args.bitrate):
        raise SystemExit("error: --bitrate must look like '192k' or '256000'")
    for flag, path in (("--intro-sound", args.intro_sound),
                       ("--outro-sound", args.outro_sound)):
        if path is not None and not path.exists():
            raise SystemExit(f"error: {flag} file not found: {path}")


def _setup_logging(mode: str, verbose: bool, reporter: progress.Reporter) -> None:
    """Route logging to match the progress mode.

    Under the live display, log lines go through a rich handler tied to the same
    console so they print cleanly above the bars; otherwise the plain formatter
    is kept unchanged. 'none' quiets everything below warnings.
    """
    level = logging.DEBUG if verbose else logging.INFO
    if mode == "rich" and reporter.console is not None:
        from rich.logging import RichHandler
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=reporter.console, show_path=False,
                                  show_time=False, markup=False, rich_tracebacks=True)],
        )
    elif mode == "none":
        logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")
    else:
        logging.basicConfig(level=level, format="%(levelname)-7s %(message)s")


def _effective_bookends(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    """Bookends after the --nocut policy. An intro shifts the entire timeline,
    defeating --nocut's sample-alignment purpose, so both are ignored there."""
    if not args.nocut:
        return args.intro_sound, args.outro_sound
    ignored = [name for name, given in (("--intro-sound", args.intro_sound),
                                        ("--outro-sound", args.outro_sound)) if given]
    if ignored:
        log.warning("%s ignored because --nocut keeps the original timeline",
                    ", ".join(ignored))
    return None, None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = progress.resolve_mode(args.progress, args.verbose)
    reporter = progress.make_reporter(mode)
    _setup_logging(mode, args.verbose, reporter)
    _validate(args)
    if args.no_fillers and args.filler_sensitivity is not None:
        log.warning("--filler-sensitivity is ignored because --no-fillers was given")
    if args.nocut:
        ignored = [name for name, given in (
            ("--filler-sensitivity", args.filler_sensitivity is not None),
            ("--max-pause", args.max_pause is not None),
            ("--target-pause", args.target_pause is not None),
        ) if given]
        if ignored:
            log.warning("%s ignored because --nocut keeps the original timeline",
                        ", ".join(ignored))

    intro_sound, outro_sound = _effective_bookends(args)

    cfg = Config(
        keep_stems=args.keep_stems,
        nocut=args.nocut,
        strength=args.strength,
        dropouts=not args.no_dropouts,
        declip=not args.no_declip,
        dehum=not args.no_dehum,
        align=not args.no_align,
        denoise=not args.no_denoise,
        dereverb=not args.no_dereverb,
        tonebalance=not args.no_tonebalance,
        declick=not args.no_declick,
        plosives=not args.no_plosives,
        deess=not args.no_deess,
        resonance=not args.no_resonance,
        gate=not args.no_gate,
        breath=not args.no_breath,
        fillers=not args.no_fillers,
        filler_sensitivity=0.0 if args.no_fillers else args.filler_sensitivity,
        whisper_model=args.whisper_model,
        language=args.language,
        leveler=not args.no_leveler,
        tighten=not args.no_tighten,
        max_pause_s=args.max_pause,
        target_pause_s=args.target_pause,
        master=not args.no_master,
        exciter=not args.no_exciter,
        intro_sound=intro_sound,
        outro_sound=outro_sound,
        lufs=args.lufs,
        out_sr=args.out_sr,
        lossy_bitrate=args.bitrate,
    )
    try:
        in_dur, out_dur = pipeline.run(args.inputs, args.output, cfg, reporter=reporter)
    except (FfmpegError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        # The live display is already torn down by progress.use()'s finally; just
        # report a clean stop instead of dumping a traceback. 130 = 128 + SIGINT.
        log.warning("interrupted — stopping")
        return 130
    summary = f"{{m}} {args.output}  ({in_dur / 60:.1f} min in {{a}} {out_dur / 60:.1f} min out)"
    try:
        print(summary.format(m="✓", a="→"))
    except UnicodeEncodeError:  # non-UTF-8 console (e.g. legacy Windows codepage)
        print(summary.format(m="OK", a="->"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
