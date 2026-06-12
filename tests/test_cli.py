"""CLI argument validation fails fast on nonsense, before any processing."""

import logging
from pathlib import Path

import pytest

from podcare import cli
from podcare.cli import _validate, build_parser
from podcare.pipeline import STAGES


@pytest.fixture
def infile(tmp_path):
    f = tmp_path / "in.wav"
    f.write_bytes(b"\x00")
    return f


def _validate_args(infile, tmp_path, *extra):
    args = build_parser().parse_args(
        [str(infile), "-o", str(tmp_path / "out.wav"), *extra])
    _validate(args)


def test_accepts_sane_defaults(infile, tmp_path):
    _validate_args(infile, tmp_path)  # no raise


@pytest.mark.parametrize("flag,value", [
    ("--out-sr", "0"),
    ("--out-sr", "-5"),
    ("--out-sr", "4000"),
    ("--lufs", "0"),
    ("--lufs", "-100"),
    ("--bitrate", "loud"),
    ("--strength", "2"),
])
def test_rejects_bad_values(infile, tmp_path, flag, value):
    with pytest.raises(SystemExit):
        _validate_args(infile, tmp_path, flag, value)


def test_rejects_unsupported_output_format(infile, tmp_path):
    args = build_parser().parse_args([str(infile), "-o", str(tmp_path / "out.ogg")])
    with pytest.raises(SystemExit):
        _validate(args)


def test_progress_flag_defaults_to_auto(infile, tmp_path):
    args = build_parser().parse_args([str(infile), "-o", str(tmp_path / "out.wav")])
    assert args.progress == "auto"


def test_progress_flag_accepts_choices(infile, tmp_path):
    for mode in ("auto", "rich", "plain", "none"):
        args = build_parser().parse_args(
            [str(infile), "-o", str(tmp_path / "out.wav"), "--progress", mode])
        assert args.progress == mode


def test_progress_flag_rejects_unknown(infile, tmp_path):
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [str(infile), "-o", str(tmp_path / "out.wav"), "--progress", "fancy"])


# --------------------------------------------------------------------------- #
# New stages + intro/outro bookends
# --------------------------------------------------------------------------- #

def _args(argv):
    return cli.build_parser().parse_args(argv)


def test_new_stages_registered_in_order():
    names = [s.name for s in STAGES]
    assert names.index("dropouts") < names.index("repair")
    assert names.index("deess") < names.index("resonance") < names.index("gate")


def test_bookend_flags_parse(tmp_path):
    intro = tmp_path / "i.wav"
    intro.write_bytes(b"")
    args = _args(["in.wav", "-o", "out.mp3", "--intro-sound", str(intro)])
    assert args.intro_sound == intro and args.outro_sound is None


def test_nocut_ignores_bookends(caplog):
    args = _args(["in.wav", "-o", "out.mp3", "--nocut",
                  "--intro-sound", "i.wav", "--outro-sound", "o.wav"])
    with caplog.at_level(logging.WARNING):
        intro, outro = cli._effective_bookends(args)
    assert intro is None and outro is None
    assert "--intro-sound" in caplog.text and "--nocut" in caplog.text


def test_bookends_kept_without_nocut():
    args = _args(["in.wav", "-o", "out.mp3", "--intro-sound", "i.wav"])
    intro, outro = cli._effective_bookends(args)
    assert intro == Path("i.wav") and outro is None


def test_validate_rejects_missing_bookend(tmp_path):
    src = tmp_path / "in.wav"
    src.write_bytes(b"")
    args = _args([str(src), "-o", str(tmp_path / "out.mp3"),
                  "--intro-sound", str(tmp_path / "missing.wav")])
    with pytest.raises(SystemExit, match="intro-sound"):
        cli._validate(args)
