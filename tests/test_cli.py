"""CLI argument validation fails fast on nonsense, before any processing."""

import pytest

from podcare.cli import _validate, build_parser


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
