"""Smoke tests for the MCP server layer over the pipeline.

Skipped entirely when the optional ``mcp`` dependency isn't installed
(`uv sync --extra mcp`).
"""

import asyncio
import json

import pytest
import soundfile as sf

pytest.importorskip("mcp")

from podcare import mcp_server as m  # noqa: E402

from conftest import SR, speech_like  # noqa: E402


def _call(name: str, args: dict) -> dict:
    """Invoke a tool and parse its structured JSON result."""
    blocks = asyncio.run(m.mcp.call_tool(name, args))
    return json.loads(blocks[0].text)


def _two_mics(tmp_path):
    a, b = tmp_path / "host.wav", tmp_path / "guest.wav"
    sf.write(a, speech_like(4, seed=1), SR, subtype="FLOAT")
    sf.write(b, speech_like(4, seed=2), SR, subtype="FLOAT")
    return a, b


def test_every_stage_is_a_tool():
    names = {t.name for t in asyncio.run(m.mcp.list_tools())}
    # One tool per pipeline stage toggle, plus mixdown/process orchestration.
    assert m._TOGGLEABLE - {"declip"} <= names  # 'declip' is exposed as the 'repair' tool
    assert "repair" in names
    assert {"mixdown", "master", "process"} <= names


def test_track_stage_writes_one_output_per_input(tmp_path):
    a, b = _two_mics(tmp_path)
    res = _call("deess", {"input_paths": [str(a), str(b)],
                          "output_dir": str(tmp_path / "out"), "strength": 0.7})
    assert res["stage"] == "deess"
    assert len(res["outputs"]) == 2
    for p in res["outputs"]:
        assert sf.info(p).samplerate == SR


def test_session_stage_mixes_to_one(tmp_path):
    a, b = _two_mics(tmp_path)
    res = _call("mixdown", {"input_paths": [str(a), str(b)],
                            "output_dir": str(tmp_path / "mix")})
    assert len(res["outputs"]) == 1


def test_master_encodes_delivery_file(tmp_path):
    a, _ = _two_mics(tmp_path)
    out = tmp_path / "final.mp3"
    res = _call("master", {"input_path": str(a), "output_path": str(out),
                           "strength": 0.5})
    assert res["output"] == str(out)
    assert out.exists() and out.stat().st_size > 0


def test_process_runs_full_pipeline(tmp_path):
    a, b = _two_mics(tmp_path)
    out = tmp_path / "episode.wav"
    # Disable the ML filler stage so the test never downloads a Whisper model.
    res = _call("process", {"input_paths": [str(a), str(b)], "output_path": str(out),
                            "strength": 0.5, "disable": ["fillers", "dereverb"]})
    assert out.exists() and out.stat().st_size > 0
    assert res["output_minutes"] > 0


def test_process_rejects_unknown_stage(tmp_path):
    a, _ = _two_mics(tmp_path)
    with pytest.raises(Exception, match="unknown stage"):
        _call("process", {"input_paths": [str(a)], "output_path": str(tmp_path / "o.wav"),
                          "disable": ["nope"]})
