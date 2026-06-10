"""Distortion repair: declick, declip and rumble high-pass via ffmpeg."""

from __future__ import annotations

import logging

from .. import audio_io
from ..config import Config
from ..session import Track

log = logging.getLogger(__name__)


def repair_track(track: Track, cfg: Config) -> Track:
    filters = []
    if cfg.declip:
        filters += ["adeclick", "adeclip"]
    if cfg.hpf_hz > 0:
        filters.append(f"highpass=f={cfg.hpf_hz}:poles=2")
    if not filters:
        return track
    audio = audio_io.filter_array(track.audio, cfg.sr, ",".join(filters))
    log.info("repair: %s — applied %s", track.name, ", ".join(filters))
    return Track(track.name, audio)
