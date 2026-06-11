"""Progress reporting is opt-in and never changes pipeline output."""

import numpy as np

from podcare import progress
from podcare.dsp import process_chunked

SR = 48000


class _CountingReporter(progress.NullReporter):
    """Records sub-bar calls so we can assert the hot loop reports progress."""

    def __init__(self):
        self.subs = 0
        self.advances = 0
        self.ends = 0

    def begin_sub(self, total, unit, label=""):
        self.subs += 1
        self.last_total = total
        self.last_label = label

    def advance_sub(self, amount=1.0):
        self.advances += 1

    def end_sub(self):
        self.ends += 1


def test_default_reporter_is_null():
    assert isinstance(progress.active(), progress.NullReporter)


def test_use_installs_and_restores():
    rep = _CountingReporter()
    assert isinstance(progress.active(), progress.NullReporter)
    with progress.use(rep):
        assert progress.active() is rep
    assert isinstance(progress.active(), progress.NullReporter)


def test_process_chunked_reports_each_chunk():
    audio = np.linspace(-1, 1, SR * 5, dtype=np.float32)  # 5 s > chunk+overlap
    rep = _CountingReporter()
    with progress.use(rep):
        process_chunked(audio, SR, lambda c: c, chunk_s=1.0, overlap_s=0.25,
                        label="denoise · host")
    assert rep.subs == 1
    assert rep.ends == 1
    assert rep.advances == rep.last_total  # advanced exactly once per counted chunk
    assert rep.last_label == "denoise · host"


def test_process_chunked_output_identical_with_and_without_reporter():
    # The reporter must be a pure side-channel: identical samples either way.
    audio = np.sin(np.linspace(0, 40 * np.pi, SR * 5)).astype(np.float32)
    plain = process_chunked(audio, SR, lambda c: c * 0.5, chunk_s=1.0, overlap_s=0.25)
    with progress.use(_CountingReporter()):
        live = process_chunked(audio, SR, lambda c: c * 0.5, chunk_s=1.0, overlap_s=0.25)
    assert np.array_equal(plain, live)


def test_short_input_fast_path_does_not_report():
    # The single-chunk fast path has no per-chunk loop, so no sub-bar.
    rep = _CountingReporter()
    with progress.use(rep):
        process_chunked(np.zeros(SR, dtype=np.float32), SR, lambda c: c, chunk_s=60.0)
    assert rep.subs == 0


def test_resolve_mode():
    assert progress.resolve_mode("rich", verbose=False) == "rich"
    assert progress.resolve_mode("plain", verbose=False) == "plain"
    assert progress.resolve_mode("none", verbose=True) == "none"
    # verbose forces plain so debug logs don't fight a live display
    assert progress.resolve_mode("auto", verbose=True) == "plain"


def test_resolve_mode_auto_without_tty(monkeypatch):
    monkeypatch.setattr("sys.stderr.isatty", lambda: False, raising=False)
    assert progress.resolve_mode("auto", verbose=False) == "plain"


def test_make_reporter():
    assert isinstance(progress.make_reporter("plain"), progress.NullReporter)
    assert isinstance(progress.make_reporter("none"), progress.NullReporter)
    rich_rep = progress.make_reporter("rich")
    assert isinstance(rich_rep, progress.RichReporter)
    assert rich_rep.console is not None
