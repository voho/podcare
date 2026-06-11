"""Live progress reporting for the processing pipeline.

A `Reporter` is installed for the duration of a run via `use()`; pipeline code
and the hot DSP loops report through the module-level `active()` reporter, which
is a no-op `NullReporter` by default. Only the CLI, when attached to a terminal,
installs the `RichReporter` that draws the live bars — so library use, tests and
piped/CI runs are unaffected and produce byte-identical output.

The display has two parts: a persistent *overall* bar counting completed stages
(with ETA), and a transient *sub* bar for the active long stage (chunk count for
denoise/dereverb, transcribed seconds for fillers, or an indeterminate spinner
for opaque stages like decode and master/encode).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Protocol


class Reporter(Protocol):
    """What pipeline/DSP code calls. `NullReporter` is the silent default."""

    console: object  # rich Console when live, else None (used to route logging)

    def start(self, total_stages: int) -> None: ...
    def begin_stage(self, idx: int, total: int, name: str, detail: str) -> None: ...
    def end_stage(self, name: str, seconds: float) -> None: ...
    def begin_sub(self, total: float, unit: str, label: str = "") -> None: ...
    def advance_sub(self, amount: float = 1.0) -> None: ...
    def end_sub(self) -> None: ...
    def open(self) -> None: ...
    def close(self) -> None: ...


class NullReporter:
    """Default no-op reporter: progress is silent unless a real one is installed."""

    console = None

    def start(self, total_stages: int) -> None:
        pass

    def begin_stage(self, idx: int, total: int, name: str, detail: str) -> None:
        pass

    def end_stage(self, name: str, seconds: float) -> None:
        pass

    def begin_sub(self, total: float, unit: str, label: str = "") -> None:
        pass

    def advance_sub(self, amount: float = 1.0) -> None:
        pass

    def end_sub(self) -> None:
        pass

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass


_active: Reporter = NullReporter()


def active() -> Reporter:
    """The reporter currently installed (a `NullReporter` outside a run)."""
    return _active


@contextmanager
def use(reporter: Reporter):
    """Install `reporter` as the active one and manage its live display."""
    global _active
    prev = _active
    _active = reporter
    reporter.open()
    try:
        yield reporter
    finally:
        reporter.close()
        _active = prev


class RichReporter:
    """Draws a live overall stage bar plus a sub-bar for the active long stage.

    rich is imported lazily here so the module stays import-light for tests and
    library users who never install a live reporter.
    """

    def __init__(self) -> None:
        from rich.console import Console, Group
        from rich.live import Live
        from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                                    SpinnerColumn, TaskProgressColumn,
                                    TextColumn, TimeElapsedColumn,
                                    TimeRemainingColumn)

        # Progress draws on stderr so a piped stdout (the final '✓' line) stays clean.
        self.console = Console(stderr=True)
        self._overall = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("[dim]{task.completed:.0f}/{task.total:.0f} stages[/]"),
            TimeElapsedColumn(),
            TextColumn("[dim]ETA[/]"),
            TimeRemainingColumn(),
            console=self.console,
        )
        self._sub = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        # A single Live drives the refresh of both Progress renderables; their own
        # auto-refresh threads are never started (we don't call Progress.start()).
        self._live = Live(Group(self._overall, self._sub), console=self.console,
                          refresh_per_second=12, transient=False)
        self._overall_task = None
        self._sub_task = None
        self._sub_total: float | None = None
        self._total_stages = 0

    # -- lifecycle ---------------------------------------------------------- #
    def open(self) -> None:
        self._live.start()

    def close(self) -> None:
        self._drop_sub()
        if self._overall_task is not None:
            self._overall.update(self._overall_task, completed=self._total_stages)
        self._live.stop()

    # -- overall bar -------------------------------------------------------- #
    def start(self, total_stages: int) -> None:
        self._total_stages = total_stages
        self._overall_task = self._overall.add_task("starting", total=total_stages)

    def begin_stage(self, idx: int, total: int, name: str, detail: str) -> None:
        if self._overall_task is None:  # defensive: start() should have run first
            self.start(total)
        self._overall.update(self._overall_task, description=name, completed=idx - 1)
        # Indeterminate spinner for the active stage until a sub-bar refines it.
        self._new_sub(name, total=None)

    def end_stage(self, name: str, seconds: float) -> None:
        self._drop_sub()
        if self._overall_task is not None:
            self._overall.advance(self._overall_task, 1)

    # -- sub bar ------------------------------------------------------------ #
    def begin_sub(self, total: float, unit: str, label: str = "") -> None:
        desc = label or "working"
        total_f = float(total) if total and total > 0 else None
        self._new_sub(f"  {desc}", total=total_f)

    def advance_sub(self, amount: float = 1.0) -> None:
        if self._sub_task is not None:
            self._sub.advance(self._sub_task, amount)

    def end_sub(self) -> None:
        if self._sub_task is not None and self._sub_total:
            self._sub.update(self._sub_task, completed=self._sub_total)

    # -- helpers ------------------------------------------------------------ #
    def _new_sub(self, label: str, total: float | None) -> None:
        self._drop_sub()
        self._sub_total = total
        self._sub_task = self._sub.add_task(label, total=total)

    def _drop_sub(self) -> None:
        if self._sub_task is not None:
            self._sub.remove_task(self._sub_task)
            self._sub_task = None
            self._sub_total = None


def resolve_mode(mode: str, verbose: bool) -> str:
    """Resolve a `--progress` choice to a concrete mode: 'rich' | 'plain' | 'none'.

    'auto' draws the live display only on an interactive terminal and not under
    --verbose (whose debug stream would fight the live bars).
    """
    if mode in ("rich", "plain", "none"):
        return mode
    if verbose:
        return "plain"
    return "rich" if sys.stderr.isatty() else "plain"


def make_reporter(mode: str) -> Reporter:
    """Build the reporter for a resolved mode; degrades to silent if rich is absent."""
    if mode == "rich":
        try:
            return RichReporter()
        except Exception:  # noqa: BLE001 — never let display setup abort a render
            return NullReporter()
    return NullReporter()
