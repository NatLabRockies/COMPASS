"""COMPASS CLI progress bars"""

import logging
from datetime import timedelta
from contextlib import contextmanager

from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    SpinnerColumn,
    ProgressColumn,
)
from rich.console import Group
from rich.text import Text

from compass.exceptions import COMPASSValueError, COMPASSNotInitializedError


logger = logging.getLogger(__name__)


class _TimeElapsedColumn(ProgressColumn):
    """Renders time elapsed"""

    def render(self, task):  # noqa: PLR6301
        """Show time elapsed"""
        elapsed = task.finished_time if task.finished else task.elapsed
        if elapsed is None:
            return Text("[-:--:--]", style="white")
        delta = timedelta(seconds=max(0, int(elapsed)))
        return Text(f"[{delta}]", style="white")


class _MofNCompleteColumn(ProgressColumn):
    """Renders completed count/total, e.g. '   10/1000'"""

    def render(self, task):  # noqa: PLR6301
        """Show completed/total"""
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))
        return Text(f"   {completed:{total_width}d}/{total}", style="white")


class _COMPASSProgressBars:
    """COMPASS progress bar configurations"""

    def __init__(self, console=None):
        """

        Parameters
        ----------
        console : rich.Console, optional
            Optional Console instance. Default is an internal Console
            instance writing to stdout. By default, ``None``.
        """
        self.console = console
        self._main = Progress(
            SpinnerColumn(style="dim"),
            TextColumn("{task.description}"),
            _TimeElapsedColumn(),
            BarColumn(
                complete_style="progress.elapsed",
                finished_style="progress.spinner",
            ),
            _MofNCompleteColumn(),
            console=console,
        )
        self._group = Group(self._main)
        self._main_task = None
        self._jd_pbs = {}
        self._jd_tasks = {}

    @property
    def group(self):
        """rich.console.Group: Group of renderable progress bars."""
        return self._group

    def create_main_task(self, num_jurisdictions):
        """Set up main task to track number of jurisdictions processed

        Parameters
        ----------
        num_jurisdictions : int
            Number of jurisdictions that are being processed.

        Raises
        ------
        COMPASSValueError
            If the main task has already been set up.
        """
        if self._main_task is not None:
            msg = "Main task has already been set!"
            raise COMPASSValueError(msg)

        logger.trace(
            "Starting main progress bar with %d jurisdiction(s)",
            num_jurisdictions,
        )
        if num_jurisdictions == 1:
            text = "[bold cyan]Searching 1 Jurisdiction"
        else:
            text = f"[bold cyan]Searching {num_jurisdictions:,} Jurisdictions"

        self._main_task = self._main.add_task(
            f"{text:<40}", total=num_jurisdictions
        )

    def progress_main_task(self):
        """Advance the main task one step

        In other words, mark one jurisdiction as complete.

        Raises
        ------
        COMPASSNotInitializedError
            If the main task has not been set up (i.e.
            `create_main_task` has not been called).
        """
        if self._main_task is None:
            msg = (
                "Main task has not been created! Call the "
                "`create_main_task` method first"
            )
            raise COMPASSNotInitializedError(msg)

        self._main.update(self._main_task, advance=1)

    @contextmanager
    def jurisdiction_prog_bar(self, location, progress_main=True):
        """Set a progress bar for the processing of one jurisdiction

        Parameters
        ----------
        location : str
            Name of jurisdiction being processed.
        progress_main : bool, default=True
            Option to progress the main task when exiting this context
            manager.

        Yields
        ------
        rich.progress.Progress
            `rich` progress bar initialized for this jurisdiction.

        Raises
        ------
        COMPASSValueError
            If a progress bar already exists for this location.
        """
        if location in self._jd_pbs:
            msg = f"Progress bar already exists for {location}"
            raise COMPASSValueError(msg)

        pb = Progress(
            TextColumn("    "),
            SpinnerColumn(style="dim"),
            TextColumn(f"[progress.percentage]{location:<30}"),
            _TimeElapsedColumn(),
            TextColumn("[bar.back]{task.description}"),
            console=self.console,
        )
        self._jd_pbs[location] = pb
        self._group.renderables.append(pb)
        self._jd_tasks[location] = pb.add_task("")

        try:
            yield pb
        finally:
            self._remove_jurisdiction_prog_bar(location)
            if progress_main:
                self.progress_main_task()

    def _remove_jurisdiction_prog_bar(self, location):
        """Remove jurisdiction prog bar and associated task (if any)"""
        pb = self._jd_pbs.pop(location)
        if task_id := self._jd_tasks.get(location):
            pb.remove_task(task_id)

        self._group.renderables.remove(pb)

    def update_jurisdiction_task(self, location, *args, **kwargs):
        """Update the task corresponding to the jurisdiction

        Parameters
        ----------
        location : str
            Name of jurisdiction being processed.
        *args, **kwargs
            Parameters to pass to the `task.update` function in the
            `rich` python library.
        """
        task_id = self._jd_tasks[location]
        self._jd_pbs[location].update(task_id, *args, **kwargs)

    @contextmanager
    def jurisdiction_sub_prog(self, location):
        """Start a sub-progress update area for location

        This type of sub-progress does not have a bar, so it's useful
        for tasks with an unknown length/duration.

        Parameters
        ----------
        location : str
            Name of jurisdiction being processed.

        Yields
        ------
        rich.progress.Progress
            `rich` sub-progress initialized for this jurisdiction.
        """
        pb = Progress(
            TextColumn("        "),
            SpinnerColumn(style="dim"),
            TextColumn("{task.description}"),
            _TimeElapsedColumn(),
            console=self.console,
        )

        jd_pb = self._jd_pbs.get(location)
        if jd_pb:
            insert_index = self._group.renderables.index(jd_pb) + 1
        else:
            insert_index = len(self._group.renderables)

        self._group.renderables.insert(insert_index, pb)

        try:
            yield pb
        finally:
            self._group.renderables.remove(pb)

    @contextmanager
    def jurisdiction_sub_prog_bar(self, location):
        """Start a sub-progress bar for location

        Parameters
        ----------
        location : str
            Name of jurisdiction being processed.

        Yields
        ------
        rich.progress.Progress
            `rich` sub-progress bar initialized for this jurisdiction.
        """
        pb = Progress(
            TextColumn("        "),
            SpinnerColumn(style="dim"),
            TextColumn("{task.description}"),
            _TimeElapsedColumn(),
            BarColumn(
                bar_width=30,
                complete_style="progress.elapsed",
                finished_style="progress.spinner",
            ),
            _MofNCompleteColumn(),
            TextColumn("[bar.back]{task.fields[just_parsed]}"),
            console=self.console,
        )

        jd_pb = self._jd_pbs.get(location)
        if jd_pb:
            insert_index = self._group.renderables.index(jd_pb) + 1
        else:
            insert_index = len(self._group.renderables)

        self._group.renderables.insert(insert_index, pb)

        try:
            yield pb
        finally:
            self._group.renderables.remove(pb)


COMPASS_PB = _COMPASSProgressBars()
"""Compass progress bars instance (singleton)"""
