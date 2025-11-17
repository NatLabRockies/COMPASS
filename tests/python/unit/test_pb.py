"""Tests for `compass.pb` progress bar helpers"""

from contextlib import ExitStack
from io import StringIO

import pytest
from rich.console import Console

import compass.pb
from compass.exceptions import COMPASSNotInitializedError, COMPASSValueError


@pytest.fixture
def console():
    """Console directed to an in-memory buffer for deterministic tests"""
    return Console(file=StringIO(), color_system=None, force_terminal=False)


@pytest.fixture
def progress_bars(console):
    """Fresh progress bar manager for each test"""
    manager = compass.pb._COMPASSProgressBars(console=console)
    try:
        yield manager
    finally:
        manager._main.stop()


class _DummyTask:
    """Lightweight object exposing attributes used by rich tasks"""

    def __init__(
        self,
        *,
        finished=False,
        finished_time=None,
        elapsed=None,
        completed=0,
        total=None,
        fields=None,
        description="",
    ):
        self.finished = finished
        self.finished_time = finished_time
        self.elapsed = elapsed
        self.completed = completed
        self.total = total
        self.fields = fields or {}
        self.description = description


def test_time_elapsed_column_handles_missing_elapsed():
    """Render placeholder when elapsed time is missing"""
    column = compass.pb._TimeElapsedColumn()
    task = _DummyTask()

    text = column.render(task)
    assert text.plain == "[-:--:--]"


def test_time_elapsed_column_uses_finished_time():
    """Render finished time when task completes"""
    column = compass.pb._TimeElapsedColumn()
    task = _DummyTask(finished=True, finished_time=125.7)

    text = column.render(task)
    assert text.plain == "[0:02:05]"


def test_m_of_n_complete_column_with_total_known():
    """Display counts when total is known"""
    column = compass.pb._MofNCompleteColumn(style="green")
    task = _DummyTask(completed=5, total=42)

    text = column.render(task)
    assert text.plain == "    5/42"


def test_m_of_n_complete_column_with_unknown_total():
    """Display counts when total is unknown"""

    column = compass.pb._MofNCompleteColumn()
    task = _DummyTask(completed=7)

    text = column.render(task)
    assert text.plain == "   7/?"


def test_total_cost_column_with_and_without_cost():
    """Show cost text only when present"""
    column = compass.pb._TotalCostColumn()

    zero_text = column.render(_DummyTask(fields={"total_cost": 0}))
    cost_text = column.render(_DummyTask(fields={"total_cost": 3.456}))

    assert not zero_text.plain
    assert cost_text.plain == "($3.46)"


def test_group_property_returns_group(progress_bars):
    """Expose default progress group"""
    assert progress_bars.group.renderables == [progress_bars._main]


def test_create_main_task_single_and_duplicate_error(console):
    """Guard against duplicate main tasks"""
    bars = compass.pb._COMPASSProgressBars(console=console)
    bars.create_main_task(1)

    task = bars._main.tasks[bars._main_task]
    assert "Searching 1 Jurisdiction" in task.description

    with pytest.raises(COMPASSValueError):
        bars.create_main_task(1)

    bars._main.stop()


def test_create_main_task_multiple(progress_bars):
    """Format description for multiple jurisdictions"""
    progress_bars.create_main_task(3)

    task = progress_bars._main.tasks[progress_bars._main_task]
    assert "Searching 3 Jurisdictions" in task.description


def test_progress_main_task_requires_initialization(progress_bars):
    """Require initialization before advancing main task"""

    with pytest.raises(COMPASSNotInitializedError):
        progress_bars.progress_main_task()


def test_progress_main_task_advances(progress_bars):
    """Advance main task increments progress"""
    progress_bars.create_main_task(2)
    progress_bars.progress_main_task()

    task = progress_bars._main.tasks[progress_bars._main_task]
    assert task.completed == 1


def test_update_total_cost_handles_add_and_replace(progress_bars):
    """Handle cost updates for add and replace paths"""
    progress_bars.create_main_task(1)

    progress_bars.update_total_cost(1.25)
    assert progress_bars._total_cost == pytest.approx(1.25)

    progress_bars.update_total_cost(0.5, replace=True)
    assert progress_bars._total_cost == pytest.approx(1.25)

    progress_bars.update_total_cost(5.0, replace=True)
    assert progress_bars._total_cost == pytest.approx(5.0)

    task = progress_bars._main.tasks[progress_bars._main_task]
    assert task.fields["total_cost"] == pytest.approx(5.0)


def test_update_total_cost_without_main(progress_bars):
    """Track cost updates without main task"""
    progress_bars.update_total_cost(2.0)
    assert progress_bars._total_cost == pytest.approx(2.0)


def test_jurisdiction_prog_bar_lifecycle(progress_bars):
    """Manage lifecycle for jurisdiction bars"""
    progress_bars.create_main_task(1)

    with progress_bars.jurisdiction_prog_bar("Denmark") as jd_pb:
        assert "Denmark" in progress_bars._jd_pbs
        progress_bars.update_jurisdiction_task("Denmark", description="Work")

        def enter_duplicate():
            with ExitStack() as stack:
                stack.enter_context(
                    progress_bars.jurisdiction_prog_bar("Denmark")
                )

        with pytest.raises(COMPASSValueError):
            enter_duplicate()

        progress_bars._jd_tasks["Denmark"] = jd_pb.add_task("extra")

    task = progress_bars._main.tasks[progress_bars._main_task]
    assert task.completed == 1
    assert "Denmark" not in progress_bars._jd_pbs


def test_jurisdiction_prog_bar_without_progress_main(progress_bars):
    """Skip main progress when requested"""
    progress_bars.create_main_task(1)

    with progress_bars.jurisdiction_prog_bar("Spain", progress_main=False):
        pass

    task = progress_bars._main.tasks[progress_bars._main_task]
    assert task.completed == 0


def test_jurisdiction_sub_prog_inserts_after_parent(progress_bars):
    """Insert sub progress after parent bar"""
    progress_bars.create_main_task(1)

    with ExitStack() as stack:
        stack.enter_context(progress_bars.jurisdiction_prog_bar("Sweden"))
        sub_pb = stack.enter_context(
            progress_bars.jurisdiction_sub_prog("Sweden")
        )
        assert sub_pb in progress_bars.group.renderables

    assert not any(
        "Sweden" in str(item) for item in progress_bars.group.renderables
    )


def test_jurisdiction_sub_prog_without_parent(progress_bars):
    """Create sub progress without parent bar"""

    assert "Norway" not in progress_bars._jd_pbs
    start_len = len(progress_bars.group.renderables)

    with progress_bars.jurisdiction_sub_prog("Norway") as sub_pb:
        assert sub_pb in progress_bars.group.renderables
        assert progress_bars.group.renderables.index(sub_pb) == start_len

    assert sub_pb not in progress_bars.group.renderables


def test_jurisdiction_sub_prog_bar_updates_fields(progress_bars):
    """Update jurisdiction sub progress fields"""
    progress_bars.create_main_task(1)

    with ExitStack() as stack:
        stack.enter_context(progress_bars.jurisdiction_prog_bar("Iceland"))
        sub_pb = stack.enter_context(
            progress_bars.jurisdiction_sub_prog_bar("Iceland")
        )
        task_id = sub_pb.add_task(
            "downloading",
            total=3,
            just_parsed="none",
        )
        sub_pb.advance(task_id)
        sub_pb.update(task_id, just_parsed="doc.pdf")

    assert "Iceland" not in progress_bars._jd_pbs


def test_jurisdiction_sub_prog_bar_without_parent(progress_bars):
    """Place sub progress bar at end when no parent exists"""
    start_len = len(progress_bars.group.renderables)

    with progress_bars.jurisdiction_sub_prog_bar("Portugal") as sub_pb:
        assert progress_bars._jd_pbs.get("Portugal") is None
        assert progress_bars.group.renderables.index(sub_pb) == start_len

        task_id = sub_pb.add_task("extracting", total=2)
        sub_pb.advance(task_id)

    assert sub_pb not in progress_bars.group.renderables


@pytest.mark.asyncio()
async def test_file_download_prog_bar_async_lifecycle(
    progress_bars, monkeypatch
):
    """Drive async lifecycle for download progress bar"""
    calls = []
    original_sleep = compass.pb.asyncio.sleep

    async def fake_sleep(duration):
        calls.append(duration)
        await original_sleep(0)

    monkeypatch.setattr(compass.pb.asyncio, "sleep", fake_sleep)

    progress_bars.create_main_task(1)

    async with progress_bars.file_download_prog_bar(
        "Finland", num_downloads=2
    ) as dl_pb:
        dl_task = progress_bars._dl_tasks["Finland"]
        dl_pb.advance(dl_task)
        progress_bars.update_download_task("Finland", completed=1)
        progress_bars._dl_tasks["Finland"] = dl_pb.add_task("extra")

    assert calls == [1]
    assert "Finland" not in progress_bars._dl_pbs


@pytest.mark.asyncio()
async def test_file_download_prog_bar_positions_after_jurisdiction(
    progress_bars, monkeypatch
):
    """Insert download progress immediately after jurisdiction bar"""
    _patch_sleep(monkeypatch)
    progress_bars.create_main_task(1)

    with progress_bars.jurisdiction_prog_bar("Poland"):
        async with progress_bars.file_download_prog_bar(
            "Poland", num_downloads=1
        ) as dl_pb:
            jd_index = progress_bars.group.renderables.index(
                progress_bars._jd_pbs["Poland"]
            )
            dl_index = progress_bars.group.renderables.index(dl_pb)
            assert dl_index == jd_index + 1

            progress_bars._dl_tasks["Poland"] = None

    progress_bars._dl_tasks.pop("Poland", None)


@pytest.mark.asyncio()
async def test_start_file_download_prog_bar_duplicate(
    progress_bars, monkeypatch
):
    """Prevent duplicate download progress bars"""
    original_sleep = compass.pb.asyncio.sleep

    async def fake_sleep(_):
        await original_sleep(0)

    monkeypatch.setattr(compass.pb.asyncio, "sleep", fake_sleep)

    dl_pb, task = progress_bars.start_file_download_prog_bar("France", 1)

    with pytest.raises(COMPASSValueError):
        progress_bars.start_file_download_prog_bar("France", 1)

    progress_bars._dl_tasks["France"] = dl_pb.add_task("extra")
    await progress_bars.tear_down_file_download_prog_bar(
        "France", 1, dl_pb, task
    )

    assert "France" not in progress_bars._dl_pbs


def _patch_sleep(monkeypatch):
    """Patch asyncio.sleep and capture durations"""
    calls = []
    original_sleep = compass.pb.asyncio.sleep

    async def fake_sleep(duration):
        calls.append(duration)
        await original_sleep(0)

    monkeypatch.setattr(compass.pb.asyncio, "sleep", fake_sleep)
    return lambda: calls


@pytest.mark.asyncio()
async def test_website_crawl_prog_bar(progress_bars, monkeypatch):
    """Drive async lifecycle for website crawl progress bar"""
    calls_getter = _patch_sleep(monkeypatch)

    progress_bars.create_main_task(1)

    async with progress_bars.website_crawl_prog_bar(
        "Germany", num_pages=3
    ) as wc_pb:
        progress_bars.update_website_crawl_doc_found("Germany")
        progress_bars.update_website_crawl_doc_found("Germany")
        progress_bars.update_website_crawl_task("Germany", completed=1)
        progress_bars._wc_tasks["Germany"] = wc_pb.add_task("extra")

    assert "Germany" not in progress_bars._wc_pbs
    assert calls_getter() == [1]


@pytest.mark.asyncio()
async def test_website_crawl_prog_bar_positions_after_jurisdiction(
    progress_bars, monkeypatch
):
    """Insert website crawl progress immediately after jurisdiction bar"""
    calls_getter = _patch_sleep(monkeypatch)
    progress_bars.create_main_task(1)

    with progress_bars.jurisdiction_prog_bar("Poland"):
        async with progress_bars.website_crawl_prog_bar(
            "Poland", num_pages=1
        ) as wc_pb:
            jd_index = progress_bars.group.renderables.index(
                progress_bars._jd_pbs["Poland"]
            )
            wc_index = progress_bars.group.renderables.index(wc_pb)
            assert wc_index == jd_index + 1

            progress_bars._wc_tasks["Poland"] = None

    assert calls_getter() == [1]
    progress_bars._wc_tasks.pop("Poland", None)


@pytest.mark.asyncio()
async def test_file_download_prog_bar_positions_without_jurisdiction(
    progress_bars, monkeypatch
):
    """Place download progress at end when no jurisdiction bar exists"""
    calls_getter = _patch_sleep(monkeypatch)

    dl_pb, task = progress_bars.start_file_download_prog_bar("Greece", 2)

    assert progress_bars.group.renderables[-1] is dl_pb

    await progress_bars.tear_down_file_download_prog_bar(
        "Greece", 2, dl_pb, task
    )

    assert calls_getter() == [1]


@pytest.mark.asyncio()
async def test_website_crawl_prog_bar_duplicate(progress_bars, monkeypatch):
    """Prevent duplicate website crawl progress bars"""
    _patch_sleep(monkeypatch)

    async with progress_bars.website_crawl_prog_bar(
        "Italy", num_pages=1
    ) as wc_pb:
        with pytest.raises(COMPASSValueError):
            async with progress_bars.website_crawl_prog_bar(
                "Italy", num_pages=1
            ):
                pass

        progress_bars._wc_tasks["Italy"] = wc_pb.add_task("extra")

    assert "Italy" not in progress_bars._wc_pbs


@pytest.mark.asyncio()
async def test_compass_website_crawl_prog_bar(progress_bars, monkeypatch):
    """Drive async lifecycle for compass crawl progress bar"""
    calls_getter = _patch_sleep(monkeypatch)

    progress_bars.create_main_task(1)

    async with progress_bars.compass_website_crawl_prog_bar(
        "Hungary",
        num_pages=2,
    ) as cwc_pb:
        progress_bars.update_compass_website_crawl_doc_found("Hungary")
        progress_bars.update_compass_website_crawl_doc_found("Hungary")
        progress_bars.update_compass_website_crawl_task("Hungary", completed=1)
        progress_bars._cwc_tasks["Hungary"] = cwc_pb.add_task("extra")

    assert "Hungary" not in progress_bars._cwc_pbs
    assert calls_getter() == [1]


@pytest.mark.asyncio()
async def test_compass_website_crawl_prog_bar_positions_after_jurisdiction(
    progress_bars, monkeypatch
):
    """Insert compass crawl progress immediately after jurisdiction bar"""
    calls_getter = _patch_sleep(monkeypatch)
    progress_bars.create_main_task(1)

    with progress_bars.jurisdiction_prog_bar("Poland"):
        async with progress_bars.compass_website_crawl_prog_bar(
            "Poland", num_pages=1
        ) as cwc_pb:
            jd_index = progress_bars.group.renderables.index(
                progress_bars._jd_pbs["Poland"]
            )
            cwc_index = progress_bars.group.renderables.index(cwc_pb)
            assert cwc_index == jd_index + 1

            progress_bars._cwc_tasks["Poland"] = None

    assert calls_getter() == [1]
    progress_bars._cwc_tasks.pop("Poland", None)


@pytest.mark.asyncio()
async def test_compass_website_crawl_prog_bar_duplicate(
    progress_bars, monkeypatch
):
    """Prevent duplicate compass crawl progress bars"""
    _patch_sleep(monkeypatch)

    async with progress_bars.compass_website_crawl_prog_bar(
        "Iceland",
        num_pages=1,
    ) as cwc_pb:
        with pytest.raises(COMPASSValueError):
            async with progress_bars.compass_website_crawl_prog_bar(
                "Iceland", num_pages=1
            ):
                pass

        progress_bars._cwc_tasks["Iceland"] = cwc_pb.add_task("extra")

    assert "Iceland" not in progress_bars._cwc_pbs


def test_singleton_instance_accessible(console):
    """Expose singleton progress bar instance"""
    assert isinstance(compass.pb.COMPASS_PB, compass.pb._COMPASSProgressBars)
