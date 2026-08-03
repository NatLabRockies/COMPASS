"""Test COMPASS CPU Services"""

import logging
import sys
import time
import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from compass.services.cpu import (
    TIMEOUT_PARAMS,
    ProcessPoolService,
    _read_docling,
    _read_docling_without_timeout,
    _run_docling_in_subprocess,
)
from compass.services.provider import RunningAsyncServices
from compass.utilities.logs import LocationFileLog, LogListener


logger = logging.getLogger("compass")


def _log_from_process():
    """Call logger instance from a process"""
    msg = "A DEBUG LOG"
    logger.debug(msg)
    msg = "HELLO WORLD"
    logger.info(msg)
    return msg


def _write_to_process_streams():
    """Write to stdout/stderr from a worker process"""
    print("PROCESS STDOUT", flush=True)
    print("PROCESS STDERR", file=sys.stderr, flush=True)
    return "STREAMED"


def _return_from_subprocess(value):
    """Return a serializable value from a child process"""
    return value


def _block_subprocess(seconds):
    """Block a child process long enough for a deadline to expire"""
    time.sleep(seconds)


@pytest.mark.asyncio
async def test_logging_within_service(tmp_path):
    """Test that child-process logs are forwarded to the listener"""

    class ProcessLogging(ProcessPoolService):
        """Subclass for testing"""

        @property
        def can_process(self):
            return True

        async def process(self):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.pool, _log_from_process)

    log_listener = LogListener(["compass"], level="DEBUG")
    services = [ProcessLogging()]
    captured_records = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured_records.append(record)

    async with RunningAsyncServices(services), log_listener as ll:
        capture_handler = _CaptureHandler(level=logging.DEBUG)
        ll.addHandler(capture_handler)
        with LocationFileLog(ll, tmp_path, location="test_loc", level="DEBUG"):
            msg = await ProcessLogging.call()
            for _ in range(30):
                messages = {record.message for record in captured_records}
                if "[compass] HELLO WORLD" in messages:
                    break
                await asyncio.sleep(0.1)
        ll.removeHandler(capture_handler)

    assert msg == "HELLO WORLD"
    assert any(
        record.message == "[compass] HELLO WORLD"
        for record in captured_records
    ), {record.message for record in captured_records}
    assert not any(
        record.message == "[compass] A DEBUG LOG"
        for record in captured_records
    ), {record.message for record in captured_records}


@pytest.mark.asyncio
async def test_process_streams_are_forwarded_to_logs(capfd):
    """Test that worker stdout/stderr are logged instead of printed"""

    class ProcessStreamLogging(ProcessPoolService):
        """Subclass for testing"""

        @property
        def can_process(self):
            return True

        async def process(self):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.pool, _write_to_process_streams
            )

    log_listener = LogListener(["compass"], level="DEBUG")
    services = [ProcessStreamLogging()]
    captured_records = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured_records.append(record)

    async with RunningAsyncServices(services), log_listener as ll:
        capture_handler = _CaptureHandler(level=logging.INFO)
        ll.addHandler(capture_handler)
        msg = await ProcessStreamLogging.call()
        for _ in range(30):
            messages = {record.message for record in captured_records}
            if {
                "[compass.subprocess.stdout] PROCESS STDOUT",
                "[compass.subprocess.stderr] PROCESS STDERR",
            } <= messages:
                break
            await asyncio.sleep(0.1)
        ll.removeHandler(capture_handler)

    assert msg == "STREAMED"
    assert any(
        record.message == "[compass.subprocess.stdout] PROCESS STDOUT"
        for record in captured_records
    ), {record.message for record in captured_records}
    assert any(
        record.message == "[compass.subprocess.stderr] PROCESS STDERR"
        for record in captured_records
    ), {record.message for record in captured_records}


def test_read_docling_converts_missing_confidences_to_none(monkeypatch):
    """Test that missing Docling confidence values are normalized"""

    class FakeDocumentConverter:
        def __init__(self, format_options):
            self.format_options = format_options

        def convert(self, stream, headers=None):
            return SimpleNamespace(
                confidence=SimpleNamespace(
                    mean_score=np.nan,
                    low_score=pd.NA,
                    pages={0: SimpleNamespace(ocr_score=np.nan)},
                ),
                input=SimpleNamespace(
                    file=Path("sample.html"),
                    format=SimpleNamespace(value="html"),
                ),
                pages=["page 1"],
                document=SimpleNamespace(
                    # ruff:ignore[unused-lambda-argument]
                    export_to_markdown=lambda **kwargs: "markdown body"
                ),
                status=SimpleNamespace(value="success"),
            )

    monkeypatch.setattr(
        "compass.services.cpu.DocumentConverter", FakeDocumentConverter
    )

    doc = _read_docling_without_timeout(b"<html></html>", "sample.html")

    assert doc.pages == ["markdown body"]
    assert doc.attrs["mean_confidence"] is None
    assert doc.attrs["low_score_confidence"] is None


def test_read_docling_uses_process_deadline(monkeypatch):
    """Docling deadlines should run outside the process-pool worker"""
    captured = {}
    expected = object()
    configured_options = {"document_timeout": 120}

    def _run_in_subprocess(fn, *, args, kwargs, timeout):
        captured["fn"] = fn
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["timeout"] = timeout
        return expected

    monkeypatch.setattr(
        "compass.services.cpu._run_docling_in_subprocess",
        _run_in_subprocess,
    )

    result = _read_docling(
        b"%PDF",
        "sample.pdf",
        pdf_pipeline_options=configured_options,
    )

    assert result is expected
    assert captured["fn"] is _read_docling_without_timeout
    assert captured["args"] == (b"%PDF", "sample.pdf")
    assert captured["kwargs"]["pdf_pipeline_options"] == {
        "document_timeout": 120
    }
    assert captured["timeout"] == 132
    assert configured_options == {"document_timeout": 120}


def test_docling_subprocess_returns_result():
    """The Docling child process should return completed conversions"""
    result = _run_docling_in_subprocess(
        _return_from_subprocess,
        args=("converted",),
        kwargs={},
        timeout=60,
    )

    assert result == "converted"


def test_docling_subprocess_enforces_deadline(monkeypatch):
    """The Docling child process should be stopped at its deadline"""
    monkeypatch.setitem(TIMEOUT_PARAMS, "shutdown_timeout", 0.1)
    monkeypatch.setitem(TIMEOUT_PARAMS, "force_shutdown_timeout", 0.1)

    start_time = time.monotonic()
    with pytest.raises(TimeoutError, match="Docling conversion exceeded"):
        _run_docling_in_subprocess(
            _block_subprocess,
            args=(10,),
            kwargs={},
            timeout=0.1,
        )

    assert time.monotonic() - start_time < 1


def test_process_pool_release_resources_graceful_shutdown():  # ruff:ignore[complex-structure]
    """Graceful process-pool shutdown should not force worker exit"""

    class DummyService(ProcessPoolService):
        @property
        def can_process(self):
            return True

        async def process(self):
            return None

    class FakeThread:
        def __init__(self):
            self.join_calls = []
            self._alive = True

        def join(self, timeout=None):
            self.join_calls.append(timeout)
            self._alive = False

        def is_alive(self):
            return self._alive

    class FakeProcess:
        def __init__(self):
            self.terminate_calls = 0
            self.kill_calls = 0
            self.join_calls = []

        def is_alive(self):
            return False

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1

        def join(self, timeout=None):
            self.join_calls.append(timeout)

    class FakePool:
        def __init__(self):
            self.shutdown_calls = []
            self._executor_manager_thread = FakeThread()
            self._processes = {0: FakeProcess()}

        def shutdown(self, wait=True, cancel_futures=True):
            self.shutdown_calls.append((wait, cancel_futures))

    service = DummyService()
    pool = FakePool()
    service.pool = pool

    service.release_resources()

    assert service.pool is None
    assert pool.shutdown_calls == [(False, True)]
    assert pool._executor_manager_thread.join_calls == [
        TIMEOUT_PARAMS["shutdown_timeout"]
    ]
    process = pool._processes[0]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0


def test_process_pool_release_resources_forces_stuck_shutdown():  # ruff:ignore[complex-structure]
    """Stuck process-pool shutdown should terminate then kill workers"""

    class DummyService(ProcessPoolService):
        @property
        def can_process(self):
            return True

        async def process(self):
            return None

    class FakeThread:
        def __init__(self):
            self.join_calls = []
            self._alive = True

        def join(self, timeout=None):
            self.join_calls.append(timeout)
            if len(self.join_calls) > 1:
                self._alive = False

        def is_alive(self):
            return self._alive

    class FakeProcess:
        def __init__(self):
            self.terminate_calls = 0
            self.kill_calls = 0
            self.join_calls = []
            self._alive = True

        def is_alive(self):
            return self._alive

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1
            self._alive = False

        def join(self, timeout=None):
            self.join_calls.append(timeout)

    class FakePool:
        def __init__(self):
            self.shutdown_calls = []
            self._executor_manager_thread = FakeThread()
            self._processes = {0: FakeProcess()}

        def shutdown(self, wait=True, cancel_futures=True):
            self.shutdown_calls.append((wait, cancel_futures))

    service = DummyService()
    pool = FakePool()
    service.pool = pool

    service.release_resources()

    assert service.pool is None
    assert pool.shutdown_calls == [(False, True)]
    assert pool._executor_manager_thread.join_calls == [
        TIMEOUT_PARAMS["shutdown_timeout"],
        TIMEOUT_PARAMS["force_shutdown_timeout"],
    ]
    process = pool._processes[0]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.join_calls == [
        TIMEOUT_PARAMS["force_shutdown_timeout"],
        TIMEOUT_PARAMS["force_shutdown_timeout"],
    ]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
