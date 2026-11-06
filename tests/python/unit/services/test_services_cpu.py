"""Test COMPASS CPU Services"""

import sys
import asyncio
import logging
from pathlib import Path

import pytest

from compass.services.cpu import TIMEOUT_PARAMS, ProcessPoolService
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


# ruff:ignore[complex-structure]
def test_process_pool_release_resources_graceful_shutdown():
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


# ruff:ignore[complex-structure]
def test_process_pool_release_resources_forces_stuck_shutdown():
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
