"""Test COMPASS CPU Services"""

import logging
import sys
import asyncio
from pathlib import Path

import pytest

from compass.services.cpu import ProcessPoolService
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
                if captured_records:
                    break
                await asyncio.sleep(0.1)
        ll.removeHandler(capture_handler)

    assert msg == "HELLO WORLD"
    assert any(record.message == "HELLO WORLD" for record in captured_records)
    assert not any(
        record.message == "A DEBUG LOG" for record in captured_records
    )


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
            if len(captured_records) >= 2:
                break
            await asyncio.sleep(0.1)
        ll.removeHandler(capture_handler)

    assert msg == "STREAMED"
    assert any(
        record.message == "PROCESS STDOUT" for record in captured_records
    )
    assert any(
        record.message == "PROCESS STDERR" for record in captured_records
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
