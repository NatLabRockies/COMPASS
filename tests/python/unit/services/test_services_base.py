"""Test Ordinances Base Services"""

import asyncio
from pathlib import Path

import pytest

from compass.services.base import LLMService, Service
from compass.services.usage import TimeBoundedUsageTracker


def test_base_llm_limited_service(patched_clock):
    """Test base implementation of `LLMService` class"""

    class TestService(LLMService):
        """Simple service implementation for tests"""

        async def process(self, *args, **kwargs):
            """Always return 0"""
            return 0

    timed_tracker = TimeBoundedUsageTracker(max_seconds=0.1)
    service = TestService(
        model_name="test", rate_limit=100, timed_tracker=timed_tracker
    )

    assert service.can_process
    service.timed_tracker.add(50)
    assert service.can_process
    patched_clock.advance(0.01)
    service.timed_tracker.add(75)
    assert not service.can_process
    patched_clock.advance(0.099)
    assert service.can_process


class _Service(Service):
    """Service returning a fixed answer, or raising one error"""

    def __init__(self, error=None):
        self.error = error

    @property
    def can_process(self):
        return True

    async def process(self, *_, **__):
        if self.error is not None:
            raise self.error
        return "answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, ValueError("failed")])
async def test_result_is_dropped_when_the_caller_stopped_waiting(error):
    """A caller that timed out cancels its future; that is not an error"""
    fut = asyncio.get_running_loop().create_future()
    fut.cancel()
    await _Service(error).process_using_futures(fut)
    assert fut.cancelled()


@pytest.mark.asyncio
async def test_result_reaches_a_waiting_caller():
    """The usual case is unchanged"""
    fut = asyncio.get_running_loop().create_future()
    await _Service().process_using_futures(fut)
    assert fut.result() == "answer"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
