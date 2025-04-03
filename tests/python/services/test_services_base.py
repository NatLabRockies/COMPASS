"""Test Ordinances Base Services"""

import time
from pathlib import Path

import pytest

from compass.services.base import LLMService
from compass.services.usage import TimeBoundedUsageTracker


def test_base_llm_limited_service():
    """Test base implementation of `LLMService` class"""

    class TestService(LLMService):
        """Simple service implementation for tests"""

        async def process(self, *args, **kwargs):
            """Always return 0"""
            return 0

    rate_tracker = TimeBoundedUsageTracker(max_seconds=0.1)
    service = TestService(
        model_name="test", rate_limit=100, rate_tracker=rate_tracker
    )

    assert service.can_process
    service.rate_tracker.add(50)
    assert service.can_process
    service.rate_tracker.add(75)
    assert not service.can_process
    time.sleep(0.1)
    assert service.can_process


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
