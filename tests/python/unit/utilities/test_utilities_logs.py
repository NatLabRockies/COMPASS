"""Test COMPASS Ordinance logging logic."""

import logging
import asyncio
from pathlib import Path

import pytest

from compass.services.provider import RunningAsyncServices
from compass.utilities.logs import LocationFileLog, LogListener


@pytest.mark.asyncio
async def test_logs_sent_to_separate_files(tmp_path, service_base_class):
    """Test that logs are correctly sent to individual files."""

    logger = logging.getLogger("ords")
    test_locations = ["a", "bc", "def", "ghij"]
    __, TestService = service_base_class

    assert not logger.handlers

    class AlwaysThreeService(TestService):
        """Test service that returns ``3``."""

        NUMBER = 3
        LEN_SLEEP = 0.1

    async def process_single(val):
        """Call `AlwaysThreeService`."""
        logger.info(f"This location is {val!r}")
        return await AlwaysThreeService.call(len(val))

    async def process_location_with_logs(listener, log_dir, location):
        """Process location and record logs for tests."""
        with LocationFileLog(listener, log_dir, location=location):
            logger.info("A generic test log")
            return await process_single(location)

    log_dir = tmp_path / "ord_logs"
    services = [AlwaysThreeService()]
    loggers = ["ords"]

    async with RunningAsyncServices(services), LogListener(loggers) as ll:
        producers = [
            asyncio.create_task(
                process_location_with_logs(ll, log_dir, loc), name=loc
            )
            for loc in test_locations
        ]
        await asyncio.gather(*producers)

    assert not logger.handlers

    log_files = list(log_dir.glob("*.log"))
    json_log_files = list(log_dir.glob("*.json"))
    assert len(log_files) == len(test_locations)
    assert len(json_log_files) == len(test_locations)
    for loc in test_locations:
        expected_log_file = log_dir / f"{loc}.log"
        assert expected_log_file.exists()
        log_text = expected_log_file.read_text(encoding="utf-8")
        assert "A generic test log" in log_text
        assert f"This location is {loc!r}" in log_text


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
