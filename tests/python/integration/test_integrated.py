"""Ordinance integration tests"""

import time
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

import aiohttp
import httpx
import pytest
import openai
import elm.web.html_pw
from elm.web.search.dux import DuxDistributedGlobalSearch
from elm.web.file_loader import AsyncWebFileLoader
from elm.web.document import HTMLDocument
from flaky import flaky

from compass.services.usage import TimeBoundedUsageTracker, UsageTracker
from compass.services.openai import OpenAIService, usage_from_response
from compass.services.threaded import TempFileCache
from compass.services.provider import RunningAsyncServices
from compass.utilities.enums import LLMUsageCategory
from compass.utilities.logs import LocationFileLog, LogListener


class MockResponse:
    def __init__(self, read_return):
        self.read_return = read_return
        self.content_type = "application/pdf"
        self.charset = "utf-8"

    async def read(self):
        return self.read_return


@pytest.fixture
def sample_file(test_data_files_dir):
    """Sample file with contents to use for integration test"""
    return test_data_files_dir / "Whatcom.txt"


@pytest.fixture
def mock_get_methods(sample_file):
    """Return patched get methods"""

    @asynccontextmanager
    async def patched_get(session, url, *args, **kwargs):  # noqa: RUF029
        if url == "Whatcom":
            with sample_file.open("rb") as fh:
                content = fh.read()

        yield MockResponse(content)

    async def patched_get_html(url, *args, **kwargs):  # noqa: RUF029
        with sample_file.open(encoding="utf-8") as fh:
            return fh.read()

    return patched_get, patched_get_html


@pytest.mark.asyncio
async def test_openai_query(sample_openai_response, monkeypatch):
    """Test querying OpenAI while tracking limits and usage"""

    start_time = None
    elapsed_times = []
    time_limit = 0.1
    sleep_mult = 1.2

    async def _test_response(*args, **kwargs):  # noqa: RUF029
        time_elapsed = time.monotonic() - start_time
        elapsed_times.append(time_elapsed)
        if time_elapsed < time_limit:
            response = httpx.Response(404)
            response.request = httpx.Request(method="test", url="test")
            raise openai.RateLimitError(
                "for testing", response=response, body=None
            )

        if kwargs.get("bad_request"):
            response = httpx.Response(404)
            response.request = httpx.Request(method="test", url="test")
            raise openai.NotFoundError(
                "for testing", response=response, body=None
            )
        return sample_openai_response()

    client = openai.AsyncOpenAI(api_key="dummy")
    monkeypatch.setattr(
        client.chat.completions,
        "create",
        _test_response,
        raising=True,
    )
    rate_tracker = TimeBoundedUsageTracker(
        max_seconds=time_limit * sleep_mult * 0.8
    )
    openai_service = OpenAIService(
        client, model_name="gpt-4", rate_limit=3, rate_tracker=rate_tracker
    )

    usage_tracker = UsageTracker("my_county", usage_from_response)
    async with RunningAsyncServices([openai_service]):
        start_time = time.monotonic()
        message = await openai_service.call(usage_tracker=usage_tracker)
        message2 = await openai_service.call()

        assert openai_service.rate_tracker.total == 13
        assert message == "test_response"
        assert message2 == "test_response"
        assert len(elapsed_times) == 3
        assert elapsed_times[0] < 1
        assert elapsed_times[1] >= time_limit + 1
        assert elapsed_times[2] >= time_limit * 3

        assert usage_tracker == {
            "gpt-4": {
                LLMUsageCategory.DEFAULT: {
                    "requests": 1,
                    "prompt_tokens": 100,
                    "response_tokens": 10,
                }
            }
        }

        time.sleep(time_limit * sleep_mult)
        assert openai_service.rate_tracker.total == 0

        start_time = time.monotonic() - time_limit - 1
        await openai_service.call()
        await openai_service.call()
        assert len(elapsed_times) == 5
        assert elapsed_times[-2] - time_limit - 1 < 1
        assert (
            elapsed_times[-1] - time_limit - 1 > time_limit * sleep_mult * 0.8
        )

        time.sleep(time_limit * sleep_mult)
        start_time = time.monotonic() - time_limit - 1
        assert openai_service.rate_tracker.total == 0

        with pytest.raises(openai.NotFoundError):
            message = await openai_service.call(
                usage_tracker=usage_tracker, bad_request=True
            )

        assert openai_service.rate_tracker.total <= 3
        assert usage_tracker == {
            "gpt-4": {
                LLMUsageCategory.DEFAULT: {
                    "requests": 1,
                    "prompt_tokens": 100,
                    "response_tokens": 10,
                }
            }
        }


@flaky(max_runs=3, min_passes=1)
@pytest.mark.asyncio
async def test_google_search_with_logging(tmp_path):
    """Test searching google for some locations with logging"""

    assert not list(tmp_path.glob("*"))

    logger = logging.getLogger("search_test")
    test_locations = ["El Paso County, Colorado", "Decatur County, Indiana"]
    num_requested_links = 5

    async def search_single(location):
        logger.info("This location is %r", location)
        search_engine = DuxDistributedGlobalSearch(backend="all")
        return await search_engine.results(
            f"Wind energy zoning ordinance {location}",
            num_results=num_requested_links,
        )

    async def search_location_with_logs(
        listener, log_dir, location, level="INFO"
    ):
        with LocationFileLog(
            listener, log_dir, location=location, level=level
        ):
            logger.info("A generic test log")
            return await search_single(location)

    log_dir = tmp_path / "logs"
    log_listener = LogListener(["search_test"], level="DEBUG")
    async with log_listener as ll:
        searchers = [
            asyncio.create_task(
                search_location_with_logs(ll, log_dir, loc, level="DEBUG"),
                name=loc,
            )
            for loc in test_locations
        ]
        output = await asyncio.gather(*searchers)

    expected_words = ["paso", "decatur"]
    assert len(output) == 2
    for query_results, expected_word in zip(
        output, expected_words, strict=False
    ):
        assert len(query_results) == 1
        assert len(query_results[0]) == num_requested_links
        assert any(expected_word in link for link in query_results[0])

    log_files = list(log_dir.glob("*.log"))
    json_log_files = list(log_dir.glob("*.json"))
    assert len(log_files) == 2
    assert len(json_log_files) == 2
    for fp in log_files:
        text = fp.read_text()
        assert "A generic test log" in text
        assert f"This location is {fp.stem!r}" in text


@pytest.mark.asyncio
async def test_async_file_loader_with_temp_cache(
    monkeypatch, mock_get_methods, sample_file
):
    """Test `AsyncWebFileLoader` with a `TempFileCache` service"""

    get_meth, get_html = mock_get_methods
    monkeypatch.setattr(aiohttp.ClientSession, "get", get_meth, raising=True)
    monkeypatch.setattr(elm.web.html_pw, "_load_html", get_html, raising=True)

    with sample_file.open(encoding="utf-8") as fh:
        content = fh.read()

    truth = HTMLDocument([content])

    async with RunningAsyncServices([TempFileCache()]):
        loader = AsyncWebFileLoader(file_cache_coroutine=TempFileCache.call)
        doc = await loader.fetch("Whatcom")
        assert doc.text == truth.text
        assert doc.attrs["source"] == "Whatcom"
        cached_fp = doc.attrs["cache_fn"]
        assert cached_fp.exists()
        assert cached_fp.read_text(encoding="utf-8") == doc.text


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
