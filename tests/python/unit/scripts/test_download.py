"""Tests for compass.scripts.download"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import compass.scripts.download as download_module
from compass.scripts.download import (
    download_jurisdiction_ordinances_from_website_compass_crawl as crawl,
)
from compass.utilities.enums import LLMTasks


@pytest.mark.asyncio
async def test_find_jurisdiction_website_returns_base_domain(monkeypatch):
    """Return the canonical root URL for the selected website"""

    async def fake_search_with_fallback(**_kwargs):  # ruff:ignore[unused-async]
        return [
            "https://prattvilleal.gov/venue/autauga-county-commission/",
            "https://prattvilleal.gov/government/mayor",
            "https://example.org/other-page",
        ]

    class DummyValidator:
        def __init__(self, **_kwargs):
            pass

        async def check(self, url, jurisdiction):
            return url == "https://prattvilleal.gov/"

    monkeypatch.setattr(
        download_module,
        "search_with_fallback",
        fake_search_with_fallback,
    )
    monkeypatch.setattr(
        download_module,
        "JurisdictionWebsiteValidator",
        DummyValidator,
    )

    jurisdiction = SimpleNamespace(
        full_name="Autauga County, Alabama",
        full_name_the_prefixed="Autauga County, Alabama",
    )
    model_config = SimpleNamespace(
        llm_service=object(),
        llm_call_kwargs={},
    )

    out = await download_module.find_jurisdiction_website(
        jurisdiction, {LLMTasks.DEFAULT: model_config}
    )

    assert out == "https://prattvilleal.gov/"


@pytest.mark.asyncio
async def test_elm_crawl_tracks_accepted_partial_results(monkeypatch):
    """ELM crawl should retain accepted docs and completed pages early"""

    class DummyLoader:
        def __init__(self, **_kwargs):
            pass

    class DummyCrawler:
        def __init__(self, validator, **_kwargs):
            self.validator = validator

        async def run_with_timeout(
            self, _website, crawl_timeout_s, on_result_hook=None
        ):
            assert crawl_timeout_s == 3600
            result = SimpleNamespace(url="https://example.com/page")
            if on_result_hook:
                await on_result_hook(result)
            assert await self.validator(SimpleNamespace(text="keep", attrs={}))
            assert not await self.validator(
                SimpleNamespace(text="discard", attrs={})
            )
            return SimpleNamespace(
                documents=[SimpleNamespace(text="keep", attrs={})],
                raw_results=[result],
            )

    monkeypatch.setattr(download_module, "AsyncWebFileLoader", DummyLoader)
    monkeypatch.setattr(download_module, "COMPASSWebFileLoader", DummyLoader)
    monkeypatch.setattr(download_module, "ELMWebsiteCrawler", DummyCrawler)

    heuristic = SimpleNamespace(check=lambda text: text == "keep")

    (
        docs,
        results,
    ) = await download_module.download_jurisdiction_ordinances_from_website(
        "https://example.com",
        heuristic,
        {"ordinance": 1},
        return_c4ai_results=True,
    )

    assert [doc.text for doc in docs] == ["keep"]
    assert [result.url for result in results] == ["https://example.com/page"]


@pytest.mark.asyncio
async def test_compass_crawl_tracks_accepted_partial_results(monkeypatch):
    """COMPASS crawl should retain accepted docs before an early exit"""
    crawler_kwargs = {}

    class DummyCrawler:
        def __init__(self, validator, **kwargs):
            self.validator = validator
            crawler_kwargs.update(kwargs)

        async def run(self, _website, crawl_timeout_s, **_kwargs):
            assert crawl_timeout_s == 3600
            assert await self.validator(SimpleNamespace(text="keep", attrs={}))
            assert not await self.validator(
                SimpleNamespace(text="discard", attrs={})
            )
            return [SimpleNamespace(text="keep", attrs={})]

    monkeypatch.setattr(download_module, "COMPASSCrawler", DummyCrawler)

    heuristic = SimpleNamespace(check=lambda text: text == "keep")
    url_ignore_substrings = ["blocked.example"]
    url_keep_substrings = ["trusted.example"]

    docs = await crawl(
        "https://example.com",
        heuristic,
        {"ordinance": 1},
        url_ignore_substrings=url_ignore_substrings,
        url_keep_substrings=url_keep_substrings,
    )

    assert [doc.text for doc in docs] == ["keep"]
    assert crawler_kwargs["url_ignore_substrings"] is url_ignore_substrings
    assert crawler_kwargs["url_keep_substrings"] is url_keep_substrings


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
