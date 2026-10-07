"""Tests for collection-step loader configuration"""

from pathlib import Path
from types import SimpleNamespace
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, Mock

import pytest

import compass.pipeline.collection.steps as steps_module
import compass.scripts.download as download_module
from compass.pipeline.collection.steps import (
    CompassWebsiteCrawlStep,
    ElmWebsiteCrawlStep,
)
from compass.utilities.enums import LLMTasks
from compass.pipeline.data_classes import WebSearchParams


class _DummyExtractor:
    """Provide async crawl inputs for collection-step tests"""

    async def get_heuristic(self):
        """Return a placeholder heuristic"""
        return object()

    async def get_website_keywords(self):
        """Return placeholder keyword points"""
        return {"ordinance": 1}


class _DummyValidator:
    """Capture validator kwargs for assertions"""

    last_init_kwargs = None

    def __init__(self, **kwargs):
        self.__class__.last_init_kwargs = kwargs

    async def check(self, website, jurisdiction):
        """Return success without changing the workflow"""
        return True


def _build_workflow(*, website="https://example.com", models=None):
    """Build a minimal workflow for collection-step tests"""
    model_config = SimpleNamespace(llm_service=object(), llm_call_kwargs={})
    if models is None:
        models = {
            LLMTasks.DEFAULT: model_config,
            LLMTasks.DOCUMENT_JURISDICTION_VALIDATION: model_config,
        }
    runtime = SimpleNamespace(
        file_loader_kwargs={
            "pdf_ocr_read_coroutine": object(),
            "loader_mode": "ocr",
        },
        file_loader_kwargs_no_ocr={"loader_mode": "no-ocr"},
        crawl_semaphore=AsyncExitStack(),
        browser_semaphore=None,
        search_engine_semaphore=None,
        search_params=SimpleNamespace(
            url_ignore_substrings=(),
            se_kwargs={},
            website_crawl_timeout_seconds=3600,
        ),
        models=models,
    )
    return SimpleNamespace(
        perform_website_search=True,
        jurisdiction_website=website,
        jurisdiction=SimpleNamespace(full_name="Example Township"),
        extractor=_DummyExtractor(),
        runtime=runtime,
        last_scrape_results=[],
        usage_tracker=None,
    )


@pytest.mark.asyncio
async def test_elm_website_crawl_uses_ocr_loader(monkeypatch):
    """ELM website collection should keep OCR-enabled loader kwargs"""
    workflow = _build_workflow()
    captured = {}

    async def fake_redirect(url, **kwargs):  # ruff:ignore[unused-async]
        return url

    async def fake_download(url, **kwargs):  # ruff:ignore[unused-async]
        captured.update(kwargs)
        return [], []

    monkeypatch.setattr(steps_module, "get_redirected_url", fake_redirect)
    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinances_from_website",
        fake_download,
    )

    docs = await ElmWebsiteCrawlStep().collect(workflow)

    assert docs == []
    assert (
        captured["file_loader_kwargs"] is workflow.runtime.file_loader_kwargs
    )


@pytest.mark.asyncio
async def test_compass_website_crawl_uses_ocr_loader(monkeypatch):
    """COMPASS website collection should keep OCR-enabled loader kwargs"""
    workflow = _build_workflow()
    workflow.last_scrape_results = [
        [SimpleNamespace(url="https://seen.example")]
    ]
    captured = {}

    async def fake_download(url, **kwargs):  # ruff:ignore[unused-async]
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinances_from_website_compass_crawl",
        fake_download,
    )

    docs = await CompassWebsiteCrawlStep().collect(workflow)

    assert docs == []
    assert (
        captured["file_loader_kwargs"] is workflow.runtime.file_loader_kwargs
    )
    assert captured["already_visited"] == {"https://seen.example"}


@pytest.mark.asyncio
async def test_elm_website_crawl_uses_provided_website_without_discovery(
    monkeypatch,
):
    """Provided jurisdiction websites should bypass discovery"""
    workflow = _build_workflow(website="https://user-provided.example/path")
    captured = {}

    async def fake_get_base_website(url):  # ruff:ignore[unused-async]
        return "https://user-provided.example"

    async def fail_if_discovery_called(workflow):  # ruff:ignore[unused-async]
        raise AssertionError("website discovery should not be attempted")

    async def fake_download(url, **kwargs):  # ruff:ignore[unused-async]
        captured["url"] = url
        return [], []

    monkeypatch.setattr(
        steps_module, "_get_base_website", fake_get_base_website
    )
    monkeypatch.setattr(
        steps_module,
        "_find_jurisdiction_website_for_workflow",
        fail_if_discovery_called,
    )
    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinances_from_website",
        fake_download,
    )

    docs = await ElmWebsiteCrawlStep().collect(workflow)

    assert docs == []
    assert captured["url"] == "https://user-provided.example"
    assert workflow.jurisdiction_website == "https://user-provided.example"


@pytest.mark.asyncio
async def test_elm_website_crawl_attempts_discovery_when_models_present(
    monkeypatch,
):
    """Missing jurisdiction websites should be discovered when models exist"""
    workflow = _build_workflow(website=None)
    calls = {"discover": 0}
    captured = {}

    async def fake_discover(workflow):  # ruff:ignore[unused-async]
        calls["discover"] += 1
        return "https://discovered.example/home"

    async def fake_get_base_website(url):  # ruff:ignore[unused-async]
        return "https://discovered.example"

    async def fake_download(url, **kwargs):  # ruff:ignore[unused-async]
        captured["url"] = url
        return [], []

    monkeypatch.setattr(
        steps_module,
        "_find_jurisdiction_website_for_workflow",
        fake_discover,
    )
    monkeypatch.setattr(
        steps_module, "_get_base_website", fake_get_base_website
    )
    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinances_from_website",
        fake_download,
    )

    docs = await ElmWebsiteCrawlStep().collect(workflow)

    assert docs == []
    assert calls["discover"] == 1
    assert captured["url"] == "https://discovered.example"
    assert workflow.jurisdiction_website == "https://discovered.example"


@pytest.mark.asyncio
async def test_elm_website_crawl_skips_discovery_without_models(monkeypatch):
    """Missing websites should short-circuit when no models are available"""
    workflow = _build_workflow(website=None, models={})

    async def fail_if_discovery_called(workflow):  # ruff:ignore[unused-async]
        raise AssertionError("website discovery should not be attempted")

    async def fail_if_download_called(url, **kwargs):  # ruff:ignore[unused-async]
        raise AssertionError("website crawl should not be attempted")

    monkeypatch.setattr(
        steps_module,
        "_find_jurisdiction_website_for_workflow",
        fail_if_discovery_called,
    )
    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinances_from_website",
        fail_if_download_called,
    )

    docs = await ElmWebsiteCrawlStep().collect(workflow)

    assert docs == []
    assert workflow.jurisdiction_website is None


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])


@pytest.mark.parametrize("priority_search", [{}, {"max_pages": 15}])
async def test_priority_search_replaces_standard_search(
    monkeypatch, priority_search
):
    """Explicit options route collection to the priority queue."""
    workflow = _build_workflow()
    workflow.perform_se_search = True
    workflow.runtime.search_params.priority_search = priority_search
    expected = [object()]
    prioritized = AsyncMock(return_value=expected)
    old_search = AsyncMock(side_effect=AssertionError("Unexpected old search"))
    monkeypatch.setattr(
        steps_module, "download_prioritized_ordinances", prioritized
    )
    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinance_using_search_engine",
        old_search,
    )
    assert (
        await steps_module.SearchEngineDocumentsStep().collect(workflow)
        == expected
    )
    prioritized.assert_awaited_once_with(workflow)
    old_search.assert_not_awaited()


@pytest.mark.parametrize("settings", [{}, {"priority_search": None}])
async def test_standard_search_does_not_expand_links(monkeypatch, settings):
    """Default retrieval returns search documents without starting a crawl."""
    workflow = _build_workflow()
    workflow.perform_se_search = True
    workflow.runtime.search_params = WebSearchParams(**settings)
    workflow.extractor.get_query_templates = AsyncMock(
        return_value=["{jurisdiction} ordinance"]
    )
    doc = SimpleNamespace(attrs={"source": "https://example.com/ordinance"})
    search = AsyncMock(
        return_value={"results": [{"url": doc.attrs["source"]}]}
    )
    download = AsyncMock(return_value=[doc])
    prioritized = AsyncMock()
    monkeypatch.setattr(
        download_module.COMPASS_PB, "update_jurisdiction_task", Mock()
    )
    monkeypatch.setattr(
        download_module, "search_single_jurisdiction",
        search,
    )
    monkeypatch.setattr(download_module, "_docs_from_urls", download)
    monkeypatch.setattr(
        steps_module, "download_prioritized_ordinances", prioritized
    )
    workflow.extractor.get_website_keywords = AsyncMock(
        side_effect=AssertionError("Unexpected link expansion")
    )
    workflow.extractor.get_heuristic = AsyncMock(
        side_effect=AssertionError("Unexpected link expansion")
    )

    docs = await steps_module.SearchEngineDocumentsStep().collect(workflow)

    assert docs == [doc]
    assert doc.attrs["compass_crawl"] is False
    assert doc.attrs["check_correct_jurisdiction"] is True
    search.assert_awaited_once()
    assert search.call_args.args[:2] == (
        ["{jurisdiction} ordinance"], workflow.jurisdiction,
    )
    assert search.call_args.args[2] == 5
    assert search.call_args.kwargs["simple"] is True
    download.assert_awaited_once()
    assert download.call_args.args[0] == [doc.attrs["source"]]
    prioritized.assert_not_awaited()
