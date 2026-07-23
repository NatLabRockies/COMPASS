"""Tests for collection-step loader configuration"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import compass.pipeline.collection.steps as steps_module
from compass.pipeline.collection.steps import (
    CompassWebsiteCrawlStep,
    ElmWebsiteCrawlStep,
)
from compass.utilities.enums import LLMTasks


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
        crawl_semaphore=None,
        browser_semaphore=None,
        search_engine_semaphore=None,
        search_params=SimpleNamespace(
            url_ignore_substrings=(),
            se_kwargs={},
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
