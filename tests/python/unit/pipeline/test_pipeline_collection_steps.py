"""Tests for collection-step loader configuration"""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, Mock

import pytest
from elm.web.document import HTMLDocument, MDDocument, PDFDocument

import compass.pipeline.collection.steps as steps_module
import compass.scripts.download as download_module
from compass.pipeline.collection.steps import (
    CompassWebsiteCrawlStep,
    ElmWebsiteCrawlStep,
    SearchEngineDocumentsStep,
    SearchResultsCrawlStep,
)
from compass.pipeline.collection.dedupe import DocumentDeDuplicator
from compass.pipeline.data_classes import WebSearchParams
from compass.utilities.enums import LLMTasks, COMPASSDocumentCollectionStep


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


def _add_search_docs(workflow, candidates, budget):
    """Populate collected documents with search provenance and ranks"""
    workflow.num_search_results_to_crawl = budget
    docs = [
        SimpleNamespace(
            attrs={
                "source": candidate.get("resolved_url", candidate["url"]),
                "collection_step_rank": candidate.get("overall_rank", index),
                "search_engines": candidate.get("search_engines", []),
                "doc_type": candidate.get("doc_type"),
            }
        )
        for index, candidate in enumerate(candidates, start=1)
    ]
    workflow.collection.de_duplicator.add_docs(
        docs, step_name=str(COMPASSDocumentCollectionStep.SEARCH_ENGINE)
    )


def _build_workflow(
    *,
    website="https://example.com",
    models=None,
    num_search_results_to_crawl=0,
):
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
            num_urls_to_check_per_jurisdiction=2,
            simple_se_result_sort=True,
            num_search_results_to_crawl=num_search_results_to_crawl,
            search_results_crawl_depth=3,
            url_ignore_substrings=("blocked.example",),
            url_keep_substrings=("trusted.example",),
            se_kwargs={},
            website_crawl_timeout_seconds=3600,
            priority_search=None,
        ),
        models=models,
    )
    return SimpleNamespace(
        perform_website_search=True,
        perform_se_search=True,
        perform_search_based_crawl=num_search_results_to_crawl > 0,
        jurisdiction_website=website,
        jurisdiction=SimpleNamespace(full_name="Example Township"),
        extractor=_DummyExtractor(),
        runtime=runtime,
        last_scrape_results=[],
        usage_tracker=None,
        num_search_results_to_crawl=num_search_results_to_crawl,
        collection=SimpleNamespace(de_duplicator=DocumentDeDuplicator()),
    )


class _SearchExtractor(_DummyExtractor):
    async def get_query_templates(self):
        return ["{jurisdiction} ordinance"]


def test_search_candidates_never_refresh_completed_search(monkeypatch):
    """An empty search checkpoint must not trigger another search"""
    workflow = _build_workflow(num_search_results_to_crawl=4)
    workflow.extractor = _SearchExtractor()
    workflow.collection = SimpleNamespace(de_duplicator=DocumentDeDuplicator())
    workflow.num_search_results_to_crawl = 2

    def unexpected_search(*_args, **_kwargs):
        pytest.fail("A completed search must not be refreshed")

    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinance_using_search_engine",
        unexpected_search,
    )

    assert steps_module._get_search_crawl_candidates(workflow) == []
    assert workflow.num_search_results_to_crawl == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "direct_budget,crawl_budget", [(2, 0), (2, 1), (2, 2), (2, 4), (0, 4)]
)
async def test_search_engine_collects_direct_documents_without_candidates(
    monkeypatch, caplog, direct_budget, crawl_budget
):
    """Search preserves crawl budgets without separate candidates"""
    workflow = _build_workflow(num_search_results_to_crawl=crawl_budget)
    workflow.runtime.search_params.num_urls_to_check_per_jurisdiction = (
        direct_budget
    )
    workflow.extractor = _SearchExtractor()
    captured = {}

    async def fake_search(*_args, **kwargs):  # ruff:ignore[unused-async]
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinance_using_search_engine",
        fake_search,
    )

    await SearchEngineDocumentsStep().collect(workflow)
    assert "search_crawl_candidates" not in captured
    assert "num_search_results_to_crawl" not in captured
    assert captured["num_urls"] == direct_budget
    assert workflow.num_search_results_to_crawl == crawl_budget
    assert "exceeds direct collection budget" not in caplog.text


@pytest.mark.parametrize(
    "doc_class,doc_type,eligible",
    [
        (HTMLDocument, None, True),
        (MDDocument, "HTML", True),
        (MDDocument, None, False),
        (MDDocument, "pdf", False),
        (HTMLDocument, "pdf", False),
        (PDFDocument, "html", False),
    ],
)
def test_search_crawl_uses_existing_document_type(
    doc_class, doc_type, eligible
):
    """Use document classes and native attrs without search type copies"""
    workflow = _build_workflow(num_search_results_to_crawl=1)
    workflow.num_search_results_to_crawl = 1
    doc = doc_class(pages=["Ordinance text"])
    doc.attrs.update(
        source="https://example.com/ordinance",
        collection_step_rank=1,
        doc_type=doc_type,
    )
    workflow.collection.de_duplicator.add_docs(
        [doc], step_name=COMPASSDocumentCollectionStep.SEARCH_ENGINE
    )

    candidates = steps_module._get_search_crawl_candidates(workflow)

    assert bool(candidates) is eligible
    if eligible:
        assert candidates == [
            {
                "url": doc.attrs["source"],
                "overall_rank": 1,
                "search_engines": [],
                "doc_type": "html",
            }
        ]


@pytest.mark.parametrize(
    "source,eligible",
    [
        ("/home/user/ordinance.html", False),
        ("documents/ordinance.html", False),
        (r"C:\documents\ordinance.html", False),
        (Path("documents/ordinance.html"), False),
        ("file:///home/user/ordinance.html", False),
        (None, False),
        ("http://example.com/ordinance", True),
        ("https://example.com/ordinance", True),
    ],
)
def test_search_crawl_skips_local_sources(source, eligible):
    """Only web sources are eligible for search-result crawling"""
    workflow = _build_workflow(num_search_results_to_crawl=1)
    doc = HTMLDocument(pages=["Ordinance text"])
    doc.attrs.update(source=source, collection_step_rank=1)
    workflow.collection.de_duplicator.add_docs(
        [doc], step_name=COMPASSDocumentCollectionStep.SEARCH_ENGINE
    )

    candidates = steps_module._get_search_crawl_candidates(workflow)

    assert bool(candidates) is eligible
    if eligible:
        assert candidates[0]["url"] == source


@pytest.mark.asyncio
async def test_search_results_crawl_uses_first_n_html_candidates_and_settings(
    monkeypatch,
):
    """Only HTML candidates in the ranked prefix are crawled and merged"""
    workflow = _build_workflow(num_search_results_to_crawl=4)
    workflow.runtime.search_params.search_results_crawl_depth = 5
    candidates = [
        {"url": "https://pdf.example/file.pdf", "doc_type": "pdf"},
        {
            "url": "https://html.example/start",
            "resolved_url": "https://resolved.example/start",
            "overall_rank": 2,
            "search_engines": ["engine"],
            "doc_type": "html",
        },
        {"url": "https://unknown.example", "doc_type": None},
        {"url": "https://html-two.example", "doc_type": "html"},
        {"url": "https://outside-prefix.example", "doc_type": "html"},
    ]
    _add_search_docs(workflow, candidates, budget=4)
    workflow.runtime.search_params.num_search_results_to_crawl = 10
    crawl_calls = []
    doc = SimpleNamespace(attrs={"checksum": "shared", "source": "doc.pdf"})

    async def fake_crawl(url, **kwargs):  # ruff:ignore[unused-async]
        crawl_calls.append((url, kwargs))
        return [doc]

    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinances_from_website_compass_crawl",
        fake_crawl,
    )

    docs = await SearchResultsCrawlStep().collect(workflow)

    assert docs == [doc]
    assert [call[0] for call in crawl_calls] == [
        "https://resolved.example/start",
        "https://html-two.example",
    ]
    forwarded = crawl_calls[0][1]
    assert forwarded["max_depth"] == 5
    assert forwarded["browser_semaphore"] is None
    assert forwarded["url_ignore_substrings"] == ("blocked.example",)
    assert forwarded["file_loader_kwargs"] is not (
        workflow.runtime.file_loader_kwargs
    )
    assert doc.attrs["search_crawl_seeds"] == [
        {
            "url": candidates[1]["resolved_url"],
            "overall_rank": 2,
            "search_engines": ["engine"],
            "doc_type": "html",
        },
        {
            **candidates[3],
            "overall_rank": 4,
            "search_engines": [],
        },
    ]


@pytest.mark.asyncio
async def test_search_results_crawl_continues_after_seed_failure(monkeypatch):
    """A failed candidate does not prevent later seeds from crawling"""
    workflow = _build_workflow(num_search_results_to_crawl=2)
    candidates = [
        {"url": "https://fail.example", "doc_type": "html"},
        {"url": "https://ok.example", "doc_type": "html"},
    ]
    _add_search_docs(workflow, candidates, budget=2)
    success = SimpleNamespace(attrs={"source": "result.pdf"})
    calls = []

    async def fake_crawl(url, **kwargs):  # ruff:ignore[unused-async]
        calls.append(url)
        if "fail" in url:
            raise RuntimeError("seed failed")
        return [success]

    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinances_from_website_compass_crawl",
        fake_crawl,
    )

    assert await SearchResultsCrawlStep().collect(workflow) == [success]
    assert calls == ["https://fail.example", "https://ok.example"]


@pytest.mark.asyncio
async def test_search_results_crawl_does_not_search_missing_saved_docs(
    monkeypatch,
):
    """Missing search documents never trigger a replacement search"""
    workflow = _build_workflow(num_search_results_to_crawl=2)
    workflow.extractor = _SearchExtractor()

    async def fake_search(*_args, **kwargs):  # ruff:ignore[unused-async]
        pytest.fail("Missing saved search documents must not refresh search")

    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinance_using_search_engine",
        fake_search,
    )
    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinances_from_website_compass_crawl",
        lambda *_args, **_kwargs: pytest.fail("PDF seed must not crawl"),
    )

    assert await SearchResultsCrawlStep().collect(workflow) == []


@pytest.mark.asyncio
async def test_search_results_crawl_skips_when_disabled_or_budget_zero(
    monkeypatch,
):
    """Disabled search and a zero budget perform no search or crawl"""
    workflow = _build_workflow()

    async def fail_if_called(*_args, **_kwargs):  # ruff:ignore[unused-async]
        raise AssertionError("search-results crawl should be disabled")

    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinance_using_search_engine",
        fail_if_called,
    )
    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinances_from_website_compass_crawl",
        fail_if_called,
    )

    assert await SearchResultsCrawlStep().collect(workflow) == []
    workflow = _build_workflow(num_search_results_to_crawl=2)
    workflow.perform_se_search = False
    workflow.perform_search_based_crawl = False
    assert await SearchResultsCrawlStep().collect(workflow) == []


@pytest.mark.asyncio
async def test_search_results_crawl_propagates_cancellation(monkeypatch):
    """Cancellation during a seed crawl must not be swallowed"""
    workflow = _build_workflow(num_search_results_to_crawl=1)
    candidates = [{"url": "https://seed.example", "doc_type": "html"}]
    _add_search_docs(workflow, candidates, budget=1)

    async def cancel_crawl(*_args, **_kwargs):  # ruff:ignore[unused-async]
        raise asyncio.CancelledError

    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinances_from_website_compass_crawl",
        cancel_crawl,
    )

    with pytest.raises(asyncio.CancelledError):
        await SearchResultsCrawlStep().collect(workflow)


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
    assert (
        captured["url_ignore_substrings"]
        is workflow.runtime.search_params.url_ignore_substrings
    )
    assert (
        captured["url_keep_substrings"]
        is workflow.runtime.search_params.url_keep_substrings
    )


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


@pytest.mark.asyncio
async def test_elm_website_crawl_recovers_partial_docs(monkeypatch):
    """ELM crawl interruption should return docs accepted before it ended"""
    workflow = _build_workflow()
    workflow.runtime.search_params.website_crawl_timeout_seconds = 0.001
    partial_doc = SimpleNamespace(attrs={})
    partial_result = SimpleNamespace(url="https://example.com/seen")

    async def fake_download(_url, **kwargs):
        await asyncio.sleep(0)
        assert kwargs["timeout_seconds"] == pytest.approx(0.001)
        return [partial_doc], [partial_result]

    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinances_from_website",
        fake_download,
    )

    docs = await ElmWebsiteCrawlStep().collect(workflow)

    assert docs == [partial_doc]
    assert partial_doc.attrs == {
        "compass_crawl": False,
        "check_correct_jurisdiction": True,
    }
    assert workflow.last_scrape_results == [partial_result]


@pytest.mark.asyncio
async def test_compass_website_crawl_recovers_partial_docs(monkeypatch):
    """COMPASS crawl interruption should return accepted docs"""
    workflow = _build_workflow()
    workflow.runtime.search_params.website_crawl_timeout_seconds = 0.001
    prior_result = SimpleNamespace(url="https://example.com/seen")
    workflow.last_scrape_results = [[prior_result]]
    partial_doc = SimpleNamespace(attrs={})
    captured = {}

    async def fake_download(_url, **kwargs):
        await asyncio.sleep(0)
        captured.update(kwargs)
        return [partial_doc]

    monkeypatch.setattr(
        steps_module,
        "download_jurisdiction_ordinances_from_website_compass_crawl",
        fake_download,
    )

    docs = await CompassWebsiteCrawlStep().collect(workflow)

    assert docs == [partial_doc]
    assert partial_doc.attrs == {
        "compass_crawl": True,
        "check_correct_jurisdiction": True,
    }
    assert captured["already_visited"] == {"https://example.com/seen"}
    assert workflow.last_scrape_results == [[prior_result]]


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
