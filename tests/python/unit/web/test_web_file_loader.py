"""COMPASS web file loader tests"""

import json
import asyncio
import runpy
from importlib.util import find_spec
from pathlib import Path
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from elm.web.document import HTMLDocument, MDDocument
from elm.web.file_loader import AsyncFetchWithRetry, AsyncWebFileLoader

import compass.web.file_loader as file_loader_module
from compass.utilities.event_log import EventLog, use_event_log
from compass.web.file_loader import (
    AsyncDoclingWebFileLoader,
    AsyncLocalDoclingFileLoader,
    COMPASSWebFileLoader,
)


requires_docling = pytest.mark.skipif(
    find_spec("docling_rs") is None or find_spec("docling_core") is None,
    reason="Requires the docling extra",
)


def _doc(source, doc_type="pdf", empty=False, conversion_status="success"):
    return SimpleNamespace(
        attrs={
            "source": source,
            "doc_type": doc_type,
            "conversion_status": conversion_status,
        },
        empty=empty,
    )


class _Fetcher:
    async def fetch(self, url):
        return b"content", "application/pdf", None, {}


class _FailedFetcher:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    async def fetch_all(self, *sources):
        self.calls.append(sources)
        return [self.docs[source] for source in sources]


def _parsed_doc(doc_type="pdf", status="success", text="Ordinance text"):
    """Build parsed output using the real document abstraction"""
    return MDDocument(
        pages=[text] if text else [],
        attrs={"doc_type": doc_type, "conversion_status": status},
    )


@pytest.fixture
def document_cache(tmp_path):
    """Cache loader output on disk without starting threaded services"""
    cached = []

    async def cache_document(doc, content):
        await asyncio.sleep(0)
        target = tmp_path / f"document_{len(cached)}.{doc.FILE_EXTENSION}"
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
        cached.append((doc, content, target))
        return target

    return cache_document, cached


@pytest.mark.asyncio
@pytest.mark.parametrize("doc_type", ["pdf", "html", "docx"])
@requires_docling
async def test_docling_web_fetch_preserves_inputs_and_caches_content(
    monkeypatch, document_cache, doc_type
):
    """Forward web inputs and cache binary or parsed HTML content"""
    cache_document, cached = document_cache
    url = "https://example.com/download?id=123"
    headers = {
        "Content-Disposition": f'attachment; filename="code.{doc_type}"'
    }
    loader = AsyncDoclingWebFileLoader(
        file_cache_coroutine=cache_document,
        to_md_kwargs={"image_placeholder": ""},
        pdf_pipeline_options={"document_timeout": 30},
    )
    loader.content_fetcher.fetch = AsyncMock(
        return_value=(b"raw document", None, None, headers)
    )
    parsed = _parsed_doc(doc_type)
    converter = AsyncMock(return_value=parsed)
    monkeypatch.setattr(
        "compass.services.docling.read_docling_web_file", converter
    )

    doc = await loader.fetch(url)

    converter.assert_awaited_once_with(
        b"raw document",
        url=f"code.{doc_type}",
        source_uri=url,
        headers=headers,
        pdf_pipeline_options={"document_timeout": 30},
        image_placeholder="",
    )
    assert doc is parsed
    assert doc.attrs["source"] == url
    assert len(cached) == 1
    assert doc.attrs["cache_fn"] == cached[0][2]
    if doc_type == "html":
        assert cached[0][2].read_text(encoding="utf-8") == doc.text
    else:
        assert doc_type == doc.FILE_EXTENSION
        assert doc.WRITE_KWARGS == {"mode": "wb"}
        assert cached[0][2].read_bytes() == b"raw document"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["download", "timeout", "empty", "error"])
@requires_docling
async def test_docling_web_fetch_failures_are_not_cached(
    monkeypatch, document_cache, failure
):
    """Return empty documents without caching failed web conversions"""
    cache_document, cached = document_cache
    loader = AsyncDoclingWebFileLoader(file_cache_coroutine=cache_document)
    loader.content_fetcher.fetch = AsyncMock(
        return_value=None
        if failure == "download"
        else (b"invalid", None, None, {})
    )
    converter = AsyncMock(return_value=_parsed_doc(text=""))
    if failure == "timeout":
        converter.side_effect = TimeoutError("deadline")
    elif failure == "error":
        converter.side_effect = ValueError("invalid document")
    monkeypatch.setattr(
        "compass.services.docling.read_docling_web_file", converter
    )

    assert (await loader.fetch("https://example.com/code.pdf")).empty
    assert await loader.fetch_all("https://example.com/code.pdf") == []
    assert cached == []
    if failure == "download":
        converter.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("doc_type", ["pdf", "html"])
@requires_docling
async def test_docling_local_fetch_preserves_metadata_and_options(
    monkeypatch, tmp_path, document_cache, doc_type
):
    """Preserve local metadata and select the correct cached content"""
    source = tmp_path / f"source.{doc_type}"
    source.write_bytes(b"local document")
    cache_document, cached = document_cache
    parsed = _parsed_doc(doc_type)
    converter = AsyncMock(return_value=(parsed, source.read_bytes()))
    monkeypatch.setattr(
        "compass.services.docling.read_docling_local_file", converter
    )
    loader = AsyncLocalDoclingFileLoader(
        file_cache_coroutine=cache_document,
        doc_attrs={
            "source": "https://example.com/code",
            "jurisdiction": "Test",
        },
        to_md_kwargs={"page_break_placeholder": "<!-- page break -->"},
        pdf_pipeline_options={"do_table_structure": False},
    )

    doc = await loader.fetch(source)

    converter.assert_awaited_once_with(
        source,
        pdf_pipeline_options={"do_table_structure": False},
        page_break_placeholder="<!-- page break -->",
    )
    assert doc.attrs["source_fp"] == source
    assert doc.attrs["source"] == "https://example.com/code"
    assert doc.attrs["jurisdiction"] == "Test"
    assert doc.attrs["cache_fn"] == cached[0][2]
    if doc_type == "html":
        assert cached[0][2].read_text(encoding="utf-8") == doc.text
    else:
        assert doc.WRITE_KWARGS == {"mode": "wb"}
        assert doc.FILE_EXTENSION == "pdf"
        assert cached[0][2].read_bytes() == b"local document"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "empty", "missing"])
@requires_docling
async def test_docling_local_fetch_failures_are_not_cached(
    monkeypatch, tmp_path, document_cache, failure
):
    """Avoid caching local files that cannot be converted"""
    cache_document, cached = document_cache
    converter = AsyncMock(return_value=(_parsed_doc(text=""), b"invalid"))
    if failure == "timeout":
        converter.side_effect = TimeoutError("deadline")
    elif failure == "missing":
        converter.side_effect = FileNotFoundError("missing.pdf")
    monkeypatch.setattr(
        "compass.services.docling.read_docling_local_file", converter
    )
    loader = AsyncLocalDoclingFileLoader(file_cache_coroutine=cache_document)

    doc = await loader.fetch(tmp_path / "missing.pdf")

    assert doc.empty
    assert cached == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback_kind", ["cached", "uncached", "empty"])
@pytest.mark.parametrize("status", ["partial_success", "failure", None])
@requires_docling
async def test_docling_partial_conversion_selects_valid_fallback(
    monkeypatch, fallback_kind, status
):
    """Replace partial output only with nonempty cached ELM output"""
    url = "https://example.com/code.pdf"
    partial = _parsed_doc(status=status)
    fallback = _parsed_doc(
        text="" if fallback_kind == "empty" else "ELM ordinance text"
    )
    if fallback_kind == "cached":
        fallback.attrs["cache_fn"] = "cached.pdf"
    loader = AsyncDoclingWebFileLoader(re_fetch_failed_with_elm=True)
    loader.content_fetcher.fetch = AsyncMock(
        return_value=(b"document", None, None, {})
    )
    monkeypatch.setattr(
        "compass.services.docling.read_docling_web_file",
        AsyncMock(return_value=partial),
    )
    loader.failed_fetcher.fetch_all = AsyncMock(return_value=[fallback])

    docs = await loader.fetch_all(url)

    assert docs == [fallback if fallback_kind == "cached" else partial]
    assert docs[0].attrs["source"] == url
    loader.failed_fetcher.fetch_all.assert_awaited_once_with(url)


@pytest.mark.asyncio
@requires_docling
async def test_docling_html_and_pdf_results_skip_unnecessary_fallback(
    monkeypatch, document_cache
):
    """Retain initial and browser HTML results without retrying good PDFs"""
    cache_document, cached = document_cache
    urls = ["https://example.com/code.pdf", "https://example.com/code.html"]
    loader = AsyncDoclingWebFileLoader(
        re_fetch_failed_with_elm=True, file_cache_coroutine=cache_document
    )
    loader.content_fetcher.fetch = AsyncMock(
        return_value=(b"document", None, None, {})
    )
    monkeypatch.setattr(
        "compass.services.docling.read_docling_web_file",
        AsyncMock(side_effect=[_parsed_doc("pdf"), _parsed_doc("html")]),
    )
    rendered = HTMLDocument(pages=["Browser ordinance"])
    loader.html_loader._html_loader.fetch = AsyncMock(return_value=rendered)
    loader.html_loader.file_cache_coroutine = cache_document
    loader.failed_fetcher.fetch_all = AsyncMock()

    docs = await loader.fetch_all(*urls)

    assert len(docs) == 3
    assert [doc.attrs["source"] for doc in docs] == [*urls, urls[1]]
    assert docs[-1] is rendered
    assert len(cached) == 3
    loader.html_loader._html_loader.fetch.assert_awaited_once_with(urls[1])
    loader.failed_fetcher.fetch_all.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("browser_failure", ["empty", "error"])
@requires_docling
async def test_docling_browser_failure_does_not_block_partial_fallback(
    monkeypatch, browser_failure
):
    """Retry partial HTML even when browser rendering fails"""
    url = "https://example.com/code.html"
    loader = AsyncDoclingWebFileLoader(re_fetch_failed_with_elm=True)
    loader.content_fetcher.fetch = AsyncMock(
        return_value=(b"HTML", None, None, {})
    )
    monkeypatch.setattr(
        "compass.services.docling.read_docling_web_file",
        AsyncMock(return_value=_parsed_doc("html", status="partial_success")),
    )
    browser_fetch = AsyncMock(return_value=HTMLDocument(pages=[]))
    if browser_failure == "error":
        browser_fetch.side_effect = RuntimeError("Browser failed")
    loader.html_loader._html_loader.fetch = browser_fetch
    fallback = _parsed_doc("html", text="ELM ordinance fallback")
    fallback.attrs["cache_fn"] = "cached.html"
    loader.failed_fetcher.fetch_all = AsyncMock(return_value=[fallback])

    assert await loader.fetch_all(url) == [fallback]
    loader.failed_fetcher.fetch_all.assert_awaited_once_with(url)
    assert fallback.attrs["source"] == url


@pytest.mark.asyncio
@requires_docling
async def test_docling_empty_source_list_does_not_retry():
    """Return no documents or fallback requests for an empty input list"""
    loader = AsyncDoclingWebFileLoader(re_fetch_failed_with_elm=True)
    loader.failed_fetcher.fetch_all = AsyncMock()

    assert await loader.fetch_all() == []
    loader.failed_fetcher.fetch_all.assert_not_awaited()


@pytest.mark.parametrize("backend", [None, "elm", "docling", "unknown"])
@requires_docling
def test_docling_backend_selection_isolated_from_other_tests(
    monkeypatch, backend
):
    """Select both loaders from the import-time backend setting"""
    if backend is None:
        monkeypatch.delenv("COMPASS_FILE_LOAD_BACKEND", raising=False)
    else:
        monkeypatch.setenv("COMPASS_FILE_LOAD_BACKEND", backend)
    namespace = runpy.run_path(file_loader_module.__file__)
    prefix = "AsyncDoclingWeb" if backend == "docling" else "AsyncWeb"
    local = "AsyncLocalDocling" if backend == "docling" else "AsyncLocal"
    # the web loader adds event recording on top of the chosen backend
    assert issubclass(
        namespace["COMPASSWebFileLoader"], namespace[f"{prefix}FileLoader"]
    )
    assert (
        namespace["COMPASSLocalFileLoader"] is namespace[f"{local}FileLoader"]
    )


@pytest.mark.asyncio
@requires_docling
async def test_docling_web_file_loader_fetch_all_falls_back_to_elm(
    monkeypatch,
):
    """Retry only missing sources with the ELM fallback loader"""
    loader = AsyncDoclingWebFileLoader()
    fallback_doc = _doc("missing")
    failed_fetcher = _FailedFetcher({"missing": fallback_doc})
    loader.failed_fetcher = failed_fetcher

    async def _fetch(source):  # ruff:ignore[unused-async]
        if source == "missing":
            return None
        return _doc(source)

    async def _fetch_html_docs(docs):  # ruff:ignore[unused-async]
        return docs

    monkeypatch.setattr(loader, "fetch", _fetch)
    monkeypatch.setattr(
        loader, "_fetch_html_docs_again_using_playwright", _fetch_html_docs
    )

    docs = await loader.fetch_all("kept", "missing")

    assert [doc.attrs["source"] for doc in docs] == ["kept", "missing"]
    assert failed_fetcher.calls == [("missing",)]


@pytest.mark.asyncio
@requires_docling
async def test_docling_web_file_loader_handles_sourceless_elm_failure(
    monkeypatch,
):
    """Keep partial Docling output when ELM fails before adding source"""
    loader = AsyncDoclingWebFileLoader()
    partial_doc = _doc("failed", conversion_status="partial_success")
    fallback_doc = SimpleNamespace(attrs={}, empty=True)
    failed_fetcher = _FailedFetcher({"failed": fallback_doc})
    loader.failed_fetcher = failed_fetcher

    async def _fetch(source):  # ruff:ignore[unused-async]
        return partial_doc

    async def _fetch_html_docs(docs):  # ruff:ignore[unused-async]
        return docs

    monkeypatch.setattr(loader, "fetch", _fetch)
    monkeypatch.setattr(
        loader, "_fetch_html_docs_again_using_playwright", _fetch_html_docs
    )

    docs = await loader.fetch_all("failed")

    assert docs == [partial_doc]
    assert docs[0].attrs["source"] == "failed"
    assert failed_fetcher.calls == [("failed",)]


@pytest.mark.asyncio
@requires_docling
async def test_docling_html_rendering_preserves_source(monkeypatch):
    """Keep the input source on both initial and rendered HTML docs"""
    loader = AsyncDoclingWebFileLoader()
    url = "https://example.com/requested"
    resolved_url = "https://example.com/resolved"
    rendered_requests = []

    async def fetch_doc(source):  # ruff:ignore[unused-async]
        doc = MDDocument(pages=["Initial HTML"])
        doc.attrs.update(source=resolved_url, doc_type="html")
        return doc, None

    async def render_doc(source):  # ruff:ignore[unused-async]
        rendered_requests.append(source)
        doc = HTMLDocument(pages=["Rendered HTML"])
        doc.attrs["source"] = resolved_url
        return doc, None

    monkeypatch.setattr(loader, "_fetch_doc", fetch_doc)
    monkeypatch.setattr(loader.html_loader, "_fetch_doc", render_doc)

    docs = await loader.fetch_all(url)

    assert len(docs) == 2
    assert rendered_requests == [url]
    assert all(doc.attrs["source"] == url for doc in docs)
    assert all("requested_url" not in doc.attrs for doc in docs)


@pytest.mark.asyncio
@requires_docling
async def test_docling_web_loader_passes_configured_deadline(monkeypatch):
    """Configured Docling deadlines should reach the converter"""
    captured = {}
    loader = AsyncDoclingWebFileLoader(
        pdf_pipeline_options={"document_timeout": 120}
    )
    loader.content_fetcher = _Fetcher()

    async def _read_docling_web_file(*args, **kwargs):
        await asyncio.sleep(0)
        captured.update(kwargs)
        return _doc("https://example.com/sample.pdf")

    monkeypatch.setattr(
        "compass.services.docling.read_docling_web_file",
        _read_docling_web_file,
    )

    doc, raw_content = await loader._fetch_doc(
        "https://example.com/sample.pdf"
    )

    assert doc.attrs["doc_type"] == "pdf"
    assert raw_content == b"content"
    assert captured["pdf_pipeline_options"] == {"document_timeout": 120}


@pytest.mark.parametrize("empty", [False, True])
async def test_timing_preserves_backend_result(monkeypatch, caplog, empty):
    """Timing retains the backend result and reports empty downloads."""
    doc = _doc("https://city.gov/code", empty=empty)
    fetch = AsyncMock(return_value=doc)
    monkeypatch.setattr(COMPASSWebFileLoader.__bases__[1], "fetch", fetch)
    with caplog.at_level("INFO"):
        result = await COMPASSWebFileLoader().fetch(doc.attrs["source"])
    assert result is doc
    fetch.assert_awaited_once_with(doc.attrs["source"])
    assert "Finished document retrieval" in caplog.text
    assert f"empty={empty}" in caplog.text


async def test_fetch_records_document_event(monkeypatch, tmp_path):
    """Every COMPASS download is recorded in the event log."""
    doc = _doc("https://city.gov/code")
    monkeypatch.setattr(
        COMPASSWebFileLoader.__bases__[1], "fetch", AsyncMock(return_value=doc)
    )
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        await COMPASSWebFileLoader().fetch(doc.attrs["source"])
    (event,) = [json.loads(x) for x in log.path.read_text().splitlines()]
    assert event["type"] == "document_fetched"
    assert event["requested"] == "https://city.gov/code"
    assert event["empty"] is False


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])


@pytest.fixture
async def challenge_server():
    """Local server: a firewall-style 307 page, and a redirect to a PDF"""

    async def challenge(_request):  # ruff:ignore[unused-async]
        return web.Response(
            status=307,
            text="<html>You are being redirected...</html>",
            content_type="text/html",
            headers={"Server": "Sucuri/Cloudproxy"},
        )

    async def moved(_request):  # ruff:ignore[unused-async]
        raise web.HTTPFound("/doc.pdf")

    async def pdf(_request):  # ruff:ignore[unused-async]
        return web.Response(
            body=b"%PDF-1.7 data", content_type="application/pdf"
        )

    app = web.Application()
    app.add_routes(
        [
            web.get("/challenge", challenge),
            web.get("/moved", moved),
            web.get("/doc.pdf", pdf),
        ]
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    await runner.cleanup()


def test_docling_fallback_loaders_are_recorded():
    """The Docling backend's fallback loaders record their fetches too"""
    assert COMPASSWebFileLoader.HTML_LOADER_CLASS.FETCH_STAGE == (
        "browser render"
    )
    fallback = COMPASSWebFileLoader.FALLBACK_LOADER_CLASS()
    assert fallback.FETCH_STAGE == "ELM fallback"
    hook = fallback.content_fetcher.on_response_hook
    assert hook == fallback._record_exchange


@pytest.mark.asyncio
async def test_fallback_fetch_is_labelled_in_the_log(monkeypatch, tmp_path):
    """A document recovered by a fallback is recorded with its stage"""
    doc = _doc("https://city.gov/code", empty=False)
    monkeypatch.setattr(
        AsyncWebFileLoader, "fetch", AsyncMock(return_value=doc)
    )
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        await COMPASSWebFileLoader.FALLBACK_LOADER_CLASS().fetch(
            doc.attrs["source"]
        )
    (event,) = [json.loads(x) for x in log.path.read_text().splitlines()]
    assert event["type"] == "document_fetched"
    assert event["stage"] == "ELM fallback"
    assert event["empty"] is False


def test_web_loader_listens_for_http_responses():
    """The loader asks ELM's fetcher to report each response to it"""
    loader = COMPASSWebFileLoader()
    hook = loader.content_fetcher.on_response_hook
    assert hook == loader._record_exchange
    assert loader.content_fetcher.on_error_hook == loader._record_error


@pytest.mark.asyncio
async def test_downloads_record_why_no_response_came():
    """A fetch that never got a response logs the connection error"""
    loader = SimpleNamespace(_exchanges={})
    url = "https://unreachable.example/doc.pdf"
    await COMPASSWebFileLoader._record_error(
        loader, url, ConnectionError("connection refused")
    )
    assert COMPASSWebFileLoader._http_exchange(loader, url) == {
        "http": {"error": "ConnectionError: connection refused"}
    }


@pytest.mark.asyncio
async def test_downloads_record_their_http_exchange(
    challenge_server, tmp_path
):
    """Status, headers, redirects, and a body summary reach the log"""
    loader = SimpleNamespace(_exchanges={})
    fetcher = AsyncFetchWithRetry(
        on_response_hook=partial(COMPASSWebFileLoader._record_exchange, loader)
    )
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        await fetcher.fetch(f"{challenge_server}/challenge")
        await fetcher.fetch(f"{challenge_server}/moved")
        blocked = COMPASSWebFileLoader._http_exchange(
            loader, f"{challenge_server}/challenge"
        )["http"]
        moved = COMPASSWebFileLoader._http_exchange(
            loader, f"{challenge_server}/moved"
        )["http"]
    assert blocked["response"]["status"] == 307
    assert blocked["response"]["headers"]["server"] == "Sucuri/Cloudproxy"
    assert "You are being redirected" in blocked["response"]["preview"]
    assert blocked["response"]["bytes"] == 40
    assert moved["response"]["status"] == 200
    assert moved["response"]["history"][0]["status"] == 302
    assert moved["response"]["history"][0]["location"] == "/doc.pdf"
    # bodies are described in the log, never saved as files
    assert not [p for p in tmp_path.iterdir() if p.name != "events.jsonl"]
    assert loader._exchanges == {}
