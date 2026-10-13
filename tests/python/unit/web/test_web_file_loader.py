"""COMPASS web file loader tests"""

import asyncio
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from elm.web.document import HTMLDocument, MDDocument

import compass.web.file_loader as file_loader_module
from compass.web.file_loader import (
    AsyncDoclingWebFileLoader,
    AsyncLocalDoclingFileLoader,
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
        "compass.web.file_loader.read_docling_web_file", converter
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
        "compass.web.file_loader.read_docling_web_file", converter
    )

    assert (await loader.fetch("https://example.com/code.pdf")).empty
    assert await loader.fetch_all("https://example.com/code.pdf") == []
    assert cached == []
    if failure == "download":
        converter.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("doc_type", ["pdf", "html"])
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
        "compass.web.file_loader.read_docling_local_file", converter
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
        "compass.web.file_loader.read_docling_local_file", converter
    )
    loader = AsyncLocalDoclingFileLoader(file_cache_coroutine=cache_document)

    doc = await loader.fetch(tmp_path / "missing.pdf")

    assert doc.empty
    assert cached == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback_kind", ["cached", "uncached", "empty"])
@pytest.mark.parametrize("status", ["partial_success", "failure", None])
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
        "compass.web.file_loader.read_docling_web_file",
        AsyncMock(return_value=partial),
    )
    loader.failed_fetcher.fetch_all = AsyncMock(return_value=[fallback])

    docs = await loader.fetch_all(url)

    assert docs == [fallback if fallback_kind == "cached" else partial]
    assert docs[0].attrs["source"] == url
    loader.failed_fetcher.fetch_all.assert_awaited_once_with(url)


@pytest.mark.asyncio
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
        "compass.web.file_loader.read_docling_web_file",
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
        "compass.web.file_loader.read_docling_web_file",
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
async def test_docling_empty_source_list_does_not_retry():
    """Return no documents or fallback requests for an empty input list"""
    loader = AsyncDoclingWebFileLoader(re_fetch_failed_with_elm=True)
    loader.failed_fetcher.fetch_all = AsyncMock()

    assert await loader.fetch_all() == []
    loader.failed_fetcher.fetch_all.assert_not_awaited()


@pytest.mark.parametrize("backend", [None, "elm", "docling", "unknown"])
def test_docling_backend_selection_isolated_from_other_tests(
    monkeypatch, backend
):
    """Select both loader aliases from the import-time backend setting"""
    if backend is None:
        monkeypatch.delenv("COMPASS_FILE_LOAD_BACKEND", raising=False)
    else:
        monkeypatch.setenv("COMPASS_FILE_LOAD_BACKEND", backend)
    namespace = runpy.run_path(file_loader_module.__file__)
    prefix = "AsyncDoclingWeb" if backend == "docling" else "AsyncWeb"
    local = "AsyncLocalDocling" if backend == "docling" else "AsyncLocal"
    assert (
        namespace["COMPASSWebFileLoader"] is namespace[f"{prefix}FileLoader"]
    )
    assert (
        namespace["COMPASSLocalFileLoader"] is namespace[f"{local}FileLoader"]
    )


@pytest.mark.asyncio
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
        "compass.web.file_loader.read_docling_web_file", _read_docling_web_file
    )

    doc, raw_content = await loader._fetch_doc(
        "https://example.com/sample.pdf"
    )

    assert doc.attrs["doc_type"] == "pdf"
    assert raw_content == b"content"
    assert captured["pdf_pipeline_options"] == {"document_timeout": 120}


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
