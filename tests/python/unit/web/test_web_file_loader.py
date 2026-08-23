"""COMPASS web file loader tests"""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from compass.web.file_loader import AsyncDoclingWebFileLoader


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
        loader,
        "_fetch_html_docs_again_using_playwright",
        _fetch_html_docs,
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
        loader,
        "_fetch_html_docs_again_using_playwright",
        _fetch_html_docs,
    )

    docs = await loader.fetch_all("failed")

    assert docs == [partial_doc]
    assert fallback_doc.attrs["source"] == "failed"
    assert failed_fetcher.calls == [("failed",)]


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
        "compass.web.file_loader.read_docling_web_file",
        _read_docling_web_file,
    )

    doc, raw_content = await loader._fetch_doc(
        "https://example.com/sample.pdf"
    )

    assert doc.attrs["doc_type"] == "pdf"
    assert raw_content == b"content"
    assert captured["pdf_pipeline_options"] == {"document_timeout": 120}


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
