"""COMPASS web file loader tests"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from compass.web.file_loader import AsyncDoclingWebFileLoader


def _doc(source, doc_type="pdf", empty=False):
    return SimpleNamespace(
        attrs={"source": source, "doc_type": doc_type},
        empty=empty,
    )


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

    async def _fetch_playwright_html(docs):  # ruff:ignore[unused-async]
        return []

    monkeypatch.setattr(loader, "fetch", _fetch)
    monkeypatch.setattr(
        loader,
        "_fetch_playwright_html",
        _fetch_playwright_html,
    )

    docs = await loader.fetch_all("kept", "missing")

    assert [doc.attrs["source"] for doc in docs] == ["kept", "missing"]
    assert failed_fetcher.calls == [("missing",)]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
