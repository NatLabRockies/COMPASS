"""Tests for collection persistence"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import compass.pipeline.collection.persistence as persistence_module
from compass.pipeline.collection.dedupe import DocumentDeDuplicator


def _build_doc(source, pages, *, has_parsed_text=True):
    """Build a minimal collected document for persistence tests"""
    return SimpleNamespace(
        attrs={
            "source": source,
            "checksum": source,
            "is_pdf": False,
            "has_parsed_text": has_parsed_text,
        },
        pages=pages,
    )


@pytest.mark.asyncio
async def test_persist_documents_filters_docs_without_parsed_text(
    monkeypatch, tmp_path
):
    """Docs without parsed text should be omitted from persistence"""

    async def fake_move(doc, out_stem, _subdir):  # ruff:ignore[unused-async]
        suffix = Path(doc.attrs["source"]).suffix or ".txt"
        return tmp_path / f"{out_stem}{suffix}"

    async def fake_write_parsed(doc, out_stem):  # ruff:ignore[unused-async]
        if not doc.attrs["has_parsed_text"]:
            return None
        return tmp_path / f"{out_stem}.txt"

    monkeypatch.setattr(persistence_module.FileMover, "call", fake_move)
    monkeypatch.setattr(
        persistence_module.ParsedFileWriter,
        "call",
        fake_write_parsed,
    )

    jurisdiction = SimpleNamespace(
        full_name="Example Township",
        county="Example County",
        state="CO",
        subdivision_name=None,
        type="Township",
        code="12345",
    )
    valid_doc = _build_doc("https://example.com/valid.html", ["page one"])
    missing_parsed_doc = _build_doc(
        "https://example.com/missing.html",
        ["page one"],
        has_parsed_text=False,
    )
    collected_docs = DocumentDeDuplicator()
    collected_docs.add_docs(
        [valid_doc, missing_parsed_doc],
        step_name="crawl",
        jurisdiction_name=jurisdiction.full_name,
    )

    collection_info = await persistence_module.persist_documents(
        jurisdiction,
        collected_docs,
        relative_to=tmp_path,
    )

    assert collection_info["documents"] == [
        {
            "source": "https://example.com/valid.html",
            "checksum": "https://example.com/valid.html",
            "is_pdf": False,
            "has_parsed_text": True,
            "jurisdiction_name": "Example Township",
            "source_fp": Path("Example Township_1.html"),
            "parsed_fp": Path("Example Township_1.txt"),
            "check_correct_jurisdiction": True,
            "num_pages": 1,
            "from_steps": ["crawl"],
        }
    ]
    assert missing_parsed_doc.attrs["parsed_fp"] is None


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
