"""Tests for collection persistence"""

import json
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


def _build_jurisdiction(full_name="Example Township", code="12345"):
    """Build a minimal jurisdiction for persistence tests"""
    return SimpleNamespace(full_name=full_name, code=code)


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


def test_load_specific_collection_manifest_shard_returns_none(tmp_path):
    """Missing jurisdiction shard should return None"""
    jurisdiction = _build_jurisdiction()

    collection_info = (
        persistence_module._load_specific_collection_manifest_shard(
            tmp_path, jurisdiction
        )
    )

    assert collection_info is None


def test_load_specific_collection_manifest_shard_resolves_paths(tmp_path):
    """Jurisdiction shard paths should resolve from the shard directory"""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    collection_info = {
        "FIPS": "12345",
        "full_name": "Example Township",
        "documents": [
            {
                "source": "https://example.com/valid.html",
                "source_fp": "./Example Township_1.html",
                "parsed_fp": "./Example Township_1.txt",
            }
        ],
    }
    shard_fp = (
        shard_dir
        / persistence_module._collection_manifest_shard_filename(
            collection_info
        )
    )
    shard_fp.write_text(json.dumps(collection_info), encoding="utf-8")

    loaded = persistence_module._load_specific_collection_manifest_shard(
        shard_dir, _build_jurisdiction()
    )

    assert loaded["documents"][0]["source"] == "https://example.com/valid.html"
    assert (
        loaded["documents"][0]["source_fp"]
        == (shard_dir / "Example Township_1.html").resolve().as_posix()
    )
    assert (
        loaded["documents"][0]["parsed_fp"]
        == (shard_dir / "Example Township_1.txt").resolve().as_posix()
    )


def test_build_collection_manifest_computes_doc_stats():
    """Collection manifest should summarize document counts"""
    manifest = persistence_module.build_collection_manifest(
        "solar",
        [
            {"full_name": "Alpha", "documents": [{"id": 1}, {"id": 2}]},
            {"full_name": "Beta", "documents": [{"id": 3}]},
            {"full_name": "Gamma", "documents": []},
            None,
        ],
        datetime(2026, 1, 1, tzinfo=UTC),
        4,
    )

    assert manifest["num_jurisdictions_searched"] == 4
    assert manifest["num_jurisdictions_found"] == 2
    assert manifest["num_doc_stats"] == {
        "min": 1,
        "max": 2,
        "median": 1.5,
        "total": 3,
    }
    assert [
        jurisdiction["full_name"] for jurisdiction in manifest["jurisdictions"]
    ] == ["Alpha", "Beta"]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
