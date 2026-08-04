"""Tests for collection persistence"""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import compass.pipeline.collection.persistence as persistence_module
from compass.exceptions import COMPASSValueError
from compass.pipeline.collection.dedupe import DocumentDeDuplicator
from compass.services.provider import RunningAsyncServices
from compass.services.threaded import GenericFuncRunner


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
@pytest.mark.parametrize("input_type", ["str", "path"])
@pytest.mark.parametrize("is_relative", [True, False])
@pytest.mark.parametrize("has_wildcard", [True, False])
@pytest.mark.parametrize("is_list", [True, False])
# ruff:ignore[complex-structure]
async def test_load_collection_manifest_jurisdictions_path_variants(
    tmp_path, monkeypatch, input_type, is_relative, has_wildcard, is_list
):
    """Manifest inputs and persisted document paths should resolve"""
    manifest_dir = tmp_path / "manifests"
    manifest_fps = [
        manifest_dir / "first" / "manifest_first.json",
        manifest_dir / "second" / "manifest_second.json",
    ]
    expected_jurisdictions = []
    for index, manifest_fp in enumerate(manifest_fps, start=1):
        document_paths = {
            "dot": "./documents/source.html",
            "parent": "../shared/source.html",
            "normalized": "./documents/../normalized/source.html",
            "windows_dot": r".\documents\source.html",
            "windows_parent": r"..\shared\source.html",
        }
        documents = [
            {
                "path_case": path_case,
                "source_fp": source_fp,
                "parsed_fp": source_fp.replace("source.html", "parsed.txt"),
            }
            for path_case, source_fp in document_paths.items()
        ]
        jurisdiction = {"FIPS": f"{index:03d}", "documents": documents}
        manifest_fp.parent.mkdir(parents=True)
        manifest_fp.write_text(
            json.dumps(
                {
                    "tech": "solar",
                    "jurisdictions": [jurisdiction],
                }
            ),
            encoding="utf-8",
        )
        expected_jurisdictions.append(
            {
                "FIPS": f"{index:03d}",
                "documents": [
                    {
                        "path_case": doc_info["path_case"],
                        "source_fp": str(
                            (
                                manifest_fp.parent
                                / doc_info["source_fp"].replace("\\", "/")
                            )
                            .resolve()
                            .as_posix()
                        ),
                        "parsed_fp": str(
                            (
                                manifest_fp.parent
                                / doc_info["parsed_fp"].replace("\\", "/")
                            )
                            .resolve()
                            .as_posix()
                        ),
                    }
                    for doc_info in documents
                ],
            }
        )

    manifest_inputs = []
    for manifest_fp in manifest_fps:
        if is_relative:
            manifest_input = f"./{manifest_fp.relative_to(tmp_path)}"
        else:
            manifest_input = manifest_fp
        if has_wildcard:
            manifest_input = str(manifest_input).replace(
                manifest_fp.name, "*.json"
            )
        if input_type == "path":
            manifest_input = Path(manifest_input)
        else:
            manifest_input = str(manifest_input)
        manifest_inputs.append(manifest_input)

    monkeypatch.chdir(tmp_path)
    manifest_input = manifest_inputs if is_list else manifest_inputs[0]
    async with RunningAsyncServices([GenericFuncRunner()]):
        jurisdictions = (
            await persistence_module.load_collection_manifest_jurisdictions(
                manifest_input, "solar"
            )
        )

    if not is_list:
        expected_jurisdictions = expected_jurisdictions[:1]
    expected_jurisdictions = {
        jurisdiction["FIPS"]: jurisdiction
        for jurisdiction in expected_jurisdictions
    }
    assert jurisdictions == expected_jurisdictions
    for fips, jurisdiction in sorted(jurisdictions.items()):
        manifest_fp = manifest_fps[int(fips) - 1]
        for doc_info in jurisdiction["documents"]:
            for key in ("source_fp", "parsed_fp"):
                assert Path(doc_info[key]).is_absolute()
                expected_path = document_paths[doc_info["path_case"]]
                if key == "parsed_fp":
                    expected_path = expected_path.replace(
                        "source.html", "parsed.txt"
                    )
                expected_path = expected_path.replace("\\", "/")
                assert doc_info[key] == str(
                    (manifest_fp.parent / expected_path).resolve().as_posix()
                )


@pytest.mark.asyncio
async def test_load_collection_manifest_jurisdictions_recursive_wildcard(
    tmp_path,
):
    """Recursive wildcard inputs should load nested manifests"""
    manifest_dir = tmp_path / "manifests"
    manifest_fps = [
        manifest_dir / "first" / "collection_manifest.json",
        manifest_dir / "second" / "nested" / "collection_manifest.json",
    ]
    for index, manifest_fp in enumerate(manifest_fps, start=1):
        manifest_fp.parent.mkdir(parents=True)
        manifest_fp.write_text(
            json.dumps(
                {
                    "tech": "solar",
                    "jurisdictions": [{"FIPS": f"{index:03d}"}],
                }
            ),
            encoding="utf-8",
        )

    async with RunningAsyncServices([GenericFuncRunner()]):
        jurisdictions = (
            await persistence_module.load_collection_manifest_jurisdictions(
                manifest_dir / "**" / "*.json", "solar"
            )
        )

    assert jurisdictions == {"001": {"FIPS": "001"}, "002": {"FIPS": "002"}}


@pytest.mark.asyncio
async def test_load_collection_manifest_jurisdictions_rejects_duplicate_fips(
    tmp_path,
):
    """Overlapping manifests should fail instead of discarding an entry"""
    manifest_fps = [tmp_path / "first.json", tmp_path / "second.json"]
    for manifest_fp in manifest_fps:
        manifest_fp.write_text(
            json.dumps(
                {
                    "tech": "solar",
                    "jurisdictions": [{"FIPS": "12345", "documents": []}],
                }
            ),
            encoding="utf-8",
        )

    async with RunningAsyncServices([GenericFuncRunner()]):
        with pytest.raises(
            COMPASSValueError,
            match="Duplicate collection manifest entry for FIPS '12345'",
        ):
            await persistence_module.load_collection_manifest_jurisdictions(
                manifest_fps, "solar"
            )


@pytest.mark.asyncio
async def test_load_collection_manifest_jurisdictions_resolves_shard_paths(
    tmp_path,
):
    """Shard-recovered document paths should resolve from manifest root"""
    manifest_dir = tmp_path / "collection"
    shard_dir = manifest_dir / "shards"
    shard_dir.mkdir(parents=True)
    collection_info = {
        "FIPS": "12345",
        "full_name": "Example Township",
        "documents": [
            {
                "source_fp": "./downloaded/source.html",
                "parsed_fp": "./parsed/source.txt",
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

    manifest_fp = (
        manifest_dir / persistence_module.COLLECTION_MANIFEST_FILENAME
    )
    async with RunningAsyncServices([GenericFuncRunner()]):
        jurisdictions = (
            await persistence_module.load_collection_manifest_jurisdictions(
                manifest_fp, "solar"
            )
        )

    document = jurisdictions["12345"]["documents"][0]
    assert document["source_fp"] == str(
        (manifest_dir / "downloaded/source.html").resolve().as_posix()
    )
    assert document["parsed_fp"] == str(
        (manifest_dir / "parsed/source.txt").resolve().as_posix()
    )
    assert Path(document["source_fp"]).is_absolute()
    assert Path(document["parsed_fp"]).is_absolute()


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


@pytest.mark.asyncio
async def test_persist_documents_includes_collection_step_metadata(
    monkeypatch, tmp_path
):
    """Persisted collection info should include step count metadata"""

    async def fake_move(doc, out_stem, _subdir):  # ruff:ignore[unused-async]
        suffix = Path(doc.attrs["source"]).suffix or ".txt"
        return tmp_path / f"{out_stem}{suffix}"

    async def fake_write_parsed(doc, out_stem):  # ruff:ignore[unused-async]
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
    shared_doc = _build_doc("https://example.com/shared.html", ["page one"])
    search_only_doc = _build_doc(
        "https://example.com/search-only.html",
        ["page one", "page two"],
    )
    collected_docs = DocumentDeDuplicator()
    collected_docs.add_docs(
        [shared_doc, search_only_doc],
        step_name="crawl",
        jurisdiction_name=jurisdiction.full_name,
    )
    collected_docs.add_docs(
        [shared_doc],
        step_name="search",
        jurisdiction_name=jurisdiction.full_name,
    )

    collection_info = await persistence_module.persist_documents(
        jurisdiction,
        collected_docs,
        relative_to=tmp_path,
    )

    assert collection_info["num_docs"] == 2
    assert collection_info["collection_step_counts"] == {
        "crawl": 2,
        "search": 1,
    }
    assert [doc["from_steps"] for doc in collection_info["documents"]] == [
        ["crawl", "search"],
        ["crawl"],
    ]


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
    assert loaded["documents"][0]["source_fp"] == "./Example Township_1.html"
    assert loaded["documents"][0]["parsed_fp"] == "./Example Township_1.txt"


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
