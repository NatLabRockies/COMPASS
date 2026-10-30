"""Tests for collection persistence"""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import compass.pipeline.collection.persistence as persistence_module
from compass.exceptions import COMPASSValueError
from compass.pipeline.collection.dedupe import DocumentDeDuplicator
from compass.pipeline.targets import shard_failures
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
@pytest.mark.parametrize(
    "available_packages",
    [set(), {"docling_rs"}, {"docling_core"}],
)
async def test_reload_without_docling_dependencies(
    tmp_path, monkeypatch, caplog, available_packages
):
    """Reload UTF-8 text without importing an unavailable Docling backend"""
    fp = tmp_path / "parsed.md"
    text = "# Ordinance\nCaf\u00e9 <!-- retained comment -->"
    fp.write_text(text, encoding="utf-8")
    monkeypatch.setenv("COMPASS_FILE_LOAD_BACKEND", "elm")
    monkeypatch.setattr(
        persistence_module,
        "find_spec",
        lambda name: object() if name in available_packages else None,
    )
    import_service = Mock(side_effect=AssertionError("Unexpected import"))
    monkeypatch.setattr(persistence_module, "import_module", import_service)

    for _ in range(2):
        doc = await persistence_module._read_maybe_using_docling(fp)
        assert doc.pages == [text]
        assert doc.remove_comments is False

    import_service.assert_not_called()
    assert not [record for record in caplog.records if record.levelno >= 40]


@pytest.mark.asyncio
async def test_reload_with_docling_dependencies(tmp_path, monkeypatch):
    """Use Docling when both optional packages are available"""
    fp = tmp_path / "parsed.md"
    expected = object()
    read_document = AsyncMock(return_value=(expected, b"raw"))
    import_service = Mock(
        return_value=SimpleNamespace(read_docling_local_file=read_document)
    )
    find_package = Mock(return_value=object())
    monkeypatch.setattr(persistence_module, "find_spec", find_package)
    monkeypatch.setattr(persistence_module, "import_module", import_service)

    assert await persistence_module._read_maybe_using_docling(fp) is expected
    assert {call.args[0] for call in find_package.call_args_list} == {
        "docling_rs",
        "docling_core",
    }
    import_service.assert_called_once_with("compass.services.docling")
    read_document.assert_awaited_once_with(fp)


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
        jurisdiction = {"FIPS": f"{index:03d}", "results": documents}
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
                "results": [
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
        for doc_info in jurisdiction["results"]:
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
                    "jurisdictions": [{"FIPS": "12345", "results": []}],
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
        "results": [
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

    document = jurisdictions["12345"]["results"][0]
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

    monkeypatch.setattr(persistence_module.OrdFileMover, "call", fake_move)
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
    )

    collection_info = await persistence_module.persist_documents(
        jurisdiction,
        collected_docs,
        {"crawl"},
        relative_to=tmp_path,
        num_search_results_to_crawl=3,
    )

    assert collection_info["results"] == [
        {
            "source": "https://example.com/valid.html",
            "checksum": "https://example.com/valid.html",
            "is_pdf": False,
            "has_parsed_text": True,
            "archive_status": "saved",
            "source_fp": Path("Example Township_1.html"),
            "parsed_fp": Path("Example Township_1.txt"),
            "check_correct_jurisdiction": True,
            "num_pages": 1,
            "from_steps": ["crawl"],
        }
    ]
    assert missing_parsed_doc.attrs["parsed_fp"] is None
    assert collection_info["num_search_results_to_crawl"] == 3


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

    monkeypatch.setattr(persistence_module.OrdFileMover, "call", fake_move)
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
    )
    collected_docs.add_docs(
        [shared_doc],
        step_name="search",
    )

    collection_info = await persistence_module.persist_documents(
        jurisdiction,
        collected_docs,
        {"crawl", "search"},
        relative_to=tmp_path,
    )

    assert collection_info["num_docs"] == 2
    assert collection_info["completed_step_document_counts"] == {
        "crawl": 2,
        "search": 1,
    }
    assert [doc["from_steps"] for doc in collection_info["results"]] == [
        ["crawl", "search"],
        ["crawl"],
    ]


@pytest.mark.asyncio
async def test_persist_documents_preserves_restored_docs_and_step_counts(
    monkeypatch, tmp_path
):
    """Checkpoint persistence should retain old artifacts without rewrites"""

    async def fake_move(doc, out_stem, _subdir):  # ruff:ignore[unused-async]
        suffix = Path(doc.attrs["source"]).suffix or ".txt"
        return tmp_path / f"{out_stem}{suffix}"

    async def fake_write_parsed(doc, out_stem):  # ruff:ignore[unused-async]
        return tmp_path / f"{out_stem}.txt"

    monkeypatch.setattr(persistence_module.OrdFileMover, "call", fake_move)
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
    restored_doc = SimpleNamespace(
        attrs={
            "checksum": "restored",
            "source": "https://example.com/restored.html",
            "source_fp": "source_docs/restored.html",
            "parsed_fp": "parsed_docs/restored.txt",
            "from_steps": ["known_local_docs"],
        }
    )
    new_doc = _build_doc("https://example.com/new.html", ["page one"])
    collected_docs = DocumentDeDuplicator()
    collected_docs.add_docs([restored_doc])
    collected_docs.add_docs(
        [new_doc],
        step_name="search_engine",
    )

    collection_info = await persistence_module.persist_documents(
        jurisdiction,
        collected_docs,
        {"known_local_docs", "search_engine"},
        relative_to=tmp_path,
        jurisdiction_website="https://example.com",
    )

    assert collection_info["jurisdiction_website"] == "https://example.com"
    assert collection_info["results"][0] == restored_doc.attrs
    assert collection_info["results"][1]["source_fp"] == Path(
        "Example Township_2.html"
    )
    assert collection_info["completed_step_document_counts"] == {
        "known_local_docs": 1,
        "search_engine": 1,
    }


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
        "results": [
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

    assert loaded["results"][0]["source"] == "https://example.com/valid.html"
    assert loaded["results"][0]["source_fp"] == "./Example Township_1.html"
    assert loaded["results"][0]["parsed_fp"] == "./Example Township_1.txt"


def test_build_collection_manifest_computes_doc_stats():
    """Collection manifest should summarize document counts"""
    manifest = persistence_module.build_collection_manifest(
        "solar",
        [
            {"full_name": "Alpha", "results": [{"id": 1}, {"id": 2}]},
            {"full_name": "Beta", "results": [{"id": 3}]},
            {"full_name": "Gamma", "results": []},
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
    assert manifest["completed_step_document_totals"] == {}
    assert [
        jurisdiction["full_name"] for jurisdiction in manifest["jurisdictions"]
    ] == ["Alpha", "Beta"]


def test_build_collection_manifest_omits_checkpoint_only_jurisdiction():
    """No-document shards should be retained only as restart artifacts"""
    manifest = persistence_module.build_collection_manifest(
        "solar",
        [
            {
                "full_name": "No Documents",
                "results": [],
                "completed_step_document_counts": {
                    "known_local_docs": 0,
                    "search_engine": 0,
                },
            }
        ],
        datetime(2026, 1, 1, tzinfo=UTC),
        1,
    )

    assert manifest["num_jurisdictions_found"] == 0
    assert manifest["completed_step_document_totals"] == {}
    assert manifest["jurisdictions"] == []


@pytest.mark.parametrize(
    "targets, error",
    [
        ({"num_docs": -1}, "nonnegative integer"),
        ({"num_docs": 1.5}, "nonnegative integer"),
        ({"num_docs": True}, "finite numeric minimum"),
        ({"num_doc_stats.min": 1}, "Unknown collection shard target"),
        (
            {"completed_step_document_counts.unknown": 1},
            "Unknown collection shard target",
        ),
    ],
)
def test_validate_collection_targets_rejects_invalid_counts(targets, error):
    """Collection targets reject invalid counts and unknown metrics"""
    with pytest.raises(COMPASSValueError, match=error):
        persistence_module.validate_collection_targets(targets)


@pytest.mark.parametrize("saved_count", [None, 2])
def test_collection_targets_read_document_and_step_counts(saved_count):
    """Read saved counts and derive document counts for older shards"""
    record = {
        "results": [{"id": 1}, {"id": 2}],
        "completed_step_document_counts": {"known_local_docs": 2},
    }
    if saved_count is not None:
        record["num_docs"] = saved_count
    targets = {
        "num_docs": 2,
        "completed_step_document_counts.known_local_docs": 2,
        "completed_step_document_counts.search_engine": 0,
    }
    metrics = persistence_module.validate_collection_targets(targets)

    assert not shard_failures(record, targets, metrics)
    targets["completed_step_document_counts.search_engine"] = 1
    assert shard_failures(record, targets, metrics) == [
        {
            "metric": "completed_step_document_counts.search_engine",
            "actual": 0,
            "minimum": 1,
        }
    ]


@pytest.mark.parametrize("record", [None, {"results": [], "num_docs": 0}])
def test_collection_targets_require_nonempty_documents(record):
    """Zero explicit minima do not allow empty or missing shards"""
    targets = {"num_docs": 0}
    metrics = persistence_module.validate_collection_targets(targets)
    assert shard_failures(record, targets, metrics) == [
        {"metric": "results", "actual": 0, "minimum": 1}
    ]


@pytest.mark.parametrize("failures", [None, [], ["Example: documents=0"]])
def test_collection_manifest_preserves_target_failures(failures):
    """Target failures remain visible even for documentless jurisdictions"""
    manifest = persistence_module.build_collection_manifest(
        "solar", [], datetime.now(UTC), 1, failed_targets=failures
    )
    assert manifest["failed_targets"] == failures


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
