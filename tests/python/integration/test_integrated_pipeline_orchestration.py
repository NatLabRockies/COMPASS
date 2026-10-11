"""Tests for compass.pipeline orchestration"""

import json
from pathlib import Path

import pandas as pd
import pytest
from elm.web.document import MDDocument

import compass.pipeline.data_classes as data_classes_module
import compass.web.search as web_search_module
import compass.scripts.download as download_module
from compass.pipeline import (
    CollectionRequest,
    ExtractionRequest,
    ProcessRequest,
    SearchRequest,
)
from compass.pipeline.collection.persistence import (
    COLLECTION_MANIFEST_FILENAME,
)
from compass.pipeline.coordinator import run_compass
from compass._cli.main import main
from compass.plugin.base import BaseExtractionPlugin
from compass.plugin.registry import PLUGIN_REGISTRY, register_plugin
from compass.pb import COMPASS_PB
from compass.services.base import Service
from compass.utilities.enums import LLMTasks


@pytest.fixture(autouse=True)
def reset_compass_pb():
    """Reset progress bar state around each test"""
    COMPASS_PB.reset()
    yield
    COMPASS_PB.reset()


class _DummyLLMService(Service):
    """No-op service used to satisfy extraction orchestration in tests"""

    @property
    def can_process(self):
        """bool: Always ready to process"""
        return True

    async def process(self, *args, **kwargs):
        """Return a no-op response"""
        return


class _DummyModelConfig:
    """Minimal model config for deterministic extraction tests"""

    def __init__(self):
        self.name = "dummy-model"
        self.llm_service = _DummyLLMService()
        self.llm_call_kwargs = {}
        self.llm_service_rate_limit = 1
        self.text_splitter_chunk_size = 1000
        self.text_splitter_chunk_overlap = 0
        self.client_type = "test"


class _RoundtripTestPlugin(BaseExtractionPlugin):
    """Deterministic plugin for collection and extraction round trips"""

    IDENTIFIER = "roundtrip-test"

    async def get_query_templates(self):
        """Return deterministic query templates for round-trip tests"""
        return ["{jurisdiction} ordinance"]

    async def get_website_keywords(self):
        """Return empty website keywords for local-doc tests"""
        return {}

    async def get_heuristic(self):
        """Return a heuristic that keeps all docs"""

        class _KeepEverything:
            def check(self, text):
                return bool(text)

        return _KeepEverything()

    async def filter_docs(self, extraction_context, __):
        """Keep all docs for deterministic round-trip tests"""
        if not extraction_context:
            return None
        return extraction_context

    async def parse_docs_for_structured_data(self, extraction_context):
        """Turn each source doc into one structured row"""
        rows = []
        for doc in extraction_context.documents:
            await extraction_context.mark_doc_as_data_source(doc)
            rows.append(
                {
                    "jurisdiction": self.jurisdiction.full_name,
                    "source": doc.attrs.get("source"),
                    "source_kind": (
                        "pdf"
                        if str(doc.attrs.get("source", "")).endswith(".pdf")
                        else "text"
                    ),
                    "user_label": doc.attrs.get("user_label"),
                    "num_pages": len(doc.pages),
                }
            )

        extraction_context.attrs["structured_data"] = pd.DataFrame(rows)
        extraction_context.attrs["out_data_fn"] = (
            f"{self.jurisdiction.full_name} Ordinances.csv"
        )
        return extraction_context

    @classmethod
    def save_structured_data(cls, doc_infos, out_dir):
        """Write a simple combined CSV and return the row count"""
        frames = []
        for doc_info in doc_infos:
            if doc_info.get("ord_db_fp") is None:
                continue
            frames.append(pd.read_csv(doc_info["ord_db_fp"]))

        if not frames:
            return 0

        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(
            Path(out_dir) / "roundtrip_test_combined.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return len(frames)


@pytest.fixture
def registered_roundtrip_plugin():
    """Register a deterministic plugin for process round-trip tests"""
    plugin_id = _RoundtripTestPlugin.IDENTIFIER.casefold()
    already_registered = plugin_id in PLUGIN_REGISTRY
    if not already_registered:
        register_plugin(_RoundtripTestPlugin)

    yield _RoundtripTestPlugin

    if not already_registered:
        PLUGIN_REGISTRY.pop(plugin_id, None)


@pytest.fixture
def patched_model_configs(monkeypatch):
    """Replace pipeline model config setup with a deterministic stub"""

    def _dummy_build_models(model_input, rate_tracker):
        return {LLMTasks.DEFAULT: _DummyModelConfig()}

    monkeypatch.setattr(
        data_classes_module, "build_models", _dummy_build_models
    )


@pytest.fixture
def roundtrip_local_docs_inputs(tmp_path, test_data_files_dir):
    """Create jurisdiction and local-doc inputs for round-trip tests"""
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.write_text(
        "State,County,Subdivision,Jurisdiction Type\n"
        "Washington,Whatcom,,county\n"
        "New York,Allegany,Caneadea,town\n",
        encoding="utf-8",
    )

    known_local_docs = {
        "53073": [
            {
                "source_fp": test_data_files_dir / "Whatcom.txt",
                "user_label": "whatcom-text",
            }
        ],
        "3600312243": [
            {
                "source_fp": test_data_files_dir / "Caneadea New York.pdf",
                "user_label": "caneadea-pdf",
            }
        ],
    }

    return jurisdiction_fp, known_local_docs


@pytest.mark.asyncio
async def test_collect_then_extract_round_trip_from_manifest(
    tmp_path,
    registered_roundtrip_plugin,
    patched_model_configs,
    roundtrip_local_docs_inputs,
):
    """Collect docs to a manifest and then extract from that manifest"""
    jurisdiction_fp, known_local_docs = roundtrip_local_docs_inputs
    out_dir = tmp_path / "collection"

    collection_msg = await run_compass(
        CollectionRequest(
            out_dir=out_dir,
            tech="roundtrip-test",
            jurisdiction_fp=jurisdiction_fp,
            known_local_docs=known_local_docs,
            make_paths_relative=False,
            perform_se_search=False,
            perform_website_search=False,
        )
    )

    assert "2 documents collected for 2 jurisdictions" in collection_msg

    manifest_fp = out_dir / COLLECTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_fp.read_text(encoding="utf-8"))
    assert manifest["tech"] == "roundtrip-test"
    assert len(manifest["jurisdictions"]) == 2

    shard_fps = sorted(out_dir.rglob("*_collection_manifest.json"))
    assert len(shard_fps) == 2

    shard_payloads = [
        json.loads(shard_fp.read_text(encoding="utf-8"))
        for shard_fp in shard_fps
    ]
    assert {shard_payload["FIPS"] for shard_payload in shard_payloads} == {
        "53073",
        "3600312243",
    }

    whatcom = next(
        info for info in manifest["jurisdictions"] if info["FIPS"] == "53073"
    )
    caneadea = next(
        info
        for info in manifest["jurisdictions"]
        if info["FIPS"] == "3600312243"
    )

    assert whatcom["results"][0]["source_fp"] is not None
    assert Path(whatcom["results"][0]["parsed_fp"]).exists()
    assert whatcom["results"][0]["from_steps"] == ["known_local_docs"]

    assert Path(caneadea["results"][0]["source_fp"]).exists()
    assert Path(caneadea["results"][0]["parsed_fp"]).exists()
    assert caneadea["results"][0]["is_pdf"] is True
    assert whatcom in shard_payloads
    assert caneadea in shard_payloads
    for collection_info in (whatcom, caneadea):
        assert collection_info["completed_step_document_counts"] == {
            "known_local_docs": 1
        }
    assert manifest["completed_step_document_totals"] == {
        "known_local_docs": 2
    }

    COMPASS_PB.reset()
    resumed_collection_msg = await run_compass(
        CollectionRequest(
            out_dir=out_dir,
            tech="roundtrip-test",
            jurisdiction_fp=jurisdiction_fp,
            known_local_docs=known_local_docs,
            make_paths_relative=False,
            perform_se_search=False,
            perform_website_search=False,
        )
    )

    assert (
        "2 documents collected for 2 jurisdictions" in resumed_collection_msg
    )
    resumed_manifest = json.loads(manifest_fp.read_text(encoding="utf-8"))
    assert {info["FIPS"] for info in resumed_manifest["jurisdictions"]} == {
        "53073",
        "3600312243",
    }

    COMPASS_PB.reset()
    extraction_dir = tmp_path / "extracted"
    extraction_msg = await run_compass(
        ExtractionRequest(
            out_dir=extraction_dir,
            tech="roundtrip-test",
            collection_manifest_fp=manifest_fp,
            jurisdiction_fp=jurisdiction_fp,
            model=None,
        )
    )

    assert "Number of jurisdictions with extracted data: 2" in extraction_msg
    combined_fp = extraction_dir / "roundtrip_test_combined.csv"
    assert combined_fp.exists()

    combined = pd.read_csv(combined_fp)
    assert set(combined["user_label"]) == {"whatcom-text", "caneadea-pdf"}
    assert set(combined["source_kind"]) == {"text", "pdf"}


@pytest.mark.asyncio
async def test_extract_recovers_from_collection_manifest_shards(
    tmp_path,
    registered_roundtrip_plugin,
    patched_model_configs,
    roundtrip_local_docs_inputs,
):
    """Extraction should recover from per-jurisdiction manifest shards"""
    jurisdiction_fp, known_local_docs = roundtrip_local_docs_inputs
    out_dir = tmp_path / "collection"

    await run_compass(
        CollectionRequest(
            out_dir=out_dir,
            tech="roundtrip-test",
            jurisdiction_fp=jurisdiction_fp,
            known_local_docs=known_local_docs,
            make_paths_relative=True,
            perform_se_search=False,
            perform_website_search=False,
        )
    )

    manifest_fp = out_dir / COLLECTION_MANIFEST_FILENAME
    manifest_fp.unlink()

    COMPASS_PB.reset()
    extraction_dir = tmp_path / "extracted"
    extraction_msg = await run_compass(
        ExtractionRequest(
            out_dir=extraction_dir,
            tech="roundtrip-test",
            collection_manifest_fp=manifest_fp,
            jurisdiction_fp=jurisdiction_fp,
            model=None,
        )
    )

    assert "Number of jurisdictions with extracted data: 2" in extraction_msg
    combined_fp = extraction_dir / "roundtrip_test_combined.csv"
    assert combined_fp.exists()

    combined = pd.read_csv(combined_fp)
    assert set(combined["user_label"]) == {"whatcom-text", "caneadea-pdf"}


@pytest.mark.asyncio
async def test_process_writes_manifest_and_structured_outputs(
    tmp_path,
    registered_roundtrip_plugin,
    patched_model_configs,
    roundtrip_local_docs_inputs,
):
    """End-to-end process should compose collection and extraction"""
    jurisdiction_fp, known_local_docs = roundtrip_local_docs_inputs
    out_dir = tmp_path / "outputs"

    COMPASS_PB.reset()
    result = await run_compass(
        ProcessRequest(
            out_dir=out_dir,
            tech="roundtrip-test",
            jurisdiction_fp=jurisdiction_fp,
            known_local_docs=known_local_docs,
            perform_se_search=False,
            perform_website_search=False,
            model=None,
        )
    )

    assert "Number of jurisdictions with extracted data: 2" in result
    assert not (out_dir / COLLECTION_MANIFEST_FILENAME).exists()
    assert (out_dir / "roundtrip_test_combined.csv").exists()
    assert any((out_dir / "jurisdiction_dbs").glob("*.csv"))


@pytest.mark.asyncio
@pytest.mark.parametrize("request_class", [CollectionRequest, ProcessRequest])
@pytest.mark.parametrize("source", ["manifest", "shards"])
async def test_search_then_replay(
    tmp_path,
    monkeypatch,
    registered_roundtrip_plugin,
    patched_model_configs,
    roundtrip_local_docs_inputs,
    request_class,
    source,
):
    """Replay persisted search results without queries or new filtering"""
    jurisdiction_fp, _ = roundtrip_local_docs_inputs
    search_calls = []
    downloads = []

    async def search_backend(queries, **kwargs):  # ruff:ignore[unused-async]
        search_calls.append(queries)
        name = "whatcom" if "Whatcom" in kwargs["task_name"] else "caneadea"
        return [
            {
                "url": f"https://example.com/{name}/{rank}",
                "overall_rank": rank,
                "filtered_reason": None if rank < 3 else "beyond_top_n",
                "search_engine": "fixture",
                "search_engines": ["fixture"],
            }
            for rank in (2, 1, 3)
        ]

    async def fetch_doc(self, url):  # ruff:ignore[unused-async]
        downloads.append(url)
        return MDDocument(pages=[f"Ordinance at {url}"]), None

    monkeypatch.setattr(
        web_search_module, "search_with_fallback_with_attrs", search_backend
    )
    search_dir = tmp_path / "searched"
    await run_compass(
        SearchRequest(
            search_dir,
            "roundtrip-test",
            jurisdiction_fp,
            simple_se_result_sort=True,
        )
    )
    assert len(search_calls) == 2
    manifest_fp = search_dir / "search_result_manifest.json"
    manifest = json.loads(manifest_fp.read_text(encoding="utf-8"))
    assert manifest["result_stats"]["total"] == 6
    assert manifest["filtered_result_stats"]["total"] == 4
    assert len(list((search_dir / "se_results").glob("*.json"))) == 2
    assert (search_dir / "logs" / "main.log").exists()
    assert len(list((search_dir / "logs").glob("*.log"))) >= 3

    def unexpected_search(*_args, **_kwargs):
        pytest.fail("Replay must not query engines or request templates")

    monkeypatch.setattr(
        web_search_module, "search_with_fallback_with_attrs", unexpected_search
    )
    monkeypatch.setattr(
        _RoundtripTestPlugin, "get_query_templates", unexpected_search
    )
    monkeypatch.setattr(
        download_module.COMPASSWebFileLoader, "_fetch_doc", fetch_doc
    )
    saved_source = manifest_fp
    if source == "shards":
        manifest_fp.unlink()
        saved_source = search_dir / "se_results"
    COMPASS_PB.reset()
    output = tmp_path / "replayed"
    await run_compass(
        request_class(
            output,
            "roundtrip-test",
            jurisdiction_fp,
            model=None,
            search_result_manifest_fp=saved_source,
            perform_website_search=False,
            num_urls_to_check_per_jurisdiction=1,
            url_ignore_substrings=["example.com"],
        )
    )
    assert len(downloads) == 4
    for name in ("whatcom", "caneadea"):
        assert [url for url in downloads if name in url] == [
            f"https://example.com/{name}/1",
            f"https://example.com/{name}/2",
        ]
    if request_class is CollectionRequest:
        collected = json.loads(
            (output / COLLECTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert collected["num_doc_stats"]["total"] == 4
        for info in collected["jurisdictions"]:
            assert [
                doc["collection_step_rank"] for doc in info["results"]
            ] == [1, 2]
            assert all(
                doc["search_engines"] == ["fixture"]
                and doc["from_steps"] == ["search_engine"]
                for doc in info["results"]
            )
    else:
        assert len(pd.read_csv(output / "roundtrip_test_combined.csv")) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("request_class", [CollectionRequest, ProcessRequest])
@pytest.mark.parametrize("state", ["missing", "empty", "error", "disabled"])
async def test_replay_missing_empty_and_disabled(
    tmp_path,
    monkeypatch,
    registered_roundtrip_plugin,
    patched_model_configs,
    roundtrip_local_docs_inputs,
    request_class,
    state,
):
    """Skip missing jurisdictions but run other sources for empty entries"""
    jurisdiction_fp, known_local_docs = roundtrip_local_docs_inputs
    saved = tmp_path / "search.json"
    records = (
        []
        if state == "missing"
        else [
            {
                "FIPS": code,
                "results": [],
                "error": "failed" if state == "error" else None,
            }
            for code in known_local_docs
        ]
    )
    saved.write_text(
        json.dumps({"tech": "roundtrip-test", "jurisdictions": records}),
        encoding="utf-8",
    )
    if state == "disabled":
        saved.unlink()

    def unexpected_search(*_args, **_kwargs):
        pytest.fail("Saved inputs must never trigger a search")

    monkeypatch.setattr(
        web_search_module, "search_with_fallback_with_attrs", unexpected_search
    )
    monkeypatch.setattr(
        _RoundtripTestPlugin, "get_query_templates", unexpected_search
    )
    output = tmp_path / "output"
    await run_compass(
        request_class(
            output,
            "roundtrip-test",
            jurisdiction_fp,
            model=None,
            known_local_docs=known_local_docs,
            search_result_manifest_fp=saved,
            perform_se_search=state != "disabled",
            perform_website_search=False,
        )
    )
    if state == "missing":
        assert not list(output.rglob("*_collection_manifest.json"))
        assert not (output / "roundtrip_test_combined.csv").exists()
        assert "skipping jurisdiction" in (
            output / "logs" / "main.log"
        ).read_text(encoding="utf-8")
    elif request_class is CollectionRequest:
        manifest = json.loads(
            (output / COLLECTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        assert manifest["num_doc_stats"]["total"] == 2
    else:
        assert len(pd.read_csv(output / "roundtrip_test_combined.csv")) == 2


@pytest.fixture
def collection_cli_config(tmp_path, registered_roundtrip_plugin):
    """Create local document inputs for collection continuation tests"""
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.write_text(
        "County,State\nWhatcom,Washington\n", encoding="utf-8"
    )
    document_fp = tmp_path / "first.txt"
    document_fp.write_text("First ordinance document", encoding="utf-8")
    config_fp = tmp_path / "collection.json"
    config_fp.write_text(
        json.dumps(
            {
                "tech": "roundtrip-test",
                "jurisdiction_fp": str(jurisdiction_fp),
                "out_dir": str(tmp_path / "output"),
                "known_local_docs": {
                    "53073": [{"source_fp": str(document_fp)}]
                },
                "perform_se_search": False,
                "perform_website_search": False,
            }
        ),
        encoding="utf-8",
    )
    return config_fp


@pytest.mark.parametrize("no_progress", [True, False])
def test_collection_cli_reuses_passing_shard(
    cli_runner, collection_cli_config, patched_model_configs, no_progress
):
    """Passing collection shards retain their bytes and aggregate counts"""
    args = [
        "collect",
        "-c",
        str(collection_cli_config),
        "--target",
        "num_docs=1",
        "--target",
        "completed_step_document_counts.known_local_docs=1",
    ]
    if no_progress:
        args.append("--no-progress")
    initial = cli_runner.invoke(main, args)
    assert initial.exit_code == 0, f"{initial.output}\n{initial.exception}"
    output = collection_cli_config.parent / "output"
    shard_fp = next(output.rglob("*_collection_manifest.json"))
    original_bytes = shard_fp.read_bytes()
    (output / COLLECTION_MANIFEST_FILENAME).unlink()
    COMPASS_PB.reset()

    result = cli_runner.invoke(main, [*args, "--out-dir-exists", "continue"])

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert shard_fp.read_bytes() == original_bytes
    manifest = json.loads(
        (output / COLLECTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["failed_targets"] == []
    assert manifest["num_doc_stats"]["total"] == 1
    assert COMPASS_PB.console is None


def test_collection_cli_resumes_then_resets_on_next_invocation(
    cli_runner, collection_cli_config, patched_model_configs
):
    """Resume once, report unmet minima, and reset on the next continue"""
    args = ["collect", "-c", str(collection_cli_config), "--no-progress"]
    initial = cli_runner.invoke(main, args)
    assert initial.exit_code == 0, f"{initial.output}\n{initial.exception}"
    output = collection_cli_config.parent / "output"
    shard_fp = next(output.rglob("*_collection_manifest.json"))
    shard = json.loads(shard_fp.read_text(encoding="utf-8"))
    shard["completed_step_document_counts"] = {}
    shard_fp.write_text(json.dumps(shard), encoding="utf-8")
    second_fp = collection_cli_config.parent / "second.txt"
    second_fp.write_text(
        "Second distinct ordinance document", encoding="utf-8"
    )
    config = json.loads(collection_cli_config.read_text(encoding="utf-8"))
    config["known_local_docs"]["53073"] = [{"source_fp": str(second_fp)}]
    collection_cli_config.write_text(json.dumps(config), encoding="utf-8")
    unrelated_fp = output / "unrelated_collection_manifest.json"
    unrelated_fp.write_text('{"unchanged": true}', encoding="utf-8")
    COMPASS_PB.reset()
    continuing = [
        *args,
        "--out-dir-exists",
        "continue",
        "--target",
        "num_docs=3",
        "--target",
        "num_docs=1",
    ]

    resumed = cli_runner.invoke(main, continuing)

    assert resumed.exit_code == 1, f"{resumed.output}\n{resumed.exception}"
    assert "Failed 1 targets" in resumed.output
    shard = json.loads(shard_fp.read_text(encoding="utf-8"))
    assert shard["num_docs"] == 2
    assert shard["completed_step_document_counts"] == {"known_local_docs": 2}
    manifest = json.loads(
        (output / COLLECTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["failed_targets"] == [
        "Whatcom County, Washington: num_docs=2 (minimum 3)"
    ]
    COMPASS_PB.reset()

    reset = cli_runner.invoke(main, continuing)

    assert reset.exit_code == 1, f"{reset.output}\n{reset.exception}"
    shard = json.loads(shard_fp.read_text(encoding="utf-8"))
    assert shard["num_docs"] == 1
    assert len(shard["results"]) == 1
    assert "second" in shard["results"][0]["source"]
    assert unrelated_fp.read_text(encoding="utf-8") == '{"unchanged": true}'
    assert second_fp.exists()


@pytest.mark.parametrize(
    "options, error",
    [
        (["--target", "num_docs=-1"], "nonnegative integer"),
        (["--target", "num_docs=1.5"], "nonnegative integer"),
        (["--target", "num_docs=true"], "finite numeric minimum"),
        (["--target", "num_docs"], "use METRIC=MINIMUM"),
        (["--target", "num_doc_stats.min=1"], "Unknown collection shard"),
        (
            ["--target", "completed_step_document_counts.unknown=1"],
            "Unknown collection shard",
        ),
        (["--out-dir-exists", "overwrite"], "conflicts with --out-dir-exists"),
    ],
)
def test_collection_cli_invalid_targets_leave_shards_untouched(
    cli_runner, collection_cli_config, patched_model_configs, options, error
):
    """Reject invalid continuation requirements before replacing shards"""
    output = collection_cli_config.parent / "output"
    output.mkdir()
    sentinel = output / "existing_collection_manifest.json"
    sentinel.write_text('{"existing": true}', encoding="utf-8")
    result = cli_runner.invoke(
        main,
        [
            "collect",
            "-c",
            str(collection_cli_config),
            "--no-progress",
            "--out-dir-exists",
            "continue",
            "--target",
            "completed_step_document_counts.known_local_docs=0",
            *options,
        ],
    )

    assert result.exit_code == 1
    assert error in f"{result.output}\n{result.exception}"
    assert sentinel.read_text(encoding="utf-8") == '{"existing": true}'


def test_collection_cli_selectively_resets_and_collects_missing_shards(
    cli_runner, collection_cli_config, patched_model_configs
):
    """Only complete failing and missing shards recollect in a mixed run"""
    config = json.loads(collection_cli_config.read_text(encoding="utf-8"))
    Path(config["jurisdiction_fp"]).write_text(
        "County,State\nWhatcom,Washington\nAdams,Colorado\nBoulder,Colorado\n",
        encoding="utf-8",
    )
    for code in ["8001", "8013"]:
        config["known_local_docs"][code] = config["known_local_docs"]["53073"]
    collection_cli_config.write_text(json.dumps(config), encoding="utf-8")
    args = ["collect", "-c", str(collection_cli_config), "--no-progress"]
    initial = cli_runner.invoke(main, args)
    assert initial.exit_code == 0, f"{initial.output}\n{initial.exception}"
    output = collection_cli_config.parent / "output"
    shards = {
        json.loads(path.read_text(encoding="utf-8"))["FIPS"]: path
        for path in output.rglob("*_collection_manifest.json")
    }
    passing_bytes = shards["53073"].read_bytes()
    failed = json.loads(shards["8001"].read_text(encoding="utf-8"))
    failed.update(
        num_docs=0,
        documents=[],
        completed_step_document_counts={"known_local_docs": 0},
    )
    shards["8001"].write_text(json.dumps(failed), encoding="utf-8")
    shards["8013"].unlink()
    seed_fp = output / "search_input.json"
    seed_fp.write_text('{"preserved": true}', encoding="utf-8")
    COMPASS_PB.reset()

    result = cli_runner.invoke(
        main,
        [*args, "--out-dir-exists", "continue", "--target", "num_docs=1"],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert shards["53073"].read_bytes() == passing_bytes
    for code in ["8001", "8013"]:
        assert (
            json.loads(shards[code].read_text(encoding="utf-8"))["num_docs"]
            == 1
        )
    manifest = json.loads(
        (output / COLLECTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["num_doc_stats"]["total"] == 3
    assert manifest["failed_targets"] == []
    assert seed_fp.read_text(encoding="utf-8") == '{"preserved": true}'


@pytest.mark.parametrize("no_progress", [True, False])
def test_collection_cli_empty_results_save_failures(
    cli_runner, collection_cli_config, patched_model_configs, no_progress
):
    """Persist empty shards before reporting the implicit document minimum"""
    (collection_cli_config.parent / "first.txt").write_text(
        "", encoding="utf-8"
    )
    args = [
        "collect",
        "-c",
        str(collection_cli_config),
        "--target",
        "num_docs=0",
    ]
    if no_progress:
        args.append("--no-progress")
    result = cli_runner.invoke(main, args)

    assert result.exit_code == 1, f"{result.output}\n{result.exception}"
    assert "Failed 1 targets" in result.output
    output = collection_cli_config.parent / "output"
    shard_fp = next(output.rglob("*_collection_manifest.json"))
    assert json.loads(shard_fp.read_text(encoding="utf-8"))["num_docs"] == 0
    manifest = json.loads(
        (output / COLLECTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["jurisdictions"] == []
    assert manifest["failed_targets"] == [
        "Whatcom County, Washington: documents=0 (minimum 1)"
    ]
    assert COMPASS_PB.console is None


def test_collection_cli_targets_follow_output_override(
    cli_runner, collection_cli_config, patched_model_configs
):
    """Read final target failures from the overridden output directory"""
    override = collection_cli_config.parent / "overridden"
    result = cli_runner.invoke(
        main,
        [
            "collect",
            "-c",
            str(collection_cli_config),
            "--no-progress",
            "--target",
            "num_docs=1",
            f"--out_dir={override}",
        ],
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert (override / COLLECTION_MANIFEST_FILENAME).exists()
    assert not (collection_cli_config.parent / "output").exists()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
