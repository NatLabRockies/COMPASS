"""Tests for compass.pipeline orchestration"""

import logging
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import compass.pipeline.coordinator as coordinator_module
import compass.pipeline.data_classes as data_classes_module
import compass.pipeline.runtime as runtime_module
from compass.exceptions import COMPASSFileNotFoundError, COMPASSValueError
from compass.pipeline import (
    CollectionRequest,
    ExtractionRequest,
    ProcessRequest,
    SearchRequest,
)
from compass.pipeline.coordinator import run_compass
from compass.pipeline.runtime import PipelineRuntime
from compass.utilities.enums import COMPASSDocumentCollectionStep

from compass.pb import COMPASS_PB


@pytest.fixture(autouse=True)
def reset_compass_pb():
    """Reset progress bar state around each test"""
    COMPASS_PB.reset()
    yield
    COMPASS_PB.reset()


@pytest.fixture
def testing_log_file(tmp_path):
    """Logger fixture for testing"""
    log_fp = tmp_path / "test.log"
    handler = logging.FileHandler(log_fp, encoding="utf-8")
    logger = logging.getLogger("compass")
    prev_level = logger.level
    prev_propagate = logger.propagate
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    logger.addHandler(handler)

    yield log_fp

    handler.flush()
    logger.removeHandler(handler)
    handler.close()
    logger.setLevel(prev_level)
    logger.propagate = prev_propagate


@pytest.fixture
def patched_workflow(monkeypatch):
    """Patch workflow selection to a dummy pipeline workflow"""

    class DummyWorkflow:
        """Minimal workflow used to verify request dispatch"""

        LAST_MODE_USED = None

        def __init__(self, runtime):
            self.runtime = runtime

        async def run(self, jurisdictions_df):
            DummyWorkflow.LAST_MODE_USED = self.runtime.mode
            return f"processed {self.runtime.mode}"

    monkeypatch.setattr(
        data_classes_module,
        "build_models",
        lambda _model_input, **_kwargs: {},
    )
    monkeypatch.setattr(
        coordinator_module,
        "_load_jurisdictions_to_process",
        lambda _: pd.DataFrame([{"State": "Washington", "County": "Whatcom"}]),
    )
    monkeypatch.setattr(coordinator_module, "_select_workflow", DummyWorkflow)
    return DummyWorkflow


def test_known_local_docs_missing_file(tmp_path):
    """Raise when known_local_docs points to missing config"""
    missing_fp = tmp_path / "does_not_exist.json"
    request = ProcessRequest(
        out_dir=tmp_path / "outputs",
        tech="solar",
        jurisdiction_fp=tmp_path / "jurisdictions.csv",
        model=None,
        known_local_docs=str(missing_fp),
    )

    with pytest.raises(
        COMPASSFileNotFoundError, match="Configuration file does not exist"
    ):
        PipelineRuntime(request)


def test_known_local_docs_logs_missing_file(tmp_path, testing_log_file):
    """Log missing known_local_docs config to error file"""

    missing_fp = tmp_path / "does_not_exist.json"
    request = ProcessRequest(
        out_dir=tmp_path / "outputs",
        tech="solar",
        jurisdiction_fp=tmp_path / "jurisdictions.csv",
        model=None,
        known_local_docs=str(missing_fp),
    )

    with pytest.raises(
        COMPASSFileNotFoundError, match="Configuration file does not exist"
    ):
        PipelineRuntime(request)

    assert testing_log_file.exists()
    assert "Configuration file does not exist" in testing_log_file.read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize("request_class", [CollectionRequest, ProcessRequest])
@pytest.mark.parametrize("input_kind", ["mapping", "path", "string_path"])
def test_known_crawl_seeds_route_to_matching_jurisdiction(
    tmp_path, request_class, input_kind
):
    """Requests load seed mappings and isolate each jurisdiction's seeds"""
    seeds = {
        53073: ["https://example.com/whatcom/zoning"],
        "53001": ["https://example.com/adams/zoning"],
    }
    seed_input = seeds
    if input_kind != "mapping":
        seed_input = tmp_path / "seeds.json"
        seed_input.write_text(json.dumps(seeds), encoding="utf-8")
        if input_kind == "string_path":
            seed_input = str(seed_input)
    request = request_class(
        tmp_path / "outputs",
        "solar",
        "jurisdictions.csv",
        model=None,
        known_crawl_seeds=seed_input,
        perform_se_search=False,
        perform_website_search=False,
    )
    runtime = PipelineRuntime(request)
    runtime.__dict__["extractor_class"] = lambda **_kwargs: SimpleNamespace()
    mode = coordinator_module.COMPASSCollection(runtime)
    jurisdiction = SimpleNamespace(
        code="53073", full_name="Whatcom County", website_url=None
    )

    workflow = mode._create(jurisdiction)

    assert workflow.known_crawl_seeds == seeds[53073]
    assert [step.STEP_NAME for step in workflow.collection.steps] == [
        COMPASSDocumentCollectionStep.KNOWN_CRAWL_SEEDS
    ]
    jurisdiction.code = "99999"
    assert mode._create(jurisdiction).known_crawl_seeds is None


@pytest.mark.parametrize("seeds", [None, {}, {"53073": []}])
def test_empty_known_crawl_seeds_do_not_enable_collection(tmp_path, seeds):
    """An absent mapping or an empty jurisdiction list enables no crawl"""
    request = CollectionRequest(
        tmp_path / "outputs",
        "solar",
        "jurisdictions.csv",
        known_crawl_seeds=seeds,
        perform_se_search=False,
        perform_website_search=False,
    )
    runtime = PipelineRuntime(request)
    runtime.__dict__["extractor_class"] = lambda **_kwargs: SimpleNamespace()
    jurisdiction = SimpleNamespace(
        code="53073", full_name="Whatcom County", website_url=None
    )

    workflow = coordinator_module.COMPASSCollection(runtime)._create(
        jurisdiction
    )
    assert workflow.collection.steps == []


@pytest.mark.asyncio
@pytest.mark.parametrize("request_class", [CollectionRequest, ProcessRequest])
async def test_known_crawl_seeds_allow_seed_only_runs(
    tmp_path,
    request_class,
    patched_workflow,
    assert_message_was_logged,
    caplog,
):
    """The coordinator accepts seeds when all discovery is disabled"""
    request = request_class(
        tmp_path / "outputs",
        "solar",
        "jurisdictions.csv",
        model=None,
        log_level="INFO",
        known_crawl_seeds={"53073": ["https://example.com/zoning"]},
        perform_se_search=False,
        perform_website_search=False,
    )

    with caplog.at_level(logging.INFO, logger=coordinator_module.__name__):
        assert await run_compass(request) == f"processed {request.MODE}"
    assert_message_was_logged("Crawl known seed URLs", log_level="INFO")


def test_runtime_passes_docling_pipeline_options_to_local_loader(tmp_path):
    """Pass Docling configuration to known-local document loaders"""
    request = CollectionRequest(
        out_dir=tmp_path / "outputs",
        tech="solar",
        jurisdiction_fp=tmp_path / "jurisdictions.csv",
        file_loader_kwargs={
            "pdf_pipeline_options": {
                "document_timeout": 120,
                "do_table_structure": True,
            },
        },
    )

    runtime = PipelineRuntime(request)

    assert runtime.local_file_loader_kwargs["pdf_pipeline_options"] == {
        "document_timeout": 120,
        "do_table_structure": True,
    }


@pytest.mark.asyncio
async def test_search_request_rejects_existing_output_dir(tmp_path):
    """Fresh programmatic searches must not overwrite existing output"""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    existing_output = out_dir / "search_result_manifest.json"
    existing_output.write_text("{}", encoding="utf-8")
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.write_text(
        "State,County\nWashington,Whatcom\n", encoding="utf-8"
    )
    request = SearchRequest(
        out_dir=out_dir, tech="solar", jurisdiction_fp=jurisdiction_fp
    )

    with pytest.raises(COMPASSValueError, match="already exists"):
        __ = PipelineRuntime(request).dirs

    loggers = [logging.getLogger(name) for name in ("compass", "elm")]
    original_handlers = [list(logger.handlers) for logger in loggers]

    with pytest.raises(COMPASSValueError, match="already exists"):
        await run_compass(request)

    assert [logger.handlers for logger in loggers] == original_handlers
    assert existing_output.read_text(encoding="utf-8") == "{}"
    assert not (out_dir / "logs").exists()


@pytest.mark.asyncio
async def test_collect_request_uses_collection_workflow(
    tmp_path, patched_workflow
):
    """Collection requests should dispatch to the collection workflow"""
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    request = CollectionRequest(
        out_dir=tmp_path / "outputs",
        tech="solar",
        jurisdiction_fp=jurisdiction_fp,
    )
    result = await run_compass(request)

    assert result == f"processed {request.MODE}"
    assert patched_workflow.LAST_MODE_USED == request.MODE


@pytest.mark.asyncio
async def test_collect_request_allows_existing_output_dir(
    tmp_path, patched_workflow
):
    """Collection requests may reuse an existing output directory"""
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()

    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    request = CollectionRequest(
        out_dir=out_dir,
        tech="solar",
        jurisdiction_fp=jurisdiction_fp,
    )
    result = await run_compass(request)

    assert result == f"processed {request.MODE}"
    assert patched_workflow.LAST_MODE_USED == request.MODE
    assert out_dir.exists()


@pytest.mark.asyncio
async def test_extract_request_uses_extraction_workflow(
    tmp_path, patched_workflow
):
    """Extraction requests should dispatch to the extraction workflow"""
    out_dir = tmp_path / "outputs"

    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    manifest_fp = tmp_path / "manifest_fp.json"
    manifest_fp.touch()

    request = ExtractionRequest(
        out_dir=out_dir,
        tech="solar",
        jurisdiction_fp=jurisdiction_fp,
        collection_manifest_fp=manifest_fp,
        model=None,
    )
    result = await run_compass(request)

    assert result == f"processed {request.MODE}"
    assert patched_workflow.LAST_MODE_USED == request.MODE


@pytest.mark.asyncio
async def test_external_exceptions_logged_to_file(tmp_path, monkeypatch):
    """Log external exceptions to error file"""

    class RaisingWorkflow:
        """Workflow that fails inside the runtime context"""

        async def run(self, jurisdictions_df):
            raise NotImplementedError("Simulated external error")

    def _load_single_jurisdiction(_):
        return pd.DataFrame([{"State": "Washington", "County": "Whatcom"}])

    monkeypatch.setattr(
        coordinator_module,
        "_load_jurisdictions_to_process",
        _load_single_jurisdiction,
    )
    monkeypatch.setattr(
        data_classes_module,
        "build_models",
        lambda _model_input: {},
    )
    monkeypatch.setattr(
        coordinator_module,
        "_select_workflow",
        lambda __: RaisingWorkflow(),
    )

    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    with pytest.raises(NotImplementedError, match="Simulated external error"):
        await run_compass(
            ProcessRequest(
                out_dir=tmp_path / "outputs",
                tech="solar",
                jurisdiction_fp=jurisdiction_fp,
                model=None,
            )
        )

    log_files = list((tmp_path / "outputs" / "logs").glob("*"))
    assert len(log_files) == 1

    log_text = log_files[0].read_text(encoding="utf-8")
    assert "Fatal error during processing" in log_text
    assert "Simulated external error" in log_text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_class",
    [SearchRequest, CollectionRequest, ExtractionRequest, ProcessRequest],
)
@pytest.mark.parametrize(
    ("plugin_attrs", "expected_version"),
    [
        ({"SCHEMA": {"version": "1.2.3"}}, "1.2.3"),
        ({"SCHEMA": {}}, "unknown"),
        ({"SCHEMA": {"version": None}}, "unknown"),
        ({"SCHEMA": {"version": ""}}, "unknown"),
        ({}, None),
        ({"SCHEMA": None}, None),
    ],
)
async def test_plugin_schema_version_logged_at_start(
    tmp_path,
    patched_workflow,
    monkeypatch,
    request_class,
    plugin_attrs,
    expected_version,
):
    """Log schema versions once per run and omit non-schema plugins"""
    monkeypatch.setattr(
        runtime_module,
        "resolve_plugin",
        lambda _: SimpleNamespace(**plugin_attrs),
    )
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()
    request_kwargs = {
        "out_dir": tmp_path / "outputs",
        "tech": "solar",
        "jurisdiction_fp": jurisdiction_fp,
        "log_level": "INFO",
    }
    if request_class is ExtractionRequest:
        manifest_fp = tmp_path / "collection_manifest.json"
        manifest_fp.touch()
        request_kwargs["collection_manifest_fp"] = manifest_fp

    await run_compass(request_class(**request_kwargs))

    log_text = (tmp_path / "outputs" / "logs" / "main.log").read_text(
        encoding="utf-8"
    )
    if expected_version is None:
        assert "Plugin schema version:" not in log_text
    else:
        assert log_text.count("Plugin schema version:") == 1
        assert f"INFO: Plugin schema version: {expected_version}" in log_text


@pytest.mark.asyncio
async def test_process_args_logged_at_debug_to_file(
    tmp_path, patched_workflow, caplog, assert_message_was_logged
):
    """Log function arguments with DEBUG_TO_FILE level"""

    out_dir = tmp_path / "outputs"
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()
    caplog.set_level("DEBUG_TO_FILE", logger="compass")

    request = ProcessRequest(
        out_dir=out_dir,
        tech="solar",
        jurisdiction_fp=jurisdiction_fp,
        log_level="DEBUG",
    )
    result = await run_compass(request)

    assert result == f"processed {request.MODE}"

    assert_message_was_logged(
        "Called process pipeline with:",
        log_level="DEBUG_TO_FILE",
    )
    assert_message_was_logged('"out_dir": ', log_level="DEBUG_TO_FILE")
    assert_message_was_logged("outputs", log_level="DEBUG_TO_FILE")
    assert_message_was_logged('"tech": "solar"', log_level="DEBUG_TO_FILE")
    assert_message_was_logged('"jurisdiction_fp": ', log_level="DEBUG_TO_FILE")
    assert_message_was_logged("jurisdictions.csv", log_level="DEBUG_TO_FILE")
    assert_message_was_logged(
        '"log_level": "DEBUG"', log_level="DEBUG_TO_FILE"
    )
    assert_message_was_logged(
        '"model": "gpt-4o-mini"', log_level="DEBUG_TO_FILE"
    )
    assert_message_was_logged(
        '"keep_async_logs": false', log_level="DEBUG_TO_FILE"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "has_known_local_docs",
        "has_known_doc_urls",
        "perform_se_search",
        "perform_website_search",
    ),
    [
        pytest.param(
            *flags, id=("local-{}_urls-{}_se-{}_web-{}".format(*flags))
        )
        for flags in product([False, True], repeat=4)
    ],
)
async def test_process_steps_logged(
    tmp_path,
    patched_workflow,
    assert_message_was_logged,
    has_known_local_docs,
    has_known_doc_urls,
    perform_se_search,
    perform_website_search,
):
    """Log enabled processing steps for every combination of inputs"""

    out_dir = tmp_path / "outputs"
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    known_local_docs = None
    if has_known_local_docs:
        known_local_docs = {"1": [{"source_fp": tmp_path / "local_doc.pdf"}]}

    known_doc_urls = None
    if has_known_doc_urls:
        known_doc_urls = {
            "1": [{"source": "https://example.com/ordinance.pdf"}]
        }

    expected_steps = []
    if has_known_local_docs:
        expected_steps.append("Check local document")
    if has_known_doc_urls:
        expected_steps.append("Check known document URL")
    if perform_se_search:
        expected_steps.append("Look for document using search engine")
    if perform_website_search:
        expected_steps.append("Look for document on jurisdiction website")

    if not expected_steps:
        with pytest.raises(
            COMPASSValueError, match="No processing steps enabled"
        ):
            await run_compass(
                ProcessRequest(
                    out_dir=str(out_dir),
                    tech="solar",
                    jurisdiction_fp=str(jurisdiction_fp),
                    log_level="DEBUG",
                    known_local_docs=known_local_docs,
                    known_doc_urls=known_doc_urls,
                    perform_se_search=perform_se_search,
                    perform_website_search=perform_website_search,
                )
            )
        return

    request = ProcessRequest(
        out_dir=str(out_dir),
        tech="solar",
        jurisdiction_fp=str(jurisdiction_fp),
        log_level="DEBUG",
        known_local_docs=known_local_docs,
        known_doc_urls=known_doc_urls,
        perform_se_search=perform_se_search,
        perform_website_search=perform_website_search,
    )
    result = await run_compass(request)

    assert result == f"processed {request.MODE}"
    assert patched_workflow.LAST_MODE_USED == request.MODE

    assert_message_was_logged(
        "Using the following document acquisition step(s):", log_level="INFO"
    )
    assert_message_was_logged(" -> ".join(expected_steps), log_level="INFO")


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
