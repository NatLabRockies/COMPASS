"""Tests for compass.pipeline orchestration"""

import logging
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

import compass.pipeline.coordinator as coordinator_module
import compass.pipeline.data_classes as data_classes_module
from compass.exceptions import COMPASSFileNotFoundError, COMPASSValueError
from compass.pipeline import (
    CollectionRequest,
    ExtractionRequest,
    ProcessRequest,
)
from compass.pipeline.coordinator import run_compass
from compass.pipeline.runtime import PipelineRuntime
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
        "_build_models",
        lambda __: {},
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
        """Return empty query templates for local-doc tests"""
        return []

    async def get_website_keywords(self):
        """Return empty website keywords for local-doc tests"""
        return {}

    async def get_heuristic(self):
        """Return a heuristic that keeps all docs"""

        class _KeepEverything:
            def check(self, text):
                return bool(text)

        return _KeepEverything()

    async def filter_docs(self, extraction_context):
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

    def _dummy_build_models(request):
        return {LLMTasks.DEFAULT: _DummyModelConfig()}

    monkeypatch.setattr(
        data_classes_module, "_build_models", _dummy_build_models
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
        "_build_models",
        lambda __: {},
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
