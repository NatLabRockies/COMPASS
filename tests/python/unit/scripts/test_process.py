"""Tests for compass.scripts.process"""

import json
import logging
from pathlib import Path
from itertools import product

import pandas as pd
import pytest

from compass.plugin.registry import PLUGIN_REGISTRY, register_plugin
from compass.plugin.base import BaseExtractionPlugin
from compass.pb import COMPASS_PB
from compass.services.base import Service
from compass.exceptions import COMPASSValueError, COMPASSFileNotFoundError
import compass.scripts.process as process_module
from compass.scripts.process import (
    _COMPASSRunner,
    collect_jurisdiction_documents,
    extract_collected_jurisdiction_documents,
    process_jurisdictions_with_openai,
    COLLECTION_MANIFEST_FILENAME,
)
from compass.utilities.enums import LLMTasks
from compass.utilities import ProcessKwargs


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
def patched_runner(monkeypatch):
    """Patch the COMPASSRunner to a dummy that bypasses processing"""

    class DummyRunner:
        """Minimal runner that bypasses full processing"""

        LAST_MODE_USED = None

        def __init__(self, mode, **_):
            DummyRunner.LAST_MODE_USED = mode

        async def _run_collection(self, jurisdiction_fp):
            return f"collected {jurisdiction_fp}"

        async def _run_extraction(self, **kwargs):
            return f"extracted {kwargs}"

        async def run(self, jurisdiction_fp):
            return f"processed {jurisdiction_fp}"

    monkeypatch.setattr(process_module, "_COMPASSRunner", DummyRunner)
    return DummyRunner


def test_known_local_docs_missing_file(tmp_path):
    """Raise when known_local_docs points to missing config"""
    missing_fp = tmp_path / "does_not_exist.json"
    runner = _COMPASSRunner(
        dirs=None,
        log_listener=None,
        tech="solar",
        models={},
        process_kwargs=ProcessKwargs(str(missing_fp), None),
    )

    with pytest.raises(
        COMPASSFileNotFoundError, match="Configuration file does not exist"
    ):
        _ = runner.known_local_docs


def test_known_local_docs_logs_missing_file(tmp_path, testing_log_file):
    """Log missing known_local_docs config to error file"""

    missing_fp = tmp_path / "does_not_exist.json"
    runner = _COMPASSRunner(
        dirs=None,
        log_listener=None,
        tech="solar",
        models={},
        process_kwargs=ProcessKwargs(str(missing_fp), None),
    )

    with pytest.raises(
        COMPASSFileNotFoundError, match="Configuration file does not exist"
    ):
        _ = runner.known_local_docs

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

    async def filter_docs(
        self, extraction_context, need_jurisdiction_verification=True
    ):
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
    """Replace OpenAI model config setup with a deterministic stub"""

    def _dummy_initialize_model_params(user_input):
        return {LLMTasks.DEFAULT: _DummyModelConfig()}

    monkeypatch.setattr(
        process_module,
        "_initialize_model_params",
        _dummy_initialize_model_params,
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
async def test_collect_wrapper_uses_collection_runner(
    tmp_path, patched_runner
):
    """Collection wrapper should dispatch to the collection runner"""
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    result = await collect_jurisdiction_documents(
        out_dir=tmp_path / "outputs",
        tech="solar",
        jurisdiction_fp=jurisdiction_fp,
    )

    assert result == f"processed {jurisdiction_fp}"
    assert patched_runner.LAST_MODE_USED == "collect"


@pytest.mark.asyncio
async def test_extract_wrapper_uses_extraction_runner(
    tmp_path, patched_runner
):
    """Extraction wrapper should dispatch to the extraction runner"""
    out_dir = tmp_path / "outputs"

    manifest_fp = tmp_path / COLLECTION_MANIFEST_FILENAME
    manifest_fp.write_text('{"jurisdictions": []}', encoding="utf-8")

    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    manifest_fp = tmp_path / "manifest_fp.json"
    manifest_fp.touch()

    result = await extract_collected_jurisdiction_documents(
        out_dir=out_dir,
        tech="solar",
        jurisdiction_fp=jurisdiction_fp,
        collection_manifest_fp=manifest_fp,
    )

    assert "processed" in result
    assert patched_runner.LAST_MODE_USED == "extract"


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

    collection_msg = await collect_jurisdiction_documents(
        out_dir=out_dir,
        tech="roundtrip-test",
        jurisdiction_fp=jurisdiction_fp,
        known_local_docs=known_local_docs,
        perform_se_search=False,
        perform_website_search=False,
    )

    assert "2 documents collected for 2 jurisdictions" in collection_msg

    manifest_fp = out_dir / COLLECTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_fp.read_text(encoding="utf-8"))
    assert manifest["tech"] == "roundtrip-test"
    assert len(manifest["jurisdictions"]) == 2

    whatcom = next(
        info for info in manifest["jurisdictions"] if info["FIPS"] == 53073
    )
    caneadea = next(
        info
        for info in manifest["jurisdictions"]
        if info["FIPS"] == 3600312243
    )

    assert whatcom["documents"][0]["source_fp"] is not None
    assert Path(whatcom["documents"][0]["parsed_fp"]).exists()
    assert whatcom["documents"][0]["from_steps"] == ["known_local_docs"]

    assert Path(caneadea["documents"][0]["source_fp"]).exists()
    assert Path(caneadea["documents"][0]["parsed_fp"]).exists()
    assert caneadea["documents"][0]["is_pdf"] is True

    COMPASS_PB.reset()
    extraction_dir = tmp_path / "extracted"
    extraction_msg = await extract_collected_jurisdiction_documents(
        out_dir=extraction_dir,
        tech="roundtrip-test",
        collection_manifest_fp=manifest_fp,
        jurisdiction_fp=jurisdiction_fp,
    )

    assert "Number of jurisdictions with extracted data: 2" in extraction_msg
    combined_fp = extraction_dir / "roundtrip_test_combined.csv"
    assert combined_fp.exists()

    combined = pd.read_csv(combined_fp)
    assert set(combined["user_label"]) == {"whatcom-text", "caneadea-pdf"}
    assert set(combined["source_kind"]) == {"text", "pdf"}


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
    result = await process_jurisdictions_with_openai(
        out_dir=out_dir,
        tech="roundtrip-test",
        jurisdiction_fp=jurisdiction_fp,
        known_local_docs=known_local_docs,
        perform_se_search=False,
        perform_website_search=False,
    )

    assert "Number of jurisdictions with extracted data: 2" in result
    assert not (out_dir / COLLECTION_MANIFEST_FILENAME).exists()
    assert (out_dir / "roundtrip_test_combined.csv").exists()
    assert any((out_dir / "jurisdiction_dbs").glob("*.csv"))


@pytest.mark.asyncio
async def test_duplicate_tasks_logs_to_file(tmp_path):
    """Log duplicate LLM tasks to error file"""

    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    with pytest.raises(COMPASSValueError, match="Found duplicated task"):
        _ = await process_jurisdictions_with_openai(
            out_dir=tmp_path / "outputs",
            tech="solar",
            jurisdiction_fp=jurisdiction_fp,
            model=[
                {
                    "name": "gpt-4.1-mini",
                    "tasks": ["default", "date_extraction"],
                },
                {
                    "name": "gpt-4.1",
                    "tasks": [
                        "ordinance_text_extraction",
                        "permitted_use_text_extraction",
                        "date_extraction",
                    ],
                },
            ],
        )

    log_files = list((tmp_path / "outputs" / "logs").glob("*"))
    assert len(log_files) == 1
    assert "Fatal error during processing" not in log_files[0].read_text(
        encoding="utf-8"
    )
    assert "Found duplicated task" in log_files[0].read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_external_exceptions_logged_to_file(tmp_path, monkeypatch):
    """Log external exceptions to error file"""

    def _always_fail(*__, **___):
        raise NotImplementedError("Simulated external error")

    monkeypatch.setattr(
        process_module, "_initialize_model_params", _always_fail
    )

    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    with pytest.raises(NotImplementedError, match="Simulated external error"):
        _ = await process_jurisdictions_with_openai(
            out_dir=tmp_path / "outputs",
            tech="solar",
            jurisdiction_fp=jurisdiction_fp,
        )

    log_files = list((tmp_path / "outputs" / "logs").glob("*"))
    assert len(log_files) == 1

    log_text = log_files[0].read_text(encoding="utf-8")
    assert "Fatal error during processing" in log_text
    assert "Simulated external error" in log_text


@pytest.mark.asyncio
async def test_process_args_logged_at_debug_to_file(
    tmp_path, patched_runner, assert_message_was_logged
):
    """Log function arguments with DEBUG_TO_FILE level"""

    out_dir = tmp_path / "outputs"
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.touch()

    result = await process_jurisdictions_with_openai(
        out_dir=out_dir,
        tech="solar",
        jurisdiction_fp=jurisdiction_fp,
        log_level="DEBUG",
    )

    assert result == f"processed {jurisdiction_fp}"

    assert_message_was_logged(
        "Called 'process_jurisdictions_with_openai' with:",
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
    patched_runner,
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
            await process_jurisdictions_with_openai(
                out_dir=str(out_dir),
                tech="solar",
                jurisdiction_fp=str(jurisdiction_fp),
                log_level="DEBUG",
                known_local_docs=known_local_docs,
                known_doc_urls=known_doc_urls,
                perform_se_search=perform_se_search,
                perform_website_search=perform_website_search,
            )
        return

    result = await process_jurisdictions_with_openai(
        out_dir=str(out_dir),
        tech="solar",
        jurisdiction_fp=str(jurisdiction_fp),
        log_level="DEBUG",
        known_local_docs=known_local_docs,
        known_doc_urls=known_doc_urls,
        perform_se_search=perform_se_search,
        perform_website_search=perform_website_search,
    )

    assert result == f"processed {jurisdiction_fp}"
    assert patched_runner.LAST_MODE_USED == "process"

    assert_message_was_logged(
        "Using the following document acquisition step(s):", log_level="INFO"
    )
    assert_message_was_logged(" -> ".join(expected_steps), log_level="INFO")


# @pytest.mark.asyncio
# async def test_process_mode_collects_and_extracts_each_jurisdiction_in_order(
#     monkeypatch, tmp_path
# ):
#     """Process mode should extract each jurisdiction after its collection"""
#     from compass.utilities import Directories
#     from compass.utilities.logs import LogListener

#     jurisdictions_df = pd.DataFrame(
#         [
#             {
#                 "State": "Colorado",
#                 "County": "Adams",
#                 "Subdivision": None,
#                 "Jurisdiction Type": "county",
#                 "FIPS": 1,
#                 "Website": None,
#             },
#             {
#                 "State": "Colorado",
#                 "County": "Boulder",
#                 "Subdivision": None,
#                 "Jurisdiction Type": "county",
#                 "FIPS": 2,
#                 "Website": None,
#             },
#         ]
#     )
#     events = []

#     async def _collect(  # noqa
#         jurisdiction, known_local_docs=None, known_doc_urls=None
#     ):
#         events.append(("collect", jurisdiction.code))
#         return {
#             "full_name": jurisdiction.full_name,
#             "county": jurisdiction.county,
#             "state": jurisdiction.state,
#             "subdivision": jurisdiction.subdivision_name,
#             "jurisdiction_type": jurisdiction.type,
#             "FIPS": jurisdiction.code,
#             "jurisdiction_website": None,
#             "found": True,
#             "documents": [],
#         }

#     async def _extract(  # noqa
#         jurisdiction, collection_info, usage_tracker=None
#     ):
#         events.append(("extract", jurisdiction.code))
#         return {
#             "jurisdiction": jurisdiction,
#             "ord_db_fp": f"{jurisdiction.code}.csv",
#         }

#     COMPASS_PB.reset()
#     COMPASS_PB.create_main_task(num_jurisdictions=len(jurisdictions_df))

#     with LogListener(["compass"], level="INFO") as ll:
#         runner = _COMPASSRunner(
#             dirs=Directories(tmp_path),
#             log_listener=ll,
#             tech="solar",
#             models={},
#             process_kwargs=ProcessKwargs(
#                 None, None, None, None, None, None, 1
#             ),
#         )

#         monkeypatch.setattr(runner, "_collect_jurisdiction_info", _collect)
#         monkeypatch.setattr(runner, "_extracted_jurisdiction_info", _extract)

#         try:
#             collection_manifest, doc_infos = await runner._process_all(
#                 jurisdictions_df
#             )
#         finally:
#             COMPASS_PB.reset()

#     assert [info["FIPS"] for info in collection_manifest["jurisdictions"]] == [
#         1,
#         2,
#     ]
#     assert [info["ord_db_fp"] for info in doc_infos] == ["1.csv", "2.csv"]
#     assert events == [
#         ("collect", 1),
#         ("extract", 1),
#         ("collect", 2),
#         ("extract", 2),
#     ]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
