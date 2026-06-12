"""Tests for compass.pipeline orchestration"""

import json
from pathlib import Path

import pandas as pd
import pytest

import compass.pipeline.data_classes as data_classes_module
from compass.pipeline import (
    CollectionRequest,
    ExtractionRequest,
    ProcessRequest,
)
from compass.pipeline.collection.persistence import (
    COLLECTION_MANIFEST_FILENAME,
)
from compass.pipeline.coordinator import run_compass
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

    assert whatcom["documents"][0]["source_fp"] is not None
    assert Path(whatcom["documents"][0]["parsed_fp"]).exists()
    assert whatcom["documents"][0]["from_steps"] == ["known_local_docs"]

    assert Path(caneadea["documents"][0]["source_fp"]).exists()
    assert Path(caneadea["documents"][0]["parsed_fp"]).exists()
    assert caneadea["documents"][0]["is_pdf"] is True
    assert whatcom in shard_payloads
    assert caneadea in shard_payloads

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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
