"""Tests for the fixture-driven offline integration harness"""

import json
from pathlib import Path

import pandas as pd
import pytest

from compass._cli.main import main
from compass.pipeline import CollectionRequest, ExtractionRequest
from compass.pipeline.collection.persistence import (
    COLLECTION_MANIFEST_FILENAME,
)
from compass.pipeline.coordinator import run_compass
from compass.plugin.base import BaseExtractionPlugin
from compass.plugin.registry import PLUGIN_REGISTRY, register_plugin
from compass.pb import COMPASS_PB
from compass.utilities.enums import LLMTasks
from integration_harness import OfflineScenario


class _OfflineHarnessPlugin(BaseExtractionPlugin):
    """Plugin that exercises replayed LLM calls over collected documents"""

    IDENTIFIER = "offline-harness"

    async def get_query_templates(self):
        """Return one deterministic search query"""
        return ["ordinance {jurisdiction}"]

    async def get_website_keywords(self):
        """Return deterministic crawl keywords"""
        return {"ordinance": 1}

    async def get_heuristic(self):
        """Return a heuristic that accepts replay documents"""

        class _AcceptAll:
            def check(self, text):
                return bool(text)

        return _AcceptAll()

    async def filter_docs(self, extraction_context, max_num_docs=None):
        """Keep all replay documents"""
        return extraction_context

    async def parse_docs_for_structured_data(self, extraction_context):
        """Extract one replayed result from every document"""
        service = self.model_configs[LLMTasks.DEFAULT].llm_service
        rows = []
        for doc in extraction_context.documents:
            response = await service.call(
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the ordinance identifier.",
                    },
                    {"role": "user", "content": doc.text},
                ]
            )
            row = json.loads(response)
            row["source"] = doc.attrs["source"]
            rows.append(row)
            await extraction_context.mark_doc_as_data_source(doc)

        extraction_context.attrs["structured_data"] = pd.DataFrame(rows)
        extraction_context.attrs["out_data_fn"] = "offline_harness.csv"
        return extraction_context

    @classmethod
    def save_structured_data(cls, doc_infos, out_dir):
        """Combine jurisdiction outputs"""
        frames = [
            pd.read_csv(info["ord_db_fp"])
            for info in doc_infos
            if info["ord_db_fp"] is not None
        ]
        if not frames:
            return 0
        pd.concat(frames, ignore_index=True).to_csv(
            Path(out_dir) / "offline_harness_combined.csv", index=False
        )
        return len(frames)


@pytest.fixture(autouse=True)
def reset_compass_pb():
    """Reset progress state around every harness test"""
    COMPASS_PB.reset()
    yield
    COMPASS_PB.reset()


@pytest.fixture
def offline_harness_plugin():
    """Register the harness plugin for one test"""
    plugin_id = _OfflineHarnessPlugin.IDENTIFIER.casefold()
    already_registered = plugin_id in PLUGIN_REGISTRY
    if not already_registered:
        register_plugin(_OfflineHarnessPlugin)
    yield
    if not already_registered:
        PLUGIN_REGISTRY.pop(plugin_id, None)


@pytest.fixture
def offline_scenario(monkeypatch, tmp_path, test_data_dir):
    """Install the external service replay scenario"""
    scenario = OfflineScenario.from_file(
        test_data_dir / "integration" / "offline_scenario.json",
        tmp_path / "replay_cache",
    )
    scenario.install(monkeypatch)
    return scenario


@pytest.mark.asyncio
async def test_offline_collection_and_extraction_harness(
    tmp_path, offline_harness_plugin, offline_scenario
):
    """Replay web acquisition and LLM extraction without external calls"""
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.write_text(
        "State,County,Subdivision,Jurisdiction Type,FIPS,Website\n"
        "Washington,Whatcom,,county,53073,https://example.test\n",
        encoding="utf-8",
    )
    known_doc_urls = {
        "53073": [
            {"source": item["source"]}
            for item in offline_scenario.config["known_urls"]
        ]
    }
    collection_dir = tmp_path / "collection"

    collection_message = await run_compass(
        CollectionRequest(
            out_dir=collection_dir,
            tech="offline-harness",
            jurisdiction_fp=jurisdiction_fp,
            known_doc_urls=known_doc_urls,
            perform_se_search=True,
            perform_website_search=True,
            make_paths_relative=True,
        )
    )

    assert "4 documents collected for 1 jurisdiction" in collection_message
    manifest_fp = collection_dir / COLLECTION_MANIFEST_FILENAME
    manifest = json.loads(manifest_fp.read_text(encoding="utf-8"))
    assert manifest["completed_step_document_totals"] == {
        "known_doc_urls": 1,
        "search_engine": 1,
        "website_search_elm": 1,
        "website_search_compass": 1,
    }
    assert {
        step
        for document in manifest["jurisdictions"][0]["documents"]
        for step in document["from_steps"]
    } == {
        "known_doc_urls",
        "search_engine",
        "website_search_elm",
        "website_search_compass",
    }

    COMPASS_PB.reset()
    extraction_dir = tmp_path / "extraction"
    extraction_message = await run_compass(
        ExtractionRequest(
            out_dir=extraction_dir,
            tech="offline-harness",
            jurisdiction_fp=jurisdiction_fp,
            collection_manifest_fp=manifest_fp,
            model="offline-replay",
        )
    )

    assert (
        "Number of jurisdictions with extracted data: 1"
        in extraction_message
    )
    output = pd.read_csv(extraction_dir / "offline_harness_combined.csv")
    assert set(output["ordinance_id"]) == {
        "known-url",
        "search",
        "elm-crawl",
        "compass-crawl",
    }
    offline_scenario.assert_consumed()


def test_offline_process_cli_end_to_end(
    tmp_path, monkeypatch, cli_runner, offline_harness_plugin
):
    """Run the process CLI from configuration to structured output"""
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.write_text(
        "State,County,Subdivision,Jurisdiction Type,FIPS,Website\n"
        "Washington,Whatcom,,county,53073,\n",
        encoding="utf-8",
    )
    source = "https://documents.test/process-ordinance.html"
    scenario = OfflineScenario(
        {
            "known_urls": [
                {
                    "source": source,
                    "content": (
                        "CLI process ordinance with identifier cli-process."
                    ),
                }
            ],
            "llm_responses": [
                {
                    "prompt_contains": "CLI process ordinance",
                    "response": '{"ordinance_id": "cli-process"}',
                }
            ],
        },
        tmp_path / "cli_replay_cache",
    )
    scenario.install(monkeypatch)

    out_dir = tmp_path / "cli_output"
    config_fp = tmp_path / "process_config.json"
    config_fp.write_text(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "tech": "offline-harness",
                "jurisdiction_fp": str(jurisdiction_fp),
                "known_doc_urls": {"53073": [{"source": source}]},
                "perform_se_search": False,
                "perform_website_search": False,
                "model": "offline-replay",
            }
        ),
        encoding="utf-8",
    )

    result = cli_runner.invoke(
        main,
        [
            "process",
            "--config",
            str(config_fp),
            "--no-progress",
            "--out-dir-exists",
            "fail",
        ],
    )

    assert result.exit_code == 0, result.output
    output = pd.read_csv(out_dir / "offline_harness_combined.csv")
    assert output.to_dict(orient="records") == [
        {"ordinance_id": "cli-process", "source": source}
    ]
    assert (out_dir / "jurisdictions.json").exists()
    assert (out_dir / "usage.json").exists()
    scenario.assert_consumed()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
