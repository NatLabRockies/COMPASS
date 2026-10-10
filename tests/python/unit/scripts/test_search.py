"""Tests for compass.scripts.search"""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

import compass.scripts.search as search_module
import compass.web.search as web_search_module
from compass.exceptions import COMPASSFileNotFoundError, COMPASSValueError
from compass.pipeline.runtime import PipelineRuntime
from compass.pipeline.data_classes import SearchRequest
from compass.pipeline.coordinator import COMPASSSearch
from compass.pb import COMPASS_PB
from compass.utilities.jurisdictions import load_jurisdictions_from_fp


@pytest.mark.asyncio
@pytest.mark.parametrize("with_results", [True, False])
async def test_search_report_metadata(tmp_path, monkeypatch, with_results):
    """Report run metadata and aggregate successful, empty, failed searches"""
    jurisdiction_fp = tmp_path / "jurisdictions.csv"
    jurisdiction_fp.write_text(
        "County,State\n"
        "Adams,Colorado\n"
        "Boulder,Colorado\n"
        "Denver,Colorado\n"
        "Jefferson,Colorado\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    request = SearchRequest(
        out_dir=tmp_path / "output",
        tech="wind",
        jurisdiction_fp=jurisdiction_fp,
        num_urls_to_check_per_jurisdiction=7,
        search_engines=[
            {"se_name": "PlaywrightGoogleLinkSearch"},
            {"se_name": "APIDuckDuckGoSearch"},
        ],
    )
    runtime = PipelineRuntime(request)
    query_templates = await runtime.extractor_class(
        None, None
    ).get_query_templates()
    first_results = [
        {"url": "https://example.com/a", "search_engine": "Google"},
        {"url": "https://example.com/b", "search_engine": "Bing"},
    ]
    second_results = [
        {"url": "https://example.com/c", "search_engine": "Google"},
        {"url": "https://example.com/d", "search_engine": "Google"},
        {"url": "https://example.com/e", "search_engine": "Bing"},
        {"url": "https://example.com/f", "search_engine": "Bing"},
    ]
    search_backend = AsyncMock(
        side_effect=[
            first_results if with_results else [],
            second_results if with_results else [],
            [],
            RuntimeError("Search unavailable"),
        ]
    )
    monkeypatch.setattr(
        web_search_module, "_run_holistic_sort_search", search_backend
    )

    before = datetime.now(UTC)
    jurisdictions_df = load_jurisdictions_from_fp(request.jurisdiction_fp)
    COMPASS_PB.reset()
    COMPASS_PB.create_main_task(num_jurisdictions=len(jurisdictions_df))
    try:
        async with runtime:
            await COMPASSSearch(runtime).run(jurisdictions_df)
    finally:
        COMPASS_PB.reset()
    report = json.loads(
        (
            runtime.dirs.out / search_module.SEARCH_RESULT_MANIFEST_FILENAME
        ).read_text(encoding="utf-8")
    )
    after = datetime.now(UTC)

    assert search_backend.await_count == 4
    assert report["tech"] == "wind"
    assert report["versions"] == {
        "compass": search_module.compass_version,
        "elm": search_module.elm_version,
    }
    assert report["num_urls_requested"] == 7
    assert report["search_engines"] == [
        "PlaywrightGoogleLinkSearch",
        "APIDuckDuckGoSearch",
    ]
    assert report["query_templates"] == list(query_templates)
    start = datetime.fromisoformat(report["time_start_utc"])
    end = datetime.fromisoformat(report["time_end_utc"])
    assert start.tzinfo == end.tzinfo == UTC
    assert before <= start <= end <= after
    assert report["total_time"] == (end - start).total_seconds()
    assert report["total_time_string"] == str(end - start)
    assert report["num_jurisdictions_searched"] == 4
    assert report["num_jurisdictions_found"] == (2 if with_results else 0)
    assert report["search_engine_totals"] == (
        {"Google": 3, "Bing": 3} if with_results else {}
    )
    assert report["result_stats"] == {
        "min": 0,
        "max": 4 if with_results else 0,
        "median": 1 if with_results else 0,
        "total": 6 if with_results else 0,
    }
    assert [jur["num_results"] for jur in report["jurisdictions"]] == (
        [2, 4, 0, 0] if with_results else [0, 0, 0, 0]
    )
    assert report["jurisdictions"][-1]["error"] == (
        "RuntimeError: Search unavailable"
    )
    shards = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in runtime.dirs.se_shards.glob("*.json")
    ]
    assert {shard["FIPS"]: shard for shard in shards} == {
        result["FIPS"]: result for result in report["jurisdictions"]
    }
    location_logs = list(runtime.dirs.logs.glob("*.log"))
    assert len(location_logs) >= 4
    for path in location_logs:
        if path.name == "main.log":
            continue
        text = path.read_text(encoding="utf-8")
        assert "Kicking off search for jurisdiction:" in text
        assert "Completed search for" in text
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize("jur_results", [[], [None]])
async def test_build_search_report_without_results(tmp_path, jur_results):
    """Aggregate empty runs and failed workflows without invalid records"""
    runtime = PipelineRuntime(SearchRequest(tmp_path, "wind", None))
    report = await search_module.build_search_report(
        runtime, jur_results, datetime.now(UTC)
    )
    assert report["num_jurisdictions_searched"] == len(jur_results)
    assert report["num_jurisdictions_found"] == 0
    assert report["jurisdictions"] == []
    assert report["search_engine_totals"] == {}
    assert (
        report["result_stats"]
        == report["filtered_result_stats"]
        == {
            "min": 0,
            "max": 0,
            "median": 0,
            "total": 0,
        }
    )


def test_summary_keeps_only_unfiltered_and_sorted():
    """Render only unfiltered rows sorted by overall rank"""
    report = {
        "tech": "wind",
        "timestamp": "2026-01-01T00:00:00Z",
        "num_urls_requested": 2,
        "jurisdictions": [
            {
                "jurisdiction": "Example County, Test",
                "error": None,
                "results": [
                    {
                        "overall_rank": 2,
                        "query_rank": 1,
                        "search_engine": "A",
                        "query": "q2",
                        "url": "https://example.com/rank2",
                        "filtered_reason": None,
                    },
                    {
                        "overall_rank": 1,
                        "query_rank": 1,
                        "search_engine": "A",
                        "query": "q1",
                        "url": "https://example.com/rank1",
                        "filtered_reason": None,
                    },
                    {
                        "overall_rank": None,
                        "query_rank": 2,
                        "search_engine": "A",
                        "query": "q-dup",
                        "url": "https://example.com/dup",
                        "filtered_reason": "duplicate",
                    },
                ],
            }
        ],
    }

    output = search_module.summary(report)

    first_rank = output.index("[1]")
    second_rank = output.index("[2]")
    assert first_rank < second_rank
    assert "https://example.com/rank1" in output
    assert "https://example.com/rank2" in output
    assert "https://example.com/dup" not in output


@pytest.mark.parametrize(
    "source", ["manifest", "run", "shards", "glob", "current"]
)
def test_load_search_results(tmp_path, monkeypatch, source):
    """Load aggregates and existing shards without dropping empty records"""
    records = [
        {"FIPS": "08001", "full_name": "Adams", "results": []},
        {
            "FIPS": "08013",
            "full_name": "Boulder",
            "results": [{"url": "https://example.com", "overall_rank": 1}],
        },
    ]
    shard_dir = tmp_path / "se_results"
    for record in records:
        web_search_module.write_search_result_shard(
            shard_dir, record, SimpleNamespace(full_name=record["full_name"])
        )
    manifest = tmp_path / search_module.SEARCH_RESULT_MANIFEST_FILENAME
    search_module.write_search_report(
        {"tech": "wind", "jurisdictions": records}, manifest
    )
    inputs = {
        "manifest": manifest,
        "run": tmp_path,
        "shards": shard_dir,
        "glob": str(shard_dir / "*.json"),
        "current": ".",
    }
    monkeypatch.chdir(tmp_path)
    loaded = search_module.load_search_result_jurisdictions(
        inputs[source], "wind"
    )
    assert loaded == {record["FIPS"]: record for record in records}
    manifest.unlink()
    assert (
        search_module.load_search_result_jurisdictions(tmp_path, "wind")
        == loaded
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"tech": "solar", "jurisdictions": []}, "technology"),
        ({"FIPS": None, "results": []}, "FIPS"),
        ({"FIPS": "08001", "results": [{}]}, "Invalid search results"),
        (
            {
                "FIPS": "08001",
                "results": [
                    {"url": "https://example.com", "overall_rank": "first"}
                ],
            },
            "Invalid search results",
        ),
        ({"jurisdictions": {}}, "jurisdictions"),
        (
            {"jurisdictions": [{"FIPS": "08001", "results": []}] * 2},
            "Duplicate",
        ),
    ],
)
def test_reject_invalid_search_results(tmp_path, payload, message):
    """Reject invalid inputs before collecting any documents"""
    manifest = tmp_path / "search.json"
    search_module.write_search_report(payload, manifest)
    with pytest.raises(COMPASSValueError, match=message):
        search_module.load_search_result_jurisdictions(manifest, "wind")


def test_missing_search_results(tmp_path):
    """Reject nonexistent inputs and empty shard directories"""
    for path in (tmp_path / "missing.json", tmp_path):
        with pytest.raises(COMPASSFileNotFoundError):
            search_module.load_search_result_jurisdictions(path, "wind")


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
