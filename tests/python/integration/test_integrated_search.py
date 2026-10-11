"""Integration tests for compass search"""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import compass.scripts.search as search_module
import compass.web.search as web_search_module
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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
