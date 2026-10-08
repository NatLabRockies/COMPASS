"""Tests for compass.scripts.search"""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import compass.scripts.search as search_module
import compass.web.search as web_search_module
from compass.pipeline.data_classes import CollectionRequest


@pytest.mark.asyncio
@pytest.mark.parametrize("with_results", [True, False])
async def test_run_search_report_metadata(tmp_path, monkeypatch, with_results):
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
    config_path = Path("config.json") if with_results else None
    if config_path is not None:
        config_path.write_text("{}", encoding="utf-8")
    request = CollectionRequest(
        out_dir=tmp_path / "output",
        tech="wind",
        jurisdiction_fp=jurisdiction_fp,
        num_urls_to_check_per_jurisdiction=7,
        search_engines=[
            {"se_name": "PlaywrightGoogleLinkSearch"},
            {"se_name": "APIDuckDuckGoSearch"},
        ],
    )
    query_templates = (
        await search_module.PipelineRuntime(request)
        .extractor_class(None, None)
        .get_query_templates()
    )
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
    report = await search_module.run_search_from_request(
        request, config_path=config_path
    )
    after = datetime.now(UTC)

    assert search_backend.await_count == 4
    assert report["tech"] == "wind"
    assert report["versions"] == {
        "compass": search_module.compass_version,
        "elm": search_module.elm_version,
    }
    assert report["config_path"] == (
        str(config_path.resolve()) if config_path is not None else None
    )
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
    assert json.loads(json.dumps(report)) == report


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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
