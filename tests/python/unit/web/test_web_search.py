"""Tests for compass.pipeline.search.persistence"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest
from elm.web.search.run import SEARCH_ENGINE_OPTIONS

import compass.web.search as search_module
from compass.utilities.event_log import EventLog, use_event_log
from compass.utilities.jurisdictions import Jurisdiction


_JURISDICTION_QUERY_TEMPLATE = "{jurisdiction} zoning ordinance"
_WEBSITE_QUERY_TEMPLATE = "site:{jurisdiction_website} zoning ordinance"


@pytest.mark.asyncio
@pytest.mark.parametrize("simple", [True, False])
@pytest.mark.parametrize("outcome", ["results", "filtered", "empty", "error"])
async def test_search_returns_jurisdiction_and_result_metadata(
    monkeypatch, simple, outcome
):
    """Return jurisdiction metadata and counts on success or failure"""
    results = (
        [
            {"url": "https://example.com/a", "search_engine": "Google"},
            {"url": "https://example.com/b", "search_engine": "Google"},
            {
                "url": "https://example.com/c",
                "search_engine": "Bing",
                "filtered_reason": "beyond_top_n"
                if outcome == "filtered"
                else None,
            },
        ]
        if outcome in {"results", "filtered"}
        else []
    )
    search_backend = AsyncMock(
        return_value=results,
        side_effect=RuntimeError("Search unavailable")
        if outcome == "error"
        else None,
    )
    monkeypatch.setattr(
        search_module,
        "_run_simple_sort_search" if simple else "_run_holistic_sort_search",
        search_backend,
    )
    jurisdiction = Jurisdiction(
        "city",
        "Colorado",
        county="Adams",
        subdivision_name="Thornton",
        code="0877290",
        website_url="https://www.thorntonco.gov/government",
    )

    output = await search_module.search_single_jurisdiction(
        [_JURISDICTION_QUERY_TEMPLATE], jurisdiction, simple=simple
    )

    search_backend.assert_awaited_once()
    assert output == {
        "full_name": "City of Thornton, Adams County, Colorado",
        "state": "Colorado",
        "county": "Adams",
        "subdivision": "Thornton",
        "jurisdiction_type": "City",
        "FIPS": "0877290",
        "jurisdiction_website": "https://www.thorntonco.gov/government",
        "queries": [
            "City of Thornton, Adams County, Colorado zoning ordinance"
        ],
        "num_results": len(results),
        "num_kept_results": len(results) - (outcome == "filtered"),
        "search_engine_counts": {"Google": 2, "Bing": 1}
        if outcome in {"results", "filtered"}
        else {},
        "results": results,
        "error": "RuntimeError: Search unavailable"
        if outcome == "error"
        else None,
    }


@pytest.mark.asyncio
async def test_search_formats_query_with_known_jurisdiction_website(
    monkeypatch,
):
    """Include website query when the jurisdiction website is known"""
    search_backend = AsyncMock(return_value=[])
    monkeypatch.setattr(
        search_module, "_run_simple_sort_search", search_backend
    )
    jurisdiction = Jurisdiction(
        "county",
        "Colorado",
        county="Adams",
        website_url="https://www.adcogov.org/government?lang=en#offices",
    )
    query_templates = [
        _JURISDICTION_QUERY_TEMPLATE,
        _WEBSITE_QUERY_TEMPLATE,
    ]

    output = await search_module.search_single_jurisdiction(
        query_templates, jurisdiction
    )

    submitted_queries = search_backend.await_args.args[0]
    assert submitted_queries == [
        "Adams County, Colorado zoning ordinance",
        "site:https://www.adcogov.org zoning ordinance",
    ]
    assert output["queries"] == submitted_queries


@pytest.mark.asyncio
async def test_search_discards_website_query_without_known_website(
    monkeypatch,
):
    """Discard website query when the jurisdiction website is unknown"""
    search_backend = AsyncMock(return_value=[])
    monkeypatch.setattr(
        search_module, "_run_simple_sort_search", search_backend
    )
    jurisdiction = Jurisdiction("county", "Colorado", county="Adams")
    query_templates = [
        _JURISDICTION_QUERY_TEMPLATE,
        _WEBSITE_QUERY_TEMPLATE,
    ]

    output = await search_module.search_single_jurisdiction(
        query_templates, jurisdiction
    )

    submitted_queries = search_backend.await_args.args[0]
    assert submitted_queries == ["Adams County, Colorado zoning ordinance"]
    assert output["queries"] == submitted_queries


@pytest.mark.asyncio
async def test_search_records_the_queries_it_runs(monkeypatch, tmp_path):
    """The formatted queries are logged before the search is made"""
    monkeypatch.setattr(
        search_module, "_run_simple_sort_search", AsyncMock(return_value=[])
    )
    jurisdiction = Jurisdiction("county", "Colorado", county="Adams")
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        await search_module.search_single_jurisdiction(
            [_JURISDICTION_QUERY_TEMPLATE], jurisdiction, num_urls=7
        )
    (event,) = [json.loads(x) for x in log.path.read_text().splitlines()]
    assert event["type"] == "search_queries"
    assert event["queries"] == ["Adams County, Colorado zoning ordinance"]
    assert event["num_urls"] == 7
    assert event["tags"]["step"] == "collect.search_engine.queries"


@pytest.mark.asyncio
async def test_search_skips_backend_when_no_queries_remain(monkeypatch):
    """Return an empty result when all query templates are discarded"""
    search_backend = AsyncMock()
    monkeypatch.setattr(
        search_module, "_run_simple_sort_search", search_backend
    )
    jurisdiction = Jurisdiction("county", "Colorado", county="Adams")

    output = await search_module.search_single_jurisdiction(
        [_WEBSITE_QUERY_TEMPLATE], jurisdiction
    )

    search_backend.assert_not_awaited()
    assert output == {
        "full_name": "Adams County, Colorado",
        "state": "Colorado",
        "county": "Adams",
        "subdivision": None,
        "jurisdiction_type": "County",
        "FIPS": None,
        "jurisdiction_website": None,
        "queries": [],
        "num_results": 0,
        "num_kept_results": 0,
        "search_engine_counts": {},
        "results": [],
        "error": None,
    }


@pytest.mark.asyncio
async def test_search_formats_bare_jurisdiction_website(monkeypatch):
    """Strip paths from jurisdiction websites without a URL scheme"""
    search_backend = AsyncMock(return_value=[])
    monkeypatch.setattr(
        search_module, "_run_simple_sort_search", search_backend
    )
    jurisdiction = Jurisdiction(
        "county",
        "Colorado",
        county="Adams",
        website_url="www.adcogov.org/government?lang=en#offices",
    )

    await search_module.search_single_jurisdiction(
        [_WEBSITE_QUERY_TEMPLATE], jurisdiction
    )

    submitted_queries = search_backend.await_args.args[0]
    assert submitted_queries == ["site:www.adcogov.org zoning ordinance"]


@pytest.mark.asyncio
async def test_run_simple_sort_search_preserves_search_engine_attributes(
    monkeypatch,
):
    """Simple search should retain Elm's engine metadata"""
    expected = [
        {
            "url": "https://example.com/ordinance.pdf",
            "search_engines": ["SerpAPIGoogleSearch"],
        }
    ]

    async def fake_search_with_fallback_with_attrs(  # ruff:ignore[unused-async]
        queries, **kwargs
    ):
        assert queries == ["Example, CO ordinance"]
        assert kwargs == {
            "num_urls": 5,
            "url_ignore_substrings": ["ignore"],
            "url_keep_substrings": ["keep"],
            "browser_semaphore": "semaphore",
            "task_name": "Example, CO",
        }
        return expected

    monkeypatch.setattr(
        search_module,
        "search_with_fallback_with_attrs",
        fake_search_with_fallback_with_attrs,
    )

    result = await search_module._run_simple_sort_search(
        ["Example, CO ordinance"],
        5,
        ["ignore"],
        ["keep"],
        "semaphore",
        "Example, CO",
    )

    assert result == expected
    assert result[0]["search_engines"] == ["SerpAPIGoogleSearch"]


def test_apply_blacklist_filters_is_case_insensitive():
    """Blacklist should match URL substrings regardless of case"""
    results = [
        {
            "url": "https://EXAMPLE.com/WIKIPEDIA.org/page",
            "query": "q1",
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 1,
            "overall_rank": None,
            "filtered_reason": None,
        },
        {
            "url": "https://example.com/keep",
            "query": "q1",
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 2,
            "overall_rank": None,
            "filtered_reason": None,
        },
    ]

    search_module._apply_blacklist_filters(results, ["wikipedia.org"], None)

    assert results[0]["filtered_reason"] == "blacklist:wikipedia.org"
    assert results[1]["filtered_reason"] is None


def test_apply_blacklist_filters_honors_whitelist_and_first_match():
    """Whitelist matches should override the first matching blacklist part"""
    results = [
        {
            "url": "https://example.com/drop",
            "filtered_reason": None,
        },
        {
            "url": "https://example.com/trusted/drop",
            "filtered_reason": None,
        },
    ]

    search_module._apply_blacklist_filters(
        results,
        ["", "EXAMPLE.COM", "drop"],
        ["TRUSTED"],
    )

    assert results[0]["filtered_reason"] == "blacklist:example.com"
    assert results[1]["filtered_reason"] is None


def test_apply_duplicate_filters_keeps_best_and_tracks_duplicates():
    """Keep best duplicate candidate and track collapsed rows"""
    results = [
        {
            "url": "https://example.com/a.pdf",
            "query": "q1",
            "query_index": 0,
            "se_order": 0,
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 2,
            "overall_rank": None,
            "filtered_reason": None,
            "_order": 0,
        },
        {
            "url": "https://example.com/a.pdf",
            "query": "q2",
            "query_index": 1,
            "se_order": 0,
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 1,
            "overall_rank": None,
            "filtered_reason": None,
            "_order": 1,
        },
    ]

    search_module._apply_duplicate_filters(results)

    winner = results[1]
    loser = results[0]

    assert winner["filtered_reason"] is None
    assert winner["duplicates"] == [
        {
            "url": "https://example.com/a.pdf",
            "query": "q1",
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 2,
        }
    ]
    assert winner["search_engines"] == ["SerpAPIGoogleSearch"]
    assert loser["filtered_reason"] == "duplicate"


def test_apply_duplicate_filters_collapses_across_search_engines():
    """Same URL from different engines should be marked duplicate

    When ``query_rank`` and ``query_index`` tie, the entry from the
    search engine listed first in the config (lower ``se_order``)
    wins.
    """
    results = [
        {
            "url": "https://example.com/a.pdf",
            "query": "q1",
            "query_index": 0,
            "se_order": 1,
            "search_engine": "TestSearch",
            "query_rank": 1,
            "overall_rank": None,
            "filtered_reason": None,
            "_order": 0,
        },
        {
            "url": "https://example.com/a.pdf",
            "query": "q1",
            "query_index": 0,
            "se_order": 0,
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 1,
            "overall_rank": None,
            "filtered_reason": None,
            "_order": 1,
        },
    ]

    search_module._apply_duplicate_filters(results)

    winner = results[1]
    loser = results[0]

    assert winner["filtered_reason"] is None
    assert winner["search_engine"] == "SerpAPIGoogleSearch"
    assert winner["search_engines"] == [
        "SerpAPIGoogleSearch",
        "TestSearch",
    ]
    assert winner["duplicates"] == [
        {
            "url": "https://example.com/a.pdf",
            "query": "q1",
            "search_engine": "TestSearch",
            "query_rank": 1,
        }
    ]
    assert loser["filtered_reason"] == "duplicate"


def test_apply_top_n_filters_assigns_overall_rank_and_beyond_top_n():
    """Assign overall rank and mark entries beyond requested top-N"""
    results = [
        {
            "url": "https://example.com/1",
            "query": "q1",
            "query_index": 0,
            "se_order": 0,
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 1,
            "overall_rank": None,
            "filtered_reason": None,
            "_order": 0,
        },
        {
            "url": "https://example.com/2",
            "query": "q2",
            "query_index": 1,
            "se_order": 0,
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 1,
            "overall_rank": None,
            "filtered_reason": None,
            "_order": 1,
        },
        {
            "url": "https://example.com/3",
            "query": "q3",
            "query_index": 2,
            "se_order": 0,
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 2,
            "overall_rank": None,
            "filtered_reason": None,
            "_order": 2,
        },
    ]

    search_module._apply_top_n_filters(results, num_urls=2)

    assert results[0]["overall_rank"] == 1
    assert results[1]["overall_rank"] == 2
    assert results[2]["overall_rank"] == 3
    assert results[2]["filtered_reason"] == "beyond_top_n"


def test_apply_top_n_filters_prioritizes_more_duplicates_on_tie():
    """Rank tied rows by descending number of duplicates"""
    results = [
        {
            "url": "https://example.com/dup-winner",
            "query": "q-dup-1",
            "query_index": 5,
            "se_order": 0,
            "search_engine": "ZEngine",
            "query_rank": 1,
            "overall_rank": None,
            "filtered_reason": None,
            "_order": 0,
        },
        {
            "url": "https://example.com/dup-winner",
            "query": "q-dup-2",
            "query_index": 6,
            "se_order": 0,
            "search_engine": "ZEngine",
            "query_rank": 2,
            "overall_rank": None,
            "filtered_reason": None,
            "_order": 1,
        },
        {
            "url": "https://example.com/no-dup",
            "query": "q-other",
            "query_index": 0,
            "se_order": 0,
            "search_engine": "AEngine",
            "query_rank": 1,
            "overall_rank": None,
            "filtered_reason": None,
            "_order": 2,
        },
    ]

    search_module._apply_duplicate_filters(results)
    search_module._apply_top_n_filters(results, num_urls=1)

    assert results[0]["overall_rank"] == 1
    assert len(results[0]["duplicates"]) == 1
    assert results[2]["overall_rank"] == 2
    assert results[2]["filtered_reason"] == "beyond_top_n"


def test_apply_filters_orders_phases_and_cleans_internal_fields():
    """Apply blacklist, duplicate, and top-N in deterministic order"""
    results = [
        [
            [
                {
                    "url": "https://site.com/wiki",
                    "query": "q-blacklist",
                    "query_index": 0,
                    "se_order": 0,
                    "search_engine": "SerpAPIGoogleSearch",
                    "query_rank": 1,
                    "overall_rank": None,
                    "filtered_reason": None,
                },
                {
                    "url": "https://example.com/dup",
                    "query": "q-dup-2",
                    "query_index": 0,
                    "se_order": 0,
                    "search_engine": "SerpAPIGoogleSearch",
                    "query_rank": 2,
                    "overall_rank": None,
                    "filtered_reason": None,
                },
                {
                    "url": "https://example.com/dup",
                    "query": "q-dup-1",
                    "query_index": 1,
                    "se_order": 0,
                    "search_engine": "SerpAPIGoogleSearch",
                    "query_rank": 1,
                    "overall_rank": None,
                    "filtered_reason": None,
                },
                {
                    "url": "https://example.com/other",
                    "query": "q-other",
                    "query_index": 2,
                    "se_order": 0,
                    "search_engine": "SerpAPIGoogleSearch",
                    "query_rank": 1,
                    "overall_rank": None,
                    "filtered_reason": None,
                },
            ]
        ]
    ]

    output = search_module._apply_filters(
        results,
        url_blacklist=["WIKI"],
        url_whitelist=None,
        num_urls=1,
    )

    assert output[2]["filtered_reason"].startswith("blacklist:")
    assert output[3]["filtered_reason"] == "duplicate"
    assert output[0]["filtered_reason"] is None
    assert output[0]["overall_rank"] == 1
    assert output[0]["duplicates"][0]["query"] == "q-dup-2"
    assert output[1]["filtered_reason"] == "beyond_top_n"
    assert output[1]["overall_rank"] == 2

    for row in output:
        assert "_order" not in row
        assert "se_order" not in row
        assert "query_index" not in row


class _FakeEngine:
    """Search engine stand-in with ELM's ``_search`` signature"""

    _SE_NAME = "Fake Engine"

    def __init__(self, fail=False):
        self.api_key = "secret-key"
        self.verify = False
        self.fail = fail

    async def _search(self, query, num_results=10, raw=False):
        if self.fail:
            raise httpx.ReadTimeout("timed out")
        return [f"https://example.com/{query}"]


async def test_search_calls_are_recorded(tmp_path):
    """Requests, results, and timeouts appear in the event log"""
    recorded = search_module._recorded_search(_FakeEngine)
    log = EventLog(tmp_path / "events.jsonl")
    with use_event_log(log):
        await recorded(_FakeEngine(), "zoning")
        with pytest.raises(httpx.ReadTimeout):
            await recorded(_FakeEngine(fail=True), "zoning")
    events = [json.loads(x) for x in log.path.read_text().splitlines()]
    assert [e["type"] for e in events] == [
        "search_request",
        "search_response",
        "search_request",
        "search_response",
    ]
    assert events[0]["engine"] == "Fake Engine"
    assert events[0]["query"] == "zoning"
    assert events[0]["parameters"] == {"verify": False, "fail": False}
    assert events[1]["results"] == ["https://example.com/zoning"]
    assert events[3]["error"] == "ReadTimeout: timed out"
    assert events[3]["call"] == events[2]["call"]


def test_registered_engines_record_calls():
    """ELM's registry resolves to the recording engine classes"""
    option = SEARCH_ENGINE_OPTIONS["SerpAPIGoogleSearch"]
    assert option.se_class._records_events
    assert option.se_class.__name__ == "SerpAPIGoogleSearch"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
