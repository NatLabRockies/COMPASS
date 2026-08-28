"""Tests for compass.scripts.search"""

from pathlib import Path

import pytest

import compass.web.search as search_module
from compass.utilities.jurisdictions import Jurisdiction


@pytest.mark.asyncio
async def test_search_formats_query_with_known_jurisdiction_website(
    monkeypatch,
):
    """Include website query when the jurisdiction website is known"""
    submitted_queries = None

    async def capture_queries(queries, *args, **kwargs):
        nonlocal submitted_queries
        submitted_queries = queries
        return []

    monkeypatch.setattr(
        search_module, "_run_simple_sort_search", capture_queries
    )
    jurisdiction = Jurisdiction(
        "county",
        "Colorado",
        county="Adams",
        website_url="https://www.adcogov.org",
    )
    query_templates = [
        "{jurisdiction} zoning ordinance",
        "site:{jurisdiction_website} zoning ordinance",
    ]

    output = await search_module.search_single_jurisdiction(
        query_templates, jurisdiction
    )

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
    submitted_queries = None

    async def capture_queries(queries, *args, **kwargs):
        nonlocal submitted_queries
        submitted_queries = queries
        return []

    monkeypatch.setattr(
        search_module, "_run_simple_sort_search", capture_queries
    )
    jurisdiction = Jurisdiction("county", "Colorado", county="Adams")
    query_templates = [
        "{jurisdiction} zoning ordinance",
        "site:{jurisdiction_website} zoning ordinance",
    ]

    output = await search_module.search_single_jurisdiction(
        query_templates, jurisdiction
    )

    assert submitted_queries == ["Adams County, Colorado zoning ordinance"]
    assert output["queries"] == submitted_queries


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


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
