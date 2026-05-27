"""Tests for compass.scripts.search_only"""

from pathlib import Path

import pytest

import compass.scripts.search_only as search_only_module
from compass.utilities.base import WebSearchParams


def test_resolve_search_engines_uses_defaults_when_not_configured():
    """Use module defaults when search engines are not configured"""
    wsp = WebSearchParams(search_engines=None)

    se_names, init_kwargs = search_only_module._resolve_search_engines(wsp)

    assert se_names == list(search_only_module._DEFAULT_SEARCH_ENGINES)
    assert set(init_kwargs) == set(se_names)


def test_resolve_search_engines_uses_custom_order_and_kwargs():
    """Preserve configured order and map init kwargs per engine"""
    wsp = WebSearchParams(
        search_engines=[
            {
                "se_name": "DuxDistributedGlobalSearch",
                "region": "us-en",
            },
            {
                "se_name": "PlaywrightGoogleLinkSearch",
                "headless": True,
            },
        ]
    )

    se_names, init_kwargs = search_only_module._resolve_search_engines(wsp)

    assert se_names == [
        "DuxDistributedGlobalSearch",
        "PlaywrightGoogleLinkSearch",
    ]
    assert init_kwargs["DuxDistributedGlobalSearch"] == {
        "region": "us-en"
    }
    assert init_kwargs["PlaywrightGoogleLinkSearch"] == {
        "headless": True
    }


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

    search_only_module._apply_blacklist_filters(results, ["wikipedia.org"])

    assert results[0]["filtered_reason"] == "blacklist:wikipedia.org"
    assert results[1]["filtered_reason"] is None


def test_apply_duplicate_filters_keeps_best_and_tracks_duplicates():
    """Keep best duplicate candidate and track collapsed rows"""
    results = [
        {
            "url": "https://example.com/a.pdf",
            "query": "q1",
            "query_index": 0,
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
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 1,
            "overall_rank": None,
            "filtered_reason": None,
            "_order": 1,
        },
    ]

    search_only_module._apply_duplicate_filters(results)

    winner = results[1]
    loser = results[0]

    assert winner["filtered_reason"] is None
    assert winner["duplicates"] == [
        {
            "url": "https://example.com/a.pdf",
            "query": "q1",
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 2,
            "overall_rank": None,
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
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 2,
            "overall_rank": None,
            "filtered_reason": None,
            "_order": 2,
        },
    ]

    search_only_module._apply_top_n_filters(results, num_urls=2)

    assert results[0]["overall_rank"] == 1
    assert results[1]["overall_rank"] == 2
    assert results[2]["overall_rank"] == 3
    assert results[2]["filtered_reason"] == "beyond_top_n"


def test_apply_filters_orders_phases_and_cleans_internal_fields():
    """Apply blacklist, duplicate, and top-N in deterministic order"""
    results = [
        {
            "url": "https://site.com/wiki",
            "query": "q-blacklist",
            "query_index": 0,
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 1,
            "overall_rank": None,
            "filtered_reason": None,
        },
        {
            "url": "https://example.com/dup",
            "query": "q-dup-2",
            "query_index": 0,
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 2,
            "overall_rank": None,
            "filtered_reason": None,
        },
        {
            "url": "https://example.com/dup",
            "query": "q-dup-1",
            "query_index": 1,
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 1,
            "overall_rank": None,
            "filtered_reason": None,
        },
        {
            "url": "https://example.com/other",
            "query": "q-other",
            "query_index": 2,
            "search_engine": "SerpAPIGoogleSearch",
            "query_rank": 1,
            "overall_rank": None,
            "filtered_reason": None,
        },
    ]

    output = search_only_module._apply_filters(
        results,
        blacklist=["WIKI"],
        num_urls=1,
    )

    assert output[0]["filtered_reason"].startswith("blacklist:")
    assert output[1]["filtered_reason"] == "duplicate"
    assert output[2]["filtered_reason"] is None
    assert output[2]["overall_rank"] == 1
    assert output[2]["duplicates"][0]["query"] == "q-dup-2"
    assert output[3]["filtered_reason"] == "beyond_top_n"
    assert output[3]["overall_rank"] == 2

    for row in output:
        assert "_order" not in row
        assert "query_index" not in row


def test_format_search_only_report_human_keeps_only_unfiltered_and_sorted():
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

    output = search_only_module.format_search_only_report_human(report)

    first_rank = output.index("[1]")
    second_rank = output.index("[2]")
    assert first_rank < second_rank
    assert "https://example.com/rank1" in output
    assert "https://example.com/rank2" in output
    assert "https://example.com/dup" not in output


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
