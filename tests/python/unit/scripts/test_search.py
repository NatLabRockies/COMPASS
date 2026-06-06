"""Tests for compass.scripts.search"""

from pathlib import Path

import pytest

import compass.scripts.search as search_module


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
