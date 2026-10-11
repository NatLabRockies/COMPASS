"""Tests for stage-independent shard minimum targets"""

from pathlib import Path

import pytest

from compass.exceptions import COMPASSValueError
from compass.pipeline.targets import (
    evaluate_targets,
    normalize_targets,
    parse_search_target_options,
)


def test_parse_targets_preserves_literal_metric_names():
    """Keep engine labels intact and combine repeated minimums"""
    assert parse_search_target_options(
        [
            "num_results=5",
            "num_results=3",
            "search_engine_counts.Example (v1.2)=10",
        ]
    ) == {"num_results": 5, "search_engine_counts.Example (v1.2)": 10}


@pytest.mark.parametrize(
    "option",
    [
        "count",
        "=3",
        "count=",
        "count=foo",
        "count=true",
        "count=NaN",
        "count=Infinity",
        "count=null",
    ],
)
def test_reject_invalid_target_options(option):
    """Reject malformed expressions and nonnumeric lower bounds"""
    with pytest.raises(COMPASSValueError):
        parse_search_target_options([option])


def test_targets_are_stage_independent():
    """Use arbitrary numeric metrics without importing search behavior"""
    metrics = {
        "temperature": lambda record: record.get("temperature"),
        "count": lambda record: record.get("count"),
    }
    targets = {"temperature": -1.5, "count": 3}
    assert (
        evaluate_targets({"temperature": -1.5, "count": 3}, targets, metrics)
        == []
    )
    assert evaluate_targets({"count": 2}, targets, metrics) == [
        {"metric": "temperature", "actual": None, "minimum": -1.5},
        {"metric": "count", "actual": 2, "minimum": 3},
    ]


@pytest.mark.parametrize("targets", [[], {"count": True}, {None: 3}])
def test_reject_invalid_target_mappings(targets):
    """Require named numeric minimums in configuration mappings"""
    with pytest.raises(COMPASSValueError):
        normalize_targets(targets)


def test_unknown_metric_and_invalid_actual():
    """Do not silently accept unsupported or malformed shard metrics"""
    with pytest.raises(COMPASSValueError, match="Unknown"):
        evaluate_targets({}, {"unknown": 3}, {})
    with pytest.raises(COMPASSValueError, match="finite"):
        evaluate_targets({}, {"count": 3}, {"count": lambda _record: True})


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
