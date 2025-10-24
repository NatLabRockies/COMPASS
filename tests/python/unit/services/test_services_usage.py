"""Test COMPASS Ordinance service usage functions and classes"""

import time
from pathlib import Path

import pytest
from flaky import flaky

from compass.services.usage import (
    TimedEntry,
    TimeBoundedUsageTracker,
    UsageTracker,
)


def _sample_response_parser(current_usage, response):
    """Sample response to usage conversion function"""
    current_usage["requests"] = current_usage.get("requests", 0) + 1
    if "tokens" in response:
        current_usage["tokens"] = response["tokens"]
    inputs = current_usage.get("inputs", 0)
    current_usage["inputs"] = inputs + response.get("inputs", 0)
    return current_usage


def test_timed_entry():
    """Test `TimedEntry` class"""

    a = TimedEntry(100)
    assert a <= time.monotonic()

    time.sleep(0.2)
    sample_time = time.monotonic()
    time.sleep(0.2)
    b = TimedEntry(10000)
    assert b > sample_time
    assert a < sample_time

    assert a.value == 100
    assert b.value == 10000


@flaky(max_runs=10, min_passes=1)
def test_time_bounded_usage_tracker():
    """Test the `TimeBoundedUsageTracker` class"""

    tracker = TimeBoundedUsageTracker(max_seconds=0.2)
    assert tracker.total == 0
    tracker.add(500)
    assert tracker.total == 500
    time.sleep(0.1)
    tracker.add(200)
    assert tracker.total == 700
    time.sleep(0.1)
    assert tracker.total == 200
    time.sleep(0.1)
    assert tracker.total == 0


def test_usage_tracker():
    """Test the `UsageTracker` class"""

    tracker = UsageTracker("test", response_parser=_sample_response_parser)
    assert tracker == {}
    assert tracker.totals == {}

    tracker.update_from_model()
    assert tracker == {}
    assert tracker.totals == {}

    tracker.update_from_model(response={})
    assert tracker == {
        UsageTracker.UNKNOWN_MODEL_LABEL: {
            "default": {"requests": 1, "inputs": 0}
        }
    }
    assert tracker.totals == {
        UsageTracker.UNKNOWN_MODEL_LABEL: {"requests": 1, "inputs": 0}
    }

    tracker.update_from_model(response={"inputs": 100}, sub_label="parsing")
    tracker.update_from_model(
        model="my_model", response={"inputs": 200}, sub_label="parsing"
    )
    tracker.update_from_model()

    assert tracker == {
        UsageTracker.UNKNOWN_MODEL_LABEL: {
            "default": {"requests": 1, "inputs": 0},
            "parsing": {"requests": 1, "inputs": 100},
        },
        "my_model": {"parsing": {"requests": 1, "inputs": 200}},
    }
    assert tracker.totals == {
        UsageTracker.UNKNOWN_MODEL_LABEL: {"requests": 2, "inputs": 100},
        "my_model": {"requests": 1, "inputs": 200},
    }

    tracker.update_from_model(response={"tokens": 5})

    assert tracker == {
        UsageTracker.UNKNOWN_MODEL_LABEL: {
            "default": {"requests": 2, "inputs": 0, "tokens": 5},
            "parsing": {"requests": 1, "inputs": 100},
        },
        "my_model": {"parsing": {"requests": 1, "inputs": 200}},
    }
    assert tracker.totals == {
        UsageTracker.UNKNOWN_MODEL_LABEL: {
            "requests": 3,
            "inputs": 100,
            "tokens": 5,
        },
        "my_model": {"requests": 1, "inputs": 200},
    }

    output = {"some": "value"}
    tracker.add_to(output)
    expected_out = {**tracker, "tracker_totals": tracker.totals}
    assert output == {"some": "value", "test": expected_out}


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
