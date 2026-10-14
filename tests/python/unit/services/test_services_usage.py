"""Test COMPASS Ordinance service usage functions and classes"""

from pathlib import Path
from collections import UserDict

import pytest

from compass.services.usage import (
    LLM_USAGE_RATES_KEY,
    LLMRateTracker,
    LLMUsageTracker,
    TimedEntry,
    TimeBoundedUsageTracker,
)


def _sample_response_parser(current_usage, response):
    """Sample response to usage conversion function"""
    current_usage["requests"] = current_usage.get("requests", 0) + 1
    if "tokens" in response:
        current_usage["tokens"] = response["tokens"]
    inputs = current_usage.get("inputs", 0)
    current_usage["inputs"] = inputs + response.get("inputs", 0)
    return current_usage


def test_timed_entry(patched_clock):
    """Test `TimedEntry` class"""

    a = TimedEntry(100)
    assert a <= patched_clock()

    patched_clock.advance(0.2)
    sample_time = patched_clock()
    patched_clock.advance(0.2)
    b = TimedEntry(10000)
    assert b > sample_time
    assert a < sample_time

    assert a.value == 100
    assert b.value == 10000


def test_time_bounded_usage_tracker(patched_clock):
    """Test the `TimeBoundedUsageTracker` class"""

    tracker = TimeBoundedUsageTracker(max_seconds=0.2)
    assert tracker.total == 0
    tracker.add(500)
    assert tracker.total == 500
    patched_clock.advance(0.1)
    tracker.add(200)
    assert tracker.total == 700
    patched_clock.advance(0.1 + 1e-6)
    assert tracker.total == 200
    patched_clock.advance(0.2)
    assert tracker.total == 0


def test_rate_tracker(patched_clock):
    """Track call-driven fixed-window LLM usage rates"""

    tracker = LLMRateTracker()
    tracker.record_request("model_a", patched_clock())

    patched_clock.advance(1.2)
    tracker.record_request("model_a", patched_clock())

    patched_clock.advance(1)
    tracker.record_tokens("model_a", 10, patched_clock())

    patched_clock.advance(59.3)
    rates = tracker.snapshot()

    expected_requests_per_second = {"min": 1, "mean": 1, "max": 1}
    expected_requests_per_minute = {"min": 2, "mean": 2, "max": 2}
    expected_tokens_per_minute = {"min": 10, "mean": 10, "max": 10}
    expected_concurrent_requests = {"min": 0, "mean": 0, "max": 0}
    expected = {
        "requests_per_second": expected_requests_per_second,
        "requests_per_minute": expected_requests_per_minute,
        "tokens_per_minute": expected_tokens_per_minute,
        "concurrent_requests": expected_concurrent_requests,
    }

    assert rates == {"overall": expected, "models": {"model_a": expected}}
    assert isinstance(tracker, UserDict)

    output = {"some": "value"}
    tracker.add_to(output)
    assert output == {"some": "value", LLM_USAGE_RATES_KEY: tracker.data}


def test_rate_tracker_tracks_concurrent_request_attempts():
    """Track concurrent requests run-wide and per model"""

    tracker = LLMRateTracker()
    tracker.start_request_attempt("model_a")
    tracker.start_request_attempt("model_b")

    active_rates = tracker.snapshot()
    assert active_rates["overall"]["concurrent_requests"] == {
        "min": 1,
        "mean": pytest.approx(5 / 3),
        "max": 2,
    }
    assert active_rates["models"]["model_a"]["concurrent_requests"] == {
        "min": 1,
        "mean": 1,
        "max": 1,
    }
    assert active_rates["models"]["model_b"]["concurrent_requests"] == {
        "min": 1,
        "mean": 1,
        "max": 1,
    }

    tracker.end_request_attempt("model_a")
    tracker.end_request_attempt("model_b")

    completed_rates = tracker.snapshot()
    assert completed_rates["overall"]["concurrent_requests"] == {
        "min": 0,
        "mean": 1,
        "max": 2,
    }
    assert completed_rates["models"]["model_a"]["concurrent_requests"] == {
        "min": 0,
        "mean": pytest.approx(0.5),
        "max": 1,
    }
    assert completed_rates["models"]["model_b"]["concurrent_requests"] == {
        "min": 0,
        "mean": pytest.approx(0.5),
        "max": 1,
    }


def test_rate_tracker_snapshot_does_not_mutate(patched_clock):
    """Keep current windows open after generating a rate snapshot"""

    tracker = LLMRateTracker()
    tracker.record_request("model_a", patched_clock())

    patched_clock.advance(1.5)
    first_snapshot = tracker.snapshot()
    tracker.record_request("model_a", patched_clock())
    tracker.record_tokens("model_b", 20, patched_clock())

    expected_request_rates = {"min": 1, "mean": 1, "max": 1}
    expected_token_rates = {"min": 20, "mean": 20, "max": 20}
    rates = tracker.snapshot()

    assert first_snapshot["overall"]["requests_per_second"] == {
        "min": 1,
        "mean": 1,
        "max": 1,
    }
    assert rates["overall"]["requests_per_second"] == expected_request_rates
    assert rates["overall"]["tokens_per_minute"] == expected_token_rates
    assert rates["models"]["model_a"]["requests_per_second"] == (
        expected_request_rates
    )
    assert rates["models"]["model_b"]["requests_per_second"] == {
        "min": 0,
        "mean": 0,
        "max": 0,
    }
    assert rates["models"]["model_b"]["tokens_per_minute"] == (
        expected_token_rates
    )


def test_usage_tracker():
    """Test the `LLMUsageTracker` class"""

    tracker = LLMUsageTracker("test", response_parser=_sample_response_parser)
    assert tracker == {}
    assert tracker.totals == {}

    tracker.update_from_model()
    assert tracker == {}
    assert tracker.totals == {}

    tracker.update_from_model(response={})
    assert tracker == {
        LLMUsageTracker.UNKNOWN_MODEL_LABEL: {
            "default": {"requests": 1, "inputs": 0}
        }
    }
    assert tracker.totals == {
        LLMUsageTracker.UNKNOWN_MODEL_LABEL: {"requests": 1, "inputs": 0}
    }

    tracker.update_from_model(response={"inputs": 100}, sub_label="parsing")
    tracker.update_from_model(
        model="my_model", response={"inputs": 200}, sub_label="parsing"
    )
    tracker.update_from_model()

    assert tracker == {
        LLMUsageTracker.UNKNOWN_MODEL_LABEL: {
            "default": {"requests": 1, "inputs": 0},
            "parsing": {"requests": 1, "inputs": 100},
        },
        "my_model": {"parsing": {"requests": 1, "inputs": 200}},
    }
    assert tracker.totals == {
        LLMUsageTracker.UNKNOWN_MODEL_LABEL: {"requests": 2, "inputs": 100},
        "my_model": {"requests": 1, "inputs": 200},
    }

    tracker.update_from_model(response={"tokens": 5})

    assert tracker == {
        LLMUsageTracker.UNKNOWN_MODEL_LABEL: {
            "default": {"requests": 2, "inputs": 0, "tokens": 5},
            "parsing": {"requests": 1, "inputs": 100},
        },
        "my_model": {"parsing": {"requests": 1, "inputs": 200}},
    }
    assert tracker.totals == {
        LLMUsageTracker.UNKNOWN_MODEL_LABEL: {
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
