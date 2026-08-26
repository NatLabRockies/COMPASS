"""Test COMPASS Ordinance openai services"""

from pathlib import Path

import httpx
import pytest
import openai

from compass.services.openai import (
    count_tokens,
    usage_from_response,
    OpenAIService,
)
from compass.services.usage import LLMRateTracker, UsageTracker
from compass.utilities.enums import LLMUsageCategory


TEST_MESSAGES_1 = [
    {"role": "system", "content": "You are a friendly bot"},
    {"role": "user", "content": "How are you?"},
]
TEST_MESSAGES_2 = [
    {"role": "system", "content": "You are a friendly bot"},
    {"role": "user", "content": "I have 5 apples."},
    {"role": "system", "content": "Great!"},
    {"role": "user", "content": "How many apples do you have?."},
]


@pytest.mark.parametrize(
    "messages, model, token_count",
    [(TEST_MESSAGES_1, "gpt-4", 20), (TEST_MESSAGES_2, "gpt-4", 39)],
)
def test_count_tokens(messages, model, token_count):
    """Test `count_tokens` function"""
    assert count_tokens(messages, model) == token_count


@pytest.mark.parametrize(
    "usage_input, expected_output",
    [
        ({}, {"requests": 1, "prompt_tokens": 100, "response_tokens": 10}),
        (
            {"requests": 10, "response_tokens": 100},
            {"requests": 11, "prompt_tokens": 100, "response_tokens": 110},
        ),
    ],
)
def test_usage_from_response(
    usage_input, expected_output, sample_openai_response
):
    """Test `usage_from_response` function"""
    response = sample_openai_response()
    assert usage_from_response(usage_input, response) == expected_output


@pytest.mark.asyncio
async def test_openai_service(
    sample_openai_response, monkeypatch, patched_clock
):
    """Test querying OpenAI while tracking limits and usage"""

    async def _test_response(*args, **kwargs):  # ruff:ignore[unused-async]
        if kwargs.get("bad_request"):
            response = httpx.Response(404)
            response.request = httpx.Request(method="test", url="test")
            raise openai.NotFoundError(
                "for testing",
                response=response,
                body=None,
            )
        return sample_openai_response(kwargs=kwargs)

    client = openai.AsyncOpenAI(api_key="dummy")
    monkeypatch.setattr(
        client.chat.completions,
        "create",
        _test_response,
        raising=True,
    )
    rate_tracker = LLMRateTracker()
    openai_service = OpenAIService(client, model_name="gpt-4")

    usage_tracker = UsageTracker("my_county", usage_from_response)

    message = await openai_service.process(
        usage_tracker=usage_tracker, rate_tracker=rate_tracker
    )
    assert openai_service.timed_tracker.total == 13
    assert message == "test_response"

    assert usage_tracker == {
        "gpt-4": {
            LLMUsageCategory.DEFAULT: {
                "requests": 1,
                "prompt_tokens": 100,
                "response_tokens": 10,
            }
        }
    }
    assert rate_tracker.snapshot()["overall"] == {
        "requests_per_second": {"min": 1, "mean": 1, "max": 1},
        "requests_per_minute": {"min": 1, "mean": 1, "max": 1},
        "tokens_per_minute": {"min": 110, "mean": 110, "max": 110},
        "concurrent_requests": {"min": 0, "mean": 0.5, "max": 1},
    }

    with pytest.raises(openai.NotFoundError):
        message = await openai_service.process(
            usage_tracker=usage_tracker,
            rate_tracker=rate_tracker,
            bad_request=True,
        )

    assert openai_service.timed_tracker.total == 16
    assert usage_tracker == {
        "gpt-4": {
            LLMUsageCategory.DEFAULT: {
                "requests": 1,
                "prompt_tokens": 100,
                "response_tokens": 10,
            }
        }
    }
    assert rate_tracker.snapshot()["overall"] == {
        "requests_per_second": {"min": 2, "mean": 2, "max": 2},
        "requests_per_minute": {"min": 2, "mean": 2, "max": 2},
        "tokens_per_minute": {"min": 110, "mean": 110, "max": 110},
        "concurrent_requests": {
            "min": 0,
            "mean": pytest.approx(2 / 3),
            "max": 1,
        },
    }

    await openai_service.process()
    assert usage_tracker == {
        "gpt-4": {
            LLMUsageCategory.DEFAULT: {
                "requests": 1,
                "prompt_tokens": 100,
                "response_tokens": 10,
            }
        }
    }


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
