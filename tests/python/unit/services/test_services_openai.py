"""Test COMPASS Ordinance openai services"""

from pathlib import Path

import httpx
import pytest
import openai

from compass.services.openai import (
    count_tokens,
    usage_from_response,
    OpenAIService,
    _MAX_UNSUPPORTED_KWARG_DROPS,
    _unsupported_call_kwarg,
)
from compass.services.usage import UsageTracker
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
async def test_openai_service(sample_openai_response, monkeypatch):
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
    openai_service = OpenAIService(client, model_name="gpt-4")

    usage_tracker = UsageTracker("my_county", usage_from_response)

    message = await openai_service.process(usage_tracker=usage_tracker)
    assert openai_service.rate_tracker.total == 13
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

    with pytest.raises(openai.NotFoundError):
        message = await openai_service.process(
            usage_tracker=usage_tracker, bad_request=True
        )

    assert openai_service.rate_tracker.total == 16
    assert usage_tracker == {
        "gpt-4": {
            LLMUsageCategory.DEFAULT: {
                "requests": 1,
                "prompt_tokens": 100,
                "response_tokens": 10,
            }
        }
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


@pytest.mark.asyncio
async def test_openai_service_retries_without_unsupported_kwarg(
    sample_openai_response, monkeypatch
):
    """Unsupported kwargs are removed and retried once dynamically"""
    call_kwargs = []

    async def _test_response(*args, **kwargs):  # ruff:ignore[unused-async]
        call_kwargs.append(kwargs.copy())
        if len(call_kwargs) == 1:
            response = httpx.Response(400)
            response.request = httpx.Request(method="POST", url="https://test")
            raise openai.BadRequestError(
                "unsupported parameter",
                response=response,
                body={
                    "error": {
                        "message": (
                            "Unsupported value: 'temperature' does not "
                            "support 0"
                        ),
                        "type": "invalid_request_error",
                        "param": "temperature",
                        "code": "400",
                    }
                },
            )

        return sample_openai_response(kwargs=kwargs)

    client = openai.AsyncOpenAI(api_key="dummy")
    monkeypatch.setattr(
        client.chat.completions,
        "create",
        _test_response,
        raising=True,
    )
    openai_service = OpenAIService(client, model_name="gpt-5.6-terra")

    message = await openai_service.process(
        messages=TEST_MESSAGES_1, temperature=0, timeout=300
    )
    assert message == "test_response"
    assert len(call_kwargs) == 2
    assert "temperature" in call_kwargs[0]
    assert "temperature" not in call_kwargs[1]
    assert "temperature" in openai_service._unsupported_call_kwargs


@pytest.mark.asyncio
async def test_openai_service_caps_unsupported_kwarg_retries(monkeypatch):
    """Unsupported kwarg drops are capped to avoid unbounded retries"""
    call_kwargs = []
    rejected_kwargs = [
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "seed",
    ]

    async def _test_response(*args, **kwargs):  # ruff:ignore[unused-async]
        call_kwargs.append(kwargs.copy())
        param = rejected_kwargs[len(call_kwargs) - 1]
        response = httpx.Response(400)
        response.request = httpx.Request(method="POST", url="https://test")
        raise openai.BadRequestError(
            "unsupported parameter",
            response=response,
            body={
                "error": {
                    "message": (
                        f"Unsupported value: {param!r} does not support 0"
                    ),
                    "type": "invalid_request_error",
                    "param": param,
                    "code": "400",
                }
            },
        )

    client = openai.AsyncOpenAI(api_key="dummy")
    monkeypatch.setattr(
        client.chat.completions,
        "create",
        _test_response,
        raising=True,
    )
    openai_service = OpenAIService(client, model_name="gpt-5.6-terra")

    with pytest.raises(openai.BadRequestError):
        await OpenAIService._call_gpt.__wrapped__(
            openai_service,
            messages=TEST_MESSAGES_1,
            temperature=0,
            top_p=1,
            presence_penalty=0,
            frequency_penalty=0,
            seed=42,
        )

    assert len(call_kwargs) == _MAX_UNSUPPORTED_KWARG_DROPS + 1


def test_unsupported_call_kwarg_ignores_non_recoverable_bad_request():
    """Context-length bad requests are not treated as unsupported kwargs"""
    response = httpx.Response(400)
    response.request = httpx.Request(method="POST", url="https://test")
    error = openai.BadRequestError(
        "context too long",
        response=response,
        body={
            "error": {
                "message": "Input tokens exceed the configured limit",
                "type": "invalid_request_error",
                "param": "messages",
                "code": "context_length_exceeded",
            }
        },
    )
    unsupported = _unsupported_call_kwarg(
        error, {"messages": TEST_MESSAGES_2, "timeout": 300}
    )
    assert unsupported is None


def test_unsupported_call_kwarg_parses_param_from_error_message():
    """Fallback parser recovers kwarg when error body is unstructured"""
    response = httpx.Response(400)
    response.request = httpx.Request(method="POST", url="https://test")
    error = openai.BadRequestError(
        "Error code: 400 - {'error': {'message': \"Unsupported value: "
        "'temperature' does not support 0\", 'type': "
        "'invalid_request_error', 'param': 'temperature', 'code': '400'}}",
        response=response,
        body=None,
    )
    unsupported = _unsupported_call_kwarg(
        error, {"messages": TEST_MESSAGES_2, "temperature": 0}
    )
    assert unsupported == "temperature"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
