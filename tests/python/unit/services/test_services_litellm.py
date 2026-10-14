"""Test COMPASS Ordinance litellm client adapter"""

from pathlib import Path

import httpx
import pytest
import litellm
import openai
from litellm import ModelResponse
from litellm.types.utils import Usage, Choices, Message

from compass.pb import COMPASS_PB
from compass.services.litellm import (
    LiteLLMClient,
    register_litellm_model_cost,
)
from compass.services.openai import OpenAIService, usage_from_response
from compass.services.usage import LLMUsageTracker
from compass.utilities import LLM_COST_REGISTRY, compute_cost_from_totals
from compass.utilities.enums import LLMUsageCategory


TEST_MESSAGES = [
    {"role": "system", "content": "You are a friendly bot"},
    {"role": "user", "content": "How are you?"},
]


@pytest.fixture(autouse=True)
def restore_cost_registry():
    """Undo model prices registered during each test"""
    original = dict(LLM_COST_REGISTRY)
    yield
    LLM_COST_REGISTRY.clear()
    LLM_COST_REGISTRY.update(original)


def _litellm_response(model, prompt_tokens=100, completion_tokens=10):
    """Build a litellm response with the given usage"""
    return ModelResponse(
        id="1",
        model=model,
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=Message(content="test_response", role="assistant"),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@pytest.mark.asyncio
async def test_openai_service_with_litellm_client(monkeypatch, patched_clock):
    """Usage, cost, and client kwargs flow through the shared service"""
    seen = {}

    async def _acompletion(**kwargs):  # ruff:ignore[unused-async]
        seen.update(kwargs)
        return _litellm_response(kwargs["model"], 1_000_000, 0)

    monkeypatch.setattr(litellm, "acompletion", _acompletion, raising=True)
    LLM_COST_REGISTRY["proxy/model"] = {"prompt": 0.15, "response": 0.6}
    COMPASS_PB.reset()
    COMPASS_PB.create_main_task(1)

    service = OpenAIService(
        LiteLLMClient(api_key="secret", api_base="http://proxy"),
        "proxy/model",
    )
    usage_tracker = LLMUsageTracker("my_county", usage_from_response)
    message = await service.process(
        usage_tracker=usage_tracker, messages=TEST_MESSAGES
    )

    assert message == "test_response"
    assert seen["api_key"] == "secret"
    assert seen["api_base"] == "http://proxy"
    assert seen["model"] == "proxy/model"
    assert seen["messages"] == TEST_MESSAGES
    assert usage_tracker == {
        "proxy/model": {
            LLMUsageCategory.DEFAULT: {
                "requests": 1,
                "prompt_tokens": 1_000_000,
                "response_tokens": 0,
            }
        }
    }
    assert COMPASS_PB._total_cost == pytest.approx(0.15)


@pytest.mark.asyncio
async def test_litellm_errors_handled_as_openai_errors(monkeypatch):
    """litellm exceptions propagate as their openai counterparts"""

    async def _acompletion(**kwargs):  # ruff:ignore[unused-async]
        response = httpx.Response(404)
        response.request = httpx.Request(method="test", url="test")
        raise litellm.NotFoundError(
            "for testing",
            model=kwargs["model"],
            llm_provider="openai",
            response=response,
        )

    monkeypatch.setattr(litellm, "acompletion", _acompletion, raising=True)

    service = OpenAIService(LiteLLMClient(), "openai/gpt-4o-mini")
    with pytest.raises(openai.NotFoundError):
        await service.process(messages=TEST_MESSAGES)


@pytest.mark.parametrize(
    "model_name",
    ["openai/gpt-5-mini", "anthropic/claude-sonnet-4-5", "gpt-5-mini"],
)
def test_register_litellm_model_cost(model_name):
    """Model prices come from litellm's model price table"""
    LLM_COST_REGISTRY.pop(model_name, None)
    prices = litellm.model_cost.get(model_name) or litellm.model_cost.get(
        model_name.split("/", 1)[-1]
    )
    expected = (
        prices["input_cost_per_token"] + prices["output_cost_per_token"]
    ) * 1e6

    register_litellm_model_cost(model_name)

    totals = {
        model_name: {"prompt_tokens": 1_000_000, "response_tokens": 1_000_000}
    }
    assert compute_cost_from_totals(totals) == pytest.approx(expected)
    assert compute_cost_from_totals(totals) > 0


def test_register_litellm_model_cost_keeps_existing_entry():
    """User-provided `llm_costs` entries are never overwritten"""
    LLM_COST_REGISTRY["openai/gpt-5-mini"] = {"prompt": 1, "response": 1}

    register_litellm_model_cost("openai/gpt-5-mini")

    assert LLM_COST_REGISTRY["openai/gpt-5-mini"] == {
        "prompt": 1,
        "response": 1,
    }


def test_register_litellm_model_cost_skips_unpriced_model():
    """Models unknown to litellm are left out of the registry"""
    register_litellm_model_cost("openai/not-a-real-model-xyz")

    assert "openai/not-a-real-model-xyz" not in LLM_COST_REGISTRY


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
