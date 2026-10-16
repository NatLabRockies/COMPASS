"""Test COMPASS Ordinance litellm helpers"""

from pathlib import Path

import pytest
import litellm

from compass.services.litellm import register_litellm_model_cost
from compass.utilities import LLM_COST_REGISTRY, compute_cost_from_totals


@pytest.fixture(autouse=True)
def restore_cost_registry():
    """Undo model prices registered during each test"""
    original = dict(LLM_COST_REGISTRY)
    yield
    LLM_COST_REGISTRY.clear()
    LLM_COST_REGISTRY.update(original)


@pytest.mark.parametrize(
    "model_name",
    [
        "openai/gpt-5-mini",
        "anthropic/claude-sonnet-4-5",
        "gemini/gemini-3.5-flash",
    ],
)
def test_register_litellm_model_cost(model_name):
    """Model prices come from litellm's model price table"""
    LLM_COST_REGISTRY.pop(model_name, None)
    bare_name, _, _, _ = litellm.get_llm_provider(model=model_name)
    prices = litellm.model_cost.get(model_name) or litellm.model_cost.get(
        bare_name
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
