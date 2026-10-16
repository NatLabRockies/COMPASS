"""COMPASS LiteLLM helpers"""

import logging

import litellm

from compass.utilities import LLM_COST_REGISTRY


logger = logging.getLogger(__name__)


def register_litellm_model_cost(model_name):
    """Add litellm's token prices for a model to the cost registry

    Entries already in :obj:`~compass.utilities.costs.LLM_COST_REGISTRY`
    (including user-supplied ``llm_costs``) are never overwritten.
    Models litellm cannot price are skipped and cost $0, as before.

    Parameters
    ----------
    model_name : str
        LiteLLM model name or proxy alias (e.g. ``"gpt-5.4-mini"``).
    """
    if model_name in LLM_COST_REGISTRY:
        return

    prices = litellm.model_cost.get(model_name, {})
    if not prices and "/" in model_name:
        bare_name, provider = _litellm_model_and_provider(model_name)
        bare_prices = litellm.model_cost.get(bare_name, {})
        if bare_prices.get("litellm_provider") == provider:
            prices = bare_prices
    prompt_cost = prices.get("input_cost_per_token")
    if not prompt_cost:
        logger.debug("No litellm price found for model %r", model_name)
        return

    model_costs = {"prompt": prompt_cost * 1e6}
    response_cost = prices.get("output_cost_per_token")
    if response_cost:
        model_costs["response"] = response_cost * 1e6
    LLM_COST_REGISTRY.setdefault(model_name, model_costs)
def _litellm_model_and_provider(model_name):
    """Return the normalized model name and provider from LiteLLM"""
    bare_name, provider, _, _ = litellm.get_llm_provider(model=model_name)
    return bare_name, provider
