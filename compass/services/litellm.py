"""COMPASS LiteLLM helpers"""

import logging

import litellm

from compass.utilities import LLM_COST_REGISTRY


logger = logging.getLogger(__name__)


def _lookup_litellm_prices(model_name):
    """Return LiteLLM pricing metadata for a model when available"""
    model_cost = getattr(litellm, "model_cost", None)
    if not model_cost:
        return {}

    prices = model_cost.get(model_name, {})
    if prices or "/" not in model_name:
        return prices

    provider, bare_name = model_name.split("/", 1)
    bare_prices = model_cost.get(bare_name, {})
    if bare_prices.get("litellm_provider") in {None, provider}:
        return bare_prices
    return {}


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

    if "/" in model_name:
        _, bare_name = model_name.split("/", 1)
        bare_costs = LLM_COST_REGISTRY.get(bare_name, {})
        if bare_costs:
            LLM_COST_REGISTRY.setdefault(model_name, dict(bare_costs))
            return

    prices = _lookup_litellm_prices(model_name)
    prompt_cost = prices.get("input_cost_per_token")
    if not prompt_cost:
        logger.debug("No litellm price found for model %r", model_name)
        return

    model_costs = {"prompt": prompt_cost * 1e6}
    response_cost = prices.get("output_cost_per_token")
    if response_cost:
        model_costs["response"] = response_cost * 1e6
    LLM_COST_REGISTRY.setdefault(model_name, model_costs)
