"""COMPASS LiteLLM client adapter

Lets :class:`~compass.services.openai.OpenAIService` call any provider
supported by litellm, so rate limiting, retries, usage tracking, and
cost reporting are shared with the OpenAI path instead of duplicated.
"""

import logging
from types import SimpleNamespace

import litellm

from compass.utilities import LLM_COST_REGISTRY


logger = logging.getLogger(__name__)


class LiteLLMClient:
    """Async client exposing the ``chat.completions.create`` call

    litellm exceptions subclass the matching ``openai`` exceptions and
    litellm responses share the OpenAI response shape, so this client
    is a drop-in replacement for ``openai.AsyncOpenAI`` inside
    :class:`~compass.services.openai.OpenAIService`.
    """

    def __init__(self, **client_kwargs):
        """

        Parameters
        ----------
        **client_kwargs
            Keyword arguments forwarded to ``litellm.acompletion``
            on every call (e.g. ``api_key``, ``api_base``).
        """
        self.client_kwargs = client_kwargs
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        """Submit a chat completion request through litellm"""
        return await litellm.acompletion(**self.client_kwargs, **kwargs)


def register_litellm_model_cost(model_name):
    """Add litellm's token prices for a model to the cost registry

    Entries already in :obj:`~compass.utilities.costs.LLM_COST_REGISTRY`
    (including user-supplied ``llm_costs``) are never overwritten.
    Models litellm cannot price are skipped and cost $0, as before.

    Parameters
    ----------
    model_name : str
        litellm-style model name (e.g. ``"openai/gpt-5-mini"``).
    """
    if model_name in LLM_COST_REGISTRY:
        return

    prices = litellm.model_cost.get(model_name) or litellm.model_cost.get(
        model_name.split("/", 1)[-1], {}
    )
    prompt_cost = prices.get("input_cost_per_token")
    if not prompt_cost:
        logger.debug("No litellm price found for model %r", model_name)
        return

    model_costs = {"prompt": prompt_cost * 1e6}
    response_cost = prices.get("output_cost_per_token")
    if response_cost:
        model_costs["response"] = response_cost * 1e6
    LLM_COST_REGISTRY.setdefault(model_name, model_costs)
