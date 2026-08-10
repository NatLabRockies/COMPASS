"""COMPASS cost computation utilities"""

LLM_COST_REGISTRY = {
    "o1": {"prompt": 15, "response": 60},
    "o3-mini": {"prompt": 1.1, "response": 4.4},
    "gpt-4.5": {"prompt": 75, "response": 150},
    "gpt-4o": {"prompt": 2.5, "response": 10},
    "gpt-4o-mini": {"prompt": 0.15, "response": 0.6},
    "gpt-4.1": {"prompt": 2, "response": 8},
    "gpt-4.1-mini": {"prompt": 0.4, "response": 1.6},
    "gpt-4.1-nano": {"prompt": 0.1, "response": 0.4},
    "gpt-5": {"prompt": 1.25, "response": 10},
    "gpt-5-mini": {"prompt": 0.25, "response": 2},
    "gpt-5-nano": {"prompt": 0.05, "response": 0.4},
    "gpt-5-chat-latest": {"prompt": 1.25, "response": 10},
    "gpt-5.4": {"prompt": 2.50, "response": 15},
    "gpt-5.4-mini": {"prompt": 0.75, "response": 4.5},
    "gpt-5.4-nano": {"prompt": 0.20, "response": 1.25},
    "gpt-5.5": {"prompt": 5, "response": 30},
    "gpt-5.6-sol": {"prompt": 5, "response": 30},
    "gpt-5.6-terra": {"prompt": 2, "response": 12},
    "gpt-5.6-luna": {"prompt": 0.2, "response": 1.2},
    "compassop-gpt-4o": {"prompt": 2.5, "response": 10},
    "compassop-gpt-4o-mini": {"prompt": 0.15, "response": 0.6},
    "compassop-gpt-4.1": {"prompt": 2, "response": 8},
    "compassop-gpt-4.1-mini": {"prompt": 0.4, "response": 1.6},
    "compassop-gpt-4.1-nano": {"prompt": 0.1, "response": 0.4},
    "compassop-gpt-5": {"prompt": 1.25, "response": 10},
    "compassop-gpt-5-mini": {"prompt": 0.25, "response": 2},
    "compassop-gpt-5-nano": {"prompt": 0.05, "response": 0.4},
    "compassop-gpt-5-chat-latest": {"prompt": 1.25, "response": 10},
    "compassop-gpt-5.4": {"prompt": 2.50, "response": 15},
    "compassop-gpt-5.4-mini": {"prompt": 0.75, "response": 4.5},
    "compassop-gpt-5.4-nano": {"prompt": 0.20, "response": 1.25},
    "compassop-gpt-5.5": {"prompt": 5, "response": 30},
    "egswaterord-gpt4.1-mini": {"prompt": 0.4, "response": 1.6},
    "wetosa-gpt-4o": {"prompt": 2.5, "response": 10},
    "wetosa-gpt-4o-mini": {"prompt": 0.15, "response": 0.6},
    "wetosa-gpt-4.1": {"prompt": 2, "response": 8},
    "wetosa-gpt-4.1-mini": {"prompt": 0.4, "response": 1.6},
    "wetosa-gpt-4.1-nano": {"prompt": 0.1, "response": 0.4},
    "wetosa-gpt-5": {"prompt": 1.25, "response": 10},
    "wetosa-gpt-5-mini": {"prompt": 0.25, "response": 2},
    "wetosa-gpt-5-nano": {"prompt": 0.05, "response": 0.4},
    "wetosa-gpt-5-chat-latest": {"prompt": 1.25, "response": 10},
    "text-embedding-ada-002": {"prompt": 0.10},
}
"""LLM Costs registry

The registry maps model names to a dictionary that contains the cost
(in $/million tokens) for both prompt and response tokens.
"""


def cost_for_model(model_name, prompt_tokens, completion_tokens):
    """Compute the API costs for a model given the token usage

    Parameters
    ----------
    model_name : str
        Name of the model. Needs to be registered as a key in
        :obj:`LLM_COST_REGISTRY` for this method to return a non-zero
        value.
    prompt_tokens, completion_tokens : int
        Number of prompt and completion tokens used, respectively.

    Returns
    -------
    float
        Total cost based on the token usage.
    """
    model_costs = LLM_COST_REGISTRY.get(model_name, {})
    prompt_cost = prompt_tokens / 1e6 * model_costs.get("prompt", 0)
    response_cost = completion_tokens / 1e6 * model_costs.get("response", 0)
    return prompt_cost + response_cost


def compute_cost_from_totals(totals):
    """Compute total cost from total tracked usage

    Parameters
    ----------
    totals : dict
        Dictionary where keys are model names and their corresponding
        usage statistics are values. Each usage statistics dictionary
        should contain "prompt_tokens" and "response_tokens" keys
        indicating the number of tokens used for prompts and responses,
        respectively. This dictionary is typically obtained from the
        `tracker_totals` property of a
        :class:`compass.services.usage.UsageTracker` instance.

    Returns
    -------
    float
        Total cost based on the tracked usage.
    """
    return sum(
        cost_for_model(
            model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("response_tokens", 0),
        )
        for model, usage in totals.items()
    )


def compute_total_tokens_from_totals(totals):
    """Compute total prompt/response token counts from tracked usage

    Parameters
    ----------
    totals : dict
        Same shape as :func:`compute_cost_from_totals`: maps model name
        to a dict with ``"prompt_tokens"`` and ``"response_tokens"``.

    Returns
    -------
    dict
        ``{"prompt_tokens": int, "response_tokens": int}`` summed across
        all models in ``totals``.
    """
    return {
        "prompt_tokens": sum(
            u.get("prompt_tokens", 0) for u in totals.values()
        ),
        "response_tokens": sum(
            u.get("response_tokens", 0) for u in totals.values()
        ),
    }


def compute_total_cost_and_token_from_totals(totals):
    """Compute total cost and token counts together from tracked usage

    Returns
    -------
    dict
        Keys: ``"cost"`` (float), ``"prompt_tokens"`` (int),
        ``"response_tokens"`` (int).
    """
    return {
        "cost": compute_cost_from_totals(totals),
        **compute_total_tokens_from_totals(totals),
    }


def compute_total_cost_from_usage(tracked_usage):
    """Compute total cost from total tracked usage

    Parameters
    ----------
    tracked_usage : compass.services.usage.UsageTracker or dict
        Dictionary where keys are usage categories (typically
        jurisdiction names) and values are dictionaries containing usage
        details. The usage details dictionaries should have a
        "tracker_totals" key, which maps to another dictionary. This
        innermost dictionary should have model names as keys and their
        corresponding usage statistics as values. Each usage statistics
        dictionary should contain "prompt_tokens" and "response_tokens"
        keys indicating the number of tokens used for prompts and
        responses, respectively.

    Returns
    -------
    float
        Total LLM cost based on the tracked usage.
    """
    return sum(
        compute_cost_from_totals(usage.get("tracker_totals", {}))
        for usage in tracked_usage.values()
    )
