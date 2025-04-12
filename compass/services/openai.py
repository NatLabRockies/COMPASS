"""COMPASS Ordinances OpenAI service amd utils"""

import logging

import openai
from elm.base import ApiBase
from elm.utilities.retry import async_retry_with_exponential_backoff

from compass.services.base import LLMService
from compass.services.usage import TimeBoundedUsageTracker
from compass.utilities import LLM_COST_REGISTRY
from compass.utilities.enums import LLMUsageCategory
from compass.pb import COMPASS_PB


logger = logging.getLogger(__name__)


def usage_from_response(current_usage, response):
    """OpenAI usage parser

    Parameters
    ----------
    current_usage : dict
        Dictionary containing current usage information. For OpenAI
        trackers, this may contain the keys ``"requests"``,
        ``"prompt_tokens"``, and ``"response_tokens"`` if there is
        already existing tracking information. Empty dictionaries are
        allowed, in which case the three keys above will be added to
        this input.
    response : openai.Completion
        OpenAI Completion object. Must contain a ``usage`` attribute
        that

    Returns
    -------
    dict
        Dictionary with updated usage statistics.
    """
    current_usage["requests"] = current_usage.get("requests", 0) + 1
    current_usage["prompt_tokens"] = (
        current_usage.get("prompt_tokens", 0) + response.usage.prompt_tokens
    )
    current_usage["response_tokens"] = (
        current_usage.get("response_tokens", 0)
        + response.usage.completion_tokens
    )
    return current_usage


def count_tokens(messages, model):
    """Count the number of tokens in an outgoing set of messages

    Parameters
    ----------
    messages : list
        A list of message objects, where the latter is represented
        using a dictionary. Each message dictionary must have a
        "content" key containing the string to count tokens for.
    model : str
        The OpenAI model being used. This input will be passed to
        :func:`tiktoken.encoding_for_model`.

    Returns
    -------
    int
        Total number of tokens in the set of messages outgoing to
        OpenAI.

    References
    ----------
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    message_total = sum(
        ApiBase.count_tokens(message["content"], model=model) + 4
        for message in messages
    )
    return message_total + 3


class OpenAIService(LLMService):
    """OpenAI Chat GPT query service

    Purpose:
        Orchestrate OpenAI API calls.
    Responsibilities:
        1. Monitor OpenAI call queue.
        2. Submit calls to OpenAI API if rate limit has not been
           exceeded.
        3. Track token usage, both instantaneous (rate) and total (if
           user requests it).
        4. Parse responses into `str` and pass back to calling function.
    Key Relationships:
        Must be activated with
        :class:`~compass.services.provider.RunningAsyncServices`
        context.
    """

    def __init__(
        self,
        client,
        model_name,
        rate_limit=1e3,
        rate_tracker=None,
        service_tag=None,
    ):
        """

        Parameters
        ----------
        client : openai.AsyncOpenAI or openai.AsyncAzureOpenAI
            Async OpenAI client instance. Must have an async
            `client.chat.completions.create` method.
        model_name : str
            Name of model being used.
        rate_limit : int or float, optional
            Token rate limit (typically per minute, but the time
            interval is ultimately controlled by the `rate_tracker`
            instance). By default, ``1e3``.
        rate_tracker : TimeBoundedUsageTracker, optional
            A TimeBoundedUsageTracker instance. This will be used to
            track usage per time interval and compare to `rate_limit`.
            If ``None``, a `TimeBoundedUsageTracker` instance is created
            with default parameters. By default, ``None``.
        service_tag : str, optional
            optional tag to use to distinguish service (i.e. make unique
            from other services). Must set this if multiple models with
            the same name are run concurrently. By default, ``None``.
        """
        super().__init__(
            model_name=model_name,
            rate_limit=rate_limit,
            rate_tracker=rate_tracker or TimeBoundedUsageTracker(),
            service_tag=service_tag,
        )
        self.client = client

    async def process(
        self,
        usage_tracker=None,
        usage_sub_label=LLMUsageCategory.DEFAULT,
        **kwargs,
    ):
        """Process a call to OpenAI Chat GPT

        Note that this method automatically retries queries (with
        backoff) if a rate limit error is throw by the API.

        Parameters
        ----------
        model : str
            OpenAI GPT model to query.
        usage_tracker : `compass.services.usage.UsageTracker`, optional
            UsageTracker instance. Providing this input will update your
            tracker with this call's token usage info.
            By default, ``None``.
        usage_sub_label : str, optional
            Optional label to categorize usage under. This can be used
            to track usage related to certain categories.
            By default, ``"default"``.
        **kwargs
            Keyword arguments to be passed to
            `client.chat.completions.create`.

        Returns
        -------
        str or None
            Chat GPT response as a string, or ``None`` if the call
            failed.
        """
        self._record_prompt_tokens(kwargs)
        response = await self._call_gpt(model=self.model_name, **kwargs)
        self._record_completion_tokens(response)
        self._record_usage(response, usage_tracker, usage_sub_label)
        self._update_pb_cost(response)
        return _get_response_message(response)

    def _record_prompt_tokens(self, kwargs):
        """Add prompt token count to rate tracker"""
        num_tokens = count_tokens(kwargs.get("messages", []), self.model_name)
        self.rate_tracker.add(num_tokens)

    def _record_completion_tokens(self, response):
        """Add completion token count to rate tracker"""
        if response is None:
            return
        self.rate_tracker.add(response.usage.completion_tokens)

    def _record_usage(self, response, usage_tracker, usage_sub_label):
        """Record token usage for user"""
        if usage_tracker is None:
            return

        usage_tracker.update_from_model(
            model=self.model_name, response=response, sub_label=usage_sub_label
        )

    def _update_pb_cost(self, response):
        """Update the cost displayed on the progress bar"""
        if response is None:
            return

        model_costs = LLM_COST_REGISTRY.get(self.model_name, {})
        prompt_cost = (
            response.usage.prompt_tokens / 1e6 * model_costs.get("prompt", 0)
        )
        response_cost = (
            response.usage.completion_tokens
            / 1e6
            * model_costs.get("response", 0)
        )
        COMPASS_PB.update_total_cost(prompt_cost + response_cost)

    @async_retry_with_exponential_backoff()
    async def _call_gpt(self, **kwargs):
        """Query Chat GPT with user inputs"""
        try:
            return await self.client.chat.completions.create(**kwargs)
        except openai.BadRequestError:
            logger.exception("Got 'BadRequestError'")


def _get_response_message(response):
    """Get message as string from response object"""
    if response is None:
        return None
    return response.choices[0].message.content
