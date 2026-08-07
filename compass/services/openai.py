"""COMPASS Ordinances OpenAI service and utils"""

import logging
import re

import openai
from elm.base import ApiBase
from elm.utilities.retry import async_retry_with_exponential_backoff

from compass.services.base import LLMService
from compass.services.usage import TimeBoundedUsageTracker
from compass.utilities import cost_for_model
from compass.utilities.enums import LLMUsageCategory
from compass.pb import COMPASS_PB


logger = logging.getLogger(__name__)
_UNSUPPORTED_PARAM_MSG_MARKERS = (
    "does not support",
    "unsupported value",
    "unsupported parameter",
    "unknown parameter",
    "only the default",
    "not supported",
)
_MAX_UNSUPPORTED_KWARG_DROPS = 3


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
    response : object
        OpenAI Completion object (``openai.Completion``). Must contain a
        ``usage`` attribute that contains ``prompt_tokens`` and
        ``completion_tokens`` counts as attributes.

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
        ``tiktoken.encoding_for_model``.

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
        client : object
            Async OpenAI client instance (``openai.AsyncOpenAI`` or
            ``openai.AsyncAzureOpenAI``). Must have an async
            ``client.chat.completions.create`` method.
        model_name : str
            Name of model being used.
        rate_limit : int or float, optional
            Token rate limit (typically per minute, but the time
            interval is ultimately controlled by the `rate_tracker`
            instance). By default, ``1e3``.
        rate_tracker : TimeBoundedUsageTracker, optional
            Instance used to track usage per time interval and compare
            to `rate_limit` input. If ``None``, a
            TimeBoundedUsageTracker instance is created with default
            parameters. By default, ``None``.
        service_tag : str, optional
            Optional tag to use to distinguish service (i.e. make unique
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
        self._unsupported_call_kwargs = set()

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
        usage_tracker : UsageTracker, optional
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

        response_cost = cost_for_model(
            self.model_name,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        COMPASS_PB.update_total_cost(response_cost)

    @async_retry_with_exponential_backoff(
        base_delay=1,
        exponential_base=4,
        jitter=True,
        max_retries=3,
        errors=(
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.BadRequestError,
            openai.APIConnectionError,
        ),
    )
    async def _call_gpt(self, **kwargs):
        """Query Chat GPT with user inputs"""
        active_kwargs = dict(kwargs)
        self._drop_known_unsupported_kwargs(active_kwargs)
        num_dropped_kwargs = 0

        while True:
            try:
                return await self.client.chat.completions.create(
                    **active_kwargs
                )
            except openai.BadRequestError as error:
                unsupported_kwarg = _unsupported_call_kwarg(
                    error, active_kwargs
                )
                if (
                    unsupported_kwarg is not None
                    and num_dropped_kwargs >= _MAX_UNSUPPORTED_KWARG_DROPS
                ):
                    logger.warning(
                        "Exceeded maximum number of unsupported kwarg "
                        "retries (%d) for %r call; last rejected kwarg "
                        "was %r",
                        _MAX_UNSUPPORTED_KWARG_DROPS,
                        self.model_name,
                        unsupported_kwarg,
                    )
                    unsupported_kwarg = None

                if unsupported_kwarg is None:
                    messages = kwargs.get("messages")
                    if messages:
                        logger.exception(
                            "Got 'BadRequestError' for the following "
                            "messages:\n\t%s",
                            "\n\t".join([f"{m!r}" for m in messages]),
                        )
                    else:
                        logger.exception("Got 'BadRequestError'")
                    raise

                self._unsupported_call_kwargs.add(unsupported_kwarg)
                active_kwargs.pop(unsupported_kwarg, None)
                num_dropped_kwargs += 1
                logger.warning(
                    "Retrying %r call without unsupported kwarg %r after "
                    "provider rejected it",
                    self.model_name,
                    unsupported_kwarg,
                )

    def _drop_known_unsupported_kwargs(self, kwargs):
        """Drop kwargs previously rejected for this service instance"""
        for kwarg in self._unsupported_call_kwargs:
            kwargs.pop(kwarg, None)


def _get_response_message(response):
    """Get message as string from response object"""
    if response is None:
        return None
    return response.choices[0].message.content


def _unsupported_call_kwarg(error, request_kwargs):
    """str | None: Recoverable unsupported top-level call kwarg name"""
    error_body = getattr(error, "body", {}) or {}
    kwarg = None
    message = str(error)

    if isinstance(error_body, dict):
        error_info = error_body.get("error", {})
        kwarg = error_info.get("param")
        message = error_info.get("message") or message
    elif isinstance(error_body, str):
        message = error_body or message

    if not isinstance(kwarg, str):
        match = re.search(r"['\"]param['\"]:\s*['\"]([^'\"]+)['\"]", message)
        if match:
            kwarg = match.group(1)

    message = message.casefold()

    if not isinstance(kwarg, str):
        return None
    if kwarg not in request_kwargs:
        return None
    if kwarg in {"messages", "model"}:
        return None
    if any(marker in message for marker in _UNSUPPORTED_PARAM_MSG_MARKERS):
        return kwarg
    return None
