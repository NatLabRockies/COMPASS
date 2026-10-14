"""Ordinances LLM Configurations"""

import os
import logging
from collections import Counter
from abc import ABC, abstractmethod
from functools import partial, cached_property

import openai
from elm import ApiBase
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from compass.services.openai import OpenAIService
from compass.utilities import RTS_SEPARATORS
from compass.exceptions import COMPASSValueError


logger = logging.getLogger(__name__)


class _PrintableRecursiveCharacterTextSplitter(RecursiveCharacterTextSplitter):
    """RecursiveCharacterTextSplitter with __str__ method"""

    def __str__(self):
        return (
            f"RecursiveCharacterTextSplitter with"
            f"\n\t-num_separators={len(self._separators):,d}, "
            f"\n\t-chunk_size={self._chunk_size:,d}, "
            f"\n\t-chunk_overlap={self._chunk_overlap:,d}, "
            f"\n\t-is_separator_regex={self._is_separator_regex}"
        )


class LLMConfig(ABC):
    """Abstract base class representing a single LLM configuration"""

    def __init__(
        self,
        name,
        llm_call_kwargs=None,
        llm_service_rate_limit=500000,
        text_splitter_chunk_size=10_000,
        text_splitter_chunk_overlap=500,
    ):
        """

        Parameters
        ----------
        name : str
            Name of LLM.
        llm_call_kwargs : dict, optional
            Keyword arguments to be passed to the llm service ``call``
            method (i.e. `llm_service.call(**kwargs)`).
            Should *not* contain the following keys:

                - usage_tracker
                - usage_sub_label
                - messages

            These arguments are provided by the LLM Caller object.
            By default, ``None``.
        llm_service_rate_limit : int, optional
            Token rate limit (i.e. tokens per minute) of LLM service
            being used. By default, ``10_000``.
        text_splitter_chunk_size : int, optional
            Chunk size used to split the ordinance text. Parsing is
            performed on each individual chunk. Units are in token count
            of the model in charge of parsing ordinance text. Keeping
            this value low can help reduce token usage since (free)
            heuristics checks may be able to throw away irrelevant
            chunks of text before passing to the LLM.
            By default, ``10000``.
        text_splitter_chunk_overlap : int, optional
            Overlap of consecutive chunks of the ordinance text. Parsing
            is performed on each individual chunk. Units are in token
            count of the model in charge of parsing ordinance text.
            By default, ``1000``.
        """
        self.name = name
        self.llm_call_kwargs = {"timeout": 300}
        self.llm_call_kwargs.update(llm_call_kwargs or {})
        self.llm_service_rate_limit = llm_service_rate_limit
        self.text_splitter_chunk_size = text_splitter_chunk_size
        self.text_splitter_chunk_overlap = text_splitter_chunk_overlap

    @cached_property
    def text_splitter(self):
        """`TextSplitter <https://reference.langchain.com/python/langchain-text-splitters/base/TextSplitter>`_: Text splitter for ordinance text"""  # ruff:ignore[doc-line-too-long, line-too-long]
        return _PrintableRecursiveCharacterTextSplitter(
            RTS_SEPARATORS,
            chunk_size=self.text_splitter_chunk_size,
            chunk_overlap=self.text_splitter_chunk_overlap,
            length_function=partial(_count_tokens_safely, model=self.name),
            is_separator_regex=True,
        )

    @property
    @abstractmethod
    def llm_service(self):
        """LLMService: Object that can be used to submit calls to LLM"""
        raise NotImplementedError


class OpenAIConfig(LLMConfig):
    """OpenAI LLM configuration"""

    SUPPORTED_CLIENTS = {
        "openai": openai.AsyncOpenAI,
        "azure": openai.AsyncAzureOpenAI,
    }
    """Currently-supported OpenAI LLM clients"""

    _OPENAI_MODEL_NAMES = Counter()

    def __init__(
        self,
        name="gpt-4o-mini",
        llm_call_kwargs=None,
        llm_service_rate_limit=500000,
        text_splitter_chunk_size=10_000,
        text_splitter_chunk_overlap=500,
        client_type="azure",
        client_kwargs=None,
        tag=None,
    ):
        """

        Parameters
        ----------
        name : str, optional
            Name of OpenAI LLM. By default, ``"gpt-4o"``.
        llm_call_kwargs : dict, optional
            Keyword arguments to be passed to the llm service ``call``
            method (i.e. `llm_service.call(**kwargs)`).
            Should *not* contain the following keys:

                - usage_tracker
                - usage_sub_label
                - messages

            These arguments are provided by the LLM Caller object.
            By default, ``None``.
        llm_service_rate_limit : int, optional
            Token rate limit (i.e. tokens per minute) of LLM service
            being used. By default, ``10_000``.
        text_splitter_chunk_size : int, optional
            Chunk size used to split the ordinance text. Parsing is
            performed on each individual chunk. Units are in token count
            of the model in charge of parsing ordinance text. Keeping
            this value low can help reduce token usage since (free)
            heuristics checks may be able to throw away irrelevant
            chunks of text before passing to the LLM.
            By default, ``10000``.
        text_splitter_chunk_overlap : int, optional
            Overlap of consecutive chunks of the ordinance text. Parsing
            is performed on each individual chunk. Units are in token
            count of the model in charge of parsing ordinance text.
            By default, ``1000``.
        client_type : str, default="azure"
            Type of client to set up for this calling instance. Must be
            one of :obj:`OpenAIConfig.SUPPORTED_CLIENTS`.
            By default, ``"azure"``.
        client_kwargs : dict, optional
            Keyword-value pairs to pass to underlying LLM client. These
            typically include things like API keys and endpoints.
            By default, ``None``.
        tag : str, optional
            Optional tag to distinguish this model config from another
            model config for the same model `name`. This is useful if
            you have the same model (e.g. `gpt-4o-mini`) running on two
            different endpoints. If you have duplicate model names and
            don't specify this tag, one will be created for you. By
            default, ``None``.
        """
        super().__init__(
            name=name,
            llm_call_kwargs=llm_call_kwargs,
            llm_service_rate_limit=llm_service_rate_limit,
            text_splitter_chunk_size=text_splitter_chunk_size,
            text_splitter_chunk_overlap=text_splitter_chunk_overlap,
        )
        self.client_type = client_type.casefold()
        self._client_kwargs = client_kwargs or {}
        self._tag = tag or ""

        self._validate_client_type()
        self._validate_tag()

    def _validate_client_type(self):
        """Validate that user input a known client type"""
        if self.client_type not in self.SUPPORTED_CLIENTS:
            msg = (
                f"Unknown client type: {self.client_type!r}. Supported "
                f"clients: {list(self.SUPPORTED_CLIENTS)}"
            )
            raise COMPASSValueError(msg)

    def _validate_tag(self):
        """Update tag if needed"""
        self._OPENAI_MODEL_NAMES.update([self.name])
        num_models = self._OPENAI_MODEL_NAMES.get(self.name, 1)
        if num_models > 1 and not self._tag:
            self._tag = f"{num_models - 1}"

        if self._tag and not self._tag.startswith("-"):
            self._tag = f"-{self._tag}"

    @cached_property
    def client_kwargs(self):
        """dict: Parameters to pass to client initializer"""

        arg_env_pairs = []
        if self.client_type == "azure":
            arg_env_pairs = [
                ("api_key", "AZURE_OPENAI_API_KEY"),
                ("api_version", "AZURE_OPENAI_VERSION"),
                ("azure_endpoint", "AZURE_OPENAI_ENDPOINT"),
            ]
        elif self.client_type == "openai":
            arg_env_pairs = [
                ("api_key", "OPENAI_API_KEY"),
                ("base_url", "OPENAI_BASE_URL"),
            ]

        for key, env_var in arg_env_pairs:
            val = self._client_kwargs.get(key)
            self._client_kwargs[key] = val or os.environ.get(env_var)

        return self._client_kwargs

    @cached_property
    def llm_service(self):
        """LLMService: Object that can be used to submit calls to LLM"""
        client = self.SUPPORTED_CLIENTS[self.client_type](**self.client_kwargs)
        return OpenAIService(
            client,
            self.name,
            rate_limit=self.llm_service_rate_limit,
            service_tag=self._tag,
        )


class LiteLLMConfig(OpenAIConfig):
    """LiteLLM configuration for provider-prefixed model names

    Calls are made through litellm (e.g.
    ``"anthropic/claude-sonnet-4-5"`` or ``"openai/gpt-5-mini"``) using
    the same :class:`~compass.services.openai.OpenAIService` as
    :class:`OpenAIConfig`, so rate limiting, retries, usage tracking,
    and cost reporting behave identically. Token prices are taken
    from litellm's model price table unless the model is already in
    :obj:`~compass.utilities.costs.LLM_COST_REGISTRY` (e.g. via
    ``llm_costs``).

    Endpoint and key resolution (explicit ``client_kwargs`` win):

        - ``api_base`` falls back to ``LITELLM_API_BASE`` and then
          ``OPENAI_BASE_URL``.
        - ``api_key`` falls back to ``LITELLM_API_KEY`` and then
          ``OPENAI_API_KEY``, but only when an ``api_base`` is set.
          Without a proxy, litellm reads the provider's own key
          (e.g. ``ANTHROPIC_API_KEY``), so proxy keys are never sent
          to a provider directly.
    """

    def __init__(
        self,
        name="openai/gpt-4o-mini",
        llm_call_kwargs=None,
        llm_service_rate_limit=500000,
        text_splitter_chunk_size=10_000,
        text_splitter_chunk_overlap=500,
        client_kwargs=None,
        tag=None,
    ):
        """

        Parameters
        ----------
        name : str, default="openai/gpt-4o-mini"
            litellm-style model name. By default,
            ``"openai/gpt-4o-mini"``.
        llm_call_kwargs : dict, optional
            Keyword arguments to be passed to the llm service ``call``
            method. See :class:`OpenAIConfig`. By default, ``None``.
        llm_service_rate_limit : int, default=500000
            Token rate limit (i.e. tokens per minute) of LLM service
            being used. By default, ``500000``.
        text_splitter_chunk_size : int, default=10000
            Chunk size used to split the ordinance text. By default,
            ``10000``.
        text_splitter_chunk_overlap : int, default=500
            Overlap of consecutive chunks of the ordinance text.
            By default, ``500``.
        client_kwargs : dict, optional
            Keyword arguments forwarded to ``litellm.acompletion``
            on every call (e.g. ``api_key``, ``api_base``).
            By default, ``None``.
        tag : str, optional
            Optional tag to distinguish this model config from another
            model config for the same model `name`. By default,
            ``None``.
        """
        super().__init__(
            name=name,
            llm_call_kwargs=llm_call_kwargs,
            llm_service_rate_limit=llm_service_rate_limit,
            text_splitter_chunk_size=text_splitter_chunk_size,
            text_splitter_chunk_overlap=text_splitter_chunk_overlap,
            client_type="openai",
            client_kwargs=client_kwargs,
            tag=tag,
        )
        self.client_type = "litellm"

    @cached_property
    def client_kwargs(self):
        """dict: Parameters forwarded to ``litellm.acompletion``"""
        kwargs = dict(self._client_kwargs)
        api_base = kwargs.get("api_base") or kwargs.get("base_url")
        if not api_base:
            api_base = _first_env_value("LITELLM_API_BASE", "OPENAI_BASE_URL")
            if api_base:
                kwargs["api_base"] = api_base

        if api_base and not kwargs.get("api_key"):
            api_key = _first_env_value("LITELLM_API_KEY", "OPENAI_API_KEY")
            if api_key:
                kwargs["api_key"] = api_key

        return kwargs

    @cached_property
    def llm_service(self):
        """LLMService: Object that can be used to submit calls to LLM"""
        from compass.services.litellm import (  # ruff:ignore[import-outside-top-level]
            LiteLLMClient,
            register_litellm_model_cost,
        )

        register_litellm_model_cost(self.name)
        return OpenAIService(
            LiteLLMClient(**self.client_kwargs),
            self.name,
            rate_limit=self.llm_service_rate_limit,
            service_tag=self._tag,
        )


def _first_env_value(*env_vars):
    """Return the first non-empty value among environment variables"""
    for env_var in env_vars:
        if value := os.environ.get(env_var):
            return value
    return None


def _count_tokens_safely(text, model):
    """Count tokens with a conservative fallback

    ``len(text.encode("utf-8"))`` is a conservative upper bound on BPE
    token count, so it will not permit oversized chunks
    """
    try:
        return ApiBase.count_tokens(text, model=model)
    except ValueError as err:
        if "Max stack size exceeded for backtracking" not in str(err):
            raise

        logger.warning(
            "Using byte length after tokenizer backtracking failure "
            "for %d characters",
            len(text),
        )
        return len(text.encode("utf-8"))
