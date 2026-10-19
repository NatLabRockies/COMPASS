"""Ordinances LLM Configurations"""

import os
import logging
from collections import Counter
from abc import ABC, abstractmethod
from functools import partial, cached_property

import openai
from elm import ApiBase
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from compass.services import litellm as litellm_registry
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
        """Text splitter for ordinance text.

        This uses the LangChain text splitter base behavior for
        chunking ordinance text before model parsing tasks.
        """
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
    """LiteLLM proxy configuration using the shared OpenAI service

    Calls go through an OpenAI-compatible LiteLLM proxy while reusing
    :class:`~compass.services.openai.OpenAIService` for retries, usage
    tracking, and rate limiting. Token prices are taken from litellm's
    model price table unless the model is already in
    :obj:`~compass.utilities.costs.LLM_COST_REGISTRY` (e.g. via
    ``llm_costs``).

    When ``client_kwargs`` omits proxy details:

        - ``base_url`` falls back to ``LITELLM_BASE_URL``, then
          ``LITELLM_API_BASE``, then the legacy ``OPENAI_BASE_URL``.
        - ``api_key`` falls back to ``LITELLM_API_KEY``, then the
          legacy ``OPENAI_API_KEY``.

    Legacy ``OPENAI_*`` fallback is accepted only as a matched pair so
    COMPASS does not mix credentials across different endpoint
    namespaces.
    """

    def __init__(
        self,
        name="gpt-4o-mini",
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
        name : str, default="gpt-4o-mini"
            Model name or proxy alias forwarded unchanged to the
            LiteLLM proxy. By default, ``"gpt-4o-mini"``.
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
            Keyword arguments forwarded to the OpenAI-compatible proxy
            client on every call (e.g. ``api_key``, ``base_url``).
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
        """dict: Parameters forwarded to the LiteLLM proxy client"""
        kwargs = dict(self._client_kwargs)
        if "base_url" not in kwargs and "api_base" in kwargs:
            kwargs["base_url"] = kwargs.pop("api_base")

        explicit_base_url = kwargs.get("base_url")
        explicit_api_key = kwargs.get("api_key")
        litellm_base_url = os.environ.get(
            "LITELLM_BASE_URL"
        ) or os.environ.get("LITELLM_API_BASE")
        litellm_api_key = os.environ.get("LITELLM_API_KEY")
        legacy_base_url = os.environ.get("OPENAI_BASE_URL")
        legacy_api_key = os.environ.get("OPENAI_API_KEY")

        if explicit_base_url:
            if explicit_api_key:
                return kwargs

            if explicit_base_url == litellm_base_url and litellm_api_key:
                kwargs["api_key"] = litellm_api_key
                return kwargs

            if (
                explicit_base_url == legacy_base_url
                and legacy_api_key
                and not (litellm_base_url or litellm_api_key)
            ):
                kwargs["api_key"] = legacy_api_key
                return kwargs

            msg = (
                "LiteLLM proxy configs with an explicit base_url must "
                "also provide a matching api_key via client_kwargs or "
                "the corresponding proxy environment variables."
            )
            raise COMPASSValueError(msg)

        if litellm_base_url or litellm_api_key:
            if not (litellm_base_url and litellm_api_key):
                msg = (
                    "LiteLLM proxy configs require both "
                    "LITELLM_BASE_URL (or LITELLM_API_BASE) and "
                    "LITELLM_API_KEY."
                )
                raise COMPASSValueError(msg)
            kwargs["base_url"] = litellm_base_url
            kwargs["api_key"] = litellm_api_key
            return kwargs

        if legacy_base_url or legacy_api_key:
            if not (legacy_base_url and legacy_api_key):
                msg = (
                    "Legacy LiteLLM proxy fallback requires both "
                    "OPENAI_BASE_URL and OPENAI_API_KEY."
                )
                raise COMPASSValueError(msg)
            kwargs["base_url"] = legacy_base_url
            kwargs["api_key"] = legacy_api_key
            return kwargs

        msg = (
            "LiteLLM proxy configs require a proxy base_url and api_key "
            "via client_kwargs, LITELLM_BASE_URL/LITELLM_API_KEY, or "
            "legacy OPENAI_BASE_URL/OPENAI_API_KEY."
        )
        raise COMPASSValueError(msg)

    @cached_property
    def llm_service(self):
        """LLMService: Object that can be used to submit calls to LLM"""
        litellm_registry.register_litellm_model_cost(self.name)
        return OpenAIService(
            openai.AsyncOpenAI(**self.client_kwargs),
            self.name,
            rate_limit=self.llm_service_rate_limit,
            service_tag=self._tag,
        )


def _count_tokens_safely(text, model):
    """Count tokens with a conservative fallback.

    ``len(text.encode("utf-8"))`` is a conservative upper bound on BPE
    token count, so it will not permit oversized chunks.
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
