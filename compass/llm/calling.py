"""Ordinances LLM Calling classes"""

import os
import logging
from functools import partial, cached_property

import openai
from elm import ApiBase
from langchain.text_splitter import RecursiveCharacterTextSplitter

from compass.services.openai import OpenAIService
from compass.utilities import llm_response_as_json, RTS_SEPARATORS
from compass.utilities.enums import LLMUsageCategory
from compass.exceptions import COMPASSValueError


logger = logging.getLogger(__name__)
_JSON_INSTRUCTIONS = "Return your answer in JSON format"


class BaseLLMCaller:
    """Class to support LLM calling functionality

    Purpose:
        Helper classes to call LLMs.
    Responsibilities:
        1. Use a service (e.g.
           :class:`~compass.services.openai.OpenAIService`) to query an
           LLM.
        2. Maintain a useful context to simplify LLM query.
            - Typically these classes are initialized with a single LLM
              model (and optionally a usage tracker)
            - This context is passed to every ``Service.call``
              invocation, allowing user to focus on only the message.
        3. Track message history
           (:class:`~compass.llm.calling.ChatLLMCaller`)
           or convert output into JSON
           (:class:`~compass.llm.calling.StructuredLLMCaller`).
    Key Relationships:
        Delegates most of work to underlying ``Service`` class.
    """

    def __init__(self, llm_service, usage_tracker=None, **kwargs):
        """

        Parameters
        ----------
        llm_service : compass.services.base.Service
            LLM service used for queries.
        usage_tracker : compass.services.usage.UsageTracker, optional
            Optional tracker instance to monitor token usage during
            LLM calls. By default, ``None``.
        **kwargs
            Keyword arguments to be passed to the underlying service
            processing function (i.e. `llm_service.call(**kwargs)`).
            Should *not* contain the following keys:

                - usage_tracker
                - usage_sub_label
                - messages

            These arguments are provided by this caller object.
        """
        self.llm_service = llm_service
        self.usage_tracker = usage_tracker
        self.kwargs = kwargs


class LLMCaller(BaseLLMCaller):
    """Simple LLM caller, with no memory and no parsing utilities."""

    async def call(
        self, sys_msg, content, usage_sub_label=LLMUsageCategory.DEFAULT
    ):
        """Call LLM.

        Parameters
        ----------
        sys_msg : str
            The LLM system message.
        content : str
            Your chat message for the LLM.
        usage_sub_label : str, optional
            Label to store token usage under. By default, ``"default"``.

        Returns
        -------
        str or None
            The LLM response, as a string, or ``None`` if something went
            wrong during the call.
        """
        return await self.llm_service.call(
            usage_tracker=self.usage_tracker,
            usage_sub_label=usage_sub_label,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": content},
            ],
            **self.kwargs,
        )


class ChatLLMCaller(BaseLLMCaller):
    """Class to support chat-like LLM calling functionality."""

    def __init__(
        self, llm_service, system_message, usage_tracker=None, **kwargs
    ):
        """

        Parameters
        ----------
        llm_service : compass.services.base.Service
            LLM service used for queries.
        system_message : str
            System message to use for chat with LLM.
        usage_tracker : compass.services.usage.UsageTracker, optional
            Optional tracker instance to monitor token usage during
            LLM calls. By default, ``None``.
        **kwargs
            Keyword arguments to be passed to the underlying service
            processing function (i.e. `llm_service.call(**kwargs)`).
            Should *not* contain the following keys:

                - usage_tracker
                - usage_sub_label
                - messages

            These arguments are provided by this caller object.
        """
        super().__init__(llm_service, usage_tracker, **kwargs)
        self.messages = [{"role": "system", "content": system_message}]

    async def call(self, content, usage_sub_label=LLMUsageCategory.CHAT):
        """Chat with the LLM.

        Parameters
        ----------
        content : str
            Your chat message for the LLM.
        usage_sub_label : str, optional
            Label to store token usage under. By default, ``"chat"``.

        Returns
        -------
        str or None
            The LLM response, as a string, or ``None`` if something went
            wrong during the call.
        """
        self.messages.append({"role": "user", "content": content})

        response = await self.llm_service.call(
            usage_tracker=self.usage_tracker,
            usage_sub_label=usage_sub_label,
            messages=self.messages,
            **self.kwargs,
        )
        if response is None:
            self.messages = self.messages[:-1]
            return None

        self.messages.append({"role": "assistant", "content": response})
        return response


class StructuredLLMCaller(BaseLLMCaller):
    """Class to support structured (JSON) LLM calling functionality."""

    async def call(
        self, sys_msg, content, usage_sub_label=LLMUsageCategory.DEFAULT
    ):
        """Call LLM for structured data retrieval.

        Parameters
        ----------
        sys_msg : str
            The LLM system message. If this text does not contain the
            instruction text "Return your answer in JSON format", it
            will be added.
        content : str
            LLM call content (typically some text to extract info from).
        usage_sub_label : str, optional
            Label to store token usage under. By default, ``"default"``.

        Returns
        -------
        dict
            Dictionary containing the LLM-extracted features. Dictionary
            may be empty if there was an error during the LLM call.
        """
        sys_msg = _add_json_instructions_if_needed(sys_msg)

        response = await self.llm_service.call(
            usage_tracker=self.usage_tracker,
            usage_sub_label=usage_sub_label,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": content},
            ],
            **self.kwargs,
        )
        return llm_response_as_json(response) if response else {}


class LLMCallerArgs:
    """Arguments to set up an LLM Caller instance initializer"""

    SUPPORTED_CLIENTS = {
        "openai": openai.AsyncOpenAI,
        "azure": openai.AsyncAzureOpenAI,
    }
    """Currently-supported LLM clients"""

    def __init__(
        self,
        name="gpt-4o",
        llm_call_kwargs=None,
        llm_service_rate_limit=4000,
        text_splitter_chunk_size=10_000,
        text_splitter_chunk_overlap=1000,
        client_type="azure",
        client_kwargs=None,
    ):
        """

        Parameters
        ----------
        name : str, optional
            Name of LLM model to perform scraping.
            By default, ``"gpt-4o"``.
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
            one of :obj:`LLMCallerArgs.SUPPORTED_CLIENTS`.
            By default, ``"azure"``.
        client_kwargs : dict, optional
            Keyword-value pairs to pass to underlying LLM client. These
            typically include things like API keys and endpoints.
            By default, ``None``.
        """
        self.name = name
        self.llm_call_kwargs = llm_call_kwargs or {}
        self.llm_service_rate_limit = llm_service_rate_limit
        self.text_splitter_chunk_size = text_splitter_chunk_size
        self.text_splitter_chunk_overlap = text_splitter_chunk_overlap
        self.client_type = client_type.casefold()
        self._client_kwargs = client_kwargs or {}

        self._validate_client_type()

    def _validate_client_type(self):
        """Validate that user input a known client type"""
        if self.client_type not in self.SUPPORTED_CLIENTS:
            msg = (
                f"Unknown client type: {self.client_type!r}. Supported "
                f"clients: {list(self.SUPPORTED_CLIENTS)}"
            )
            raise COMPASSValueError(msg)

    @cached_property
    def client_kwargs(self):
        """dict: Parameters to pass to client initializer"""
        if self.client_type == "azure":
            arg_env_pairs = [
                ("api_key", "AZURE_OPENAI_API_KEY"),
                ("api_version", "AZURE_OPENAI_VERSION"),
                ("azure_endpoint", "AZURE_OPENAI_ENDPOINT"),
            ]
            for key, env_var in arg_env_pairs:
                if self._client_kwargs.get(key) is None:
                    self._client_kwargs[key] = os.environ.get(env_var)

        return self._client_kwargs

    @cached_property
    def text_splitter(self):
        """TextSplitter: Object that can be used to chunk text"""
        return RecursiveCharacterTextSplitter(
            RTS_SEPARATORS,
            chunk_size=self.text_splitter_chunk_size,
            chunk_overlap=self.text_splitter_chunk_overlap,
            length_function=partial(ApiBase.count_tokens, model=self.name),
            is_separator_regex=True,
        )

    @cached_property
    def llm_service(self):
        """LLMService: Object that can be used to submit calls to LLM"""
        client = self.SUPPORTED_CLIENTS[self.client_type](**self.client_kwargs)
        return OpenAIService(
            client, self.name, rate_limit=self.llm_service_rate_limit
        )


def _add_json_instructions_if_needed(system_message):
    """Add JSON instruction to system message if needed."""
    if "JSON format" not in system_message:
        logger.debug(
            "JSON instructions not found in system message. Adding..."
        )
        system_message = f"{system_message}\n{_JSON_INSTRUCTIONS}."
        logger.debug("New system message:\n%s", system_message)
    return system_message
