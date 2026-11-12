"""Ordinances LLM Calling classes"""

import logging
import time

from compass.utilities import llm_response_as_json
from compass.utilities.event_log import (
    ResponseRecorder,
    current_event_log,
    loggable_headers,
)
from compass.utilities.enums import LLMUsageCategory


logger = logging.getLogger(__name__)
_JSON_INSTRUCTIONS = "Return your answer as a dictionary in JSON format"


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

        3. Track message history (ChatLLMCaller) or convert output into
           JSON (JSONFromTextLLMCaller).

    Key Relationships:
        Delegates most of work to underlying ``Service`` class.
    """

    def __init__(self, llm_service, usage_tracker=None, **kwargs):
        """

        Parameters
        ----------
        llm_service : Service
            LLM service used for queries.
        usage_tracker : LLMUsageTracker, optional
            Optional tracker instance to monitor token usage during
            LLM calls. By default, ``None``.
        **kwargs
            Keyword arguments to be passed to the underlying service
            processing function (i.e. ``llm_service.call(**kwargs)``).
            Should **not** contain the following keys:

                - usage_sub_label
                - messages

            These arguments are provided by this caller object.
        """
        self.llm_service = llm_service
        self.usage_tracker = usage_tracker
        self.kwargs = kwargs

    async def _call_service(
        self,
        usage_sub_label,
        messages,
        logged_messages=None,
        conversation=None,
        **extra,
    ):
        """Call the LLM service, recording the exchange in the event log

        Parameters
        ----------
        usage_sub_label : str
            Label to store token usage under; also recorded as the
            call's purpose.
        messages : list of dict
            Messages sent to the LLM.
        logged_messages : list of dict, optional
            Messages to record instead of `messages` (e.g. only the
            newest turn of a long chat). By default, ``None``.
        conversation : str, optional
            ID the call is grouped under in the event log. By default,
            ``None``, which records the call as its own conversation.
        **extra
            Additional arguments for ``llm_service.call``.
        """
        log = current_event_log()
        if log is None:
            return await self.llm_service.call(
                usage_tracker=self.usage_tracker,
                usage_sub_label=usage_sub_label,
                messages=messages,
                **extra,
                **self.kwargs,
            )

        call_id = log.next_call_id()
        response_format = extra.get("response_format")
        log.emit(
            "llm_request",
            call=call_id,
            conversation=conversation or call_id,
            purpose=str(usage_sub_label),
            model=getattr(self.llm_service, "model_name", None),
            messages=logged_messages or messages,
            history_messages=(
                len(messages) - len(logged_messages) if logged_messages else 0
            ),
            response_format=_format_name(response_format),
            schema_id=(
                log.schema_id(response_format)
                if isinstance(response_format, dict)
                else None
            ),
            parameters=_loggable_parameters(self.kwargs),
        )
        recorder = ResponseRecorder(self.usage_tracker)
        started = time.monotonic()
        try:
            response = await self.llm_service.call(
                usage_tracker=recorder,
                usage_sub_label=usage_sub_label,
                messages=messages,
                **extra,
                **self.kwargs,
            )
        except Exception as exc:
            log.emit(
                "llm_response",
                call=call_id,
                seconds=round(time.monotonic() - started, 3),
                error=f"{type(exc).__name__}: {exc}",
                api=recorder.metadata,
            )
            raise
        log.emit(
            "llm_response",
            call=call_id,
            seconds=round(time.monotonic() - started, 3),
            response=response,
            api=recorder.metadata,
        )
        return response


class LLMCaller(BaseLLMCaller):
    """Simple LLM caller, with no memory and no parsing utilities

    See Also
    --------
    ChatLLMCaller
        Chat-like LLM calling functionality.
    JSONFromTextLLMCaller
        LLM calling functionality that extracts structured data (JSON)
        from the **text-based response**.
    SchemaOutputLLMCaller
        LLM calling functionality that allows you to specify the
        expected output schema as part of the API call.
    """

    async def call(
        self, sys_msg, content, usage_sub_label=LLMUsageCategory.DEFAULT
    ):
        """Call LLM

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
        return await self._call_service(
            usage_sub_label,
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": content},
            ],
        )


class ChatLLMCaller(BaseLLMCaller):
    """Class to support chat-like LLM calling functionality

    See Also
    --------
    LLMCaller
        Simple LLM caller, with no memory and no parsing utilities.
    JSONFromTextLLMCaller
        LLM calling functionality that extracts structured data (JSON)
        from the **text-based response**.
    SchemaOutputLLMCaller
        LLM calling functionality that allows you to specify the
        expected output schema as part of the API call.
    """

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
        usage_tracker : LLMUsageTracker, optional
            Optional tracker instance to monitor token usage during
            LLM calls. By default, ``None``.
        **kwargs
            Keyword arguments to be passed to the underlying service
            processing function (i.e. `llm_service.call(**kwargs)`).
            Should *not* contain the following keys:

                - usage_sub_label
                - messages

            These arguments are provided by this caller object.
        """
        super().__init__(llm_service, usage_tracker, **kwargs)
        self.messages = [{"role": "system", "content": system_message}]
        self._conversation_log = None
        self._conversation_id = None

    def conversation_id(self, log):
        """str: ID every turn of this chat is recorded under in `log`

        Assigned on first use, so work that drives the chat (e.g. a
        decision tree) can announce it before the first call.
        """
        if self._conversation_log is not log:
            self._conversation_log = log
            self._conversation_id = log.next_id("conversation")
        return self._conversation_id

    async def call(self, content, usage_sub_label=LLMUsageCategory.CHAT):
        """Chat with the LLM

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

        # The first turn is logged whole, so the system message is on
        # record; later turns log only what is new
        first_turn = all(m["role"] != "assistant" for m in self.messages)
        log = current_event_log()
        response = await self._call_service(
            usage_sub_label,
            self.messages,
            logged_messages=None if first_turn else self.messages[-1:],
            conversation=log and self.conversation_id(log),
        )
        if response is None:
            self.messages = self.messages[:-1]
            return None

        self.messages.append({"role": "assistant", "content": response})
        return response


class JSONFromTextLLMCaller(BaseLLMCaller):
    """Class to support structured (JSON) LLM calling functionality

    See Also
    --------
    LLMCaller
        Simple LLM caller, with no memory and no parsing utilities.
    ChatLLMCaller
        Chat-like LLM calling functionality.
    SchemaOutputLLMCaller
        LLM calling functionality that allows you to specify the
        expected output schema as part of the API call.
    """

    async def call(
        self, sys_msg, content, usage_sub_label=LLMUsageCategory.DEFAULT
    ):
        """Call LLM for structured data retrieval

        Parameters
        ----------
        sys_msg : str
            The LLM system message. If this text does not contain the
            instruction text "Return your answer as a dictionary in JSON
            format", it will be added.
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

        response = await self._call_service(
            usage_sub_label,
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": content},
            ],
        )
        return llm_response_as_json(response) if response else {}


class SchemaOutputLLMCaller(BaseLLMCaller):
    """Class to support structured (JSON) LLM calling functionality

    This class differs from :class:`JSONFromTextLLMCaller` in that it is
    designed to work with LLM services that allow you to specify the
    expected output schema as part of the API call (e.g. OpenAI function
    calling). This allows for more direct retrieval of structured data
    from the LLM, without needing to parse JSON from text-based
    responses. The expected response format should be provided as a
    parameter to the ``call`` method, and should be formatted according
    to the specifications of the underlying LLM service for structured
    output.

    See Also
    --------
    LLMCaller
        Simple LLM caller, with no memory and no parsing utilities.
    ChatLLMCaller
        Chat-like LLM calling functionality.
    JSONFromTextLLMCaller
        LLM calling functionality that extracts structured data (JSON)
        from the **text-based response**.
    """

    async def call(
        self,
        sys_msg,
        content,
        response_format,
        usage_sub_label=LLMUsageCategory.DEFAULT,
    ):
        """Call LLM for structured data retrieval

        Parameters
        ----------
        sys_msg : str
            The LLM system message. If this text does not contain the
            instruction text "Return your answer as a dictionary in JSON
            format", it will be added.
        content : str
            LLM call content (typically some text to extract info from).
        usage_sub_label : str, optional
            Label to store token usage under. By default, ``"default"``.
        response_format : dict
            Dictionary specifying the expected response format. This
            will be passed to the underlying LLM service (e.g. OpenAI)
            and should be formatted according to that service's
            specifications for structured output. For example, for
            OpenAI GPT models, this should be a dictionary with the
            following keys:

                - `type`: Should be set to `"json_schema"` to indicate
                  that the expected output is structured JSON.
                - `json_schema`: A dictionary specifying the expected
                  JSON schema of the output. This should include the
                  following keys:

                    - `name`: A string name for this response format
                      (e.g. "extracted_features").
                    - `strict`: A boolean indicating whether the LLM
                      should strictly adhere to the provided schema
                      (i.e. not include any keys not specified in the
                      schema). If `True`, the LLM will be instructed to
                      only include keys specified in the `schema` field.
                      If `False`, the LLM may include additional keys
                      not specified in the `schema` field.
                    - `schema`: A dictionary specifying the expected
                      JSON schema of the output. This should be
                      formatted according to JSON Schema specifications,
                      and should define the expected structure of the
                      output JSON object. For example, it may specify
                      that the output should be an object with certain
                      required properties, and the expected data types
                      of those properties.

        Returns
        -------
        dict
            Dictionary containing the LLM-extracted features. Dictionary
            may be empty if there was an error during the LLM call.
        """
        response = await self._call_service(
            usage_sub_label,
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": content},
            ],
            response_format=response_format,
        )
        return llm_response_as_json(response) if response else {}


def _add_json_instructions_if_needed(system_message):
    """Add JSON instruction to system message if needed"""
    if "JSON format" not in system_message:
        logger.debug(
            "JSON instructions not found in system message. Adding..."
        )
        system_message = f"{system_message}\n{_JSON_INSTRUCTIONS}."
        logger.debug("New system message:\n%s", system_message)
    return system_message


def _format_name(response_format):
    """Name of a structured-output schema, if one was requested"""
    if not isinstance(response_format, dict):
        return None
    return response_format.get("json_schema", {}).get("name") or (
        response_format.get("type")
    )


def _loggable_parameters(kwargs):
    """LLM call settings that can be written to the event log

    Headers are redacted the way HTTP headers are, since they may carry
    credentials (e.g. ``extra_headers={"Authorization": ...}``).
    """
    return {
        key: loggable_headers(value) if "headers" in key.casefold() else value
        for key, value in kwargs.items()
        if isinstance(value, (str, int, float, bool, type(None), list, dict))
    }
