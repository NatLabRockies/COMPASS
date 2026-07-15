"""COMPASS extraction schema-based plugin component implementations"""

import json
import asyncio
import logging
from datetime import datetime
from abc import ABC, abstractmethod

import pandas as pd
from elm import ApiBase

from compass.llm.calling import SchemaOutputLLMCaller
from compass.plugin import BaseParser, BaseTextCollector, BaseTextExtractor
from compass.utilities.enums import LLMUsageCategory
from compass.utilities.parsing import merge_overlapping_texts


logger = logging.getLogger(__name__)

SCOPE_SENTINEL_ALL = "all"
SCOPE_SENTINEL_OTHER = "other"

_DEFAULT_TEXT_SCOPE_SYSTEM_PROMPT = """\
You are a structured extraction scope validator. Given a text chunk, \
determine whether the chunk is within the given extraction scope. \
Focus on any explicit scope limitations, including technology, document \
type, regulatory subject, exclusions, and similar boundaries. Be strict and \
literal: only mark the chunk in scope when the text clearly pertains to the \
intended domain. Note that a broader ordinance can still be in scope \
when the excerpt contains an explicit definition, permit, prohibition, or \
requirement for the target technology. Do not infer beyond the text. \
If the scope match is unclear, return ``true``. Keep the response concise \
and consistent.\
"""
_DEFAULT_TEXT_SCOPE_MAIN_PROMPT = """\
Determine whether this text excerpt is within the following extraction \
scope. Only decide whether the chunk matches the scope. Do not decide yet \
whether it contains enough substantive context for extraction.\

SCOPE:

{scope}

TEXT:

{text}

Think before you answer.\
"""
_DEFAULT_TEXT_COLLECTION_SYSTEM_PROMPT = """\
You are a structured extraction validator. You receive:
1) A text chunk.
2) An extraction schema that specifies the extraction criteria.

Determine whether the chunk contains content that matches any of the schema's \
extraction criteria. Be strict and literal: only mark relevant if the chunk \
clearly contains at least one extraction component. Do not infer beyond the \
text. Do not treat chapter titles, section titles, tables of contents, \
navigation links, cross-reference lists, or citation-only indexes as relevant \
on their own; mark them relevant only when they also include operative \
provisions, definitions, prohibitions, or other substantive regulatory text \
that supports extraction. If relevant, summarize the specific matching \
content; if not, state why it does not meet the schema's requirements. \
Keep the response concise and consistent.\
"""
_DEFAULT_TEXT_COLLECTION_MAIN_PROMPT = """\
Determine whether this text excerpt contains any information relevant to \
the following extraction schema:

{schema}

TEXT:

{text}

Think before you answer.\
"""
_DEFAULT_TEXT_EXTRACTOR_SYSTEM_PROMPT = """\
You are a text extraction assistant. Your job is to extract only verbatim, \
**unmodified** excerpts from the provided text. Do not interpret or \
paraphrase. Do not summarize. Only return exactly copied segments that match \
the specified extraction scope/domain. If the relevant content appears within \
a table, return the entire table, including headers and footers, exactly as \
formatted.\
"""
_DEFAULT_TEXT_EXTRACTOR_MAIN_PROMPT = """\
# CONTEXT #
We want to reduce the provided excerpt to only contain information for the \
domain relevant to the following extraction schema:

{schema}

The extracted text will be used for structured data extraction following this \
schema, so it must be both **comprehensive** (retaining all relevant details) \
and **focused** (excluding unrelated content), with **zero rewriting or \
paraphrasing**. Ensure that all retained information is **directly
applicable** to the extraction task while preserving full context and accuracy.

# OBJECTIVE #
Extract all text **pertaining to the extraction schema domain** from the \
provided excerpt.

# RESPONSE #
Follow these guidelines carefully:

1. ## Formatting & Structure ##:
- **Preserve _all_ section titles, headers, and numberings** for reference.
- **Maintain the original wording, formatting, and structure** to ensure \
accuracy.

2. ## Output Handling ##:
- This is a strict extraction task — act like a text filter, **not** a \
summarizer or writer.
- Do not add, explain, reword, or summarize anything.
- The output must be a **copy-paste** of the original excerpt. **Absolutely \
no paraphrasing or rewriting.**
- The output must consist **only** of contiguous or discontiguous verbatim \
blocks copied from the input.
- The only allowed change is to remove irrelevant sections of text. You can \
remove irrelevant text from within sections, but you cannot add any new text \
or modify the text you keep in any way.
- If **no relevant text** is found, return null.

# TEXT #

{text}

"""
_DEFAULT_DATA_PARSER_MAIN_PROMPT = """\
Extract all applicable {desc}features explicitly supported by following text:

{text}

Think before you answer\
"""
_DEFAULT_DATA_PARSER_SYSTEM_PROMPT = """\
You are a legal scholar extracting structured data from {desc}documents. \
Follow all instructions in the schema descriptions carefully.\
"""
_DATA_PARSER_ADDITIONAL_CONTEXT = """\
# ADDITIONAL CONTEXT #
- Today's date is {todays_date}. If you are extracting a moratorium or \
temporary restriction that includes an explicit end date that has already \
passed as of today, treat it as expired and omit that prohibition feature.\
"""


class SchemaBasedTextCollector(SchemaOutputLLMCaller, BaseTextCollector, ABC):
    """Text extractor based on a chain of prompts"""

    @property
    @abstractmethod
    def SCHEMA(self):  # ruff:ignore[invalid-function-name]
        """dict: Extraction schema"""
        raise NotImplementedError

    @property
    @abstractmethod
    def SCOPE_VALIDATION_OUTPUT_SCHEMA(self):  # ruff:ignore[invalid-function-name]
        """dict: Scope validation output schema"""
        raise NotImplementedError

    @property
    @abstractmethod
    def CONTENT_VALIDATION_OUTPUT_SCHEMA(self):  # ruff:ignore[invalid-function-name]
        """dict: Content validation output schema"""
        raise NotImplementedError

    _SSP = _DEFAULT_TEXT_SCOPE_SYSTEM_PROMPT
    _SMP = _DEFAULT_TEXT_SCOPE_MAIN_PROMPT

    _SP = _DEFAULT_TEXT_COLLECTION_SYSTEM_PROMPT
    _MP = _DEFAULT_TEXT_COLLECTION_MAIN_PROMPT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chunks = {}

    @property
    def relevant_text(self):
        """str: Combined extraction text from the individual chunks"""
        if not self._chunks:
            logger.debug(
                "No relevant extraction chunk(s) found in original text",
            )
            return ""

        logger.debug(
            "Grabbing %d extraction chunk(s) from original text at these "
            "indices: %s",
            len(self._chunks),
            list(self._chunks),
        )

        text = [self._chunks[ind] for ind in sorted(self._chunks)]
        return merge_overlapping_texts(text)

    async def check_chunk(self, chunk_parser, ind):
        """Check a chunk at a given ind to see if it contains ordinance

        Parameters
        ----------
        chunk_parser : ParseChunksWithMemory
            Instance that contains a ``parse_from_ind`` method.
        ind : int
            Index of the chunk to check.

        Returns
        -------
        bool
            Boolean flag indicating whether or not the text in the chunk
            contains large wind energy conversion system ordinance text.
        """
        if "$scope" in self.SCHEMA:
            passed_scope = await chunk_parser.parse_from_ind(
                ind,
                key="matches_scope",
                llm_call_callback=self._check_chunk_scope_with_prompt,
            )

            if not passed_scope:
                logger.debug(
                    "Text at ind %d did not pass collection step: scope match",
                    ind,
                )
                return False

            logger.debug(
                "Text at ind %d passed collection step: scope match", ind
            )

        passed_filter = await chunk_parser.parse_from_ind(
            ind,
            key="contains_relevant_text",
            llm_call_callback=self._check_chunk_with_prompt,
        )

        if not passed_filter:
            logger.debug(
                "Text at ind %d did not pass collection step: relevant "
                "context",
                ind,
            )
            return False

        logger.debug(
            "Text at ind %d passed collection step: relevant context", ind
        )

        self._store_chunk(chunk_parser, ind)
        logger.debug("Added text chunk at ind %d to extraction text", ind)
        return True

    async def _check_chunk_scope_with_prompt(self, key, text_chunk):
        """Call LLM on a chunk of text to check schema scope"""
        scope = self.SCHEMA.get("$scope")
        if not scope:
            logger.debug(
                "No $scope defined in schema; skipping scope check for "
                "text chunk: %s",
                text_chunk,
            )
            return True

        main_prompt = self._SMP.format(
            schema=self.SCHEMA, scope=scope, text=text_chunk
        )
        logger.debug("Checking text chunk scope with LLM: %s", text_chunk)
        logger.debug_to_file("\t- System Message:\n%s", self._SSP)
        logger.debug_to_file("\t- Main prompt:\n%s", main_prompt)
        content = await self.call(
            sys_msg=self._SSP,
            content=main_prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "chunk_scope_validation",
                    "strict": True,
                    "schema": self.SCOPE_VALIDATION_OUTPUT_SCHEMA,
                },
            },
            usage_sub_label=LLMUsageCategory.DOCUMENT_CONTENT_VALIDATION,
        )
        logger.debug("LLM scope response:\n%s", json.dumps(content, indent=4))
        return content.get(key, False)

    async def _check_chunk_with_prompt(self, key, text_chunk):
        """Call LLM on a chunk of text to check relevant context"""
        main_prompt = self._MP.format(schema=self.SCHEMA, text=text_chunk)
        logger.debug(
            "Checking text chunk for relevant context with LLM: %s",
            text_chunk,
        )
        logger.debug_to_file("\t- System Message:\n%s", self._SP)
        logger.debug_to_file("\t- Main prompt:\n%s", main_prompt)
        content = await self.call(
            sys_msg=self._SP,
            content=main_prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "chunk_validation",
                    "strict": True,
                    "schema": self.CONTENT_VALIDATION_OUTPUT_SCHEMA,
                },
            },
            usage_sub_label=LLMUsageCategory.DOCUMENT_CONTENT_VALIDATION,
        )
        logger.debug(
            "LLM relevant context response:\n%s",
            json.dumps(content, indent=4),
        )
        return content.get(key, False)

    def _store_chunk(self, parser, chunk_ind):
        """Store chunk and its neighbors if it is not already stored"""
        for offset in range(1 - parser.num_to_recall, 2):
            ind_to_grab = chunk_ind + offset
            if ind_to_grab < 0 or ind_to_grab >= len(parser.text_chunks):
                continue
            self._chunks.setdefault(
                ind_to_grab, parser.text_chunks[ind_to_grab]
            )


class SchemaBasedTextExtractor(SchemaOutputLLMCaller, BaseTextExtractor):
    """Schema-based text extractor"""

    @property
    @abstractmethod
    def SCHEMA(self):  # ruff:ignore[invalid-function-name]
        """dict: Extraction schema"""
        raise NotImplementedError

    @property
    @abstractmethod
    def OUTPUT_SCHEMA(self):  # ruff:ignore[invalid-function-name]
        """dict: Validation output schema"""
        raise NotImplementedError

    _SP = _DEFAULT_TEXT_EXTRACTOR_SYSTEM_PROMPT
    _MP = _DEFAULT_TEXT_EXTRACTOR_MAIN_PROMPT

    @property
    def parsers(self):
        """Iterable of parsers provided by this extractor

        Yields
        ------
        name : str
            Name describing the type of text output by the parser.
        parser : callable
            Async function that takes a ``text_chunks`` input and
            outputs parsed text.
        """
        yield self.OUT_LABEL, self._process

    async def _process(self, text_chunks):
        """Perform extraction processing"""

        logger.info(
            "Extracting summary text from %d text chunks asynchronously "
            "using LLM: %r...",
            len(text_chunks),
            self.llm_service.model_name,
        )
        outer_task_name = asyncio.current_task().get_name()
        summaries = [
            asyncio.create_task(
                self.call(
                    sys_msg=self._SP,
                    content=self._MP.format(schema=self.SCHEMA, text=chunk),
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "text_extraction",
                            "strict": True,
                            "schema": self.OUTPUT_SCHEMA,
                        },
                    },
                    usage_sub_label=self._USAGE_LABEL,
                ),
                name=outer_task_name,
            )
            for chunk in text_chunks
        ]
        summary_chunks = await asyncio.gather(*summaries)
        summary_chunks = [
            chunk.get("domain_relevant_text") for chunk in summary_chunks
        ]

        text_summary = merge_overlapping_texts(summary_chunks)
        logger.debug(
            "Final summary contains %d tokens",
            ApiBase.count_tokens(
                text_summary,
                model=self.kwargs.get("model", "gpt-4"),
            ),
        )
        return text_summary


class SchemaOrdinanceParser(SchemaOutputLLMCaller, BaseParser):
    """Base class for parsing structured data"""

    DATA_TYPE_SHORT_DESC = None
    """
    Optional short description of the type of data being extracted

    Examples
    --------
    - "wind energy ordinance"
    - "solar energy ordinance"
    - "water rights"
    - "resource management plan geothermal restriction"
    """
    SYSTEM_PROMPT = _DEFAULT_DATA_PARSER_SYSTEM_PROMPT
    """System prompt to use for parsing structured data with an LLM"""

    @property
    @abstractmethod
    def SCHEMA(self):  # ruff:ignore[invalid-function-name]
        """dict: Extraction schema"""
        raise NotImplementedError

    @property
    @abstractmethod
    def QUALITATIVE_FEATURES(self):  # ruff:ignore[invalid-function-name]
        """set: **Lowercase** feature names of qualitative features"""
        raise NotImplementedError

    @property
    @abstractmethod
    def POSSIBLE_OUT_COLS(self):  # ruff:ignore[invalid-function-name]
        """list: List of possible output column names"""
        raise NotImplementedError

    _MP = _DEFAULT_DATA_PARSER_MAIN_PROMPT

    async def parse(self, text, jurisdiction):
        """Parse text and extract structured data

        Parameters
        ----------
        text : str
            Text which may or may not contain information relevant to
            the current extraction.
        jurisdiction : Jurisdiction
            Jurisdiction object corresponding to the document being
            parsed. This is passed in case the system prompt needs any
            information about the jurisdiction to successfully extract
            values from the text.

        Returns
        -------
        pandas.DataFrame or None
            DataFrame containing structured extracted data. Can also
            be ``None`` if no relevant values can be parsed from the
            text.
        """
        desc = (
            f"{self.DATA_TYPE_SHORT_DESC} "
            if self.DATA_TYPE_SHORT_DESC
            else ""
        )

        todays_date = datetime.now().strftime("%B %d, %Y")
        sys_prompt = (
            f"{self.SYSTEM_PROMPT}\n\n{_DATA_PARSER_ADDITIONAL_CONTEXT}"
        )
        sys_prompt = sys_prompt.format(
            desc=desc,
            schema=self.SCHEMA,
            todays_date=todays_date,
            jur_type=jurisdiction.type,
            jur_state=jurisdiction.state,
            jur_county=jurisdiction.county or "Unknown",
            jur_subdivision_name=jurisdiction.subdivision_name or "Unknown",
            jur_code=jurisdiction.code or "unknown",
            jur_website_url=jurisdiction.website_url or "unknown",
            jur_full_name=jurisdiction.full_name,
            jur_full_name_the_prefixed=jurisdiction.full_name_the_prefixed,
            jur_full_subdivision_phrase=(
                jurisdiction.full_subdivision_phrase or "unknown subdivision"
            ),
            jur_full_subdivision_phrase_the_prefixed=(
                jurisdiction.full_subdivision_phrase_the_prefixed
                or "the unknown subdivision"
            ),
            jur_full_county_phrase=(
                jurisdiction.full_county_phrase or "Unknown County"
            ),
        )

        main_prompt = self._MP.format(desc=desc, text=text)
        logger.debug(
            "Extracting ordinances with LLM: %r", self.llm_service.model_name
        )
        logger.debug_to_file("\t- System Message:\n%s", sys_prompt)
        logger.debug_to_file("\t- Main prompt:\n%s", main_prompt)

        extraction = await self.call(
            sys_msg=sys_prompt,
            content=main_prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_data_extraction",
                    "strict": True,
                    "schema": self.SCHEMA,
                },
            },
            usage_sub_label=LLMUsageCategory.ORDINANCE_VALUE_EXTRACTION,
        )
        logger.debug_to_file(
            "LLM response:\n%s", json.dumps(extraction, indent=4)
        )
        data = extraction["outputs"]
        if not data:
            logger.debug(
                "LLM did not extract any relevant features from the text"
            )
            return None

        return self._to_dataframe(data)

    def _to_dataframe(self, data):
        """Convert LLM output to a DataFrame"""

        output_items = self.SCHEMA["properties"]["outputs"]["items"]
        all_features = output_items["properties"]["feature"]["enum"]

        quant = [
            feat.casefold() not in self.QUALITATIVE_FEATURES
            for feat in all_features
        ]

        df = pd.DataFrame(data)
        full_df = pd.DataFrame(
            {"feature": all_features, "quantitative": quant}
        )
        full_df = full_df.merge(df, on="feature", how="left")
        full_df = self._apply_scope_defaults(full_df)

        ignore_cols = {
            "county",
            "state",
            "subdivision",
            "jurisdiction_type",
            "FIPS",
            "feature",
            "quantitative",
        }
        out_cols = [
            col.name
            for col in self.POSSIBLE_OUT_COLS
            if col.name in full_df.columns and col.name not in ignore_cols
        ]
        return full_df[["feature", *out_cols, "quantitative"]]

    def _apply_scope_defaults(self, df):
        """Fill placeholder scope entries with the 'all' sentinel"""

        output_items = self.SCHEMA["properties"]["outputs"]["items"]
        if "scope" not in output_items.get("properties", {}):
            return df
        if "scope" not in df.columns:
            df["scope"] = SCOPE_SENTINEL_ALL
        df["scope"] = df["scope"].fillna(SCOPE_SENTINEL_ALL)
        return df
