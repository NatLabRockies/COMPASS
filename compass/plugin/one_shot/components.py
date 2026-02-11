"""COMPASS extraction schema-based plugin component implementations"""

import logging
from abc import ABC, abstractmethod

import pandas as pd

from compass.llm.calling import SchemaOutputLLMCaller
from compass.plugin import BaseParser, BaseTextCollector
from compass.utilities.enums import LLMUsageCategory
from compass.utilities.parsing import merge_overlapping_texts


logger = logging.getLogger(__name__)
_TEXT_COLLECTION_SYSTEM_PROMPT = """\
You are a structured extraction validator. You receive:
1) A text chunk.
2) An extraction schema that specifies the exact criteria for relevance \
(e.g., technology type, document type, required data fields).

Determine whether the chunk contains content that matches any of the \
schema's criteria. Be strict and literal: only mark relevant if the chunk \
clearly addresses the specific technology and document scope described in \
the schema. Do not infer beyond the text. If relevant, summarize the \
specific matching content; if not, state why it does not meet the schema's \
requirements. Keep the response concise and consistent.\
"""
_TEXT_COLLECTION_MAIN_PROMPT = """\
Determine whether this text excerpt contains any information relevant to \
the following extraction schema:

{schema}

TEXT:

{text}

Think before you answer.\
"""
_DATA_PARSER_MAIN_PROMPT = """\
Extract all {desc}features from the following text:

{text}

Think before you answer"""
_DATA_PARSER_SYSTEM_PROMPT = """\
You are a legal scholar extracting structured data from {desc}documents. \
Follow all instructions in the schema descriptions carefully.\
"""


class SchemaBasedTextCollector(SchemaOutputLLMCaller, BaseTextCollector, ABC):
    """Text extractor based on a chain of prompts"""

    @property
    @abstractmethod
    def SCHEMA(self):  # noqa: N802
        """dict: Extraction schema"""
        raise NotImplementedError

    @property
    @abstractmethod
    def OUTPUT_SCHEMA(self):  # noqa: N802
        """dict: Validation output schema"""
        raise NotImplementedError

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
        key = "contains_relevant_text"
        passed_filter = await chunk_parser.parse_from_ind(
            ind,
            key=key,
            llm_call_callback=self._check_chunk_with_prompt,
        )

        if not passed_filter:
            logger.debug("Text at ind %d did not pass collection step", ind)
            return False

        logger.debug("Text at ind %d passed collection step ", ind)

        self._store_chunk(chunk_parser, ind)
        logger.debug("Added text chunk at ind %d to extraction text", ind)
        return True

    async def _check_chunk_with_prompt(self, key, text_chunk):
        """Call LLM on a chunk of text to check for ordinance"""
        content = await self.call(
            sys_msg=_TEXT_COLLECTION_SYSTEM_PROMPT,
            content=_TEXT_COLLECTION_MAIN_PROMPT.format(
                schema=self.SCHEMA, text=text_chunk
            ),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "chunk_validation",
                    "strict": True,
                    "schema": self.OUTPUT_SCHEMA,
                },
            },
            usage_sub_label=LLMUsageCategory.DOCUMENT_CONTENT_VALIDATION,
        )
        logger.debug("LLM response: %s", content)
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


class SchemaOrdinanceParser(SchemaOutputLLMCaller, BaseParser):
    """Base class for parsing structured data"""

    DATA_TYPE_SHORT_DESC = None
    """Optional short description of the type of data being extracted

    Examples
    --------
    - "wind energy ordinance"
    - "solar energy ordinance"
    - "water rights"
    - "resource management plan geothermal restriction"
    """
    SYSTEM_PROMPT = _DATA_PARSER_SYSTEM_PROMPT
    """System prompt to use for parsing structured data with an LLM"""

    @property
    @abstractmethod
    def SCHEMA(self):  # noqa: N802
        """dict: Extraction schema"""
        raise NotImplementedError

    async def parse(self, text):
        """Parse text and extract structured data

        Parameters
        ----------
        text : str
            Text which may or may not contain information relevant to
            the current extraction.

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

        extraction = await self.call(
            sys_msg=self.SYSTEM_PROMPT.format(desc=desc),
            content=_DATA_PARSER_MAIN_PROMPT.format(desc=desc, text=text),
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

        known_qual_features = set(
            self.SCHEMA.get("$definitions", {})
            .get("qualitative_restrictions", {})
            .get("properties", {})
        )
        quant = [feat not in known_qual_features for feat in all_features]

        df = pd.DataFrame(data)
        full_df = pd.DataFrame(
            {"feature": all_features, "quantitative": quant}
        )
        full_df = full_df.merge(df, on="feature", how="left")

        return full_df[
            ["feature", "value", "units", "section", "summary", "quantitative"]
        ]
