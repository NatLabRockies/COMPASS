"""Helper classes for ordinance plugins"""

import asyncio
import logging
from warnings import warn
from textwrap import dedent
from itertools import chain
from functools import cached_property, partial
from abc import ABC, abstractmethod
from contextlib import contextmanager

import pandas as pd
from elm import ApiBase

from compass.llm.calling import (
    LLMCaller,
    BaseLLMCaller,
    ChatLLMCaller,
    JSONFromTextLLMCaller,
)
from compass.plugin.interface import (
    BaseHeuristic,
    BaseTextCollector,
    FilteredExtractionPlugin,
)
from compass.services.threaded import CLEANED_FP_REGISTRY
from compass.extraction import extract_ordinance_values
from compass.utilities.enums import LLMTasks, LLMUsageCategory
from compass.utilities.ngrams import convert_text_to_sentence_ngrams
from compass.utilities.parsing import (
    clean_backticks_from_llm_response,
    extract_ord_year_from_doc_attrs,
    merge_overlapping_texts,
)
from compass.utilities import num_ordinances_dataframe
from compass.warn import COMPASSWarning
from compass.exceptions import COMPASSPluginConfigurationError
from compass.pb import COMPASS_PB


logger = logging.getLogger(__name__)
EXCLUDE_FROM_ORD_DOC_CHECK = {
    # if doc only contains these, it's not good enough to count as an
    # ordinance. Note that prohibitions are explicitly not on this list
    "color",
    "decommissioning",
    "lighting",
    "visual impact",
    "glare",
    "repowering",
    "fencing",
    "climbing prevention",
    "signage",
    "soil",
    "primary use districts",
    "special use districts",
    "accessory use districts",
}


class BaseTextExtractor(BaseLLMCaller, ABC):
    """Extract succinct extraction text from input"""

    TASK_DESCRIPTION = "Condensing text for extraction"
    """Task description to show in progress bar"""

    TASK_ID = "text_extraction"
    """ID to use for this extraction for linking with LLM configs"""

    _USAGE_LABEL = LLMUsageCategory.DOCUMENT_ORDINANCE_SUMMARY

    @property
    @abstractmethod
    def IN_LABEL(self):  # noqa: N802
        """str: Identifier for text ingested by this class"""
        raise NotImplementedError

    @property
    @abstractmethod
    def OUT_LABEL(self):  # noqa: N802
        """str: Identifier for final text extracted by this class"""
        raise NotImplementedError

    @property
    @abstractmethod
    def parsers(self):
        """Generator: Generator of (key, extractor) pairs

        `extractor` should be an async callable that accepts a list of
        text chunks and returns the shortened (succinct) text to be used
        for extraction. The `key` should be a string identifier for the
        text returned by the extractor. Multiple (key, extractor) pairs
        can be chained in generator order to iteratively refine the
        text for extraction.
        """
        raise NotImplementedError


class BaseParser(ABC):
    """Extract succinct extraction text from input"""

    TASK_ID = "data_extraction"
    """ID to use for this extraction for linking with LLM configs"""

    @property
    @abstractmethod
    def IN_LABEL(self):  # noqa: N802
        """str: Identifier for text ingested by this class"""
        raise NotImplementedError

    @property
    @abstractmethod
    def OUT_LABEL(self):  # noqa: N802
        """str: Identifier for final structured data output"""
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError


class KeywordBasedHeuristic(BaseHeuristic, ABC):
    """Perform a heuristic check for mention of a technology in text"""

    _GOOD_ACRONYM_CONTEXTS = [
        " {acronym} ",
        " {acronym}\n",
        " {acronym}.",
        "\n{acronym} ",
        "\n{acronym}.",
        "\n{acronym}\n",
        "({acronym} ",
        " {acronym})",
    ]

    def check(self, text, match_count_threshold=1):
        """Check for mention of a tech in text

        This check first strips the text of any tech "look-alike" words
        (e.g. "window", "windshield", etc for "wind" technology). Then,
        it checks for particular keywords, acronyms, and phrases that
        pertain to the tech in the text. If enough keywords are mentions
        (as dictated by `match_count_threshold`), this check returns
        ``True``.

        Parameters
        ----------
        text : str
            Input text that may or may not mention the technology of
            interest.
        match_count_threshold : int, optional
            Number of keywords that must match for the text to pass this
            heuristic check. Count must be strictly greater than this
            value. By default, ``1``.

        Returns
        -------
        bool
            ``True`` if the number of keywords/acronyms/phrases detected
            exceeds the `match_count_threshold`.
        """
        heuristics_text = self._convert_to_heuristics_text(text)
        total_keyword_matches = self._count_single_keyword_matches(
            heuristics_text
        )
        total_keyword_matches += self._count_acronym_matches(heuristics_text)
        total_keyword_matches += self._count_phrase_matches(heuristics_text)
        return total_keyword_matches > match_count_threshold

    def _convert_to_heuristics_text(self, text):
        """Convert text for heuristic content parsing"""
        heuristics_text = text.casefold()
        for word in self.NOT_TECH_WORDS:
            heuristics_text = heuristics_text.replace(word, "")
        return heuristics_text

    def _count_single_keyword_matches(self, heuristics_text):
        """Count number of good tech keywords that appear in text"""
        return sum(
            keyword in heuristics_text for keyword in self.GOOD_TECH_KEYWORDS
        )

    def _count_acronym_matches(self, heuristics_text):
        """Count number of good tech acronyms that appear in text"""
        acronym_matches = 0
        for context in self._GOOD_ACRONYM_CONTEXTS:
            acronym_keywords = {
                context.format(acronym=acronym)
                for acronym in self.GOOD_TECH_ACRONYMS
            }
            acronym_matches = sum(
                keyword in heuristics_text for keyword in acronym_keywords
            )
            if acronym_matches > 0:
                break
        return acronym_matches

    def _count_phrase_matches(self, heuristics_text):
        """Count number of good tech phrases that appear in text"""
        text_ngrams = {}
        total = 0
        for phrase in self.GOOD_TECH_PHRASES:
            n = len(phrase.split(" "))
            if n <= 1:
                msg = (
                    "Make sure your GOOD_TECH_PHRASES contain at least 2 "
                    f"words! Got phrase: {phrase!r}"
                )
                warn(msg, COMPASSWarning)
                continue

            if n not in text_ngrams:
                text_ngrams[n] = set(
                    convert_text_to_sentence_ngrams(heuristics_text, n)
                )

            test_ngrams = (  # fmt: off
                convert_text_to_sentence_ngrams(phrase, n)
                + convert_text_to_sentence_ngrams(f"{phrase}s", n)
            )
            if any(t in text_ngrams[n] for t in test_ngrams):
                total += 1

        return total

    @property
    @abstractmethod
    def NOT_TECH_WORDS(self):  # noqa: N802
        """:class:`~collections.abc.Iterable`: Not tech keywords"""
        raise NotImplementedError

    @property
    @abstractmethod
    def GOOD_TECH_KEYWORDS(self):  # noqa: N802
        """:class:`~collections.abc.Iterable`: Tech keywords"""
        raise NotImplementedError

    @property
    @abstractmethod
    def GOOD_TECH_ACRONYMS(self):  # noqa: N802
        """:class:`~collections.abc.Iterable`: Tech acronyms"""
        raise NotImplementedError

    @property
    @abstractmethod
    def GOOD_TECH_PHRASES(self):  # noqa: N802
        """:class:`~collections.abc.Iterable`: Tech phrases"""
        raise NotImplementedError


class PromptBasedTextCollector(JSONFromTextLLMCaller, BaseTextCollector, ABC):
    """Text extractor based on a chain of prompts"""

    @property
    @abstractmethod
    def PROMPTS(self):  # noqa: N802
        """list: List of dicts defining the prompts for text extraction

        Each dict in the list should have the following keys:

            - **prompt**: [REQUIRED] The text filter prompt to use
              to determine if a chunk of text is relevant for the
              current extraction task. The prompt must instruct the LLM
              to return a dictionary (as JSON) with at least one key
              that outputs the filter decision. The prompt may use the
              following placeholders, which will be filled in with the
              corresponding class attributes when the prompt is applied:

                - ``"{key}"``: The key corresponding to this prompt.

            - **key**: [REQUIRED] A string identifier for the key that
              in the output JSON dictionary that represents the LLM
              filter decision (``True`` if the tech chunk should be
              kept, and ``False`` otherwise).
            - **label**: [OPTIONAL] A string label describing the type
              of relevant text this prompt is looking for (e.g. "wind
              energy conversion system ordinance text"). This is only
              used for logging purposes and does not affect the
              extraction process itself. If not provided, this will
              default to "collector step {i}".

        The prompts will be applied in the order they appear in the
        list, with the output text from each prompt being fed as input
        to the next prompt in the chain. If any of the filter decisions
        return ``False``, the text will be discarded and not passed to
        subsequent prompts. The final output of the last prompt will
        determine wether or not the chunk of text being evaluated is
        kept as relevant text for extraction.
        """
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._chunks = {}

    @property
    def relevant_text(self):
        """str: Combined ordinance text from the individual chunks"""
        if not self._chunks:
            logger.debug(
                "No relevant ordinance chunk(s) found in original text",
            )
            return ""

        logger.debug(
            "Grabbing %d ordinance chunk(s) from original text at these "
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
        for collection_step, prompt_dict in enumerate(self.PROMPTS):
            key = prompt_dict["key"]
            prompt = prompt_dict["prompt"].format(key=key)
            label = prompt_dict.get("label", collection_step)
            passed_filter = await chunk_parser.parse_from_ind(
                ind,
                key=key,
                llm_call_callback=self._check_chunk_with_prompt,
                prompt=prompt,
            )

            if not passed_filter:
                logger.debug(
                    "Text at ind %d did not pass collection step: %s",
                    ind,
                    label,
                )
                return False

            logger.debug(
                "Text at ind %d passed collection step: %s ", ind, label
            )

        self._store_chunk(chunk_parser, ind)
        logger.debug("Added text chunk at ind %d to extraction text", ind)
        return True

    async def _check_chunk_with_prompt(self, key, text_chunk, prompt):
        """Call LLM on a chunk of text to check for ordinance"""
        content = await self.call(
            sys_msg=prompt.format(key=key),
            content=text_chunk,
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


class PromptBasedTextExtractor(LLMCaller, BaseTextExtractor, ABC):
    """Text extractor based on a chain of prompts"""

    SYSTEM_MESSAGE = (
        dedent(
            """\
        You are a text extraction assistant. Your job is to extract only
        verbatim, **unmodified** excerpts from the provided text. Do not
        interpret or paraphrase. Do not summarize. Only return exactly copied
        segments that match the specified scope. If the relevant content
        appears within a table, return the entire table, including headers
        and footers, exactly as formatted.
        """
        )
        .replace("\n", " ")
        .strip()
    )
    """System message for text extraction LLM calls"""

    FORMATTING_PROMPT = (
        dedent(
            """\
        ## Formatting & Structure ##:
        - **Preserve _all_ section titles, headers, and numberings** for
          reference.
        - **Maintain the original wording, formatting, and structure** to
          ensure accuracy.
        """
        )
        .replace("\n  ", " ")
        .strip()
    )
    """Prompt component instructing model to preserve text structure"""

    OUTPUT_PROMPT = (
        dedent(
            """\
        ## Output Handling ##:
        - This is a strict extraction task — act like a text filter, **not**
          a summarizer or writer.
        - Do not add, explain, reword, or summarize anything.
        - The output must be a **copy-paste** of the original excerpt.
          **Absolutely no paraphrasing or rewriting.**
        - The output must consist **only** of contiguous or discontiguous
          verbatim blocks copied from the input.
        - The only allowed change is to remove irrelevant sections of text.
          You can remove irrelevant text from within sections, but you cannot
          add any new text or modify the text you keep in any way.
        - If **no relevant text** is found, return the response:
          'No relevant text.'
        """
        )
        .replace("\n  ", " ")
        .strip()
    )
    """Prompt component instructing model output guidelines"""

    @property
    @abstractmethod
    def PROMPTS(self):  # noqa: N802
        """list: List of dicts defining the prompts for text extraction

        Each dict in the list should have the following keys:

            - **prompt**: [REQUIRED] The text extraction prompt to use
              for the extraction. The prompt may use the following
              placeholders, which will be filled in with the
              corresponding class attributes when the prompt is applied:

                - ``"{FORMATTING_PROMPT}"``: The
                      :obj:`PromptBasedTextExtractor.FORMATTING_PROMPT`
                      class attribute, which provides instructions for
                      preserving the formatting and structure of the
                      extracted text.
                - ``"{OUTPUT_PROMPT}"``: The
                      :obj:`PromptBasedTextExtractor.OUTPUT_PROMPT`
                      class attribute, which provides instructions for
                      how the model should format the output and what
                      content to include or exclude.

            - **key**: [OPTIONAL] A string identifier for the text
              extracted by this prompt. If not provided, a default key
              ``"extracted_text_{i}"`` will be used, where ``{i}`` is
              the index of the prompt in the list. The value of this key
              from the last dictionary in the input list will be used as
              this extractor's `OUT_LABEL`, which is typically used to
              link the extracted text to the appropriate parser via the
              parser's `IN_LABEL`. All `key` values should be unique
              across all prompts in the chain.
            - **out_fn**: [OPTIONAL] A file name template that will be
              used to write the extracted text to a file. The template
              can include the placeholder ``{jurisdiction}``, which
              will be replaced with the full jurisdiction name. If not
              provided, the extracted text will not be written to a
              file. This is primarily intended for debugging and
              analysis purposes, and is not required for the extraction
              process itself.

        The prompts will be applied in the order they appear in the
        list, with the output text from each prompt being fed as input
        to the next prompt in the chain. The final output of the last
        prompt will be the output of the extractor.
        """
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return

        last_prompt = cls.PROMPTS[-1]
        last_index = len(cls.PROMPTS) - 1
        cls.OUT_LABEL = last_prompt.get("key", f"extracted_text_{last_index}")

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
        for ind, prompt_dict in enumerate(self.PROMPTS):
            key = prompt_dict.get("key", f"extracted_text_{ind}")
            instructions = prompt_dict["prompt"].format(
                FORMATTING_PROMPT=self.FORMATTING_PROMPT,
                OUTPUT_PROMPT=self.OUTPUT_PROMPT,
            )
            yield key, partial(self._process, instructions=instructions)

    async def _process(self, text_chunks, instructions, is_valid_chunk=None):
        """Perform extraction processing"""
        if is_valid_chunk is None:
            is_valid_chunk = _valid_chunk

        logger.info(
            "Extracting summary text from %d text chunks asynchronously...",
            len(text_chunks),
        )
        logger.debug("Model instructions are:\n%s", instructions)
        outer_task_name = asyncio.current_task().get_name()
        summaries = [
            asyncio.create_task(
                self.call(
                    sys_msg=self.SYSTEM_MESSAGE,
                    content=f"{instructions}\n\n# TEXT #\n\n{chunk}",
                    usage_sub_label=self._USAGE_LABEL,
                ),
                name=outer_task_name,
            )
            for chunk in text_chunks
        ]
        summary_chunks = await asyncio.gather(*summaries)
        summary_chunks = [
            clean_backticks_from_llm_response(chunk)
            for chunk in summary_chunks
            if is_valid_chunk(chunk)
        ]

        text_summary = merge_overlapping_texts(summary_chunks)
        logger.debug(
            "Final summary contains %d tokens",
            ApiBase.count_tokens(
                text_summary, model=self.kwargs.get("model", "gpt-4")
            ),
        )
        return text_summary


class OrdinanceParser(BaseLLMCaller, BaseParser):
    """Base class for parsing structured data"""

    def _init_chat_llm_caller(self, system_message):
        """Initialize a ChatLLMCaller instance for the DecisionTree"""
        return ChatLLMCaller(
            self.llm_service,
            system_message=system_message,
            usage_tracker=self.usage_tracker,
            **self.kwargs,
        )

    async def reextract_focused_features(
        self,
        text_by_feature,
        feature_list,
        schema_info,
        original_extraction,
    ):
        """
        Re-extract specific features with filtered text and explicit focus

        Parameters
        ----------
        text_by_feature : dict
            Mapping {feature_id: filtered_text} from text filter
        feature_list : list of str
            Feature IDs to re-extract
        schema_info : dict
            Schema information including feature descriptions
        original_extraction : dict or pandas.DataFrame
            Original extraction for context

        Returns
        -------
        dict
            Partial extraction containing only re-extracted features

        Notes
        -----
        This method is called during iterative refinement to extract
        specific features that were missing or incorrect. Implementations
        should focus the LLM's attention on the requested features and
        use the filtered text to reduce noise
        """
        raise NotImplementedError


class OrdinanceExtractionPlugin(FilteredExtractionPlugin):
    """Base class for COMPASS extraction plugins

    This class provides a good balance between ease of use and
    extraction flexibility, allowing implementers to provide additional
    functionality during the extraction process.

    Plugins can hook into various stages of the extraction pipeline
    to modify behavior, add custom processing, or integrate with
    external systems.

    Subclasses should implement the desired hooks and override
    methods as needed.
    """

    @property
    @abstractmethod
    def TEXT_EXTRACTORS(self):  # noqa: N802
        """list of BaseTextExtractor: Classes to condense text

        Should be an iterable of one or more classes to condense text in
        preparation for the extraction task.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def PARSERS(self):  # noqa: N802
        """list of BaseParser: Classes to extract structured data

        Should be an iterable of one or more classes to extract
        structured data from text.
        """
        raise NotImplementedError

    @cached_property
    def producers(self):
        """list: All classes that produce attributes on the doc"""
        return chain(self.PARSERS, self.TEXT_EXTRACTORS, self.TEXT_COLLECTORS)

    @cached_property
    def consumer_producer_pairs(self):
        """list: Pairs of (consumer, producer) for IN/OUT validation"""
        return [
            (self.PARSERS, chain(self.TEXT_EXTRACTORS, self.TEXT_COLLECTORS)),
            (self.TEXT_EXTRACTORS, self.TEXT_COLLECTORS),
        ]

    async def extract_ordinances_from_text(
        self, doc, parser_class, model_config
    ):
        """Extract structured data from input text

        The extracted structured data will be stored in the ``.attrs``
        dictionary of the input document under the
        ``parser_class.OUT_LABEL`` key.

        Parameters
        ----------
        doc : BaseDocument
            Document containing text to extract structured data from.
        parser_class : BaseParser
            Class to use for structured data extraction.
        model_config : LLMConfig
            Configuration for the LLM model to use for structured data
            extraction.
        """
        parser = parser_class(
            llm_service=model_config.llm_service,
            usage_tracker=self.usage_tracker,
            **model_config.llm_call_kwargs,
        )
        logger.info(
            "Extracting %s...", parser_class.OUT_LABEL.replace("_", " ")
        )
        await extract_ordinance_values(
            doc,
            parser,
            text_key=parser_class.IN_LABEL,
            out_key=parser_class.OUT_LABEL,
        )

    @classmethod
    def get_structured_data_row_count(cls, data_df):
        """Get the number of data rows extracted from a document

        Parameters
        ----------
        data_df : pandas.DataFrame or None
            DataFrame to check for extracted structured data.

        Returns
        -------
        int
            Number of data rows extracted from the document.
        """
        if data_df is None:
            return 0

        return num_ordinances_dataframe(
            data_df, exclude_features=EXCLUDE_FROM_ORD_DOC_CHECK
        )

    async def parse_docs_for_structured_data(self, extraction_context):
        """Parse documents to extract structured data/information

        Parameters
        ----------
        extraction_context : ExtractionContext
            Context containing candidate documents to parse.

        Returns
        -------
        ExtractionContext or None
            Context with extracted data/information stored in the
            ``.attrs`` dictionary, or ``None`` if no data was extracted.
        """
        for doc_for_extraction in extraction_context:
            data_df = await self.parse_single_doc_for_structured_data(
                doc_for_extraction
            )
            row_count = self.get_structured_data_row_count(data_df)
            if row_count > 0:
                await extraction_context.mark_doc_as_data_source(
                    doc_for_extraction, out_fn_stem=self.jurisdiction.full_name
                )
                extraction_context.attrs["structured_data"] = data_df
                logger.info(
                    "%d ordinance value(s) found in doc from %s for %s. ",
                    num_ordinances_dataframe(data_df),
                    doc_for_extraction.attrs.get("source", "unknown source"),
                    self.jurisdiction.full_name,
                )
                return extraction_context

        logger.debug(
            "No ordinances found; searched %d docs",
            extraction_context.num_documents,
        )
        return None

    async def parse_single_doc_for_structured_data(self, doc_for_extraction):
        """Extract all possible structured data from a document

        This method is called from the default implementation of
        `parse_docs_for_structured_data()` for each document that passed
        filtering. If you overwrite`parse_docs_for_structured_data()``,
        you can ignore this method.

        Parameters
        ----------
        doc_for_extraction : BaseDocument
            Document to extract structured data from.

        Returns
        -------
        BaseDocument
            Document with extracted structured data stored in the
            ``.attrs`` dictionary.
        """
        with self._tracked_progress():
            tasks = [
                asyncio.create_task(
                    self._try_extract_ordinances(
                        doc_for_extraction, parser_class
                    ),
                    name=self.jurisdiction.full_name,
                )
                for parser_class in filter(None, self.PARSERS)
            ]
            await asyncio.gather(*tasks)

        return self._concat_scrape_results(doc_for_extraction)

    async def _try_extract_ordinances(self, doc_for_extraction, parser_class):
        """Apply a single extractor and parser to legal text"""

        if parser_class.IN_LABEL not in doc_for_extraction.attrs:
            await self._run_text_extractors(doc_for_extraction, parser_class)

        model_config = self._get_model_config(
            primary_key=parser_class.TASK_ID,
            secondary_key=LLMTasks.DATA_EXTRACTION,
        )
        await self.extract_ordinances_from_text(
            doc_for_extraction,
            parser_class=parser_class,
            model_config=model_config,
        )

        await self.record_usage()

    async def _run_text_extractors(self, doc_for_extraction, parser_class):
        """Run text extractor(s) on document to get text for a parser"""
        te = [
            te
            for te in self.TEXT_EXTRACTORS
            if te.OUT_LABEL == parser_class.IN_LABEL
        ]
        if len(te) != 1:
            msg = (
                f"Could not find unique text extractor for parser "
                f"{parser_class.__name__} with IN_LABEL "
                f"{parser_class.IN_LABEL!r}. Got matches: {te}"
            )
            raise COMPASSPluginConfigurationError(msg)

        te = te[0]
        model_config = self._get_model_config(
            primary_key=te.TASK_ID,
            secondary_key=LLMTasks.TEXT_EXTRACTION,
        )
        logger.debug(
            "Condensing text for extraction using %r for doc from %s",
            te.__name__,
            doc_for_extraction.attrs.get("source", "unknown source"),
        )
        assert self._jsp is not None, "No progress bar set!"
        task_id = self._jsp.add_task(te.TASK_DESCRIPTION)
        await self.extract_relevant_text(doc_for_extraction, te, model_config)
        await self.record_usage()
        self._jsp.remove_task(task_id)

    @contextmanager
    def _tracked_progress(self):
        """Context manager to set up jurisdiction sub-progress bar"""
        loc = self.jurisdiction.full_name
        with COMPASS_PB.jurisdiction_sub_prog(loc) as self._jsp:
            yield

        self._jsp = None

    def _concat_scrape_results(self, doc):
        """Concatenate structured data from all parsers"""
        data = [doc.attrs.get(p.OUT_LABEL, None) for p in self.PARSERS]
        data = [df for df in data if df is not None and not df.empty]
        if len(data) == 0:
            return None

        data = data[0] if len(data) == 1 else pd.concat(data)
        data["source"] = doc.attrs.get("source")
        data["ord_year"] = extract_ord_year_from_doc_attrs(doc.attrs)
        return data

    def _get_model_config(self, primary_key, secondary_key):
        """Get model config: primary_key -> secondary_key -> default"""
        if primary_key in self.model_configs:
            return self.model_configs[primary_key]
        return self.model_configs.get(
            secondary_key, self.model_configs[LLMTasks.DEFAULT]
        )

    def validate_plugin_configuration(self):
        """[NOT PUBLIC API] Validate plugin is properly configured"""
        super().validate_plugin_configuration()
        self._validate_text_extractors()
        self._validate_parsers()
        self._validate_in_out_keys()
        self._validate_collector_prompts()
        self._validate_extractor_prompts()
        self._register_clean_file_names()

    def _validate_text_extractors(self):
        """Validate user provided at least one text extractor class"""
        try:
            extractors = self.TEXT_EXTRACTORS
        except NotImplementedError:
            msg = (
                f"Plugin class {self.__class__.__name__} is missing required "
                "property 'TEXT_EXTRACTORS'"
            )
            raise COMPASSPluginConfigurationError(msg) from None

        if len(extractors) == 0:
            msg = (
                f"Plugin class {self.__class__.__name__} has an empty "
                "'TEXT_EXTRACTORS' property! Please provide at least "
                "one text extractor class."
            )
            raise COMPASSPluginConfigurationError(msg)

        for extractor_class in extractors:
            if not issubclass(extractor_class, BaseTextExtractor):
                msg = (
                    f"Plugin class {self.__class__.__name__} has invalid "
                    "entry in 'TEXT_EXTRACTORS' property: All entries must "
                    "be subclasses of "
                    "compass.plugin.ordinance.BaseTextExtractor, but "
                    f"{extractor_class.__name__} is not!"
                )
                raise COMPASSPluginConfigurationError(msg)

    def _validate_parsers(self):
        """Validate user provided at least one parser class"""
        try:
            parsers = self.PARSERS
        except NotImplementedError:
            msg = (
                f"Plugin class {self.__class__.__name__} is missing required "
                "property 'PARSERS'"
            )
            raise COMPASSPluginConfigurationError(msg) from None

        if len(parsers) == 0:
            msg = (
                f"Plugin class {self.__class__.__name__} has an empty "
                "'PARSERS' property! Please provide at least "
                "one text extractor class."
            )
            raise COMPASSPluginConfigurationError(msg)

        for parsers_class in parsers:
            if not issubclass(parsers_class, BaseParser):
                msg = (
                    f"Plugin class {self.__class__.__name__} has invalid "
                    "entry in 'PARSERS' property: All entries must "
                    "be subclasses of "
                    "compass.plugin.ordinance.BaseParser, but "
                    f"{parsers_class.__name__} is not!"
                )
                raise COMPASSPluginConfigurationError(msg)

    def _validate_in_out_keys(self):
        """Validate that all IN_LABELs have matching OUT_LABELs"""
        out_keys = {}
        for producer in self.producers:
            out_keys.setdefault(producer.OUT_LABEL, []).append(producer)

        dupes = {k: v for k, v in out_keys.items() if len(v) > 1}
        if dupes:
            formatted = "\n".join(
                [
                    f"{key}: {[cls.__name__ for cls in classes]}"
                    for key, classes in dupes.items()
                ]
            )
            msg = (
                "Multiple processing classes produce the same OUT_LABEL key:\n"
                f"{formatted}"
            )
            raise COMPASSPluginConfigurationError(msg)

        for consumers, producers in self.consumer_producer_pairs:
            _validate_in_out_keys(consumers, producers)

    def _validate_collector_prompts(self):
        """Validate that all text collectors have prompts defined"""

        for collector in self.TEXT_COLLECTORS:
            if not issubclass(collector, PromptBasedTextCollector):
                continue
            try:
                num_prompts = len(collector.PROMPTS)
            except NotImplementedError:
                msg = (
                    f"Text collector {self.__class__.__name__} is missing "
                    "required property 'PROMPTS'"
                )
                raise COMPASSPluginConfigurationError(msg) from None

            if num_prompts == 0:
                msg = (
                    f"Text collector {self.__class__.__name__} has an empty "
                    "'PROMPTS' property! Please provide at least one prompt "
                    "dictionary."
                )
                raise COMPASSPluginConfigurationError(msg)

    def _validate_extractor_prompts(self):
        """Validate that all text extractors have prompts defined"""

        for collector in self.TEXT_EXTRACTORS:
            if not issubclass(collector, PromptBasedTextExtractor):
                continue
            try:
                num_prompts = len(collector.PROMPTS)
            except NotImplementedError:
                msg = (
                    f"Text extractor {self.__class__.__name__} is missing "
                    "required property 'PROMPTS'"
                )
                raise COMPASSPluginConfigurationError(msg) from None

            if num_prompts == 0:
                msg = (
                    f"Text extractor {self.__class__.__name__} has an empty "
                    "'PROMPTS' property! Please provide at least one prompt "
                    "dictionary."
                )
                raise COMPASSPluginConfigurationError(msg)

    def _register_clean_file_names(self):
        """Register file names for writing cleaned text outputs"""
        CLEANED_FP_REGISTRY.setdefault(self.IDENTIFIER.casefold(), {})
        for extractor_class in self.TEXT_EXTRACTORS:
            if not issubclass(extractor_class, PromptBasedTextExtractor):
                continue
            for ind, prompt_dict in enumerate(extractor_class.PROMPTS):
                out_fn = prompt_dict.get("out_fn", None)
                if not out_fn:
                    continue

                key = prompt_dict.get("key", f"extracted_text_{ind}")
                CLEANED_FP_REGISTRY[self.IDENTIFIER.casefold()][key] = out_fn


def _valid_chunk(chunk):
    """True if chunk has content"""
    return chunk and "no relevant text" not in chunk.lower()


def _validate_in_out_keys(consumers, producers):
    """Validate that all IN_LABELs have matching OUT_LABELs"""
    in_keys = {}
    out_keys = {}

    for producer_class in producers:
        out_keys.setdefault(producer_class.OUT_LABEL, []).append(
            producer_class
        )

    for consumer_class in chain(consumers):
        in_keys.setdefault(consumer_class.IN_LABEL, []).append(consumer_class)

    for in_key, classes in in_keys.items():
        formatted = f"{[cls.__name__ for cls in classes]}"
        if in_key not in out_keys:
            msg = (
                f"One or more processing classes require IN_LABEL "
                f"{in_key!r}, which is not produced by any previous "
                f"processing class: {formatted}"
            )
            raise COMPASSPluginConfigurationError(msg)
