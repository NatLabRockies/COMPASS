"""Helper classes for ordinance plugins"""

import asyncio
import logging
from abc import ABC, abstractmethod
from warnings import warn

from elm import ApiBase

from compass.llm.calling import (
    BaseLLMCaller,
    ChatLLMCaller,
    StructuredLLMCaller,
)
from compass.utilities.enums import LLMUsageCategory
from compass.utilities.ngrams import convert_text_to_sentence_ngrams
from compass.warn import COMPASSWarning
from compass.utilities.parsing import (
    merge_overlapping_texts,
    clean_backticks_from_llm_response,
)
from compass.plugin.interface import (
    BaseHeuristic,
    BaseTextCollector,
    BaseTextExtractor,
    BaseParser,
)


logger = logging.getLogger(__name__)


class OrdinanceHeuristic(BaseHeuristic, ABC):
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


class OrdinanceTextCollector(StructuredLLMCaller, BaseTextCollector):
    """Base class for ordinance text collectors"""

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

    def _store_chunk(self, parser, chunk_ind):
        """Store chunk and its neighbors if it is not already stored"""
        for offset in range(1 - parser.num_to_recall, 2):
            ind_to_grab = chunk_ind + offset
            if ind_to_grab < 0 or ind_to_grab >= len(parser.text_chunks):
                continue

            self._chunks.setdefault(
                ind_to_grab, parser.text_chunks[ind_to_grab]
            )


class OrdinanceTextExtractor(BaseTextExtractor, ABC):
    """Base implementation for a text extractor"""

    SYSTEM_MESSAGE = (
        "You are a text extraction assistant. Your job is to extract only "
        "verbatim, **unmodified** excerpts from provided legal or policy "
        "documents. Do not interpret or paraphrase. Do not summarize. Only "
        "return exactly copied segments that match the specified scope. If "
        "the relevant content appears within a table, return the entire "
        "table, including headers and footers, exactly as formatted."
    )
    """System message for text extraction LLM calls"""
    _USAGE_LABEL = LLMUsageCategory.DOCUMENT_ORDINANCE_SUMMARY

    def __init__(self, llm_caller):
        """

        Parameters
        ----------
        llm_caller : LLMCaller
            LLM Caller instance used to extract ordinance info with.
        """
        self.llm_caller = llm_caller

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
                self.llm_caller.call(
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
                text_summary,
                model=self.llm_caller.kwargs.get("model", "gpt-4"),
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


def _valid_chunk(chunk):
    """True if chunk has content"""
    return chunk and "no relevant text" not in chunk.lower()
