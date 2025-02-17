"""Ordinance document content Validation logic

These are primarily used to validate that a legal document applies to a
particular technology (e.g. Solar Energy Conversion System).
"""

import asyncio
import logging

from elm import ApiBase

from compass.validation.content import Heuristic, ValidationWithMemory
from compass.utilities.parsing import merge_overlapping_texts


logger = logging.getLogger(__name__)


RESTRICTIONS = """- buildings / structures / residences
- property lines / parcels / subdivisions
- roads / rights-of-way
- railroads
- overhead electrical transmission wires
- bodies of water including wetlands, lakes, reservoirs, streams, and rivers
- natural, wildlife, and environmental conservation areas
- noise limits
- density limits
- height limits
- minimum/maximum lot size
- maximum project size
- moratorium or bans
- decommissioning plans/requirements
- land coverage limits
- glare limits
- visual impact assessment requirements
"""
MIN_CHARS_IN_VALID_CHUNK = 20


class SolarHeuristic(Heuristic):
    """Perform a heuristic check for mention of solar farms in text"""

    NOT_TECH_WORDS = ["solaris"]
    GOOD_TECH_KEYWORDS = ["solar", "setback"]
    GOOD_TECH_ACRONYMS = ["secs", "sef", "ses"]
    GOOD_TECH_PHRASES = [
        "solar energy conversion",
        "solar energy system",
        "solar panel",
        "solar farm",
        "solar energy farm",
        "utility solar energy system",
    ]


class SolarOrdinanceValidator(ValidationWithMemory):
    """Check document text for solar ordinances

    Purpose:
        Determine wether a document contains relevant ordinance
        information.
    Responsibilities:
        1. Determine wether a document contains relevant (e.g.
        utility-scale solar zoning) ordinance information by splitting
        the text into chunks and parsing them individually using LLMs.
    Key Relationships:
        Child class of
        :class:`~compass.validation.content.ValidationWithMemory`,
        which allows the validation to look at neighboring chunks of
        text.
    """

    _HEURISTIC = SolarHeuristic()
    IS_LEGAL_TEXT_PROMPT = (
        "You extract structured data from text. Return your answer in JSON "
        "format (not markdown). Your JSON file must include exactly three "
        "keys. The first key is 'summary', which is a string that provides a "
        "short summary of the text. The second key is 'type', which is a "
        "string that best represent the type of document this text belongs "
        "to. The third key is '{key}', which is a boolean that is set to "
        "True if the type of the text (as you previously determined) is a "
        "legally-binding statute or code and False if the text is an excerpt "
        "from other non-legal text such as a news article, survey, summary, "
        "application, public notice, etc."
    )

    CONTAINS_ORD_PROMPT = (
        "You extract structured data from text. Return your answer in JSON "
        "format (not markdown). Your JSON file must include exactly three "
        "keys. The first key is 'solar_reqs', which is a string that "
        "summarizes the setbacks or other siting requirements (if any) given "
        "in the text for solar energy systems. The second key is 'reqs', "
        "which lists the quantitative values from the text excerpt that can "
        "be used to compute setbacks or other siting requirements for a "
        "solar energy system (empty list if none exist in the text). The "
        "last key is '{key}', which is a boolean that is set to True if the "
        "text excerpt provides enough info to compute quantitative setbacks "
        "or other siting requirements for a solar energy system and False "
        "otherwise. Siting is impacted by any of the following:\n"
        f"{RESTRICTIONS}"
    )

    IS_UTILITY_SCALE_PROMPT = (
        "You are a legal scholar that reads ordinance text and determines "
        "wether it applies to large solar energy systems. Large solar energy "
        "systems (SES) may also be referred to as solar panels, solar energy "
        "conversion systems (SECS), solar energy facilities (SEF), solar "
        "energy farms (SEF), solar farms (SF), utility-scale solar energy "
        "systems (USES), commercial solar energy systems, or similar. "
        "Your client is a commercial solar developer that does not "
        "care about ordinances related to private, micro, small, or medium "
        "sized solar energy systems. Ignore any text related to such systems. "
        "Return your answer in JSON format (not markdown). Your JSON file "
        "must include exactly two keys. The first key is 'summary' which "
        "contains a string that summarizes the types of solar energy systems "
        "the text applies to (if any). The second key is '{key}', which is a "
        "boolean that is set to True if any part of the text excerpt is "
        "applicable to the large solar energy conversion systems that the "
        "client is interested in and False otherwise."
    )

    def __init__(self, structured_llm_caller, text_chunks, num_to_recall=2):
        """

        Parameters
        ----------
        structured_llm_caller : compass.llm.StructuredLLMCaller
            StructuredLLMCaller instance. Used for structured validation
            queries.
        text_chunks : list of str
            List of strings, each of which represent a chunk of text.
            The order of the strings should be the order of the text
            chunks. This validator may refer to previous text chunks to
            answer validation questions.
        num_to_recall : int, optional
            Number of chunks to check for each validation call. This
            includes the original chunk! For example, if
            `num_to_recall=2`, the validator will first check the chunk
            at the requested index, and then the previous chunk as well.
            By default, ``2``.
        """
        super().__init__(
            structured_llm_caller=structured_llm_caller,
            text_chunks=text_chunks,
            num_to_recall=num_to_recall,
        )
        self._legal_text_mem = []
        self._solar_mention_mem = []
        self._ordinance_chunks = []

    @property
    def is_legal_text(self):
        """bool: ``True`` if text was found to be from a legal source"""
        if not self._legal_text_mem:
            return False
        return sum(self._legal_text_mem) >= 0.5 * len(self._legal_text_mem)

    @property
    def ordinance_text(self):
        """str: Combined ordinance text from the individual chunks"""
        inds_to_grab = set()
        for info in self._ordinance_chunks:
            inds_to_grab |= {
                info["ind"] + x for x in range(1 - self.num_to_recall, 2)
            }

        text = [
            self.text_chunks[ind]
            for ind in sorted(inds_to_grab)
            if 0 <= ind < len(self.text_chunks)
        ]
        return merge_overlapping_texts(text)

    async def parse(self, min_chunks_to_process=3):
        """Parse text chunks and look for ordinance text

        Parameters
        ----------
        min_chunks_to_process : int, optional
            Minimum number of chunks to process before checking if
            document resembles legal text and ignoring chunks that don't
            pass the solar heuristic. By default, ``3``.

        Returns
        -------
        bool
            ``True`` if any ordinance text was found in the chunks.
        """
        for ind, text in enumerate(self.text_chunks):
            self._solar_mention_mem.append(self._HEURISTIC.check(text))
            if ind >= min_chunks_to_process:
                if not self.is_legal_text:
                    return False

                # fmt: off
                if not any(self._solar_mention_mem[-self.num_to_recall:]):
                    continue

            logger.debug("Processing text at ind %d", ind)
            logger.debug("Text:\n%s", text)

            if ind < min_chunks_to_process:
                is_legal_text = await self.parse_from_ind(
                    ind, self.IS_LEGAL_TEXT_PROMPT, key="legal_text"
                )
                self._legal_text_mem.append(is_legal_text)
                if not is_legal_text:
                    logger.debug("Text at ind %d is not legal text", ind)
                    continue

                logger.debug("Text at ind %d is legal text", ind)

            contains_ord_info = await self.parse_from_ind(
                ind, self.CONTAINS_ORD_PROMPT, key="contains_ord_info"
            )
            if not contains_ord_info:
                logger.debug(
                    "Text at ind %d does not contain ordinance info", ind
                )
                continue

            logger.debug("Text at ind %d does contain ordinance info", ind)

            is_utility_scale = await self.parse_from_ind(
                ind, self.IS_UTILITY_SCALE_PROMPT, key="x"
            )
            if not is_utility_scale:
                logger.debug(
                    "Text at ind %d is not for utility-scale SEF", ind
                )
                continue

            logger.debug("Text at ind %d is for utility-scale SEF", ind)

            self._ordinance_chunks.append({"text": text, "ind": ind})
            logger.debug("Added text at ind %d to ordinances", ind)
            # mask, since we got a good result
            self._solar_mention_mem[-1] = False

        return bool(self._ordinance_chunks)


class SolarOrdinanceTextExtractor:
    """Extract succinct ordinance text from input

    Purpose:
        Extract relevant ordinance text from document.
    Responsibilities:
        1. Extract portions from chunked document text relevant to
           particular ordinance type (e.g. solar zoning for
           utility-scale systems).
    Key Relationships:
        Uses a :class:`~compass.llm.calling.StructuredLLMCaller` for
        LLM queries.
    """

    SYSTEM_MESSAGE = (
        "You extract direct excerpts from a given text based on "
        "the user's request. Maintain all original formatting and characters "
        "without any paraphrasing. If the relevant text is inside of a "
        "space-delimited table, return the entire table with the original "
        "space-delimited formatting. Never paraphrase! Only return portions "
        "of the original text directly."
    )
    MODEL_INSTRUCTIONS_RESTRICTIONS = (
        # "Extract direct text excerpts related to the restrictions "
        "Extract all portions of the text related to the restrictions "
        "of large solar energy systems with respect to any of the following:\n"
        f"{RESTRICTIONS}"
        "Include section headers (if any) for the text excerpts. Also include "
        "any text excerpts that define what kind of large solar energy "
        "conversion system the restriction applies to. If there is no text "
        "related to siting restrictions of large solar systems, simply say: "
        '"No relevant text."'
    )
    MODEL_INSTRUCTIONS_SIZE = (
        "Extract all portions of the text that apply to large solar "
        "energy systems. Large solar energy systems (SES) may also be "
        "referred to as solar energy conversion systems (SECS), "
        "solar energy facilities (SEF), solar energy farms (SEF), solar farms "
        "(SF), utility-scale solar energy systems (USES), commercial solar "
        "energy systems, or similar. Remove all text that "
        "only applies to private, micro, or small solar energy "
        "systems. Include section headers (if any) for the text excerpts. "
        "Also include any text excerpts that define what kind of large solar "
        "energy conversion system the restriction applies to. If there is no "
        "text pertaining to large solar systems, simply say: "
        '"No relevant text."'
    )

    def __init__(self, llm_caller):
        """

        Parameters
        ----------
        llm_caller : compass.llm.LLMCaller
            LLM Caller instance used to extract ordinance info with.
        """
        self.llm_caller = llm_caller

    async def _process(self, text_chunks, instructions, valid_chunk):
        """Perform extraction processing"""
        logger.info(
            "Extracting ordinance text from %d text chunks asynchronously...",
            len(text_chunks),
        )
        logger.debug("Model instructions are:\n%s", instructions)
        outer_task_name = asyncio.current_task().get_name()
        summaries = [
            asyncio.create_task(
                self.llm_caller.call(
                    sys_msg=self.SYSTEM_MESSAGE,
                    content=f"Text:\n{chunk}\n{instructions}",
                    usage_sub_label="document_ordinance_summary",
                ),
                name=outer_task_name,
            )
            for chunk in text_chunks
        ]
        summary_chunks = await asyncio.gather(*summaries)
        summary_chunks = [
            chunk for chunk in summary_chunks if valid_chunk(chunk)
        ]

        text_summary = "\n".join(summary_chunks)
        logger.debug(
            "Final summary contains %d tokens",
            ApiBase.count_tokens(
                text_summary,
                model=self.llm_caller.kwargs.get("model", "gpt-4"),
            ),
        )
        return text_summary

    async def check_for_restrictions(self, text_chunks):
        """Extract restriction ordinance text from input text chunks

        Parameters
        ----------
        text_chunks : list of str
            List of strings, each of which represent a chunk of text.
            The order of the strings should be the order of the text
            chunks.

        Returns
        -------
        str
            Ordinance text extracted from text chunks.
        """
        return await self._process(
            text_chunks=text_chunks,
            instructions=self.MODEL_INSTRUCTIONS_RESTRICTIONS,
            valid_chunk=_valid_chunk_not_short,
        )

    async def check_for_correct_size(self, text_chunks):
        """Extract ordinance text from input text chunks for large WES

        Parameters
        ----------
        text_chunks : list of str
            List of strings, each of which represent a chunk of text.
            The order of the strings should be the order of the text
            chunks.

        Returns
        -------
        str
            Ordinance text extracted from text chunks.
        """
        return await self._process(
            text_chunks=text_chunks,
            instructions=self.MODEL_INSTRUCTIONS_SIZE,
            valid_chunk=_valid_chunk,
        )

    @property
    def parsers(self):
        """Iterable of parsers provided by this extractor

        Yields
        ------
        name : str
            Name describing the type of text output by the parser.
        parser
            Parser that takes a `text_chunks` input and outputs parsed
            text.
        """
        yield "restrictions_ordinance_text", self.check_for_restrictions
        yield "cleaned_ordinance_text", self.check_for_correct_size


def _valid_chunk(chunk):
    """True if chunk has content"""
    return chunk and "no relevant text" not in chunk.lower()


def _valid_chunk_not_short(chunk):
    """True if chunk has content and is not too short"""
    return _valid_chunk(chunk) and len(chunk) > MIN_CHARS_IN_VALID_CHUNK
