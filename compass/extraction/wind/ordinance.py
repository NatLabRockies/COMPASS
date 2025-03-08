"""Ordinance document content Validation logic

These are primarily used to validate that a legal document applies to a
particular technology (e.g. Large Wind Energy Conversion Systems).
"""

import asyncio
import logging

from elm import ApiBase

from compass.validation.content import Heuristic, ValidationWithMemory
from compass.utilities.parsing import merge_overlapping_texts


logger = logging.getLogger(__name__)


_LARGE_WES_DESCRIPTION = (
    "Large wind energy systems (WES) may also be referred to as wind "
    "turbines, wind energy conversion systems (WECS), wind energy "
    "facilities (WEF), wind energy turbines (WET), large wind energy "
    "turbines (LWET), utility-scale wind energy turbines (UWET), "
    "commercial wind energy systems, or similar. "
)
_SEARCH_TERMS_AND = (
    "zoning, special permitting, siting and setback, system design, and "
    "operational requirements/restrictions"
)
_SEARCH_TERMS_OR = (
    "zoning, special permitting, siting and setback, system design, or "
    "operational requirements/restrictions"
)
_IGNORE_TYPES = "private, residential, micro, small, or medium sized"
_TRACK_BANS = (
    "Note that wind energy bans are an important restriction to track. "
)


class WindHeuristic(Heuristic):
    """Perform a heuristic check for mention of wind turbines in text"""

    NOT_TECH_WORDS = [
        "mini wecs",
        "private wecs",
        "pwecs",
        "rewind",
        "small wind",
        "swecs",
        "windbreak",
        "windiest",
        "winds",
        "windshield",
        "window",
        "windy",
        "wind attribute",
        "wind blow",
        "wind damage",
        "wind direction",
        "wind erosion",
        "wind load",
        "wind movement",
        "wind orient",
        "wind runway",
    ]
    GOOD_TECH_KEYWORDS = ["wind", "setback"]
    GOOD_TECH_ACRONYMS = ["wecs", "wes", "lwet", "uwet", "wef"]
    GOOD_TECH_PHRASES = [
        "wind energy conversion",
        "wind turbine",
        "wind tower",
        "wind farm",
        "wind energy system",
        "wind energy farm",
        "utility wind energy system",
    ]


class WindOrdinanceValidator(ValidationWithMemory):
    """Check document text for wind ordinances

    Purpose:
        Determine whether a document contains relevant ordinance
        information.
    Responsibilities:
        1. Determine whether a document contains relevant (e.g.
        utility-scale wind zoning) ordinance information by splitting
        the text into chunks and parsing them individually using LLMs.
    Key Relationships:
        Child class of
        :class:`~compass.validation.content.ValidationWithMemory`,
        which allows the validation to look at neighboring chunks of
        text.
    """

    _HEURISTIC = WindHeuristic()
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
        "format (not markdown). Your JSON file must include exactly two "
        "keys. The first key is 'wind_reqs', which is a string that "
        f"summarizes all {_SEARCH_TERMS_AND} (if given) "
        "in the text for a wind energy system (or wind turbine/tower). "
        f"{_TRACK_BANS}"
        "The last key is '{key}', which is a boolean that is set to True if "
        f"the text excerpt describes {_SEARCH_TERMS_OR} for "
        "a wind energy system (or wind turbine/tower) and False otherwise. "
    )

    IS_UTILITY_SCALE_PROMPT = (
        "You are a legal scholar that reads ordinance text and determines "
        f"whether any of it applies to {_SEARCH_TERMS_OR} for "
        f"large wind energy systems. {_LARGE_WES_DESCRIPTION}"
        "Your client is a commercial wind developer that does not "
        f"care about ordinances related to {_IGNORE_TYPES} wind energy "
        "systems. Ignore any text related to such systems. "
        "Return your answer in JSON format (not markdown). Your JSON file "
        "must include exactly two keys. The first key is 'summary' which "
        "contains a string that lists all of the types of wind energy systems "
        "the text applies to (if any). The second key is '{key}', which is a "
        "boolean that is set to True if any part of the text excerpt mentions "
        f"{_SEARCH_TERMS_OR} for the large wind energy conversion "
        "systems that the client is interested in and False otherwise."
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
        self._wind_mention_mem = []
        self._ordinance_chunk_inds = []

    @property
    def is_legal_text(self):
        """bool: ``True`` if text was found to be from a legal source"""
        if not self._legal_text_mem:
            return False
        return sum(self._legal_text_mem) >= 0.5 * len(self._legal_text_mem)

    @property
    def ordinance_text(self):
        """str: Combined ordinance text from the individual chunks"""
        logger.debug("Ordinance chunk inds: %s", self._ordinance_chunk_inds)

        inds_to_grab = set()
        for ind in self._ordinance_chunk_inds:
            inds_to_grab |= {ind + x for x in range(1 - self.num_to_recall, 2)}

        inds_to_grab = [
            ind
            for ind in sorted(inds_to_grab)
            if 0 <= ind < len(self.text_chunks)
        ]
        logger.debug(
            "Grabbing %d chunk(s) from original text at these indices: %s",
            len(inds_to_grab),
            inds_to_grab,
        )

        text = [self.text_chunks[ind] for ind in inds_to_grab]
        return merge_overlapping_texts(text)

    async def parse(self, min_chunks_to_process=3):
        """Parse text chunks and look for ordinance text

        Parameters
        ----------
        min_chunks_to_process : int, optional
            Minimum number of chunks to process before checking if
            document resembles legal text and ignoring chunks that don't
            pass the wind heuristic. By default, ``3``.

        Returns
        -------
        bool
            ``True`` if any ordinance text was found in the chunks.
        """
        for ind, text in enumerate(self.text_chunks):
            self._wind_mention_mem.append(self._HEURISTIC.check(text))
            if ind >= min_chunks_to_process:
                if not self.is_legal_text:
                    return False

                # fmt: off
                if not any(self._wind_mention_mem[-self.num_to_recall:]):
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
                    "Text at ind %d is not for utility-scale WECS", ind
                )
                continue

            logger.debug("Text at ind %d is for utility-scale WECS", ind)

            self._ordinance_chunk_inds.append(ind)
            logger.debug("Added text at ind %d to ordinances", ind)
            # mask, since we got a good result
            self._wind_mention_mem[-1] = False

        return bool(self._ordinance_chunk_inds)


class WindOrdinanceTextExtractor:
    """Extract succinct ordinance text from input

    Purpose:
        Extract relevant ordinance text from document.
    Responsibilities:
        1. Extract portions from chunked document text relevant to
           particular ordinance type (e.g. wind zoning for utility-scale
           systems).
    Key Relationships:
        Uses a :class:`~compass.llm.calling.StructuredLLMCaller` for
        LLM queries.
    """

    SYSTEM_MESSAGE = (
        "You extract one or more direct excerpts from a given text based on "
        "the user's request. Maintain all original formatting and characters "
        "without any paraphrasing. If the relevant text is inside of a "
        "space-delimited table, return the entire table with the original "
        "space-delimited formatting. Never paraphrase! Only return portions "
        "of the original text directly."
    )
    ENERGY_SYSTEM_FILTER_PROMPT = (
        "Extract the full text for all sections pertaining to energy "
        "conversion systems. Remove sections that definitely do not pertain "
        "to energy conversion systems. Note that bans on energy conversion "
        "systems are an important restriction to track. If there is no text "
        "that pertains to energy conversion systems, simply say: "
        '"No relevant text."'
    )

    WIND_ENERGY_SYSTEM_FILTER_PROMPT = (
        "Extract the full text for all sections pertaining to wind "
        "energy conversion systems. Remove sections that definitely do "
        "not pertain to wind energy conversion systems. "
        f"{_TRACK_BANS}"
        "If there is no text that pertains to wind energy conversion "
        "systems, simply say: "
        '"No relevant text."'
    )
    LARGE_WIND_ENERGY_SYSTEM_SECTION_FILTER_PROMPT = (
        "Extract the full text for all sections pertaining to large wind "
        "energy systems.  "
        f"{_TRACK_BANS}{_LARGE_WES_DESCRIPTION}"
        f"Remove all sections that explicitly only apply to {_IGNORE_TYPES} "
        "wind energy systems. Keep section headers (if any). If there is "
        "no text pertaining to large wind systems, simply say: "
        '"No relevant text."'
    )
    LARGE_WIND_ENERGY_SYSTEM_TEXT_FILTER_PROMPT = (
        "Extract all portions of the text that apply to large wind energy "
        "systems."
        f"{_TRACK_BANS}{_LARGE_WES_DESCRIPTION}"
        f"Remove all text that explicitly only applies to {_IGNORE_TYPES} "
        "wind energy systems. Keep section headers (if any). If there is "
        "no text pertaining to large wind systems, simply say: "
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

    async def extract_energy_system_section(self, text_chunks):
        """Extract ordinance text from input text chunks for energy sys

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
            instructions=self.ENERGY_SYSTEM_FILTER_PROMPT,
            valid_chunk=_valid_chunk,
        )

    async def extract_wind_energy_system_section(self, text_chunks):
        """Extract ordinance text from input text chunks for WES

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
            instructions=self.WIND_ENERGY_SYSTEM_FILTER_PROMPT,
            valid_chunk=_valid_chunk,
        )

    async def extract_large_wind_energy_system_section(self, text_chunks):
        """Extract large WES ordinance text from input text chunks

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
            instructions=self.LARGE_WIND_ENERGY_SYSTEM_SECTION_FILTER_PROMPT,
            valid_chunk=_valid_chunk,
        )

    async def extract_large_wind_energy_system_text(self, text_chunks):
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
            instructions=self.LARGE_WIND_ENERGY_SYSTEM_TEXT_FILTER_PROMPT,
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
        yield "energy_systems_text", self.extract_energy_system_section
        yield (
            "wind_energy_systems_text",
            self.extract_wind_energy_system_section,
        )
        yield (
            "large_wind_energy_systems_text",
            self.extract_large_wind_energy_system_section,
        )
        yield (
            "cleaned_ordinance_text",
            self.extract_large_wind_energy_system_text,
        )


def _valid_chunk(chunk):
    """True if chunk has content"""
    return chunk and "no relevant text" not in chunk.lower()
