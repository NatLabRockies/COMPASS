"""Ordinance document content Validation logic

These are primarily used to validate that a legal document applies to a
particular technology (e.g. Solar Energy Conversion System).
"""

import logging

from compass.extraction.common import BaseTextExtractor
from compass.validation.content import Heuristic
from compass.utilities.parsing import merge_overlapping_texts


logger = logging.getLogger(__name__)


_LARGE_SEF_DESCRIPTION = (
    "Large solar energy systems (SES) may also be referred to as solar "
    "panels, solar energy conversion systems (SECS), solar energy "
    "facilities (SEF), solar energy farms (SEF), solar farms (SF), "
    "utility-scale solar energy systems (USES), commercial solar energy "
    "systems (CSES), alternate energy systems (AES), or similar. "
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
    "Note that solar energy bans are an important restriction to track. "
)


class SolarHeuristic(Heuristic):
    """Perform a heuristic check for mention of solar farms in text"""

    NOT_TECH_WORDS = ["solaris"]
    GOOD_TECH_KEYWORDS = ["solar", "setback"]
    GOOD_TECH_ACRONYMS = ["secs", "sef", "ses", "cses"]
    GOOD_TECH_PHRASES = [
        "commercial solar energy system",
        "solar energy conversion",
        "solar energy system",
        "solar panel",
        "solar farm",
        "solar energy farm",
        "utility solar energy system",
    ]


class SolarOrdinanceTextCollector:
    """Check text chunks for ordinances and collect them if they do"""

    CONTAINS_ORD_PROMPT = (
        "You extract structured data from text. Return your answer in JSON "
        "format (not markdown). Your JSON file must include exactly two "
        "keys. The first key is 'solar_reqs', which is a string that "
        f"summarizes all {_SEARCH_TERMS_AND} (if given) "
        "in the text for solar energy systems. "
        f"{_TRACK_BANS}"
        "The last key is '{key}', which is a boolean that is set to True if "
        f"the text excerpt describes {_SEARCH_TERMS_OR} for "
        "a solar energy system and False otherwise."
    )

    IS_UTILITY_SCALE_PROMPT = (
        "You are a legal scholar that reads ordinance text and determines "
        f"whether it applies to {_SEARCH_TERMS_OR} for "
        f"large solar energy systems. {_LARGE_SEF_DESCRIPTION}"
        "Your client is a commercial solar developer that does not "
        f"care about ordinances related to {_IGNORE_TYPES} solar energy "
        "systems. Ignore any text related to such systems. "
        "Return your answer in JSON format (not markdown). Your JSON file "
        "must include exactly two keys. The first key is 'summary' which "
        "contains a string that summarizes the types of solar energy systems "
        "the text applies to (if any). The second key is '{key}', which is a "
        "boolean that is set to True if any part of the text excerpt mentions "
        f"{_SEARCH_TERMS_OR} for the large solar energy conversion "
        "systems that the client is interested in and False otherwise."
    )

    def __init__(self):
        self._ordinance_chunks = {}

    async def check_chunk(self, chunk_parser, ind):
        """Check a chunk at a given ind to see if it contains ordinance

        Parameters
        ----------
        chunk_parser : ParseChunksWithMemory
            Instance of `ParseChunksWithMemory` that contains a
            `parse_from_ind` method.
        ind : int
            Index of the chunk to check.

        Returns
        -------
        bool
            Boolean flag indicating whether or not the text in the chunk
            contains large solar energy farm ordinance text.
        """
        contains_ord_info = await chunk_parser.parse_from_ind(
            ind, self.CONTAINS_ORD_PROMPT, key="contains_ord_info"
        )
        if not contains_ord_info:
            logger.debug("Text at ind %d does not contain ordinance info", ind)
            return False

        logger.debug("Text at ind %d does contain ordinance info", ind)

        is_utility_scale = await chunk_parser.parse_from_ind(
            ind, self.IS_UTILITY_SCALE_PROMPT, key="x"
        )
        if not is_utility_scale:
            logger.debug("Text at ind %d is not for utility-scale SEF", ind)
            return False

        logger.debug("Text at ind %d is for utility-scale SEF", ind)

        _store_chunk(chunk_parser, ind, self._ordinance_chunks)
        logger.debug("Added text at ind %d to ordinances", ind)

        return True

    @property
    def contains_ord_info(self):
        """bool: Flag indicating whether text contains ordinance info"""
        return bool(self._ordinance_chunks)

    @property
    def ordinance_text(self):
        """str: Combined ordinance text from the individual chunks"""
        logger.debug(
            "Grabbing %d chunk(s) from original text at these indices: %s",
            len(self._ordinance_chunks),
            list(self._ordinance_chunks),
        )

        text = [
            self._ordinance_chunks[ind]
            for ind in sorted(self._ordinance_chunks)
        ]
        return merge_overlapping_texts(text)


class SolarPermittedUseDistrictsTextCollector:
    """Check text chunks for permitted solar districts; collect them"""

    DISTRICT_PROMPT = (
        "You are a legal scholar that reads ordinance text and determines "
        "whether the text explicitly details the districts where large "
        "solar energy farms are a permitted use. "
        f"{_LARGE_SEF_DESCRIPTION}"
        "Do not make any inferences; only answer based on information that "
        "is explicitly outlined in the text. "
        "Note that relevant information may sometimes be found in tables. "
        "Return your answer in JSON format (not markdown). Your JSON file "
        "must include exactly two keys. The first key is 'districts' which "
        "contains a string that lists all of the district names for which "
        "the text explicitly permits large solar energy farms (if any). "
        "The last key is "
        "'{key}', which is a boolean that is set to True if any part of the "
        "text excerpt mentions districts where large solar energy farms"
        "are a permitted use and False otherwise."
    )

    def __init__(self):
        self._district_chunks = {}

    async def check_chunk(self, chunk_parser, ind):
        """Check a chunk to see if it contains permitted uses

        Parameters
        ----------
        chunk_parser : ParseChunksWithMemory
            Instance of `ParseChunksWithMemory` that contains a
            `parse_from_ind` method.
        ind : int
            Index of the chunk to check.

        Returns
        -------
        bool
            Boolean flag indicating whether or not the text in the chunk
            contains large solar energy farm permitted use text.
        """

        key = "contains_district_info"
        content = await chunk_parser.slc.call(
            sys_msg=self.DISTRICT_PROMPT.format(key=key),
            content=chunk_parser.text_chunks[ind],
            usage_sub_label="document_permitted_use_content_validation",
        )
        logger.debug("LLM response: %s", str(content))  # TODO: trace
        contains_district_info = content.get(key, False)

        if contains_district_info:
            _store_chunk(chunk_parser, ind, self._district_chunks)
            logger.debug("Text at ind %d contains district info", ind)
            return True

        logger.debug("Text at ind %d does not contain district info", ind)
        return False

    @property
    def contains_district_info(self):
        """bool: Flag indicating whether text contains district info"""
        return bool(self._district_chunks)

    @property
    def permitted_use_district_text(self):
        """str: Combined permitted use districts text from the chunks"""
        logger.debug(
            "Grabbing %d chunk(s) from original text at these indices: %s",
            len(self._district_chunks),
            list(self._district_chunks),
        )

        text = [
            self._district_chunks[ind] for ind in sorted(self._district_chunks)
        ]
        return merge_overlapping_texts(text)


class SolarOrdinanceTextExtractor(BaseTextExtractor):
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

    SOLAR_ENERGY_SYSTEM_FILTER_PROMPT = (
        "Extract the full text for all sections pertaining to solar energy"
        "conversion systems (or solar farms). Remove sections that "
        "definitely do not pertain to solar energy conversion systems. "
        f"{_TRACK_BANS}"
        "If there is no text that pertains to solar energy conversion "
        "systems, simply say: "
        '"No relevant text."'
    )
    LARGE_SOLAR_ENERGY_SYSTEM_SECTION_FILTER_PROMPT = (
        "Extract the full text for all sections pertaining to large solar "
        "energy systems (or solar farms). "
        f"{_TRACK_BANS}{_LARGE_SEF_DESCRIPTION}"
        f"Remove all sections that explicitly only apply to {_IGNORE_TYPES} "
        "solar energy systems. Keep section headers (if any). If there is "
        "no text pertaining to large solar systems, simply say: "
        '"No relevant text."'
    )

    async def extract_solar_energy_system_section(self, text_chunks):
        """Extract ordinance text from input text chunks for SEF

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
            instructions=self.SOLAR_ENERGY_SYSTEM_FILTER_PROMPT,
            is_valid_chunk=_valid_chunk,
        )

    async def extract_large_solar_energy_system_section(self, text_chunks):
        """Extract large SEF ordinance text from input text chunks

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
            instructions=self.LARGE_SOLAR_ENERGY_SYSTEM_SECTION_FILTER_PROMPT,
            is_valid_chunk=_valid_chunk,
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
        yield (
            "solar_energy_systems_text",
            self.extract_solar_energy_system_section,
        )
        yield (
            "cleaned_ordinance_text",
            self.extract_large_solar_energy_system_section,
        )


class SolarPermittedUseDistrictsTextExtractor(BaseTextExtractor):
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

    _USAGE_LABEL = "document_permitted_use_districts_summary"

    PERMITTED_USES_FILTER_PROMPT = (
        "Remove all text that does not pertain to permitted use(s) for a "
        "district. "
        "Consider all permitted use kinds, including but not limited to "
        "Primary, Special, Accessory, etc. "
        "Pay extra attention to text that mentions solar. "
        "Keep all text that explains the permitted use kind, gives the "
        "district name(s), and/or mentions solar. "
        "Note that relevant information may sometimes be found in tables. "
        "If there is no text that pertains to permitted use(s) for a "
        "district, simply say: "
        '"No relevant text."'
    )

    SEF_PERMITTED_USES_FILTER_PROMPT = (
        "Remove all text that does not pertain to solar energy farms. "
        "Keep all text that explains the permitted use kind, gives the "
        "district name(s), and/or mentions the solar energy farm. "
        "Note that information about solar energy farms may sometimes be "
        "found in tables; be sure to keep the entire table in such cases. "
        "If there is no text that pertains to solar energy farms, simply say: "
        '"No relevant text."'
    )

    async def extract_permitted_uses(self, text_chunks):
        """Extract permitted uses text from input text chunks

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
            instructions=self.PERMITTED_USES_FILTER_PROMPT,
            is_valid_chunk=_valid_chunk,
        )

    async def extract_sef_permitted_uses(self, text_chunks):
        """Extract permitted uses text for large SEF from input text

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
            instructions=self.SEF_PERMITTED_USES_FILTER_PROMPT,
            is_valid_chunk=_valid_chunk,
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
        yield "permitted_use_only_text", self.extract_permitted_uses
        yield "districts_text", self.extract_sef_permitted_uses


def _valid_chunk(chunk):
    """True if chunk has content"""
    return chunk and "no relevant text" not in chunk.lower()


def _store_chunk(parser, chunk_ind, store):
    """Store chunk and its neighbors if it is not already stored"""
    for offset in range(1 - parser.num_to_recall, 2):
        ind_to_grab = chunk_ind + offset
        if ind_to_grab < 0 or ind_to_grab >= len(parser.text_chunks):
            continue

        store.setdefault(ind_to_grab, parser.text_chunks[ind_to_grab])
