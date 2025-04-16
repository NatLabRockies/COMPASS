"""Ordinance document content Validation logic

These are primarily used to validate that a legal document applies to a
particular technology (e.g. Solar Energy Conversion System).
"""

import logging

from compass.extraction.common import BaseTextExtractor
from compass.validation.content import Heuristic
from compass.utilities.enums import LLMUsageCategory
from compass.utilities.parsing import merge_overlapping_texts


logger = logging.getLogger(__name__)


_LARGE_SEF_SYNONYMS = (
    "solar panels, solar energy conversion systems (SECS), solar energy "
    "facilities (SEF), solar energy farms (SEF), solar farms (SF), "
    "utility-scale solar energy systems (USES), commercial solar energy "
    "systems (CSES), alternate energy systems (AES), or similar"
)
_SEARCH_TERMS_AND = (
    "zoning, siting, setback, system design, and operational "
    "requirements/restrictions"
)
_SEARCH_TERMS_OR = _SEARCH_TERMS_AND.replace("and", "or")
_IGNORE_TYPES = (
    "private, residential, roof-mounted, micro, small, or medium sized"
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
        "Note that solar energy bans are an important restriction to track. "
        "The last key is '{key}', which is a boolean that is set to True if "
        f"the text excerpt describes {_SEARCH_TERMS_OR} for "
        "a solar energy system and False otherwise."
    )

    IS_UTILITY_SCALE_PROMPT = (
        "You are a legal scholar that reads ordinance text and determines "
        f"whether it applies to {_SEARCH_TERMS_OR} for "
        "large solar energy systems. Large solar energy systems (SES) may "
        f"also be referred to as {_LARGE_SEF_SYNONYMS}. "
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
        "solar energy farms are a permitted use. Large solar energy systems "
        f"(SES) may also be referred to as {_LARGE_SEF_SYNONYMS}. "
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
            usage_sub_label=(
                LLMUsageCategory.DOCUMENT_PERMITTED_USE_CONTENT_VALIDATION
            ),
        )
        logger.debug("LLM response: %s", str(content))
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
        "# CONTEXT #\n"
        "We want to reduce the provided excerpt to only contain information "
        "about solar energy systems. The extracted text will be used for "
        "structured data extraction, so it must be both **comprehensive** "
        "(retaining all relevant details) and **focused** (excluding "
        "unrelated content). Ensure that all retained information is "
        "**directly applicable** to solar energy systems while preserving "
        "full context and accuracy.\n"
        "\n# OBJECTIVE #\n"
        "Extract all text **pertaining to solar energy systems** from the "
        "provided excerpt.\n"
        "\n# RESPONSE #\n"
        "Follow these guidelines carefully:\n"
        "\n1. ## Scope of Extraction ##:\n"
        "- Include **all** text that pertains to** solar energy systems**, "
        "even if they are referred to by different names such as:\n"
        f"\t{_LARGE_SEF_SYNONYMS.capitalize()}.\n"
        "- Explicitly include any text related to **bans or prohibitions** "
        "on solar energy systems.\n"
        "\n2. ## Exclusions ##:\n"
        "- Do **not** include text that does not pertain to solar energy "
        "systems.\n"
        "\n3. ## Formatting & Structure ##:\n"
        "- **Preserve all section headers and numbering** for clarity and "
        "reference.\n"
        "- **Maintain the original wording, formatting, and structure** to "
        "ensure accuracy.\n"
        "\n4. ## Output Handling ##:\n"
        "- If **no relevant text** is found, return the response: "
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
            "cleaned_ordinance_text",
            self.extract_solar_energy_system_section,
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

    _USAGE_LABEL = LLMUsageCategory.DOCUMENT_PERMITTED_USE_DISTRICTS_SUMMARY

    PERMITTED_USES_FILTER_PROMPT = (
        "# CONTEXT #\n"
        "We want to reduce the provided excerpt to only contain information "
        "detailing permitted use(s) for a district. The extracted text will "
        "be used for structured data extraction, so it must be both "
        "**comprehensive** (retaining all relevant details) and **focused** "
        "(excluding unrelated content). Ensure that all retained information "
        "is **directly applicable** to permitted use(s) for one or more "
        "districts while preserving full context and accuracy.\n"
        "\n# OBJECTIVE #\n"
        "Remove all text **not directly pertinent** to permitted use(s) for "
        "a district.\n"
        "\n# RESPONSE #\n"
        "Follow these guidelines carefully:\n"
        "\n1. ## Scope of Extraction ##:\n"
        "- Retain all text defining permitted use(s) for a district, "
        "including:\n"
        "\t- **Primary, Special, Accessory, and other permitted use types.**\n"
        "\t- **District names and zoning classifications.**\n"
        "- Pay extra attention to any references to **solar energy "
        "facilities** or related terms.\n"
        "- Ensure that **tables, lists, and structured elements** are "
        "preserved as they may contain relevant details.\n"
        "\n2. ## Exclusions ##:\n"
        "- Do **not** include unrelated regulations, procedural details, "
        "or non-use-based restrictions.\n"
        "\n3. ## Formatting & Structure ##:\n"
        "- **Preserve all section headers and numbering** for clarity and "
        "reference.\n"
        "- **Maintain the original wording, formatting, and structure** to "
        "ensure accuracy.\n"
        "\n4. ## Output Handling ##:\n"
        "- If **no relevant text** is found, return the response: "
        '"No relevant text."'
    )

    SEF_PERMITTED_USES_FILTER_PROMPT = (
        "# CONTEXT #\n"
        "We want to reduce the provided excerpt to only contain information "
        "detailing **solar energy system** permitted use(s) for a district. "
        "The extracted text will be used for structured data extraction, so "
        "it must be both **comprehensive** (retaining all relevant details) "
        "and **focused** (excluding unrelated content). Ensure that all "
        "retained information is **directly applicable** to permitted use(s) "
        "for solar energy systems in one or more districts while "
        "preserving full context and accuracy.\n"
        "\n# OBJECTIVE #\n"
        "Remove all text **not directly pertinent** to solar energy "
        "conversion system permitted use(s) for a district.\n"
        "\n# RESPONSE #\n"
        "Follow these guidelines carefully:\n"
        "\n1. ## Scope of Extraction ##:\n"
        "- Retain all text defining permitted use(s) for a district, "
        "including:\n"
        "\t- **Primary, Special, Accessory, and other permitted use types.**\n"
        "\t- **District names and zoning classifications.**\n"
        "- Ensure that **tables, lists, and structured elements** are "
        "preserved as they may contain relevant details.\n"
        "\n2. ## Exclusions ##:\n"
        "- Do not include text that does not pertain at all to solar "
        "energy systems.\n"
        "\n3. ## Formatting & Structure ##:\n"
        "- **Preserve all section headers and numbering** for clarity and "
        "reference.\n"
        "- **Maintain the original wording, formatting, and structure** to "
        "ensure accuracy.\n"
        "\n4. ## Output Handling ##:\n"
        "- If **no relevant text** is found, return the response: "
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
