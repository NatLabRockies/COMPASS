"""Ordinance document content Validation logic

These are primarily used to validate that a legal document applies to a
particular technology (e.g. Large Wind Energy Conversion Systems).
"""

import logging

from compass.extraction.common import BaseTextExtractor
from compass.validation.content import Heuristic
from compass.utilities.enums import LLMUsageCategory
from compass.utilities.parsing import merge_overlapping_texts


logger = logging.getLogger(__name__)


_LARGE_WES_SYNONYMS = (
    "wind turbines, wind energy conversion systems (WECS), wind energy "
    "facilities (WEF), wind energy turbines (WET), large wind energy "
    "turbines (LWET), utility-scale wind energy turbines (UWET), "
    "commercial wind energy conversion systems (CWECS), alternate "
    "energy systems (AES), or similar"
)
_SEARCH_TERMS_AND = (
    "zoning, siting, setback, system design, and operational "
    "requirements/restrictions"
)
_SEARCH_TERMS_OR = _SEARCH_TERMS_AND.replace("and", "or")
_IGNORE_TYPES = "private, residential, micro, small, or medium sized"


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


class WindOrdinanceTextCollector:
    """Check text chunks for ordinances and collect them if they do"""

    CONTAINS_ORD_PROMPT = (
        "You extract structured data from text. Return your answer in JSON "
        "format (not markdown). Your JSON file must include exactly two "
        "keys. The first key is 'wind_reqs', which is a string that "
        f"summarizes all {_SEARCH_TERMS_AND} (if given) "
        "in the text for a wind energy system (or wind turbine/tower). "
        "Note that wind energy bans are an important restriction to track. "
        "The last key is '{key}', which is a boolean that is set to True if "
        f"the text excerpt describes {_SEARCH_TERMS_OR} for "
        "a wind energy system (or wind turbine/tower) and False otherwise. "
    )

    IS_UTILITY_SCALE_PROMPT = (
        "You are a legal scholar that reads ordinance text and determines "
        f"whether any of it applies to {_SEARCH_TERMS_OR} for "
        "large wind energy systems. Large wind energy systems (WES) may "
        f"also be referred to as {_LARGE_WES_SYNONYMS}. "
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
            contains large wind energy conversion system ordinance text.
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
            logger.debug("Text at ind %d is not for utility-scale WECS", ind)
            return False

        logger.debug("Text at ind %d is for utility-scale WECS", ind)

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


class WindPermittedUseDistrictsTextCollector:
    """Check text chunks for permitted wind districts; collect them"""

    DISTRICT_PROMPT = (
        "You are a legal scholar that reads ordinance text and determines "
        "whether the text explicitly details the districts where large "
        "wind energy systems are a permitted use. Large wind energy systems "
        f"(WES) may also be referred to as {_LARGE_WES_SYNONYMS}. "
        "Do not make any inferences; only answer based on information that "
        "is explicitly outlined in the text. "
        "Note that relevant information may sometimes be found in tables. "
        "Return your answer in JSON format (not markdown). Your JSON file "
        "must include exactly two keys. The first key is 'districts' which "
        "contains a string that lists all of the district names for which "
        "the text explicitly permits large wind energy systems (if any). "
        "The last key is "
        "'{key}', which is a boolean that is set to True if any part of the "
        "text excerpt mentions districts where large wind energy systems"
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
            contains large wind energy conversion system permitted use
            text.
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


class WindOrdinanceTextExtractor(BaseTextExtractor):
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

    WIND_ENERGY_SYSTEM_FILTER_PROMPT = (
        "# CONTEXT #\n"
        "We want to reduce the provided excerpt to only contain information "
        "about wind energy systems. The extracted text will be used for "
        "structured data extraction, so it must be both **comprehensive** "
        "(retaining all relevant details) and **focused** (excluding "
        "unrelated content). Ensure that all retained information is "
        "**directly applicable** to wind energy systems while preserving "
        "full context and accuracy.\n"
        "\n# OBJECTIVE #\n"
        "Extract all text **pertaining to wind energy systems** from the "
        "provided excerpt.\n"
        "\n# RESPONSE #\n"
        "Follow these guidelines carefully:\n"
        "\n1. ## Scope of Extraction ##:\n"
        "- Include all text that pertains to **wind energy systems**.\n"
        "- Explicitly include any text related to **bans or prohibitions** "
        "on wind energy systems.\n"
        "\n2. ## Exclusions ##:\n"
        "- Do **not** include text that does not pertain to wind energy "
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
    LARGE_WIND_ENERGY_SYSTEM_SECTION_FILTER_PROMPT = (
        "# CONTEXT #\n"
        "We want to reduce the provided excerpt to only contain information "
        "about **large wind energy systems**. The extracted text will be "
        "used for structured data extraction, so it must be both "
        "**comprehensive** (retaining all relevant details) and **focused** "
        "(excluding unrelated content). Ensure that all retained information "
        "is **directly applicable** to large wind energy systems while "
        "preserving full context and accuracy.\n"
        "\n# OBJECTIVE #\n"
        "Extract all text **pertaining to large wind energy systems** from "
        "the provided excerpt.\n"
        "\n# RESPONSE #\n"
        "Follow these guidelines carefully:\n"
        "\n1. ## Scope of Extraction ##:\n"
        "- Include all text that pertains to **large wind energy systems**, "
        "even if they are referred to by different names such as:\n"
        f"\t{_LARGE_WES_SYNONYMS.capitalize()}.\n"
        "- Explicitly include any text related to **bans or prohibitions** "
        "on large wind energy systems.\n"
        "- **Retain all relevant technical, operational, safety, "
        "environmental, and infrastructure-related provisions** that apply "
        "to the topic, such as:\n"
        "\t- Compliance with legal or regulatory standards.\n"
        "\t- Structural or design specifications.\n"
        "\t- Environmental impact considerations.\n"
        "\t- Safety and risk mitigation measures.\n"
        "\t- Infrastructure and implementation details.\n"
        "\n2. ## Exclusions ##:\n"
        "- Do **not** include text that explicitly applies **only** to "
        f"{_IGNORE_TYPES} wind energy systems.\n"
        f"- Do **not** include text that does not pertain at all to wind "
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
            is_valid_chunk=_valid_chunk,
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
            "wind_energy_systems_text",
            self.extract_wind_energy_system_section,
        )
        yield (
            "cleaned_ordinance_text",
            self.extract_large_wind_energy_system_section,
        )


class WindPermittedUseDistrictsTextExtractor(BaseTextExtractor):
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
        "\t- **Primary, Special, Conditional, Accessory, and other permitted "
        "use types.**\n"
        "\t- **District names and zoning classifications.**\n"
        "- Pay extra attention to any references to **wind energy "
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

    WES_PERMITTED_USES_FILTER_PROMPT = (
        "# CONTEXT #\n"
        "We want to reduce the provided excerpt to only contain information "
        "detailing **wind energy system** permitted use(s) for a district. "
        "The extracted text will be used for structured data extraction, so "
        "it must be both **comprehensive** (retaining all relevant details) "
        "and **focused** (excluding unrelated content). Ensure that all "
        "retained information is **directly applicable** to permitted use(s) "
        "for wind energy systems in one or more districts while "
        "preserving full context and accuracy.\n"
        "\n# OBJECTIVE #\n"
        "Remove all text **not directly pertinent** to wind energy conversion "
        "system permitted use(s) for a district.\n"
        "\n# RESPONSE #\n"
        "Follow these guidelines carefully:\n"
        "\n1. ## Scope of Extraction ##:\n"
        "- Retain all text defining permitted use(s) for a district, "
        "including:\n"
        "\t- **Primary, Special, Conditional, Accessory, and other permitted "
        "use types.**\n"
        "\t- **District names and zoning classifications.**\n"
        "- Ensure that **tables, lists, and structured elements** are "
        "preserved as they may contain relevant details.\n"
        "\n2. ## Exclusions ##:\n"
        "- Do not include text that does not pertain at all to wind "
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

    async def extract_wes_permitted_uses(self, text_chunks):
        """Extract permitted uses text for large WES from input text

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
            instructions=self.WES_PERMITTED_USES_FILTER_PROMPT,
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
        yield "districts_text", self.extract_wes_permitted_uses


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
