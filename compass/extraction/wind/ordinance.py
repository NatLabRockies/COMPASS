"""Wind ordinance document content collection and extraction

These methods help filter down the document text to only the portions
relevant to utility-scale wind ordinances.
"""

import logging

from compass.plugin.ordinance import (
    KeywordBasedHeuristic,
    PromptBasedTextCollector,
    PromptBasedTextExtractor,
)
from compass.utilities.enums import LLMUsageCategory


logger = logging.getLogger(__name__)


_LARGE_WES_SYNONYMS = (
    "wind turbines, wind energy conversion systems (WECS), wind energy "
    "facilities (WEF), wind energy turbines (WET), large wind energy "
    "turbines (LWET), utility-scale wind energy turbines (UWET), "
    "commercial wind energy conversion systems (CWECS), alternate "
    "energy systems (AES), commercial energy production "
    "systems (CEPCS), or similar"
)
_SEARCH_TERMS_AND = (
    "zoning, siting, setback, system design, and operational "
    "requirements/restrictions"
)
_SEARCH_TERMS_OR = _SEARCH_TERMS_AND.replace("and", "or")
_IGNORE_TYPES = "private, residential, micro, small, or medium sized"

_CONTAINS_ORD_COLLECTION_PROMPT = f"""\
You extract structured data from text. Return your answer in JSON format \
(not markdown). Your JSON file must include exactly two keys. The first \
key is 'wind_reqs', which is a string that summarizes all {_SEARCH_TERMS_AND} \
that are explicitly enacted in the text for a wind energy system (or wind \
turbine/tower) for a given jurisdiction. Note that wind energy bans are \
an important restriction to track. Include any **closely related provisions** \
if they clearly pertain to the **development, operation, modification, or \
removal** of wind energy systems (or wind turbines/towers). All restrictions \
should be enforceable - ignore any text that only provides a legal definition \
of the regulation. If the text does not specify any concrete \
{_SEARCH_TERMS_OR} for a wind energy system, set this key to `null`. The last \
key is '{{key}}', which is a boolean that is set to True if the text excerpt \
explicitly details {_SEARCH_TERMS_OR} for a wind energy system (or wind \
turbine/tower) and False otherwise.\
"""

_IS_UTILITY_SCALE_COLLECTION_PROMPT = f"""\
You are a legal scholar that reads ordinance text and determines whether \
any of it applies to {_SEARCH_TERMS_OR} for **large wind energy systems**. \
Large wind energy systems (WES) may also be referred to as \
{_LARGE_WES_SYNONYMS}. Your client is a commercial wind developer that \
does not care about ordinances related to {_IGNORE_TYPES} wind energy \
systems. Ignore any text related to such systems. Return your answer as a \
dictionary in JSON format (not markdown). Your JSON file must include \
exactly two keys. The first key is 'summary' which contains a string that \
lists all of the types of wind energy systems the text applies to (if any). \
The second key is '{{key}}', which is a boolean that is set to True if any \
part of the text excerpt details {_SEARCH_TERMS_OR} for the **large wind \
energy conversion systems** (or similar) that the client is interested in \
and False otherwise.\
"""

_DISTRICTS_COLLECTION_PROMPT = f"""\
You are a legal scholar that reads ordinance text and determines whether \
the text explicitly contains relevant information to determine the districts \
(and especially the district names) where large wind energy systems are a \
permitted use (primary, special, accessory, or otherwise), as well as the \
districts where large wind energy systems are prohibited entirely. Large \
wind energy systems (WES) may also be referred to as {_LARGE_WES_SYNONYMS}. \
Do not make any inferences; only answer based on information that is \
explicitly stated in the text. Note that relevant information may sometimes \
be found in tables. Return your answer as a dictionary in JSON format (not \
markdown). Your JSON file must include exactly two keys. The first key is \
'districts' which contains a string that lists all of the district names for \
which the text explicitly permits **large wind energy systems** (if any). \
The last key is '{{key}}', which is a boolean that is set to True if any \
part of the text excerpt provides information on districts where **large \
wind energy systems** (or similar) are a permitted use in and False  \
otherwise.\
"""

_WECS_TEXT_EXTRACTION_PROMPT = """\
# CONTEXT #
We want to reduce the provided excerpt to only contain information about \
**wind energy systems**. The extracted text will be used for structured data \
extraction, so it must be both **comprehensive** (retaining all relevant \
details) and **focused** (excluding unrelated content), with **zero rewriting \
or paraphrasing**. Ensure that all retained information is **directly \
applicable to wind energy systems** while preserving full context and accuracy.

# OBJECTIVE #
Extract all text **pertaining to wind energy systems** from the provided \
excerpt.

# RESPONSE #
Follow these guidelines carefully:

1. ## Scope of Extraction ##:
- Include all text that pertains to **wind energy systems**.
- Explicitly include any text related to **bans or prohibitions** on wind \
energy systems.
- Explicitly include any text related to the adoption or enactment date of \
the ordinance (if any).

2. ## Exclusions ##:
- Do **not** include text that does not pertain to wind energy systems.

3. {FORMATTING_PROMPT}

4. {OUTPUT_PROMPT}\
"""

_LARGE_WECS_TEXT_EXTRACTION_PROMPT = f"""\
# CONTEXT #
We want to reduce the provided excerpt to only contain information about \
**large wind energy systems**. The extracted text will be used for structured \
data extraction, so it must be both **comprehensive** (retaining all relevant \
details) and **focused** (excluding unrelated content), with **zero rewriting \
or paraphrasing**. Ensure that all retained information is **directly \
applicable** to large wind energy systems while preserving full context and \
accuracy.

# OBJECTIVE #
Extract all text **pertaining to large wind energy systems** from the \
provided excerpt.

# RESPONSE #
Follow these guidelines carefully:

1. ## Scope of Extraction ##:
- Include all text that pertains to **large wind energy systems**, even if \
they are referred to by different names such as: \
{_LARGE_WES_SYNONYMS.capitalize()}
- Explicitly include any text related to **bans or prohibitions** on large \
wind energy systems.
- Explicitly include any text related to the adoption or enactment date of \
the ordinance (if any).
- **Retain all relevant technical, design, operational, safety, \
environmental, and infrastructure-related provisions** that apply to the \
topic, such as (but not limited to):
    - Compliance with legal or regulatory standards.
    - Site, structural, or design specifications.
    - Environmental impact considerations.
    - Safety and risk mitigation measures.
    - Infrastructure, implementation, operation, and maintenance details.
    - All other **closely related provisions**.

2. ## Exclusions ##:
- Do **not** include text that explicitly applies **only** to {_IGNORE_TYPES} \
wind energy systems.
- Do **not** include text that does not pertain at all to wind energy systems.

3. {{FORMATTING_PROMPT}}

4. {{OUTPUT_PROMPT}}\
"""

_PERMITTED_USES_TEXT_EXTRACTION_PROMPT = """\
# CONTEXT #
We want to reduce the provided excerpt to only contain information detailing \
permitted use(s) for a district. The extracted text will be used for \
structured data extraction, so it must be both **comprehensive** (retaining \
all relevant details) and **focused** (excluding unrelated content), with \
**zero rewriting or paraphrasing**. Ensure that all retained information is \
**directly applicable** to permitted use(s) for one or more districts while \
preserving full context and accuracy.

# OBJECTIVE #
Remove all text **not directly pertinent** to permitted use(s) for a district.

# RESPONSE #
Follow these guidelines carefully:

1. ## Scope of Extraction ##:
- Retain all text defining permitted use(s) for a district, including:
    - **Primary, Special, Conditional, Accessory, Prohibited, and any other \
use types.**
    - **District names and zoning classifications.**
- Pay extra attention to any references to **wind energy facilities** or \
related terms.
- Ensure that **tables, lists, and structured elements** are preserved as \
they may contain relevant details.

2. ## Exclusions ##:
- Do **not** include unrelated regulations, procedural details, or \
non-use-based restrictions.

3. {FORMATTING_PROMPT}

4. {OUTPUT_PROMPT}\
"""

_WECS_PERMITTED_USES_TEXT_EXTRACTION_PROMPT = """\
# CONTEXT #
We want to reduce the provided excerpt to only contain information detailing \
**wind energy system** permitted use(s) for a district. The extracted text \
will be used for structured data extraction, so it must be both \
**comprehensive** (retaining all relevant details) and **focused** (excluding \
unrelated content), with **zero rewriting or paraphrasing**. Ensure that all \
retained information is **directly applicable** to permitted use(s) for wind \
energy systems in one or more districts while preserving full context and \
accuracy.

# OBJECTIVE #
Remove all text **not directly pertinent** to wind energy conversion system \
permitted use(s) for a district.

# RESPONSE #
Follow these guidelines carefully:

1. ## Scope of Extraction ##:
- Retain all text defining permitted use(s) for a district, including:
    - **Primary, Special, Conditional, Accessory, Prohibited, and any other \
use types.**
    - **District names and zoning classifications.**
- Ensure that **tables, lists, and structured elements** are preserved as \
they may contain relevant details.

2. ## Exclusions ##:
- Do not include text that does not pertain at all to wind energy systems.

3. {FORMATTING_PROMPT}

4. {OUTPUT_PROMPT}\
"""


class WindHeuristic(KeywordBasedHeuristic):
    """Perform a heuristic check for mention of wind turbines in text"""

    NOT_TECH_WORDS = [
        "micro wecs",
        "small wecs",
        "mini wecs",
        "private wecs",
        "personal wecs",
        "pwecs",
        "rewind",
        "small wind",
        "micro wind",
        "mini wind",
        "private wind",
        "personal wind",
        "swecs",
        "windbreak",
        "windiest",
        "winds",
        "windshield",
        "window",
        "windy",
        "wind attribute",
        "wind blow",
        "wind break",
        "wind current",
        "wind damage",
        "wind data",
        "wind direction",
        "wind draft",
        "wind erosion",
        "wind energy resource atlas",
        "wind load",
        "wind movement",
        "wind orient",
        "wind resource",
        "wind runway",
        "prevailing wind",
        "downwind",
    ]
    """Words and phrases that indicate text is NOT about WECS"""
    GOOD_TECH_KEYWORDS = ["wind", "setback"]
    """Words that indicate we should keep a chunk for analysis"""
    GOOD_TECH_ACRONYMS = ["wecs", "wes", "lwet", "uwet", "wef"]
    """Acronyms for WECS that we want to capture"""
    GOOD_TECH_PHRASES = [
        "wind energy conversion",
        "wind turbine",
        "wind tower",
        "wind farm",
        "wind energy system",
        "wind energy farm",
        "utility wind energy system",
    ]
    """Phrases that indicate text is about WECS"""


class WindOrdinanceTextCollector(PromptBasedTextCollector):
    """Check text chunks for ordinances and collect them if they do"""

    OUT_LABEL = "relevant_text"
    """Identifier for text collected by this class"""

    PROMPTS = [
        {
            "key": "contains_ord_info",
            "label": "contains ordinance info",
            "prompt": _CONTAINS_ORD_COLLECTION_PROMPT,
        },
        {
            # Generic key like "x" makes the llm focus on the
            # instruction rather than using the key name to infer the
            # content, which can improve performance,
            "key": "x",
            "label": "for utility-scale WECS",
            "prompt": _IS_UTILITY_SCALE_COLLECTION_PROMPT,
        },
    ]
    """Dicts defining the prompts for ordinance text collection"""


class WindPermittedUseDistrictsTextCollector(PromptBasedTextCollector):
    """Check text chunks for permitted wind districts; collect them"""

    OUT_LABEL = "permitted_use_text"
    """Identifier for text collected by this class"""

    PROMPTS = [
        {
            "key": "contains_district_info",
            "label": "contains district info",
            "prompt": _DISTRICTS_COLLECTION_PROMPT,
        },
    ]
    """Dicts defining the prompts for permitted use text collection"""


class WindOrdinanceTextExtractor(PromptBasedTextExtractor):
    """Extract succinct ordinance text from input"""

    IN_LABEL = WindOrdinanceTextCollector.OUT_LABEL
    """Identifier for collected text ingested by this class"""

    TASK_DESCRIPTION = "Extracting wind ordinance text"
    """Task description to show in progress bar"""

    TASK_ID = "ordinance_text_extraction"
    """ID to use for this extraction for linking with LLM configs"""

    PROMPTS = [
        {
            "key": "wind_energy_systems_text",
            "out_fn": "{jurisdiction} Wind Ordinance.txt",
            "prompt": _WECS_TEXT_EXTRACTION_PROMPT,
        },
        {
            "key": "cleaned_text_for_extraction",
            "out_fn": "{jurisdiction} Utility Scale Wind Ordinance.txt",
            "prompt": _LARGE_WECS_TEXT_EXTRACTION_PROMPT,
        },
    ]
    """Dicts defining the prompts for ordinance text extraction"""


class WindPermittedUseDistrictsTextExtractor(PromptBasedTextExtractor):
    """Extract succinct permitted use districts text from input"""

    IN_LABEL = WindPermittedUseDistrictsTextCollector.OUT_LABEL
    """Identifier for collected text ingested by this class"""

    TASK_DESCRIPTION = "Extracting wind permitted use text"
    """Task description to show in progress bar"""

    TASK_ID = "permitted_use_text_extraction"
    """ID to use for this extraction for linking with LLM configs"""

    _USAGE_LABEL = LLMUsageCategory.DOCUMENT_PERMITTED_USE_DISTRICTS_SUMMARY

    PROMPTS = [
        {
            "key": "permitted_use_only_text",
            "out_fn": "{jurisdiction} Permitted Use.txt",
            "prompt": _PERMITTED_USES_TEXT_EXTRACTION_PROMPT,
        },
        {
            "key": "districts_text",
            "out_fn": "{jurisdiction} Permitted Use Districts.txt",
            "prompt": _WECS_PERMITTED_USES_TEXT_EXTRACTION_PROMPT,
        },
    ]
    """Dicts defining the prompts for permitted use text extraction"""
