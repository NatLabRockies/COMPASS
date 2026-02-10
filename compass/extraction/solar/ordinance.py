"""Solar ordinance document content collection and extraction

These methods help filter down the document text to only the portions
relevant to utility-scale solar ordinances.
"""

import logging

from compass.plugin.ordinance import (
    KeywordBasedHeuristic,
    PromptBasedTextCollector,
    PromptBasedTextExtractor,
)
from compass.utilities.enums import LLMUsageCategory


logger = logging.getLogger(__name__)


_LARGE_SEF_SYNONYMS = (
    "solar panels, solar energy conversion systems (SECS), solar energy "
    "facilities (SEF), solar energy farms (SEF), solar farms (SF), "
    "utility-scale solar energy systems (USES), commercial solar energy "
    "systems (CSES), ground-mounted solar energy systems (GSES), "
    "alternate energy systems (AES), commercial energy production "
    "systems (CEPCS), or similar"
)
_SEARCH_TERMS_AND = (
    "zoning, siting, setback, system design, and operational "
    "requirements/restrictions"
)
_SEARCH_TERMS_OR = _SEARCH_TERMS_AND.replace("and", "or")
_IGNORE_TYPES = (
    "CSP, private, residential, roof-mounted, micro, small, or medium sized"
)

_CONTAINS_ORD_COLLECTION_PROMPT = f"""\
You extract structured data from text. Return your answer in JSON format \
(not markdown). Your JSON file must include exactly two  keys. The first \
key is 'solar_reqs', which is a string that summarizes all
{_SEARCH_TERMS_AND} that are explicitly enacted in the legal text for solar  \
energy systems for a given jurisdiction. Note that solar energy bans are an \
important restriction to track. Include any **closely related provisions**  \
if they clearly pertain to the **development, operation, modification, or  \
removal** of solar energy systems (or solar panels). All restrictions should  \
be enforceable - ignore any text that only provides a legal definition of  \
the regulation. If the text does not specify any concrete {_SEARCH_TERMS_OR}  \
for a solar energy system, set this key to `null`. The last key is \
'{{key}}', which is a boolean that is set to True if the text excerpt  \
explicitly details {_SEARCH_TERMS_OR} for a solar energy system and False  \
otherwise.\
"""

_IS_UTILITY_SCALE_COLLECTION_PROMPT = f"""
You are a legal scholar that reads ordinance text and determines whether it  \
applies to {_SEARCH_TERMS_OR} for **large solar energy systems**. Large  \
solar  energy systems (SES) may also be referred to as  \
{_LARGE_SEF_SYNONYMS}. Your client is a commercial solar developer that does  \
not care about ordinances related to {_IGNORE_TYPES} solar energy systems.  \
Ignore any text related to such systems. Return your answer as a dictionary  \
in JSON format (not markdown). Your JSON file must include exactly two keys.  \
The first key is 'summary' which contains a string that summarizes the types  \
of solar energy systems the text applies to (if any). The second key is  \
'{{key}}', which is a boolean that is set to True if any part of the text  \
excerpt details {_SEARCH_TERMS_OR} for the **large solar energy conversion  \
systems** (or similar) that the client is interested in and False otherwise.\
"""

_DISTRICTS_COLLECTION_PROMPT = f"""
You are a legal scholar that reads ordinance text and determines whether it \
explicitly contains relevant information to determine the districts (and \
especially the district names) where large solar energy farms are a permitted \
use (primary, special, accessory, or otherwise), as well as the districts \
where large solar energy farms are prohibited entirely. Large solar energy \
systems (SES) may also be referred to as {_LARGE_SEF_SYNONYMS}. Do not make \
any inferences; only answer based on information that is explicitly stated in \
the text. Note that relevant information may sometimes be found in tables. \
Return your answer as a dictionary in JSON format (not markdown). Your JSON \
file must include exactly two keys. The first key is 'districts' which \
contains a string that lists all of the district names for which the text \
explicitly permits **large solar energy farms** (if any). The last key is \
'{{key}}', which is a boolean that is set to True if any part of the text \
excerpt provides information on districts where **large solar energy farms** \
(or similar) are a permitted use and False otherwise.\
"""


_SEF_TEXT_EXTRACTION_PROMPT = f"""\
# CONTEXT #
We want to reduce the provided excerpt to only contain information about \
**solar energy systems**. The extracted text will be used for structured data \
extraction, so it must be both **comprehensive** (retaining all relevant \
details) and **focused** (excluding unrelated content), with **zero rewriting \
or paraphrasing**. Ensure that all retained information is **directly \
applicable to solar energy systems** while preserving full context and \
accuracy.

# OBJECTIVE #
Extract all text **pertaining to solar energy systems** from the provided \
excerpt.

# RESPONSE #
Follow these guidelines carefully:

1. ## Scope of Extraction ##:
- Include **all** text that pertains to** solar energy systems**, even if \
they are referred to by different names such as: \
{_LARGE_SEF_SYNONYMS.capitalize()}
- Explicitly include any text related to **bans or prohibitions** on solar \
energy systems.
- Explicitly include any text related to the adoption or enactment date of \
the ordinance (if any).

2. ## Exclusions ##:
- Do **not** include text that does not pertain to solar energy systems.

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
- Pay extra attention to any references to **solar energy facilities** or \
related terms.
- Ensure that **tables, lists, and structured elements** are preserved as \
they may contain relevant details.

2. ## Exclusions ##:
- Do **not** include unrelated regulations, procedural details, or \
non-use-based restrictions.

3. {FORMATTING_PROMPT}

4. {OUTPUT_PROMPT}\
"""

_SEF_PERMITTED_USES_TEXT_EXTRACTION_PROMPT = """\
# CONTEXT #
We want to reduce the provided excerpt to only contain information detailing \
**solar energy system** permitted use(s) for a district. The extracted text \
will be used for structured data extraction, so it must be both \
**comprehensive** (retaining all relevant details) and **focused** (excluding \
unrelated content), with **zero rewriting or paraphrasing**. Ensure that all \
retained information is **directly applicable** to permitted use(s) for solar \
energy systems in one or more districts while preserving full context and \
accuracy.

# OBJECTIVE #
Remove all text **not directly pertinent** to solar energy conversion system \
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
- Do not include text that does not pertain at all to solar energy systems.

3. {FORMATTING_PROMPT}

4. {OUTPUT_PROMPT}\
"""


class SolarHeuristic(KeywordBasedHeuristic):
    """Perform a heuristic check for mention of solar farms in text"""

    NOT_TECH_WORDS = [
        "concentrated solar",
        "csp",
        "micro secs",
        "small secs",
        "mini secs",
        "private secs",
        "personal secs",
        "psecs",
        "solaris",
        "small solar",
        "micro solar",
        "mini solar",
        "private solar",
        "personal solar",
        "swecs",
        "solar break",
        "solar damage",
        "solar data",
        "solar resource",
    ]
    """Words and phrases that indicate text is NOT about solar farms"""
    GOOD_TECH_KEYWORDS = ["solar", "setback"]
    """Words that indicate we should keep a chunk for analysis"""
    GOOD_TECH_ACRONYMS = ["secs", "sef", "ses", "cses"]
    """Acronyms for solar farms that we want to capture"""
    GOOD_TECH_PHRASES = [
        "commercial solar energy system",
        "solar energy conversion",
        "solar energy system",
        "solar panel",
        "solar farm",
        "solar energy farm",
        "utility solar energy system",
    ]
    """Phrases that indicate text is about solar farms"""


class SolarOrdinanceTextCollector(PromptBasedTextCollector):
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
            "label": "for utility-scale SEF",
            "prompt": _IS_UTILITY_SCALE_COLLECTION_PROMPT,
        },
    ]
    """Dicts defining the prompts for ordinance text collection"""


class SolarPermittedUseDistrictsTextCollector(PromptBasedTextCollector):
    """Check text chunks for permitted solar districts; collect them"""

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


class SolarOrdinanceTextExtractor(PromptBasedTextExtractor):
    """Extract succinct ordinance text from input"""

    IN_LABEL = SolarOrdinanceTextCollector.OUT_LABEL
    """Identifier for collected text ingested by this class"""

    TASK_DESCRIPTION = "Extracting solar ordinance text"
    """Task description to show in progress bar"""

    TASK_ID = "ordinance_text_extraction"
    """ID to use for this extraction for linking with LLM configs"""

    PROMPTS = [
        {
            "key": "cleaned_text_for_extraction",
            "out_fn": "{jurisdiction} Utility Scale Solar Ordinance.txt",
            "prompt": _SEF_TEXT_EXTRACTION_PROMPT,
        },
    ]
    """Dicts defining the prompts for ordinance text extraction"""


class SolarPermittedUseDistrictsTextExtractor(PromptBasedTextExtractor):
    """Extract succinct permitted use districts text from input"""

    IN_LABEL = SolarPermittedUseDistrictsTextCollector.OUT_LABEL
    """Identifier for collected text ingested by this class"""

    TASK_DESCRIPTION = "Extracting solar permitted use text"
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
            "prompt": _SEF_PERMITTED_USES_TEXT_EXTRACTION_PROMPT,
        },
    ]
    """Dicts defining the prompts for permitted use text extraction"""
