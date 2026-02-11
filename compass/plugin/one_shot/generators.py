"""COMPASS one-shot extraction plugin generators"""

import operator
import importlib.resources

from elm.utilities.retry import async_retry_with_exponential_backoff

from compass.utilities.io import load_config
from compass.utilities.enums import LLMUsageCategory
from compass.exceptions import COMPASSRuntimeError


_SCHEMA_DIR = importlib.resources.files("compass.plugin.one_shot.schemas")
_QUERY_GENERATOR_SYSTEM_PROMPT = """\
You are an expert search strategist for regulatory documents. \
Goal: Given an extraction schema (JSON) for an ordinance domain, generate \
high-quality search engine query templates that will find the legal texts \
from which the schema's data can be extracted.

Input:
- schema_json: a JSON schema describing features/requirements to extract.

Output:
- Produce 5-10 query templates.
- Every template must include the literal placeholder "{jurisdiction}" \
(exactly, with braces) somewhere in the template, which will be filled in  \
**later** with a plaintext string for a specific location (e.g. "City of  \
Denver, Colorado", "Clear Fork Groundwater Conservation District, Texas",  \
etc.).
- Do not include extra keys or any markdown.

Guidelines:
- Derive terms from the schema title/description, feature names, and \
definitions. Prefer official/legal terminology in the schema.
- Do not focus on specific extraction keywords; instead target the document \
types that would include that information.
- Include a mix of broad and precise queries and both styles.
- Include at least one query with filetype:pdf.
- Include terms that indicate the governing document type \
(e.g., "ordinance", "zoning", "code", "regulations", "chapter", "section").
- Include domain-specific synonyms and abbreviations present in the schema \
(e.g., WECS, WES, wind energy conversion system for wind, SECS, SEF, solar \
energy conversion system for solar, etc.).
- If relevant to the schema, include some queries that target sites known to \
host aggregate information (e.g. municode, american legal publishing, etc. \
for ordinance documents).
- Avoid jurisdiction-specific entities other than the {jurisdiction} \
placeholder.
- Ensure templates are for locating the legal text itself (not summaries, \
news, or reports).\
"""

_KEYWORD_GENERATOR_SYSTEM_PROMPT = """\
You are an expert search strategist for regulatory documents. \
Goal: Given an extraction schema (JSON) for an ordinance domain, generate \
high-quality website keywords and weights for prioritizing crawl links.

Input:
- schema_json: a JSON schema describing features/requirements to extract.

Output:
- Produce an array of keyword/weight objects with integer weights.
- Do not include extra keys or any markdown.

Guidelines:
- Derive terms from the schema title/description, feature names, and \
definitions. Prefer official/legal terminology in the schema.
- Focus on keywords likely to appear in legal document URLs or link text.
- Include terms that indicate governing document types \
(e.g., "ordinance", "zoning", "code", "regulations", "chapter", "section").
- Include domain-specific synonyms and abbreviations present in the schema.
- Weights are relative: higher means more relevant for link prioritization.
- Avoid jurisdiction-specific entities.
"""

_HEURISTIC_GENERATOR_SYSTEM_PROMPT = """\
You are an expert in ordinance discovery and regulatory text filtering. \
Goal: Given an extraction schema (JSON) for an ordinance domain, generate \
keyword lists for a heuristic text check that detects domain-relevant \
content and excludes look-alike words.

Input:
- schema_json: a JSON schema describing features/requirements to extract.

Output:
- Provide four keyword lists in the response schema.
- Do not include extra keys or any markdown.

Guidelines:
- Derive terms from schema title/description, feature names, and \
definitions. Prefer official/legal terminology and abbreviations.
- not_tech_words should include common look-alikes or near matches that \
appear in non-domain contexts and could cause false positives. These will \
be removed from the text before performing a keyword-based relevance check. \
- good_tech_keywords should include single-word indicators likely to \
appear in ordinance text.
- good_tech_acronyms should include short acronyms and abbreviations used \
in legal documents for the domain.
- good_tech_phrases should include multi-word phrases (at least 2 words) \
that indicate domain relevance.
- Avoid jurisdiction-specific names; keep keywords general.
"""


@async_retry_with_exponential_backoff(
    base_delay=1,
    exponential_base=4,
    jitter=True,
    max_retries=3,
    errors=(COMPASSRuntimeError,),
)
async def generate_query_templates(
    schema_llm, extraction_schema, add_think_prompt=True
):
    """Generate 5-10 search query templates for document retrieval

    Parameters
    ----------
    schema_llm : SchemaOutputLLMCaller
        A LLM caller configured to output structured data according to a
        provided schema. This function relies on the LLM to generate the
        query templates, so the quality of the generated templates will
        depend on the capabilities of the LLM being used and how well it
        can interpret the provided extraction schema. Highly recommended
        to use the most powerful/capable instruction-tuned model for
        this function.
    extraction_schema : dict
        A dictionary representing the schema of the desired extraction
        task. The query templates will be generated based on the content
        of this schema, so it should be as detailed and specific as
        possible, and should include domain-specific terminology if
        applicable. See the wind ordinance schema for an example.
    add_think_prompt : bool, optional
        Option to add a "Think before you answer" instruction to the end
        of the prompt (useful for thinking models).
        By default, ``True``.

    Returns
    -------
    list of str
        List of 5-10 query templates as strings, each including the
        literal placeholder "{jurisdiction}" for later formatting.

    Raises
    ------
    COMPASSRuntimeError
        If the LLM fails to return any valid query templates after 3
        attempts.
    """

    query_schema_fp = _SCHEMA_DIR / "query_templates.json5"
    query_schema = load_config(query_schema_fp)
    main_prompt = (
        "Generate query templates for the following extraction schema:\n\n"
        f"{extraction_schema}"
    )
    if add_think_prompt:
        main_prompt = f"{main_prompt}\n\nThink before you answer"

    response = await schema_llm.call(
        sys_msg=_QUERY_GENERATOR_SYSTEM_PROMPT,
        content=main_prompt,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "query_template_generation",
                "strict": True,
                "schema": query_schema,
            },
        },
        usage_sub_label=LLMUsageCategory.PLUGIN_GENERATION,
    )
    out = [q for q in response["queries"] if _is_formattable(q)]
    if not out:
        msg = (
            "LLM did not return any valid query templates. "
            f"Received response: {response}"
        )
        raise COMPASSRuntimeError(msg)

    return out


@async_retry_with_exponential_backoff(
    base_delay=1,
    exponential_base=4,
    jitter=True,
    max_retries=3,
    errors=(COMPASSRuntimeError,),
)
async def generate_website_keywords(
    schema_llm, extraction_schema, add_think_prompt=True
):
    """Generate website keyword weights for document retrieval

    Parameters
    ----------
    schema_llm : SchemaOutputLLMCaller
        A LLM caller configured to output structured data according to a
        provided schema. This function relies on the LLM to generate the
        keyword weights, so the quality of the generated keywords will
        depend on the capabilities of the LLM being used and how well it
        can interpret the provided extraction schema. Highly recommended
        to use the most powerful/capable instruction-tuned model for
        this function.
    extraction_schema : dict
        A dictionary representing the schema of the desired extraction
        task. The keywords will be generated based on the content of
        this schema, so it should be as detailed and specific as
        possible, and should include domain-specific terminology if
        applicable. See the wind ordinance schema for an example.
    add_think_prompt : bool, optional
        Option to add a "Think before you answer" instruction to the end
        of the prompt (useful for thinking models).
        By default, ``True``.

    Returns
    -------
    dict
        Dictionary mapping keywords to integer weights for website link
        prioritization.

    Raises
    ------
    COMPASSRuntimeError
        If the LLM fails to return any valid keyword weights after 3
        attempts.
    """

    keyword_schema_fp = _SCHEMA_DIR / "website_keywords.json5"
    keyword_schema = load_config(keyword_schema_fp)
    main_prompt = (
        "Generate website keyword weights for the following extraction "
        f"schema:\n\n{extraction_schema}"
    )
    if add_think_prompt:
        main_prompt = f"{main_prompt}\n\nThink before you answer"

    response = await schema_llm.call(
        sys_msg=_KEYWORD_GENERATOR_SYSTEM_PROMPT,
        content=main_prompt,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "website_keyword_generation",
                "strict": True,
                "schema": keyword_schema,
            },
        },
        usage_sub_label=LLMUsageCategory.PLUGIN_GENERATION,
    )
    out = _normalize_website_keywords(response.get("keywords"))
    if not out:
        msg = (
            "LLM did not return any valid website keywords. "
            f"Received response: {response}"
        )
        raise COMPASSRuntimeError(msg)

    return out


@async_retry_with_exponential_backoff(
    base_delay=1,
    exponential_base=4,
    jitter=True,
    max_retries=3,
    errors=(COMPASSRuntimeError,),
)
async def generate_heuristic_keywords(
    schema_llm, extraction_schema, add_think_prompt=True
):
    """Generate keyword lists for a heuristic text check

    Parameters
    ----------
    schema_llm : SchemaOutputLLMCaller
        A LLM caller configured to output structured data according to a
        provided schema. This function relies on the LLM to generate the
        heuristic keyword lists, so the quality of the generated output
        will depend on the capabilities of the LLM being used and how
        well it can interpret the provided extraction schema.
    extraction_schema : dict
        A dictionary representing the schema of the desired extraction
        task. The keyword lists will be generated based on the content
        of this schema, so it should be as detailed and specific as
        possible, and should include domain-specific terminology if
        applicable. See the wind ordinance schema for an example.
    add_think_prompt : bool, optional
        Option to add a "Think before you answer" instruction to the end
        of the prompt (useful for thinking models).
        By default, ``True``.

    Returns
    -------
    dict
        Dictionary containing the keyword lists for a heuristic text
        check: ``not_tech_words``, ``good_tech_keywords``,
        ``good_tech_acronyms``, and ``good_tech_phrases``.

    Raises
    ------
    COMPASSRuntimeError
        If the LLM fails to return any valid heuristic keywords after 3
        attempts.
    """

    heuristic_schema_fp = _SCHEMA_DIR / "heuristic_keywords.json5"
    heuristic_schema = load_config(heuristic_schema_fp)
    main_prompt = (
        "Generate heuristic keyword lists for the following extraction "
        f"schema:\n\n{extraction_schema}"
    )
    if add_think_prompt:
        main_prompt = f"{main_prompt}\n\nThink before you answer"

    response = await schema_llm.call(
        sys_msg=_HEURISTIC_GENERATOR_SYSTEM_PROMPT,
        content=main_prompt,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "heuristic_keyword_generation",
                "strict": True,
                "schema": heuristic_schema,
            },
        },
        usage_sub_label=LLMUsageCategory.PLUGIN_GENERATION,
    )

    if not response:
        msg = (
            "LLM did not return any heuristic keywords. "
            f"Received response: {response}"
        )
        raise COMPASSRuntimeError(msg)

    return response


def _normalize_website_keywords(raw):
    """Normalize keyword weights into a deduplicated dict"""
    if not raw:
        return {}

    items = _parse_llm_kw_to_list(raw)
    if not items:
        return {}

    return _de_duplicate_keywords(items)


def _is_formattable(q):
    """True if the query template is formattable with a jurisdiction"""
    try:
        q.format(jurisdiction="test")
    except Exception:  # noqa: BLE001
        return False

    return True


def _parse_llm_kw_to_list(llm_kw):
    """Parse LLM output into a list of (keyword, weight) tuples"""
    items = []
    for item in llm_kw:
        if isinstance(item, str):
            items.append((item, 1))
        elif isinstance(item, dict):
            items.append((item.get("keyword"), item.get("weight", 1)))
    return items


def _de_duplicate_keywords(items):
    """Process keywords by normalizing and keeping the highest weight"""
    deduped = {}
    sorted_items = sorted(items, key=operator.itemgetter(1), reverse=True)
    for keyword, weight in sorted_items:
        if not isinstance(keyword, str):
            continue

        normalized = keyword.strip().casefold()
        if not normalized or normalized.isdigit():
            continue
        try:
            int_weight = int(weight)
        except (TypeError, ValueError):
            continue

        if int_weight < 1:
            continue

        deduped.setdefault(normalized, weight)

    return deduped
