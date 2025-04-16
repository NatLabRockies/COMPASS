"""COMPASS Ordinance Location Validation logic

These are primarily used to validate that a legal document applies to a
particular location.
"""

import asyncio
import logging
from abc import ABC, abstractmethod

from compass.extraction.ngrams import convert_text_to_sentence_ngrams
from compass.utilities.enums import LLMUsageCategory


logger = logging.getLogger(__name__)


class LocationValidator(ABC):
    """Validation base class using a static system prompt"""

    SYSTEM_MESSAGE = None
    """LLM system message describing validation task"""

    def __init__(self, structured_llm_caller):
        """

        Parameters
        ----------
        structured_llm_caller : `StructuredLLMCaller`
            StructuredLLMCaller instance. Used for structured validation
            queries.
        """
        self.slc = structured_llm_caller

    async def check(self, content, **fmt_kwargs):
        """Check if the content passes the validation

        The exact validation is outlined in the class `SYSTEM_MESSAGE`.

        Parameters
        ----------
        content : str
            Document content to validate.
        **fmt_kwargs
            Keyword arguments to be passed to `SYSTEM_MESSAGE.format()`.

        Returns
        -------
        bool
            ``True`` if the content passes the validation check,
            ``False`` otherwise.
        """
        if not content:
            return False
        sys_msg = self.SYSTEM_MESSAGE.format(**fmt_kwargs)
        out = await self.slc.call(
            sys_msg,
            content,
            usage_sub_label=LLMUsageCategory.DOCUMENT_LOCATION_VALIDATION,
        )
        return self._parse_output(out)

    @abstractmethod
    def _parse_output(self, props):
        """Parse LLM response and return boolean validation response"""
        raise NotImplementedError


class URLValidator(LocationValidator):
    """Validator that checks whether a URL matches a county"""

    SYSTEM_MESSAGE = (
        "You extract structured data from a URL. Return your "
        "answer in JSON format. Your JSON file must include exactly two keys. "
        "The first key is 'correct_county', which is a boolean that is set to "
        "`True` if the URL mentions {county} County in some way. DO NOT infer "
        "based on information in the URL about any US state, city, township, "
        "or otherwise. `False` if not sure. The second key is "
        "'correct_state', which is a boolean that is set to `True` if the URL "
        "mentions {state} State in some way. DO NOT infer based on "
        "information in the URL about any US county, city, township, or "
        "otherwise. `False` if not sure."
    )

    def _parse_output(self, props):  # noqa: PLR6301
        """Parse LLM response and return boolean validation response"""
        logger.debug("Parsing URL validation output:\n\t%s", props)
        check_vars = ("correct_county", "correct_state")
        return all(props.get(var) for var in check_vars)


class CountyJurisdictionValidator(LocationValidator):
    """Validator that checks whether text applies at the county level"""

    SYSTEM_MESSAGE = (
        "You extract structured data from legal text. Return your answer "
        'in JSON format with exactly three keys: `"x"`, `"y"`, and '
        '`"explanation"`.\n'
        '\n1. **`"x"` (boolean):**\n'
        "- Set this to `true` **only if** the text **explicitly states** "
        "that the legal regulations apply to a jurisdiction **other than** "
        "{county} County.\n"
        "- This includes cases where the regulations apply to **a "
        "subdivision** (e.g., a township or city within {county} County) or "
        "**a broader scope** (e.g., a state-wide or national regulation).\n"
        "- Set this to `false` if the regulations apply specifically to "
        "**{county} County-level governance**, to **all unincorporated "
        "areas** of {county} County, **or** if there is **not enough "
        "information** to determine the jurisdiction scope.\n"
        '\n2. **`"y"` (boolean):**\n'
        "- Set this to `true` **only if** the text **explicitly states** that "
        "the regulations apply to **more than one county**\n"
        "- Set this to `false` if the regulations apply to a **single county "
        "only** or if there is **not enough information** to determine the "
        "number of counties affected.\n"
        '\n3. **`"explanation"` (string):**\n'
        '- If either `"x"` or `"y"` is `true`, provide a short explanation '
        "**citing the specific text** that led to this conclusion.\n"
        "- If **both** are `false`, explain that there was not enough "
        "information to determine otherwise.\n"
        "\n### **Example Output:**\n"
        "\n#### Correct Cases:\n"
        "\n**Case 1 (Not Enough Information - Default to `False`)**\n"
        'Input text: `"This ordinance applies to wind energy systems."`\n'
        "```json\n"
        "{{\n"
        '  "x": false,\n'
        '  "y": false,\n'
        '  "explanation": "The legal text does not provide enough information '
        "to determine whether it applies beyond {county} County or to "
        'multiple counties."\n'
        "}}\n"
        "\n**Case 2 (Explicitly Applies to a City)**\n"
        'Input text: `"This ordinance applies the city of Sturgis."`\n'
        "```json\n"
        "{{\n"
        '  "x": true,\n'
        '  "y": false,\n'
        '  "explanation": "The legal text explicitly states that it applies '
        'to the city of Sturgis, which is a subdivision of a county."\n'
        "}}\n"
        "\n**Case 3 (Explicitly Applies to Multiple Counties)**\n"
        'Input text: `"These regulations apply to all counties in {state}.'
        '"`\n'
        "```json\n"
        "{{\n"
        '  "x": false,\n'
        '  "y": true,\n'
        '  "explanation": "The legal text explicitly states that it applies '
        'to multiple (all) counties in {state}."\n'
        "}}\n"
        "\n**Case 4 (Explicitly Applies to {county} County, {state})**\n"
        'Input text: `"This ordinance applies to {county} County, {state}."`\n'
        "```json\n"
        "{{\n"
        '  "x": false,\n'
        '  "y": false,\n'
        '  "explanation": "The legal text explicitly states that it applies '
        'to {county} County, {state}."\n'
        "}}\n"
    )
    META_SCORE_KEY = "Jurisdiction Validation Score"

    def _parse_output(self, props):  # noqa: PLR6301
        """Parse LLM response and return boolean validation result"""
        logger.debug(
            "Parsing county jurisdiction validation output:\n\t%s", props
        )
        check_vars = ("x", "y")
        return not any(props.get(var) for var in check_vars)


class CountyNameValidator(LocationValidator):
    """Validator that checks whether text applies to a given county"""

    SYSTEM_MESSAGE = (
        "You extract structured data from legal text. Return your answer "
        'in JSON format with exactly three keys: `"wrong_county"`, '
        '`"wrong_state"`, and `"explanation"`.\n'
        '\n1. **`"wrong_county"` (boolean):**\n'
        "- Set this to `true` **only if** the text **explicitly states** "
        "that it does **not** apply to {county} County.\n"
        "- Set this to `false` if the text applies to {county} County **or** "
        "if there is **not enough information** to determine the county.\n"
        "- Do **not** infer this based on any mention of other U.S. states, "
        "cities, or townships.\n"
        '\n2. **`"wrong_state"` (boolean):**\n'
        "- Set this to `true` **only if** the text **explicitly states** that "
        "it does **not** apply to a jurisdiction in {state}.\n"
        "- Set this to `false` if the text applies to a jurisdiction in "
        "{state} **or** if there is **not enough information** to determine "
        "the state.\n"
        "- Do **not** infer this based on any mention of other U.S. counties, "
        "cities, or townships.\n"
        '\n3. **`"explanation"` (string):**\n'
        '- If either `"wrong_county"` or `"wrong_state"` is `true`, provide a '
        "short explanation **citing the specific text** that led to this "
        "conclusion.\n"
        "- If **both** are `false`, explain that there was not enough "
        "information to determine otherwise.\n"
        "\n### **Example Output:**\n"
        "\n#### Correct Cases:\n"
        "\n**Case 1 (Not Enough Information - Default to `False`)**\n"
        'Input text: `"This ordinance applies to wind energy regulations in '
        'the county."`\n'
        "```json\n"
        "{{\n"
        '  "wrong_county": false,\n'
        '  "wrong_state": false,\n'
        '  "explanation": "The legal text does not provide enough information '
        'to determine whether it applies to {county} County or {state}."\n'
        "}}\n"
        "\n**Case 2 (Explicit Wrong County and State)**\n"
        'Input text: `"This ordinance applies to {not_county} County, '
        '{not_state}."`\n'
        "```json\n"
        "{{\n"
        '  "wrong_county": true,\n'
        '  "wrong_state": true,\n'
        '  "explanation": "The legal text explicitly states that it applies '
        "to {not_county} County, {not_state}, which is not {county} County "
        'or in {state}."\n'
        "}}\n"
        "\n**Case 3 (Explicit Wrong State)**\n"
        'Input text: `"This law applies to counties in {not_state}."`\n'
        "```json\n"
        "{{\n"
        '  "wrong_county": false,\n'
        '  "wrong_state": true,\n'
        '  "explanation": "The legal text explicitly states it applies to '
        'counties in {not_state}, which is not in {state}."\n'
        "}}\n"
        "\n**Case 4 (Explicit Wrong County)**\n"
        'Input text: `"This law applies to {not_county} County."`\n'
        "```json\n"
        "{{\n"
        '  "wrong_county": true,\n'
        '  "wrong_state": false,\n'
        '  "explanation": "The legal text explicitly states it applies to '
        '{not_county} County, which is not in {county} County."\n'
        "}}\n"
    )
    META_SCORE_KEY = "Jurisdiction Name Validation Score"

    def _parse_output(self, props):  # noqa: PLR6301
        """Parse LLM response and return boolean validation response"""
        logger.debug("Parsing county validation output:\n\t%s", props)
        check_vars = ("wrong_county", "wrong_state")
        return not any(props.get(var) for var in check_vars)


class CountyValidator:
    """COMPASS Ordinance County validator

    Combines the logic of several validators into a single class.

    Purpose:
        Determine whether a document pertains to a specific county.
    Responsibilities:
        1. Use a combination of heuristics and LLM queries to determine
           whether or not a document pertains to a particular county.
    Key Relationships:
        Uses a :class:`~compass.llm.calling.StructuredLLMCaller` for
        LLM queries and delegates sub-validation to
        :class:`~compass.validation.location.CountyNameValidator`,
        :class:`~compass.validation.location.CountyJurisdictionValidator`,
        and :class:`~compass.validation.location.URLValidator`.
    """

    def __init__(
        self, structured_llm_caller, score_thresh=0.8, text_splitter=None
    ):
        """

        Parameters
        ----------
        structured_llm_caller : `StructuredLLMCaller`
            StructuredLLMCaller instance. Used for structured validation
            queries.
        score_thresh : float, optional
            Score threshold to exceed when voting on content from raw
            pages. By default, ``0.8``.
        text_splitter : langchain.text_splitter.TextSplitter, optional
            Optional text splitter instance to attach to doc (used for
            splitting out pages in an HTML document).
            By default, ``None``.
        """
        self.score_thresh = score_thresh
        self.cn_validator = CountyNameValidator(structured_llm_caller)
        self.cj_validator = CountyJurisdictionValidator(structured_llm_caller)
        self.url_validator = URLValidator(structured_llm_caller)
        self.text_splitter = text_splitter

    async def check(self, doc, county, state):
        """Check if the document belongs to the county

        Parameters
        ----------
        doc : :class:`elm.web.document.BaseDocument`
            Document instance. Should contain a "source" key in the
            ``attrs`` that contains a URL (used for the URL validation
            check). Raw content will be parsed for county name and
            correct jurisdiction.
        county : str
            County that document should belong to.
        state : str
            State corresponding to `county` input.

        Returns
        -------
        bool
            `True` if the doc contents pertain to the input county.
            `False` otherwise.
        """
        if hasattr(doc, "text_splitter") and self.text_splitter is not None:
            old_splitter = doc.text_splitter
            doc.text_splitter = self.text_splitter
            out = await self._check(doc, county, state)
            doc.text_splitter = old_splitter
            return out

        return await self._check(doc, county, state)

    async def _check(self, doc, county, state):
        """Check if the document belongs to the county"""
        if self.text_splitter is not None:
            doc.text_splitter = self.text_splitter

        source = doc.attrs.get("source")
        kwargs = _add_not_county_kwargs(county, state)
        logger.info("Validating document from source: %s", source or "Unknown")
        logger.debug("Checking for correct for jurisdiction...")
        jurisdiction_is_county = await _validator_check_for_doc(
            validator=self.cj_validator,
            doc=doc,
            score_thresh=self.score_thresh,
            **kwargs,
        )
        if not jurisdiction_is_county:
            return False

        logger.debug(
            "Checking URL (%s) for county name...", source or "Unknown"
        )
        url_is_county = await self.url_validator.check(
            source, county=county, state=state
        )
        if url_is_county:
            return True

        logger.debug(
            "Checking text for county name (heuristic; URL: %s)...",
            source or "Unknown",
        )
        correct_county_heuristic = _heuristic_check_for_county_and_state(
            doc, county, state
        )
        logger.debug(
            "Found county name in text (heuristic): %s",
            correct_county_heuristic,
        )
        if correct_county_heuristic:
            return True

        logger.debug(
            "Checking text for county name (LLM; URL: %s)...",
            source or "Unknown",
        )
        return await _validator_check_for_doc(
            validator=self.cn_validator,
            doc=doc,
            score_thresh=self.score_thresh,
            **kwargs,
        )


def _heuristic_check_for_county_and_state(doc, county, state):
    """Check if county and state names are in doc"""
    return any(
        any(
            (county.lower() in fg and state.lower() in fg)
            for fg in convert_text_to_sentence_ngrams(t.lower(), 5)
        )
        for t in doc.pages
    )


async def _validator_check_for_doc(validator, doc, score_thresh=0.9, **kwargs):
    """Apply a validator check to a doc's raw pages"""
    outer_task_name = asyncio.current_task().get_name()
    validation_checks = [
        asyncio.create_task(
            validator.check(text, **kwargs), name=outer_task_name
        )
        for text in doc.raw_pages
    ]
    out = await asyncio.gather(*validation_checks)
    score = _weighted_vote(out, doc)
    doc.attrs[validator.META_SCORE_KEY] = score
    logger.debug(
        "%s is %.2f for doc from source %s (Pass: %s; threshold: %.2f)",
        validator.META_SCORE_KEY,
        score,
        doc.attrs.get("source", "Unknown"),
        str(score >= score_thresh),
        score_thresh,
    )
    return score >= score_thresh


def _weighted_vote(out, doc):
    """Compute weighted average of responses based on text length"""
    if not doc.raw_pages:
        return 0
    weights = [len(text) for text in doc.raw_pages]
    total = sum(
        verdict * weight for verdict, weight in zip(out, weights, strict=False)
    )
    return total / sum(weights)


def _add_not_county_kwargs(county, state):
    """Add 'not_county' and 'not_state' kwargs"""
    kwargs = {"county": county, "state": state}
    if county.casefold() != "decatur":
        kwargs["not_county"] = "Decatur"
        kwargs["not_state"] = "Indiana"
        return kwargs

    kwargs["not_county"] = "Lincoln"
    kwargs["not_state"] = "Nebraska"
    return kwargs
