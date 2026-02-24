"""Validation components for iterative extraction"""

import logging
import json
from compass.llm.calling import SchemaOutputLLMCaller
from compass.utilities.enums import LLMUsageCategory


logger = logging.getLogger(__name__)


_COMPLETENESS_SYSTEM_MESSAGE = """\
You are an extraction quality auditor. Your job is to identify features \
that appear to be present in the ordinance text but are missing from \
the extraction output.\
"""

_COMPLETENESS_PROMPT_TEMPLATE = """\
Given the following extraction schema and ordinance text, identify any \
features that appear to be present in the text but are missing from \
the extraction output.

# EXTRACTION SCHEMA #
The following features should be extracted when present in the text:

{schema_formatted}

# ORDINANCE TEXT #
{text}

# CURRENT EXTRACTION OUTPUT #
{extraction}

# TASK #
Review the ordinance text and identify any features from the schema \
that have explicit requirements in the text but are NOT present in \
the extraction output.

For each missing feature, provide:
- feature: The exact feature ID from the schema
- issue_type: "missing"
- description: Brief explanation of what requirement was found in the \
  text that should have been extracted
- text_hint: A direct quote or excerpt from the ordinance showing the \
  requirement (keep under 100 words)
- confidence: "high" | "medium" | "low" based on how certain you are

Only report features that are CLEARLY present in the text. Do not \
speculate or infer from indirect evidence.

Return your answer as a JSON array of issue objects.
"""

_CORRECTNESS_SYSTEM_MESSAGE = """\
You are an extraction quality auditor. Your job is to verify that \
extracted values match the source ordinance text.\
"""

_CORRECTNESS_PROMPT_TEMPLATE = """\
Given the following extraction schema, ordinance text, and extracted \
values, identify any values that appear to be incorrect or inconsistent \
with the source text.

# EXTRACTION SCHEMA #
{schema_formatted}

# ORDINANCE TEXT #
{text}

# CURRENT EXTRACTION OUTPUT #
{extraction}

# TASK #
For each feature in the extraction output, verify that the value, \
units, and summary are consistent with the ordinance text.

Identify any features with issues:

**incorrect_value** - The numerical value doesn't match the text
**incorrect_units** - The units are wrong or missing
**incomplete** - The extraction is partial (missing conditions, \
exceptions, or context)
**inconsistent** - The extraction contradicts the text or combines \
incompatible requirements

For each issue, provide:
- feature: The feature ID
- issue_type: One of the types above
- description: What's wrong and what it should be
- text_hint: Relevant excerpt showing the correct information
- confidence: "high" | "medium" | "low"

Be conservative: only flag issues where you are confident something \
is wrong. Small formatting differences or equivalent phrasings are OK.

Return your answer as a JSON array of issue objects.
"""

_COMPLETENESS_OUTPUT_SCHEMA = {
    "type": "object",
    "description": "Completeness validation results",
    "additionalProperties": False,
    "required": ["issues"],
    "properties": {
        "issues": {
            "type": "array",
            "description": "List of missing features",
            "items": {
                "type": "object",
                "required": [
                    "feature",
                    "issue_type",
                    "description",
                    "text_hint",
                    "confidence",
                ],
                "additionalProperties": False,
                "properties": {
                    "feature": {"type": "string"},
                    "issue_type": {"type": "string", "enum": ["missing"]},
                    "description": {"type": "string"},
                    "text_hint": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
            },
        },
    },
}

_CORRECTNESS_OUTPUT_SCHEMA = {
    "type": "object",
    "description": "Correctness validation results",
    "additionalProperties": False,
    "required": ["issues"],
    "properties": {
        "issues": {
            "type": "array",
            "description": "List of incorrect or problematic extractions",
            "items": {
                "type": "object",
                "required": [
                    "feature",
                    "issue_type",
                    "description",
                    "text_hint",
                    "confidence",
                ],
                "additionalProperties": False,
                "properties": {
                    "feature": {"type": "string"},
                    "issue_type": {
                        "type": "string",
                        "enum": [
                            "incorrect_value",
                            "incorrect_units",
                            "incomplete",
                            "inconsistent",
                        ],
                    },
                    "description": {"type": "string"},
                    "text_hint": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
            },
        },
    },
}


class ExtractionValidator(SchemaOutputLLMCaller):
    """Validate extraction outputs against schema and text"""

    async def validate(self, text, schema, extraction, strictness="moderate"):
        """Validate extraction completeness and correctness

        Parameters
        ----------
        text : str
            Original ordinance text
        schema : dict
            Extraction schema with feature descriptions
        extraction : dict or pandas.DataFrame
            Current extraction output
        strictness : str, default="moderate"
            Validation strictness level: "lenient", "moderate", or
            "strict".

        Returns
        -------
        dict
            Contains "is_valid" (bool) and "issues" (list of dicts
            with keys: feature, issue_type, description, text_hint,
            confidence)
        """
        completeness_issues = await self._check_completeness(
            text, schema, extraction
        )

        correctness_issues = await self._check_correctness(
            text, schema, extraction
        )

        all_issues = _filter_by_strictness(
            completeness_issues + correctness_issues, strictness
        )

        is_valid = len(all_issues) == 0

        logger.info(
            "Validation results: %s (%d issues found)",
            "VALID" if is_valid else "INVALID",
            len(all_issues),
        )

        return {"is_valid": is_valid, "issues": all_issues}

    async def _check_completeness(self, text, schema, extraction):
        """[NOT PUBLIC API]"""
        schema_formatted = _format_schema(schema)
        extraction_formatted = _format_extraction(extraction)

        prompt = _COMPLETENESS_PROMPT_TEMPLATE.format(
            schema_formatted=schema_formatted,
            text=text,
            extraction=extraction_formatted,
        )

        logger.debug("Checking completeness with LLM")

        response = await self.call(
            sys_msg=_COMPLETENESS_SYSTEM_MESSAGE,
            content=prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "completeness_check",
                    "strict": True,
                    "schema": _COMPLETENESS_OUTPUT_SCHEMA,
                },
            },
            usage_sub_label=LLMUsageCategory.DOCUMENT_CONTENT_VALIDATION,
        )

        issues = response.get("issues", [])
        logger.debug("Completeness check found %d issues", len(issues))
        return issues

    async def _check_correctness(self, text, schema, extraction):
        """[NOT PUBLIC API]"""
        if not extraction or (
            hasattr(extraction, "empty") and extraction.empty
        ):
            logger.debug("Skipping correctness check for empty extraction")
            return []

        schema_formatted = _format_schema(schema)
        extraction_formatted = _format_extraction(extraction)

        prompt = _CORRECTNESS_PROMPT_TEMPLATE.format(
            schema_formatted=schema_formatted,
            text=text,
            extraction=extraction_formatted,
        )

        logger.debug("Checking correctness with LLM")

        response = await self.call(
            sys_msg=_CORRECTNESS_SYSTEM_MESSAGE,
            content=prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "correctness_check",
                    "strict": True,
                    "schema": _CORRECTNESS_OUTPUT_SCHEMA,
                },
            },
            usage_sub_label=LLMUsageCategory.DOCUMENT_CONTENT_VALIDATION,
        )

        issues = response.get("issues", [])
        logger.debug("Correctness check found %d issues", len(issues))
        return issues


def _format_schema(schema):
    """[NOT PUBLIC API]"""
    if not isinstance(schema, dict):
        return str(schema)

    lines = []
    for feature, description in schema.items():
        if isinstance(description, dict):
            desc_text = description.get("description", str(description))
        else:
            desc_text = str(description)
        lines.append(f"- {feature}: {desc_text}")
    return "\n".join(lines)


def _format_extraction(extraction):
    """[NOT PUBLIC API]"""
    if hasattr(extraction, "to_dict"):
        return json.dumps(extraction.to_dict(), indent=2)
    return json.dumps(extraction, indent=2)


def _filter_by_strictness(issues, strictness):
    """[NOT PUBLIC API]"""
    if strictness == "lenient":
        return [i for i in issues if i.get("confidence") == "high"]
    if strictness == "strict":
        return issues
    return [i for i in issues if i.get("confidence") in {"high", "medium"}]
