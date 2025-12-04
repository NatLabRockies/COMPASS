"""COMPASS enum definitions"""

from enum import StrEnum, auto


class LLMUsageCategory(StrEnum):
    """Enumerate semantic buckets for tracking LLM usage

    The values in this enumeration provide consistent labels when
    recording usage metrics, billing data, and telemetry associated
    with LLM calls originating from COMPASS pipelines. Each category
    maps to a specific functional concern (e.g., ordinance value
    extraction, jurisdiction validation) allowing downstream analytics
    to aggregate usage meaningfully.

    Notes
    -----
    Values intentionally mirror the task names used when instantiating
    :class:`~compass.llm.calling.BaseLLMCaller` implementations so that
    the enumerations can be converted to strings without additional
    mapping logic.
    """

    CHAT = auto()
    """Usage related to general LLM chat calls"""
    DATE_EXTRACTION = auto()
    """Usage related to date extraction tasks"""
    DECISION_TREE = auto()
    """Usage related to decision tree calls"""
    DEFAULT = auto()
    """Usage related to default/fallback calls"""
    DOCUMENT_CONTENT_VALIDATION = auto()
    """Usage related to document content validation tasks"""
    DOCUMENT_ORDINANCE_SUMMARY = auto()
    """Usage related to ordinance summary tasks"""
    DOCUMENT_PERMITTED_USE_CONTENT_VALIDATION = auto()
    """Usage related to permitted use content validation tasks"""
    DOCUMENT_PERMITTED_USE_DISTRICTS_SUMMARY = auto()
    """Usage related to permitted use districts summary tasks"""
    DOCUMENT_JURISDICTION_VALIDATION = auto()
    """Usage related to document jurisdiction validation tasks"""
    URL_JURISDICTION_VALIDATION = auto()
    """Usage related to URL jurisdiction validation tasks"""
    JURISDICTION_MAIN_WEBSITE_VALIDATION = auto()
    """Usage related to jurisdiction main website validation tasks"""
    ORDINANCE_VALUE_EXTRACTION = auto()
    """Usage related to ordinance value extraction tasks"""
    PERMITTED_USE_VALUE_EXTRACTION = auto()
    """Usage related to permitted use value extraction tasks"""


class LLMTasks(StrEnum):
    """Human-friendly task identifiers for LLM workflows

    This enumeration exposes the set of user-facing task names that map
    onto :class:`LLMUsageCategory` entries. Pipeline components use
    these values for configuration (e.g., selecting prompt templates)
    while the paired usage categories ensure consistent metrics
    tracking.

    Notes
    -----
    When a task is defined as a direct alias of an
    :class:`LLMUsageCategory`, it inherits the corresponding usage label
    so downstream monitoring does not require additional translation.
    """

    DATE_EXTRACTION = LLMUsageCategory.DATE_EXTRACTION
    """Date extraction task"""

    DEFAULT = LLMUsageCategory.DEFAULT
    """Default fallback option for all tasks"""

    DOCUMENT_CONTENT_VALIDATION = LLMUsageCategory.DOCUMENT_CONTENT_VALIDATION
    """Document content validation task

    This represents a task like "does the document contain ordinance
    values" or "does the document contain permitted use specifications".
    """

    DOCUMENT_JURISDICTION_VALIDATION = (
        LLMUsageCategory.DOCUMENT_JURISDICTION_VALIDATION
    )
    """Document belongs to correct jurisdiction validation task

    This represents all the tasks associated with validation that the
    document pertains to a particular jurisdiction.
    """

    JURISDICTION_MAIN_WEBSITE_VALIDATION = (
        LLMUsageCategory.JURISDICTION_MAIN_WEBSITE_VALIDATION
    )
    """Webpage is main page for jurisdiction validation task

    This represents all the tasks associated with validation that the
    document pertains to a particular jurisdiction.
    """

    ORDINANCE_TEXT_EXTRACTION = auto()
    """Ordinance text extraction task

    This task represents the extraction/summarization of text containing
    ordinance values.
    """

    PERMITTED_USE_TEXT_EXTRACTION = auto()
    """Permitted use text extraction task

    This task represents the extraction/summarization of text containing
    permitted use descriptions and allowances.
    """

    ORDINANCE_VALUE_EXTRACTION = LLMUsageCategory.ORDINANCE_VALUE_EXTRACTION
    """Ordinance structured value extraction task

    This task represents the extraction of structured ordinance values.
    """

    PERMITTED_USE_VALUE_EXTRACTION = (
        LLMUsageCategory.PERMITTED_USE_VALUE_EXTRACTION
    )
    """Permitted use structured value extraction task

    This task represents the extraction of structured permitted use
    values.
    """
