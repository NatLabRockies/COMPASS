"""COMPASS enum definitions"""

from enum import StrEnum, auto


class LLMUsageCategory(StrEnum):
    """COMPASS LLM usage categories"""

    CHAT = auto()
    DATE_EXTRACTION = auto()
    DECISION_TREE = auto()
    DEFAULT = auto()
    DOCUMENT_CONTENT_VALIDATION = auto()
    DOCUMENT_ORDINANCE_SUMMARY = auto()
    DOCUMENT_PERMITTED_USE_CONTENT_VALIDATION = auto()
    DOCUMENT_PERMITTED_USE_DISTRICTS_SUMMARY = auto()
    DOCUMENT_JURISDICTION_VALIDATION = auto()
    URL_JURISDICTION_VALIDATION = auto()
    JURISDICTION_MAIN_WEBSITE_VALIDATION = auto()
    ORDINANCE_VALUE_EXTRACTION = auto()
    PERMITTED_USE_VALUE_EXTRACTION = auto()


class LLMTasks(StrEnum):
    """LLM-based COMPASS tasks"""

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
