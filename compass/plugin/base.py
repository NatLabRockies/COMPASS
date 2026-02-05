"""Base COMPASS extraction plugin interface"""

from abc import ABC, abstractmethod

from compass.pb import COMPASS_PB
from compass.services.threaded import UsageUpdater
from compass.utilities import compute_total_cost_from_usage


class BaseExtractionPlugin(ABC):
    """Base class for COMPASS extraction plugins

    This class provides the most extraction flexibility, but the
    implementer must define most functionality on their own.
    """

    def __init__(self, jurisdiction, model_configs, usage_tracker=None):
        self.jurisdiction = jurisdiction
        self.model_configs = model_configs
        self.usage_tracker = usage_tracker

    @property
    @abstractmethod
    def IDENTIFIER(self):  # noqa: N802
        """str: Identifier for extraction task (e.g. "water rights")"""
        raise NotImplementedError

    @property
    @abstractmethod
    def QUESTION_TEMPLATES(self):  # noqa: N802
        """list: List of search engine question templates for extraction

        Question templates can contain the placeholder
        ``{jurisdiction}`` which will be replaced with the full
        jurisdiction name during the search engine query.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def WEBSITE_KEYWORDS(self):  # noqa: N802
        """list: List of keywords

        List of keywords that indicate links which should be prioritized
        when performing a website scrape for a document.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def heuristic(self):
        """BaseHeuristic: Object with a ``check()`` method

        The ``check()`` method should accept a string of text and
        return ``True`` if the text passes the heuristic check and
        ``False`` otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    async def filter_docs(
        self, extraction_context, need_jurisdiction_verification=True
    ):
        """Filter down candidate documents before parsing

        Parameters
        ----------
        extraction_context : compass.plugin.context.ExtractionContext
            Context containing candidate documents to be filtered.
            Set the ``.documents`` attribute of this object to be the
            iterable of documents that should be kept for parsing.
        need_jurisdiction_verification : bool, optional
            Whether to verify that documents pertain to the correct
            jurisdiction. By default, ``True``.

        Returns
        -------
        compass.plugin.context.ExtractionContext
            Context with filtered down documents.
        """
        raise NotImplementedError

    @abstractmethod
    async def parse_docs_for_structured_data(self, extraction_context):
        """Parse documents to extract structured data/information

        Parameters
        ----------
        extraction_context : compass.plugin.context.ExtractionContext
            Context containing candidate documents to parse.

        Returns
        -------
        compass.plugin.context.ExtractionContext or None
            Context with extracted data/information stored in the
            ``.attrs`` dictionary, or ``None`` if no data was extracted.
        """
        raise NotImplementedError

    async def record_usage(self):
        """Persist usage tracking data when a tracker is available"""
        if self.usage_tracker is None:
            return

        total_usage = await UsageUpdater.call(self.usage_tracker)
        total_cost = compute_total_cost_from_usage(total_usage)
        COMPASS_PB.update_total_cost(total_cost, replace=True)
