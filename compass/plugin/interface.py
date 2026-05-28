"""COMPASS extraction plugin base class"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from compass.plugin.base import BaseExtractionPlugin
from compass.llm.calling import BaseLLMCaller
from compass.extraction import extract_relevant_text_with_ngram_validation
from compass.scripts.download import filter_ordinance_docs
from compass.services.threaded import CLEANED_FP_REGISTRY, CleanedFileWriter
from compass.utilities import doc_infos_to_db, save_db
from compass.exceptions import COMPASSPluginConfigurationError


logger = logging.getLogger(__name__)


@dataclass
class OutputColumn:
    """Column expected to appear in a structured CSV output"""

    name: str
    """Name of column in output CSV"""

    include_in_quant_output: bool = True
    """Flag indicating whether to include in the quantitative output"""

    include_in_qual_output: bool = True
    """Flag indicating whether to include in the qualitative output"""


class BaseHeuristic(ABC):
    """Base class for a heuristic check"""

    @abstractmethod
    def check(self, text):
        """Check for mention of a tech in text (or text chunk)

        Parameters
        ----------
        text : str
            Input text that may or may not mention the technology of
            interest.

        Returns
        -------
        bool
            ``True`` if the text passes the heuristic check and
            ``False`` otherwise.
        """
        raise NotImplementedError


class BaseTextCollector(BaseLLMCaller, ABC):
    """Base class for text collectors that gather relevant text"""

    @property
    @abstractmethod
    def OUT_LABEL(self):  # noqa: N802
        """str: Identifier for text collected by this class"""
        raise NotImplementedError

    @property
    @abstractmethod
    def relevant_text(self):
        """str: Combined relevant text from the individual chunks"""
        raise NotImplementedError

    @abstractmethod
    async def check_chunk(self, chunk_parser, ind):
        """Check if a text chunk is relevant for extraction

        You should validate chunks like so::

            is_correct_kind_of_text = await chunk_parser.parse_from_ind(
                ind,
                key="my_unique_validation_key",
                llm_call_callback=my_async_llm_call_function,
            )

        where the `"key"` is unique to this particular validation (it
        will be used to cache the validation result in the chunk
        parser's memory) and `my_async_llm_call_function` is an async
        function that takes in a key and text chunk and returns a
        boolean indicating whether or not the text chunk passes the
        validation. You can call `chunk_parser.parse_from_ind` as many
        times as you want within this method, but be sure to use unique
        keys for each validation.

        Parameters
        ----------
        chunk_parser : ParseChunksWithMemory
            Instance that contains a ``parse_from_ind`` method.
        ind : int
            Index of the chunk to check.

        Returns
        -------
        bool
            Boolean flag indicating whether or not the text in the chunk
            contains information relevant to the extraction task.

        See Also
        --------
        :func:`~compass.validation.content.ParseChunksWithMemory.parse_from_ind`
            Method used to parse text from a chunk with memory of prior
            chunk validations.
        """
        raise NotImplementedError


class FilteredExtractionPlugin(BaseExtractionPlugin):
    """Base class for COMPASS extraction plugins

    This class provides the standard COMPASS document filtering and text
    collection pipeline, allowing implementers to focus primarily on the
    structured data extraction step. Filtering and text collection is
    provided by subclassing the `BaseTextCollector` class and setting
    the `TEXT_COLLECTORS` property to a list of the desired text
    collectors.

    Plugins can hook into various stages of the extraction pipeline
    to modify behavior, add custom processing, or integrate with
    external systems.

    Subclasses should implement the desired hooks and override
    methods as needed.
    """

    @property
    @abstractmethod
    def IDENTIFIER(self):  # noqa: N802
        """str: Identifier for extraction task (e.g. "water rights")"""
        raise NotImplementedError

    @property
    @abstractmethod
    def QUERY_TEMPLATES(self):  # noqa: N802
        """list: List of search engine query templates for extraction

        Query templates can contain the placeholder ``{jurisdiction}``
        which will be replaced with the full jurisdiction name during
        the search engine query.
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
    def TEXT_COLLECTORS(self):  # noqa: N802
        """list of BaseTextCollector: Classes to collect text

        Should be an iterable of one or more classes to collect text
        for the extraction task.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def HEURISTIC(self):  # noqa: N802
        """BaseHeuristic: Class with a ``check()`` method

        The ``check()`` method should accept a string of text and
        return ``True`` if the text passes the heuristic check and
        ``False`` otherwise.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def OUTPUT_COLUMNS(self):  # noqa: N802
        """list: List of output columns for the extracted data

        Each entry should be an
        :class:`compass.plugin.interface.OutputColumn` instance.
        """
        raise NotImplementedError

    @classmethod
    def save_structured_data(cls, doc_infos, out_dir):
        """Write extracted water rights data to disk

        Parameters
        ----------
        doc_infos : list of dict
            List of dictionaries containing the following keys:

                - "jurisdiction": An initialized Jurisdiction object
                  representing the jurisdiction that was extracted.
                - "ord_db_fp": A path to the extracted structured data
                  stored on disk, or ``None`` if no data was extracted.

        out_dir : path-like
            Path to the output directory for the data.

        Returns
        -------
        int
            Number of unique jurisdictions that information was
            found/written for.
        """
        db, num_docs_found = doc_infos_to_db(doc_infos, cls.OUTPUT_COLUMNS)
        save_db(db, out_dir, cls.OUTPUT_COLUMNS)
        return num_docs_found

    async def pre_filter_docs_hook(self, extraction_context):  # noqa: PLR6301
        """Pre-process documents before running them through the filter

        Parameters
        ----------
        extraction_context : ExtractionContext
            Context with downloaded documents to process.

        Returns
        -------
        ExtractionContext
            Context with documents to be passed onto the filtering step.
        """
        return extraction_context

    async def post_filter_docs_hook(self, extraction_context):  # noqa: PLR6301
        """Post-process documents after running them through the filter

        Parameters
        ----------
        extraction_context : ExtractionContext
            Context with documents that passed the filtering step.

        Returns
        -------
        ExtractionContext
            Context with documents to be passed onto the parsing step.
        """
        return extraction_context

    async def extract_relevant_text(self, doc, extractor_class, model_config):
        """Condense text for extraction task

        This method takes a text extractor and applies it to the
        collected document chunks to get a concise version of the text
        that can be used for structured data extraction.

        The extracted text will be stored in the ``.attrs`` dictionary
        of the input document under the ``extractor_class.OUT_LABEL``
        key.

        Parameters
        ----------
        doc : BaseDocument
            Document containing text chunks to condense.
        extractor_class : BaseTextExtractor
            Class to use for text extraction.
        model_config : LLMConfig
            Configuration for the LLM model to use for text extraction.
        """
        extractor = extractor_class(
            llm_service=model_config.llm_service,
            usage_tracker=self.usage_tracker,
            **model_config.llm_call_kwargs,
        )
        doc = await extract_relevant_text_with_ngram_validation(
            doc,
            model_config.text_splitter,
            extractor,
            original_text_key=extractor_class.IN_LABEL,
        )
        await self._write_cleaned_text(doc)

    async def get_query_templates(self):
        """Get a list of search engine query templates for extraction

        Query templates can contain the placeholder ``{jurisdiction}``
        which will be replaced with the full jurisdiction name during
        the search engine query.
        """
        return self.QUERY_TEMPLATES

    async def get_website_keywords(self):
        """Get a dict of website search keyword scores

        Dictionary mapping keywords to scores that indicate links which
        should be prioritized when performing a website scrape for a
        document.
        """
        return self.WEBSITE_KEYWORDS

    async def get_heuristic(self):
        """Get a `BaseHeuristic` instance with a `check()` method

        The ``check()`` method should accept a string of text and return
        ``True`` if the text passes the heuristic check and ``False``
        otherwise.
        """
        return self.HEURISTIC()

    async def filter_docs(
        self, extraction_context, need_jurisdiction_verification=True
    ):
        """Filter down candidate documents before parsing

        Parameters
        ----------
        extraction_context : ExtractionContext
            Context containing candidate documents to be filtered.
        need_jurisdiction_verification : bool, optional
            Whether to verify that documents pertain to the correct
            jurisdiction. By default, ``True``.

        Returns
        -------
        iterable of BaseDocument
            Filtered documents or ``None`` if no documents remain.
        """
        if not extraction_context:
            return None

        logger.debug(
            "Passing %d document(s) in to `pre_filter_docs_hook` ",
            extraction_context.num_documents,
        )

        docs = await self.pre_filter_docs_hook(extraction_context.documents)
        logger.debug(
            "%d document(s) remaining after `pre_filter_docs_hook` for "
            "%s\n\t- %s",
            len(docs),
            self.jurisdiction.full_name,
            "\n\t- ".join(
                [doc.attrs.get("source", "Unknown source") for doc in docs]
            ),
        )

        heuristic = await self.get_heuristic()
        docs = await filter_ordinance_docs(
            docs,
            self.jurisdiction,
            self.model_configs,
            heuristic=heuristic,
            tech=self.IDENTIFIER,
            text_collectors=self.TEXT_COLLECTORS,
            usage_tracker=self.usage_tracker,
            check_for_correct_jurisdiction=need_jurisdiction_verification,
        )

        if not docs:
            return None

        logger.debug(
            "Passing %d document(s) in to `post_filter_docs_hook` ", len(docs)
        )
        docs = await self.post_filter_docs_hook(docs)
        logger.debug(
            "%d document(s) remaining after `post_filter_docs_hook` for "
            "%s\n\t- %s",
            len(docs),
            self.jurisdiction.full_name,
            "\n\t- ".join(
                [doc.attrs.get("source", "Unknown source") for doc in docs]
            ),
        )
        if not docs:
            return None

        extraction_context.documents = docs
        return extraction_context

    async def _write_cleaned_text(self, doc):
        """Write cleaned text to `clean_files` dir"""
        out_fp = await CleanedFileWriter.call(
            doc, self.IDENTIFIER, self.jurisdiction.full_name
        )
        doc.attrs["cleaned_fps"] = out_fp
        return doc

    def validate_plugin_configuration(self):
        """[NOT PUBLIC API] Validate plugin is properly configured"""
        self._validate_plugin_identifier()
        self._validate_query_templates()
        self._validate_website_keywords()
        self._validate_text_collectors()
        self._register_collected_text_file_names()

    def _validate_plugin_identifier(self):
        """Validate that the plugin has a valid IDENTIFIER property"""
        try:
            __ = self.IDENTIFIER
        except NotImplementedError:
            msg = (
                f"Plugin class {self.__class__.__name__} is missing required "
                "property 'IDENTIFIER'"
            )
            raise COMPASSPluginConfigurationError(msg) from None

    def _validate_query_templates(self):
        """Validate that the plugin has valid QUERY_TEMPLATES"""
        try:
            num_q_templates = len(self.QUERY_TEMPLATES)
        except NotImplementedError:
            msg = (
                f"Plugin class {self.__class__.__name__} is missing required "
                "property 'QUERY_TEMPLATES'"
            )
            raise COMPASSPluginConfigurationError(msg) from None

        if num_q_templates == 0:
            msg = (
                f"Plugin class {self.__class__.__name__} has an empty "
                "'QUERY_TEMPLATES' property! Please provide at least "
                "one query template."
            )
            raise COMPASSPluginConfigurationError(msg)

    def _validate_website_keywords(self):
        """Validate that the plugin has valid WEBSITE_KEYWORDS"""
        try:
            num_website_keywords = len(self.WEBSITE_KEYWORDS)
        except NotImplementedError:
            msg = (
                f"Plugin class {self.__class__.__name__} is missing required "
                "property 'WEBSITE_KEYWORDS'"
            )
            raise COMPASSPluginConfigurationError(msg) from None

        if num_website_keywords == 0:
            msg = (
                f"Plugin class {self.__class__.__name__} has an empty "
                "'WEBSITE_KEYWORDS' property! Please provide at least "
                "one website keyword."
            )
            raise COMPASSPluginConfigurationError(msg)

    def _validate_text_collectors(self):
        """Validate that the plugin has valid TEXT_COLLECTORS"""
        try:
            collectors = self.TEXT_COLLECTORS
        except NotImplementedError:
            msg = (
                f"Plugin class {self.__class__.__name__} is missing required "
                "property 'TEXT_COLLECTORS'"
            )
            raise COMPASSPluginConfigurationError(msg) from None

        if len(collectors) == 0:
            msg = (
                f"Plugin class {self.__class__.__name__} has an empty "
                "'TEXT_COLLECTORS' property! Please provide at least "
                "one text collector class."
            )
            raise COMPASSPluginConfigurationError(msg)

        for collector_class in collectors:
            if not issubclass(collector_class, BaseTextCollector):
                msg = (
                    f"Plugin class {self.__class__.__name__} has invalid "
                    "entry in 'TEXT_COLLECTORS' property: All entries must "
                    "be subclasses of "
                    "compass.plugin.interface.BaseTextCollector, but "
                    f"{collector_class.__name__} is not!"
                )
                raise COMPASSPluginConfigurationError(msg)

    def _register_collected_text_file_names(self):
        """Register file names for writing cleaned text outputs"""

        CLEANED_FP_REGISTRY.setdefault(self.IDENTIFIER.casefold(), {})
        collected_text_key = list(self.TEXT_COLLECTORS)[-1].OUT_LABEL

        CLEANED_FP_REGISTRY[self.IDENTIFIER.casefold()][collected_text_key] = (
            "{jurisdiction} Collected Text.txt"
        )
