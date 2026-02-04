"""COMPASS extraction plugin base class"""

import asyncio
import logging
from itertools import chain
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import cached_property

import pandas as pd

from .base import BaseExtractionPlugin
from compass.llm.calling import LLMCaller
from compass.extraction import (
    extract_ordinance_values,
    extract_relevant_text_with_ngram_validation,
)
from compass.scripts.download import filter_ordinance_docs
from compass.services.threaded import (
    FileMover,
    CleanedFileWriter,
    OrdDBFileWriter,
)
from compass.utilities.enums import LLMTasks
from compass.utilities import (
    num_ordinances_dataframe,
    doc_infos_to_db,
    save_db,
)
from compass.exceptions import COMPASSPluginConfigurationError
from compass.pb import COMPASS_PB

logger = logging.getLogger(__name__)


EXCLUDE_FROM_ORD_DOC_CHECK = {
    # if doc only contains these, it's not good enough to count as an
    # ordinance. Note that prohibitions are explicitly not on this list
    "color",
    "decommissioning",
    "lighting",
    "visual impact",
    "glare",
    "repowering",
    "fencing",
    "climbing prevention",
    "signage",
    "soil",
    "primary use districts",
    "special use districts",
    "accessory use districts",
}


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


class BaseTextCollector(ABC):
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
        ParseChunksWithMemory.parse_from_ind
            Method used to parse text from a chunk with memory of prior
            chunk validations.
        """
        raise NotImplementedError


class BaseTextExtractor(ABC):
    """Extract succinct extraction text from input"""

    TASK_DESCRIPTION = "Condensing text for extraction"
    """Task description to show in progress bar"""

    TASK_ID = "text_extraction"
    """ID to use for this extraction for linking with LLM configs"""

    @property
    @abstractmethod
    def IN_LABEL(self):  # noqa: N802
        """str: Identifier for text ingested by this class"""
        raise NotImplementedError

    @property
    @abstractmethod
    def OUT_LABEL(self):  # noqa: N802
        """str: Identifier for final text extracted by this class"""
        raise NotImplementedError

    @property
    @abstractmethod
    def parsers(self):
        """Generator: Generator of (key, extractor) pairs

        `extractor` should be an async callable that accepts a list of
        text chunks and returns the shortened (succinct) text to be used
        for extraction. The `key` should be a string identifier for the
        text returned by the extractor. Multiple (key, extractor) pairs
        can be chained in generator order to iteratively refine the
        text for extraction.
        """
        raise NotImplementedError


class BaseParser(ABC):
    """Extract succinct extraction text from input"""

    TASK_ID = "data_extraction"
    """ID to use for this extraction for linking with LLM configs"""

    @property
    @abstractmethod
    def IN_LABEL(self):  # noqa: N802
        """str: Identifier for text ingested by this class"""
        raise NotImplementedError

    @property
    @abstractmethod
    def OUT_LABEL(self):  # noqa: N802
        """str: Identifier for final structured data output"""
        raise NotImplementedError

    @abstractmethod
    async def parse(self, text):
        """Parse text and extract structured data

        Parameters
        ----------
        text : str
            Text which may or may not contain information relevant to
            the current extraction.

        Returns
        -------
        pandas.DataFrame or None
            DataFrame containing structured extracted data. Can also
            be ``None`` if no relevant values can be parsed from the
            text.
        """
        raise NotImplementedError


class ExtractionPlugin(BaseExtractionPlugin):
    """Base class for COMPASS extraction plugins

    This class provides a good balance between ease of use and
    extraction flexibility, allowing implementers to provide additional
    functionality during the extraction process.

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
    def TEXT_COLLECTORS(self):  # noqa: N802
        """iterable of BaseTextCollector: Classes to collect text

        Should one or more classes to collect text for the extraction
        task.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def TEXT_EXTRACTORS(self):  # noqa: N802
        """iterable of BaseTextExtractor: Classes to condense text

        Should be one or more classes to condense text in preparation
        for the extraction task.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def PARSERS(self):  # noqa: N802
        """iterable of BaseParsers: Classes to extract structured data

        Should be one or more classes to extract structured data from
        text.
        """
        raise NotImplementedError

    @property
    def heuristic(self):
        """BaseHeuristic: Object with a ``check()`` method

        The ``check()`` method should accept a string of text and
        return ``True`` if the text passes the heuristic check and
        ``False`` otherwise.
        """
        raise NotImplementedError

    @classmethod
    def save_structured_data(cls, doc_infos, out_dir):
        """Write extracted data to disk"""
        db, num_docs_found = doc_infos_to_db(doc_infos)
        save_db(db, out_dir)
        return num_docs_found

    def __init__(self, jurisdiction, model_configs, usage_tracker=None):
        self.jurisdiction = jurisdiction
        self.model_configs = model_configs
        self.usage_tracker = usage_tracker

        # TODO: This should happen during plugin registration
        self._validate_in_out_keys()

    @cached_property
    def producers(self):
        """iterable: All classes that produce attributes on the doc"""
        return chain(self.PARSERS, self.TEXT_EXTRACTORS, self.TEXT_COLLECTORS)

    @cached_property
    def consumer_producer_pairs(self):
        """list: Pairs of (consumer, producer) for IN/OUT validation"""
        return [
            (self.PARSERS, chain(self.TEXT_EXTRACTORS, self.TEXT_COLLECTORS)),
            (self.TEXT_EXTRACTORS, self.TEXT_COLLECTORS),
        ]

    def _validate_in_out_keys(self):
        """Validate that all IN_LABELs have matching OUT_LABELs"""
        out_keys = {}
        for producer in self.producers:
            out_keys.setdefault(producer.OUT_LABEL, []).append(producer)

        dupes = {k: v for k, v in out_keys.items() if len(v) > 1}
        if dupes:
            formatted = "\n".join(
                [
                    f"{key}: {[cls.__name__ for cls in classes]}"
                    for key, classes in dupes.items()
                ]
            )
            msg = (
                "Multiple processing classes produce the same OUT_LABEL key:\n"
                f"{formatted}"
            )
            raise COMPASSPluginConfigurationError(msg)

        for consumers, producers in self.consumer_producer_pairs:
            _validate_in_out_keys(consumers, producers)

    async def pre_filter_docs_hook(self, docs):  # noqa: PLR6301
        """Pre-process documents before running them through the filter

        Parameters
        ----------
        docs : iterable of elm.web.document.BaseDocument
            Downloaded documents to process.

        Returns
        -------
        iterable of elm.web.document.BaseDocument
            Documents to be passed onto the filtering step.
        """
        return docs

    async def post_filter_docs_hook(self, docs):  # noqa: PLR6301
        """Post-process documents after running them through the filter

        Parameters
        ----------
        docs : iterable of elm.web.document.BaseDocument
            Documents that passed the filtering step.

        Returns
        -------
        iterable of elm.web.document.BaseDocument
            Documents to be passed onto the parsing step.
        """
        return docs

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
        doc : elm.web.document.BaseDocument
            Document containing text chunks to condense.
        extractor_class : BaseTextExtractor
            Class to use for text extraction.
        model_config : ModelConfig
            Configuration for the LLM model to use for text extraction.
        """
        llm_caller = LLMCaller(
            llm_service=model_config.llm_service,
            usage_tracker=self.usage_tracker,
            **model_config.llm_call_kwargs,
        )
        extractor = extractor_class(llm_caller)
        doc = await extract_relevant_text_with_ngram_validation(
            doc,
            model_config.text_splitter,
            extractor,
            original_text_key=extractor_class.IN_LABEL,
        )
        await _write_cleaned_text(doc)

    async def extract_ordinances_from_text(
        self, doc, parser_class, model_config
    ):
        """Extract structured data from input text

        The extracted structured data will be stored in the ``.attrs``
        dictionary of the input document under the
        ``parser_class.OUT_LABEL`` key.

        Parameters
        ----------
        doc : elm.web.document.BaseDocument
            Document containing text to extract structured data from.
        parser_class : BaseParser
            Class to use for structured data extraction.
        model_config : ModelConfig
            Configuration for the LLM model to use for structured data
            extraction.
        """
        parser = parser_class(
            llm_service=model_config.llm_service,
            usage_tracker=self.usage_tracker,
            **model_config.llm_call_kwargs,
        )
        logger.info(
            "Extracting %s...", parser_class.OUT_LABEL.replace("_", " ")
        )
        await extract_ordinance_values(
            doc,
            parser,
            text_key=parser_class.IN_LABEL,
            out_key=parser_class.OUT_LABEL,
        )

    @classmethod
    def get_structured_data_row_count(cls, doc):
        """Get the number of data rows extracted from a document

        Parameters
        ----------
        doc : elm.web.document.BaseDocument or None
            Document to check for extracted structured_data.

        Returns
        -------
        int
            Number of data rows extracted from the document.
        """
        if doc is None or doc.attrs.get("ordinance_values") is None:
            return 0

        ord_df = doc.attrs["ordinance_values"]

        return num_ordinances_dataframe(
            ord_df, exclude_features=EXCLUDE_FROM_ORD_DOC_CHECK
        )

    async def filter_docs(self, docs, need_jurisdiction_verification=True):
        """Filter down candidate documents before parsing

        Parameters
        ----------
        docs : iterable of elm.web.document.BaseDocument
            Documents to filter.
        need_jurisdiction_verification : bool, optional
            Whether to verify that documents pertain to the correct
            jurisdiction. By default, ``True``.

        Returns
        -------
        iterable of elm.web.document.BaseDocument
            Filtered documents or ``None`` if no documents remain.
        """
        if not docs:
            return None

        logger.debug(
            "Passing %d document(s) in to `pre_filter_docs_hook` ", len(docs)
        )

        docs = await self.pre_filter_docs_hook(docs)
        logger.debug(
            "%d document(s) remaining after `pre_filter_docs_hook` for "
            "%s\n\t- %s",
            len(docs),
            self.jurisdiction.full_name,
            "\n\t- ".join(
                [doc.attrs.get("source", "Unknown source") for doc in docs]
            ),
        )

        docs = await filter_ordinance_docs(
            docs,
            self.jurisdiction,
            self.model_configs,
            heuristic=self.heuristic,
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

        return docs or None

    async def parse_docs_for_structured_data(self, docs):
        """Parse documents to extract structured data/information

        Parameters
        ----------
        docs : iterable of elm.web.document.BaseDocument
            Documents to parse.

        Returns
        -------
        elm.web.document.BaseDocument or None
            Document with extracted data/information stored in the
            ``.attrs`` dictionary, or ``None`` if no data was extracted.
        """

        for doc_for_extraction in docs:
            doc = await self.parse_single_doc_for_structured_data(
                doc_for_extraction
            )
            row_count = self.get_structured_data_row_count(doc)
            if row_count > 0:
                doc = await _move_files(doc)
                logger.info(
                    "%d ordinance value(s) found in doc from %s for %s. "
                    "Outputs are here: '%s'",
                    row_count,
                    doc_for_extraction.attrs.get("source", "unknown source"),
                    self.jurisdiction.full_name,
                    doc.attrs["ord_db_fp"],
                )
                return doc

        logger.debug("No ordinances found; searched %d docs", len(docs))
        return None

    async def parse_single_doc_for_structured_data(self, doc_for_extraction):
        """Extract all possible structured data from a document

        This method is called from the default implementation of
        `parse_docs_for_structured_data()` for each document that passed
        filtering. If you overwrite`parse_docs_for_structured_data()``,
        you can ignore this method.

        Parameters
        ----------
        doc_for_extraction : elm.web.document.BaseDocument
            Document to extract structured data from.

        Returns
        -------
        elm.web.document.BaseDocument
            Document with extracted structured data stored in the
            ``.attrs`` dictionary.
        """
        with self._tracked_progress():
            tasks = [
                asyncio.create_task(
                    self._try_extract_ordinances(
                        doc_for_extraction, parser_class
                    ),
                    name=self.jurisdiction.full_name,
                )
                for parser_class in self.PARSERS
                if parser_class is not None
            ]
            await asyncio.gather(*tasks)

        return self._concat_scrape_results(doc_for_extraction)

    async def _try_extract_ordinances(self, doc_for_extraction, parser_class):
        """Apply a single extractor and parser to legal text"""

        if parser_class.IN_LABEL not in doc_for_extraction.attrs:
            te = [
                te
                for te in self.TEXT_EXTRACTORS
                if te.OUT_LABEL == parser_class.IN_LABEL
            ]
            if len(te) != 1:
                msg = (
                    f"Could not find unique text extractor for parser "
                    f"{parser_class.__name__} with IN_LABEL "
                    f"{parser_class.IN_LABEL!r}. Got matches: {te}"
                )
                raise COMPASSPluginConfigurationError(msg)

            te = te[0]
            if te.TASK_ID in self.model_configs:
                model_config = self.model_configs[te.TASK_ID]
            else:
                model_config = self.model_configs.get(
                    LLMTasks.TEXT_EXTRACTION,
                    self.model_configs[LLMTasks.DEFAULT],
                )
            logger.debug(
                "Condensing text for extraction using %r for doc from %s",
                te.__name__,
                doc_for_extraction.attrs.get("source", "unknown source"),
            )
            assert self._jsp is not None, "No progress bar set!"
            task_id = self._jsp.add_task(te.TASK_DESCRIPTION)
            await self.extract_relevant_text(
                doc_for_extraction, te, model_config
            )
            await self.record_usage()
            self._jsp.remove_task(task_id)

        if parser_class.TASK_ID in self.model_configs:
            model_config = self.model_configs[parser_class.TASK_ID]
        else:
            model_config = self.model_configs.get(
                LLMTasks.DATA_EXTRACTION,
                self.model_configs[LLMTasks.DEFAULT],
            )
        await self.extract_ordinances_from_text(
            doc_for_extraction,
            parser_class=parser_class,
            model_config=model_config,
        )

        await self.record_usage()

    @contextmanager
    def _tracked_progress(self):
        """Context manager to set up jurisdiction sub-progress bar"""
        loc = self.jurisdiction.full_name
        with COMPASS_PB.jurisdiction_sub_prog(loc) as self._jsp:
            yield

        self._jsp = None

    def _concat_scrape_results(self, doc):
        """Concatenate structured data from all parsers"""
        data = [doc.attrs.get(p.OUT_LABEL, None) for p in self.PARSERS]
        data = [df for df in data if df is not None and not df.empty]
        if len(data) == 0:
            return doc

        if len(data) == 1:
            doc.attrs["structured_data"] = data[0]
            return doc

        doc.attrs["structured_data"] = pd.concat(data)
        return doc


def _validate_in_out_keys(consumers, producers):
    """Validate that all IN_LABELs have matching OUT_LABELs"""
    in_keys = {}
    out_keys = {}

    for producer_class in producers:
        out_keys.setdefault(producer_class.OUT_LABEL, []).append(
            producer_class
        )

    for consumer_class in chain(consumers):
        in_keys.setdefault(consumer_class.IN_LABEL, []).append(consumer_class)

    for in_key, classes in in_keys.items():
        formatted = f"{[cls.__name__ for cls in classes]}"
        if in_key not in out_keys:
            msg = (
                f"One or more processing classes require IN_LABEL "
                f"{in_key!r}, which is not produced by any previous "
                f"processing class: {formatted}"
            )
            raise COMPASSPluginConfigurationError(msg)


async def _move_files(doc):
    """Move files to output folders, if applicable"""
    doc = await _move_file_to_out_dir(doc)
    return await _write_ord_db(doc)


async def _move_file_to_out_dir(doc):
    """Move PDF or HTML text file to output directory"""
    out_fp = await FileMover.call(doc)
    doc.attrs["out_fp"] = out_fp
    return doc


async def _write_cleaned_text(doc):
    """Write cleaned text to `clean_files` dir"""
    out_fp = await CleanedFileWriter.call(doc)
    doc.attrs["cleaned_fps"] = out_fp
    return doc


async def _write_ord_db(doc):
    """Write cleaned text to `jurisdiction_dbs` dir"""
    out_fp = await OrdDBFileWriter.call(doc)
    doc.attrs["ord_db_fp"] = out_fp
    return doc
