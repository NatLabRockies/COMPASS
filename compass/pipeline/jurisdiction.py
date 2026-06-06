"""Jurisdiction-scoped process workflow"""

import logging
import time
from functools import partial

from compass.services.threaded import JurisdictionUpdater
from compass.utilities.logs import LocationFileLog
from compass.pipeline.collection import DocumentCollection
from compass.pipeline import JurisdictionResult
from compass.pipeline.collection.persistence import (
    load_collected_docs,
    write_collection_manifest_shard,
)
from compass.pipeline.extraction import DocumentExtraction
from compass.pb import COMPASS_PB


logger = logging.getLogger(__name__)


class SingleJurisdictionRun:
    """Application service that orchestrates one jurisdiction run"""

    def __init__(
        self,
        runtime,
        jurisdiction,
        extractor,
        *,
        usage_tracker=None,
        known_local_docs=None,
        known_doc_urls=None,
        perform_se_search=True,
        perform_website_search=True,
        validate_user_website_input=True,
    ):
        """

        Parameters
        ----------
        runtime : compass.pipeline.runtime.PipelineRuntime
            Runtime context containing shared services, concurrency
            controls, output directories, and request settings for the
            current pipeline run.
        jurisdiction : compass.utilities.jurisdictions.Jurisdiction
            Jurisdiction to process, including identifying metadata
            such as its full name, code, and website URL.
        extractor : compass.plugin.base.BaseExtractionPlugin
            Configured extraction plugin instance responsible for
            parsing collected documents and persisting structured
            output for this jurisdiction.
        usage_tracker : UsageTracker, optional
            Optional tracker instance used to accumulate token usage
            and cost information for LLM calls made during the
            jurisdiction workflow. By default, ``None``.
        known_local_docs : list of dict, optional
            Optional local document descriptors that should be seeded
            into collection for this jurisdiction before any search or
            crawl steps are run. By default, ``None``.
        known_doc_urls : list of dict, optional
            Optional URL-based document descriptors that should be
            seeded into collection for this jurisdiction before any
            search or crawl steps are run. By default, ``None``.
        perform_se_search : bool, optional
            Whether search-engine-driven discovery should be performed
            for this jurisdiction. By default, ``True``.
        perform_website_search : bool, optional
            Whether website-specific search and crawl steps should be
            performed for this jurisdiction. By default, ``True``.
        validate_user_website_input : bool, optional
            Whether user-supplied jurisdiction website inputs should be
            validated before being used in collection. By default,
            ``True``.
        """
        self.runtime = runtime
        self.jurisdiction = jurisdiction
        self.extractor = extractor
        self.usage_tracker = usage_tracker
        self.known_local_docs = known_local_docs
        self.known_doc_urls = known_doc_urls
        self.perform_se_search = perform_se_search
        self.perform_website_search = perform_website_search
        self.validate_user_website_input = validate_user_website_input
        self.jurisdiction_website = jurisdiction.website_url
        self.last_scrape_results = []
        self.extraction_workflow = DocumentExtraction(self)
        self.collection_workflow = DocumentCollection(self)

    async def process(self):
        """Run process mode for one jurisdiction

        Returns
        -------
        compass.pipeline.data_classes.JurisdictionResult
            The result of running the jurisdiction, including any
            structured data found and related information.
        """
        start_time = time.perf_counter()
        extraction_context = None
        logger.info(
            "Kicking off processing for jurisdiction: %s (%s)",
            self.jurisdiction.full_name,
            self.jurisdiction.code,
        )
        try:
            extraction_context = await self.collection_workflow.execute(
                eager_extract=True,
            )
        finally:
            await self.extractor.record_usage()
            await _record_jurisdiction_info(
                self.jurisdiction,
                extraction_context,
                start_time,
                self.usage_tracker,
            )
            logger.info(
                "Completed extraction for jurisdiction: %s",
                self.jurisdiction.full_name,
            )

        if extraction_context is None or isinstance(
            extraction_context, Exception
        ):
            return JurisdictionResult(jurisdiction=self.jurisdiction)

        return JurisdictionResult(
            jurisdiction=self.jurisdiction,
            ord_db_fp=extraction_context.attrs.get("ord_db_fp"),
        )

    async def collect(self, *, relative_to=None):
        """Run collection mode for one jurisdiction

        Parameters
        ----------
        relative_to : path-like, optional
            Optional directory that should be the root of all relative
            paths. By default, ``None``.

        Returns
        -------
        dict
            A dictionary containing collection information, including
            the jurisdiction's full name, county, state, subdivision,
            type, FIPS code, and a list of collected documents with
            their associated metadata.
        """
        logger.info(
            "Kicking off collection for jurisdiction: %s",
            self.jurisdiction.full_name,
        )
        collection_info = await self.collection_workflow.execute(
            eager_extract=False, relative_to=relative_to
        )

        await _safe_shard_write(
            self.runtime.dirs.jurisdiction_dbs,
            collection_info,
            self.jurisdiction.full_name,
        )

        logger.info(
            "Completed collection for jurisdiction: %s",
            self.jurisdiction.full_name,
        )
        return collection_info

    async def extract_from_collection_info(self, collection_info):
        """Run extraction mode for one jurisdiction

        Parameters
        ----------
        collection_info : dict
            Dictionary containing information about the collected
            documents for the jurisdiction, including the jurisdiction's
            full name, county, state, subdivision, type, FIPS code, and
            a list of collected documents with their associated
            metadata.

        Returns
        -------
        compass.pipeline.data_classes.JurisdictionResult
            The result of running the jurisdiction, including any
            structured data found and related information.
        """
        start_time = time.perf_counter()
        extraction_context = None
        logger.info(
            "Kicking off extraction for jurisdiction: %s",
            self.jurisdiction.full_name,
        )
        self.jurisdiction_website = collection_info.get("jurisdiction_website")
        try:
            docs = await load_collected_docs(
                collection_info, task_name=self.jurisdiction.full_name
            )
            docs = [doc for doc in docs if doc is not None]
            extraction_context = (
                await self.extraction_workflow.extract_from_docs(docs)
            )
        finally:
            await self.extractor.record_usage()
            await _record_jurisdiction_info(
                self.jurisdiction,
                extraction_context,
                start_time,
                self.usage_tracker,
            )
            logger.info(
                "Completed extraction for jurisdiction: %s",
                self.jurisdiction.full_name,
            )

        if extraction_context is None or isinstance(
            extraction_context, Exception
        ):
            return JurisdictionResult(jurisdiction=self.jurisdiction)

        return JurisdictionResult(
            jurisdiction=self.jurisdiction,
            ord_db_fp=extraction_context.attrs.get("ord_db_fp"),
        )

    async def _run_with_logging_context(
        self, runner, *, error_action, fallback
    ):
        """Run one jurisdiction action under the shared outer context"""
        async with self.runtime.jurisdiction_semaphore:
            with COMPASS_PB.jurisdiction_prog_bar(self.jurisdiction.full_name):
                async with LocationFileLog(
                    self.runtime.log_listener,
                    self.runtime.dirs.logs,
                    location=self.jurisdiction.full_name,
                    level=self.runtime.log_level,
                ):
                    try:
                        return await runner()
                    except KeyboardInterrupt:
                        raise
                    except Exception as error:
                        msg = "Encountered error of type %r while %s %s:"
                        err_type = type(error)
                        logger.exception(
                            msg,
                            err_type,
                            error_action,
                            self.jurisdiction.full_name,
                        )
                        return fallback

    async def run_process_with_logging(self):
        """Run one jurisdiction under location-scoped logging

        Returns
        -------
        compass.pipeline.data_classes.JurisdictionResult
            The result of running the jurisdiction, including any
            structured data found and related information.
        """
        return await self._run_with_logging_context(
            self.process,
            error_action="processing",
            fallback=JurisdictionResult(jurisdiction=self.jurisdiction),
        )

    async def run_collection_with_logging(self, *, relative_to=None):
        """Collect one jurisdiction under location-scoped logging

        Parameters
        ----------
        relative_to : path-like, optional
            Optional directory that should be the root of all relative
            paths. By default, ``None``.

        Returns
        -------
        dict
            A dictionary containing collection information, including
            the jurisdiction's full name, county, state, subdivision,
            type, FIPS code, and a list of collected documents with
            their associated metadata.
        """
        return await self._run_with_logging_context(
            partial(self.collect, relative_to=relative_to),
            error_action="collecting",
            fallback=None,
        )

    async def run_extraction_with_logging(self, collection_info):
        """Extract one jurisdiction under location-scoped logging

        Parameters
        ----------
        collection_info : dict
            Dictionary containing information about the collected
            documents for the jurisdiction, including the jurisdiction's
            full name, county, state, subdivision, type, FIPS code, and
            a list of collected documents with their associated
            metadata.

        Returns
        -------
        compass.pipeline.data_classes.JurisdictionResult
            The result of running the jurisdiction, including any
            structured data found and related information.
        """
        return await self._run_with_logging_context(
            partial(self.extract_from_collection_info, collection_info),
            error_action="extracting",
            fallback=JurisdictionResult(jurisdiction=self.jurisdiction),
        )


async def _record_jurisdiction_info(
    jurisdiction, extraction_context, start_time, usage_tracker
):
    """Record final jurisdiction info"""

    seconds_elapsed = time.perf_counter() - start_time
    await JurisdictionUpdater.call(
        jurisdiction, extraction_context, seconds_elapsed, usage_tracker
    )


async def _safe_shard_write(shard_dir, collection_info, jur_name):
    """Safely write a collection manifest shard"""
    try:
        shard_fp = await write_collection_manifest_shard(
            shard_dir, collection_info
        )
        logger.info(
            "Collection manifest shard for %s stored here: '%s'",
            jur_name,
            shard_fp,
        )
    except Exception:
        logger.exception(
            "Failed to write collection manifest shard for %s", jur_name
        )
