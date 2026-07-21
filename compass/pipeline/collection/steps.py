"""Fixed collection steps for the process pipeline"""

import logging
from abc import ABC, abstractmethod

from elm.web.utilities import get_redirected_url

from compass.scripts.download import (
    download_jurisdiction_ordinance_using_search_engine,
    download_jurisdiction_ordinances_from_website,
    download_jurisdiction_ordinances_from_website_compass_crawl,
    download_known_urls,
    find_jurisdiction_website,
    load_known_docs,
)
from compass.utilities.enums import COMPASSDocumentCollectionStep
from compass.utilities.url import base_website_url
from compass.pb import COMPASS_PB


logger = logging.getLogger(__name__)


class CollectionStep(ABC):
    """Strategy base class for fixed collection steps"""

    @property
    @abstractmethod
    def STEP_NAME(self):  # noqa: N802
        """Identifier for step (e.g. "known_local_docs")"""
        raise NotImplementedError

    @abstractmethod
    async def collect(self, workflow):
        """Collect documents for one step"""
        raise NotImplementedError


class KnownLocalDocumentsStep(CollectionStep):
    """Concrete Strategy for known local document collection"""

    STEP_NAME = COMPASSDocumentCollectionStep.KNOWN_LOCAL_DOCS
    """Identifier for step"""

    async def collect(self, workflow):  # noqa: PLR6301
        """Collect known local documents for this jurisdiction

        Parameters
        ----------
        workflow : compass.pipeline.jurisdiction.SingleJurisdictionRun
            The workflow for the jurisdiction being processed, which may
            or may not have user-supplied known local documents
            configured. If the workflow doesn't have any user-supplied
            known local documents, this function will return an empty
            list.

        Returns
        -------
        list
            List of documents loaded from the user-supplied known local
            document file paths, with user-supplied metadata attrs
            added.
        """
        if not workflow.known_local_docs:
            logger.debug(
                "%r processing had no known local docs configured",
                workflow.jurisdiction.full_name,
            )
            return []

        logger.debug(
            "Checking local docs for jurisdiction: %s",
            workflow.jurisdiction.full_name,
        )
        try:
            docs = await load_known_docs(
                workflow.jurisdiction,
                [info["source_fp"] for info in workflow.known_local_docs],
                local_file_loader_kwargs=(
                    workflow.runtime.local_file_loader_kwargs
                ),
            )
        except Exception:
            logger.exception(
                "Error loading known local documents for %s",
                workflow.jurisdiction.full_name,
            )
            return []

        return _add_known_doc_attrs_to_all_docs(
            docs, workflow.known_local_docs, key="source_fp"
        )


class KnownUrlDocumentsStep(CollectionStep):
    """Concrete Strategy for known URL document collection"""

    STEP_NAME = COMPASSDocumentCollectionStep.KNOWN_DOC_URLS
    """Identifier for step"""

    async def collect(self, workflow):  # noqa: PLR6301
        """Collect documents from known URL's for this jurisdiction

        Parameters
        ----------
        workflow : compass.pipeline.jurisdiction.SingleJurisdictionRun
            The workflow for the jurisdiction being processed, which may
            or may not have user-supplied known document URLs
            configured. If the workflow doesn't have any user-supplied
            known document URLs, this function will return an empty
            list.

        Returns
        -------
        list
            List of documents loaded from the user-supplied known URLs,
            with user-supplied metadata attrs added.
        """
        if not workflow.known_doc_urls:
            logger.debug(
                "%r processing had no known URLs configured",
                workflow.jurisdiction.full_name,
            )
            return []

        logger.debug(
            "Checking known URLs for jurisdiction: %s",
            workflow.jurisdiction.full_name,
        )
        try:
            docs = await download_known_urls(
                workflow.jurisdiction,
                [info["source"] for info in workflow.known_doc_urls],
                browser_semaphore=workflow.runtime.browser_semaphore,
                file_loader_kwargs=workflow.runtime.file_loader_kwargs,
            )
        except Exception:
            logger.exception(
                "Error loading known urls for %s",
                workflow.jurisdiction.full_name,
            )
            return []

        return _add_known_doc_attrs_to_all_docs(
            docs, workflow.known_doc_urls, key="source"
        )


class SearchEngineDocumentsStep(CollectionStep):
    """Concrete Strategy for search-engine document collection"""

    STEP_NAME = COMPASSDocumentCollectionStep.SEARCH_ENGINE
    """Identifier for step"""

    async def collect(self, workflow):  # noqa: PLR6301
        """Collect documents based on a search engine search

        Parameters
        ----------
        workflow : compass.pipeline.jurisdiction.SingleJurisdictionRun
            The workflow for the jurisdiction being processed, which may
            or may not have search engine document collection enabled.
            If search engine document collection is not enabled, this
            function will return an empty list.

        Returns
        -------
        list
            List of documents collected from search engine results, with
            jurisdiction verification enabled based on the workflow's
            configuration.
        """
        if not workflow.perform_se_search:
            logger.debug(
                "%r processing didn't have SE search enabled",
                workflow.jurisdiction.full_name,
            )
            return []

        logger.debug(
            "Collecting documents using a search engine for jurisdiction: %s",
            workflow.jurisdiction.full_name,
        )
        try:
            query_templates = await workflow.extractor.get_query_templates()
            runtime = workflow.runtime
            docs = await download_jurisdiction_ordinance_using_search_engine(
                query_templates,
                workflow.jurisdiction,
                num_urls=(
                    runtime.search_params.num_urls_to_check_per_jurisdiction
                ),
                simple_se_result_sort=(
                    runtime.search_params.simple_se_result_sort
                ),
                file_loader_kwargs=runtime.file_loader_kwargs,
                search_semaphore=runtime.search_engine_semaphore,
                browser_semaphore=runtime.browser_semaphore,
                url_ignore_substrings=(
                    runtime.search_params.url_ignore_substrings
                ),
                url_keep_substrings=(
                    runtime.search_params.url_keep_substrings
                ),
                **runtime.search_params.se_kwargs,
            )
        except Exception:
            logger.exception(
                "Error collecting documents using a search engine for %s",
                workflow.jurisdiction.full_name,
            )
            return []

        for doc in docs:
            doc.attrs["compass_crawl"] = False
            doc.attrs["check_correct_jurisdiction"] = True
        return docs


class ElmWebsiteCrawlStep(CollectionStep):
    """Concrete Strategy for ELM-based website crawling"""

    STEP_NAME = COMPASSDocumentCollectionStep.WEBSITE_SEARCH_ELM
    """Identifier for step"""

    async def collect(self, workflow):  # noqa: PLR6301
        """Collect documents based on an ELM website crawl

        Parameters
        ----------
        workflow : compass.pipeline.jurisdiction.SingleJurisdictionRun
            The workflow for the jurisdiction being processed, which may
            or may not have website search enabled. If website search is
            not enabled, this function will return an empty list. If
            website search is enabled but no jurisdiction website can be
            found or validated, this function will also return an empty
            list, but will first attempt to find and validate a
            jurisdiction website based on the workflow's configuration
            before giving up on website document collection for this
            jurisdiction. If website search is enabled and a
            jurisdiction website is found and validated (either from
            user input or through automatic search), this function will
            attempt to crawl the jurisdiction website for documents
            using ELM, and return a list of documents collected from the
            crawl. If any errors are encountered during the crawl, this
            function will log the error and return an empty list.

        Returns
        -------
        list
            List of documents collected from crawling the jurisdiction
            website using ELM, with jurisdiction verification enabled
            based on the workflow's configuration. If website search is
            not enabled or if no jurisdiction website can be found or
            validated, this will return an empty list.
        """

        if not workflow.perform_website_search:
            return []

        await _resolve_jurisdiction_website(workflow)
        if not workflow.jurisdiction_website:
            logger.debug(
                "No jurisdiction website found for %r; skipping "
                "ELM website document collection",
                workflow.jurisdiction.full_name,
            )
            return []

        logger.debug(
            "Collecting documents using ELM web crawl for: %s",
            workflow.jurisdiction.full_name,
        )
        try:
            out = await download_jurisdiction_ordinances_from_website(
                workflow.jurisdiction_website,
                heuristic=await workflow.extractor.get_heuristic(),
                keyword_points=(
                    await workflow.extractor.get_website_keywords()
                ),
                file_loader_kwargs=workflow.runtime.file_loader_kwargs,
                crawl_semaphore=workflow.runtime.crawl_semaphore,
                pb_jurisdiction_name=workflow.jurisdiction.full_name,
                return_c4ai_results=True,
            )
        except Exception:
            logger.exception(
                "Error collecting documents using ELM web crawl for %s",
                workflow.jurisdiction.full_name,
            )
            workflow.last_scrape_results = []
            return []

        docs, scrape_results = out
        logger.debug("Found the following docs with ELM crawl:\n%r", docs)

        workflow.last_scrape_results = scrape_results
        for doc in docs:
            doc.attrs["compass_crawl"] = False
            doc.attrs["check_correct_jurisdiction"] = True
        return docs


class CompassWebsiteCrawlStep(CollectionStep):
    """Concrete Strategy for COMPASS-based website crawling"""

    STEP_NAME = COMPASSDocumentCollectionStep.WEBSITE_SEARCH_COMPASS
    """Identifier for step"""

    async def collect(self, workflow):  # noqa: PLR6301
        """Collect documents based on a COMPASS website crawl

        Parameters
        ----------
        workflow : compass.pipeline.jurisdiction.SingleJurisdictionRun
            The workflow for the jurisdiction being processed, which may
            or may not have website search enabled. If website search is
            not enabled, this function will return an empty list. If
            website search is enabled but no jurisdiction website can be
            found or validated, this function will also return an empty
            list, but will first attempt to find and validate a
            jurisdiction website based on the workflow's configuration
            before giving up on website document collection for this
            jurisdiction. If website search is enabled and a
            jurisdiction website is found and validated (either from
            user input or through automatic search), this function will
            attempt to crawl the jurisdiction website for documents
            using COMPASS, and return a list of documents collected from
            the crawl. If any errors are encountered during the crawl,
            this function will log the error and return an empty list.

        Returns
        -------
        list
            List of documents collected from crawling the jurisdiction
            website using COMPASS, with jurisdiction verification
            enabled based on the workflow's configuration. If website
            search is not enabled or if no jurisdiction website can be
            found or validated, this will return an empty list.
        """
        if not workflow.perform_website_search:
            return []

        await _resolve_jurisdiction_website(workflow)
        if not workflow.jurisdiction_website:
            logger.debug(
                "No jurisdiction website found for %r; skipping "
                "COMPASS website document collection",
                workflow.jurisdiction.full_name,
            )
            return []

        logger.debug(
            "Collecting documents using COMPASS web crawl for: %s",
            workflow.jurisdiction.full_name,
        )
        checked_urls = set()
        for scrape_result in workflow.last_scrape_results:
            checked_urls.update({sub_res.url for sub_res in scrape_result})

        func = download_jurisdiction_ordinances_from_website_compass_crawl
        try:
            docs = await func(
                workflow.jurisdiction_website,
                heuristic=await workflow.extractor.get_heuristic(),
                keyword_points=await workflow.extractor.get_website_keywords(),
                file_loader_kwargs=workflow.runtime.file_loader_kwargs,
                already_visited=checked_urls,
                crawl_semaphore=workflow.runtime.crawl_semaphore,
                pb_jurisdiction_name=workflow.jurisdiction.full_name,
            )
        except Exception:
            logger.exception(
                "Error collecting documents using COMPASS web crawl for %s",
                workflow.jurisdiction.full_name,
            )
            return []

        for doc in docs:
            doc.attrs["compass_crawl"] = True
            doc.attrs["check_correct_jurisdiction"] = True
        return docs


async def _resolve_jurisdiction_website(workflow):
    """Try to set and resolve the website URL for this jurisdiction"""
    if workflow.jurisdiction_website:
        workflow.jurisdiction_website = await _get_base_website(
            workflow.jurisdiction_website
        )

    # only try to find a website if we don't have one and we have LLMs
    # we can use for validation
    if workflow.jurisdiction_website or not workflow.runtime.models:
        return

    if website := await _find_jurisdiction_website_for_workflow(workflow):
        workflow.jurisdiction_website = await _get_base_website(website)


async def _get_base_website(website):
    """Get the base URL for a website, following redirects if needed"""
    try:
        website = await get_redirected_url(website, timeout=30)
    except Exception:
        logger.exception("Error getting redirected URL for %s", website)
        return None
    return base_website_url(website)


async def _find_jurisdiction_website_for_workflow(workflow):
    """Search for the jurisdiction website"""
    COMPASS_PB.update_jurisdiction_task(
        workflow.jurisdiction.full_name,
        description="Searching for jurisdiction website...",
    )
    return await find_jurisdiction_website(
        workflow.jurisdiction,
        workflow.runtime.models,
        file_loader_kwargs=workflow.runtime.file_loader_kwargs_no_ocr,
        search_semaphore=workflow.runtime.search_engine_semaphore,
        browser_semaphore=workflow.runtime.browser_semaphore,
        usage_tracker=workflow.usage_tracker,
        url_ignore_substrings=(
            workflow.runtime.search_params.url_ignore_substrings
        ),
        **workflow.runtime.search_params.se_kwargs,
    )


def _add_known_doc_attrs_to_all_docs(docs, doc_infos, key):
    """Add user-defined document attrs to loaded docs"""
    for doc in docs:
        doc.attrs["check_correct_jurisdiction"] = False
        source_fp = doc.attrs.get(key)
        if not source_fp:
            continue
        _add_known_doc_attrs(doc, source_fp, doc_infos, key)
    return docs


def _add_known_doc_attrs(doc, source_fp, doc_infos, key):
    """Add user-defined attrs to one document"""
    for info in doc_infos:
        if str(info[key]) == str(source_fp):
            doc.attrs.update(info)
            return
