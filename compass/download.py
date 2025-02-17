"""Ordinance county file downloading logic"""

import logging

from elm.web.document import PDFDocument
from elm.web.search.run import web_search_links_as_docs
from elm.web.utilities import filter_documents

from compass.llm import StructuredLLMCaller
from compass.extraction import check_for_ordinance_info
from compass.services.threaded import TempFileCache
from compass.validation.location import (
    CountyJurisdictionValidator,
    CountyNameValidator,
    CountyValidator,
)


logger = logging.getLogger(__name__)


async def download_county_ordinance(
    question_templates,
    location,
    text_splitter,
    validator_class,
    num_urls=5,
    file_loader_kwargs=None,
    browser_semaphore=None,
    **kwargs,
):
    """Download the ordinance document(s) for a single county

    Parameters
    ----------
    location : :class:`compass.utilities.location.Location`
        Location objects representing the county.
    text_splitter : obj, optional
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Raw text from HTML pages will be passed through
        this splitter to split the single wep page into multiple pages
        for the output document. Langchain's text splitters should work
        for this input.
    num_urls : int, optional
        Number of unique Google search result URL's to check for
        ordinance document. By default, ``5``.
    file_loader_kwargs : dict, optional
        Dictionary of keyword-argument pairs to initialize
        :class:`elm.web.file_loader.AsyncFileLoader` with. If found, the
        "pw_launch_kwargs" key in these will also be used to initialize
        the :class:`elm.web.search.google.PlaywrightGoogleLinkSearch`
        used for the google URL search. By default, ``None``.
    browser_semaphore : :class:`asyncio.Semaphore`, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers open concurrently. If ``None``, no limits
        are applied. By default, ``None``.
    **kwargs
        Keyword-value pairs used to initialize an
        `compass.llm.LLMCaller` instance.

    Returns
    -------
    elm.web.document.BaseDocument | None
        Document instance for the downloaded document, or ``None`` if no
        document was found.
    """
    docs = await _docs_from_web_search(
        question_templates,
        location,
        text_splitter,
        num_urls,
        browser_semaphore,
        **(file_loader_kwargs or {}),
    )
    docs = await _down_select_docs_correct_location(
        docs, location=location, **kwargs
    )
    docs = await _down_select_docs_correct_content(
        docs,
        location=location,
        text_splitter=text_splitter,
        validator_class=validator_class,
        **kwargs,
    )
    logger.info(
        "Found %d potential ordinance documents for %s",
        len(docs),
        location.full_name,
    )
    return _sort_final_ord_docs(docs)


async def _docs_from_web_search(
    question_templates,
    location,
    text_splitter,
    num_urls,
    browser_semaphore,
    **file_loader_kwargs,
):
    """Download docs from web using location queries"""
    queries = [
        question.format(location=location.full_name)
        for question in question_templates
    ]
    file_loader_kwargs.update(
        {
            "html_read_kwargs": {"text_splitter": text_splitter},
            "file_cache_coroutine": TempFileCache.call,
        }
    )
    return await web_search_links_as_docs(
        queries,
        num_urls=num_urls,
        browser_semaphore=browser_semaphore,
        task_name=location.full_name,
        **file_loader_kwargs,
    )


async def _down_select_docs_correct_location(docs, location, **kwargs):
    """Remove all documents not pertaining to the location"""
    llm_caller = StructuredLLMCaller(**kwargs)
    county_validator = CountyValidator(llm_caller)
    return await filter_documents(
        docs,
        validation_coroutine=county_validator.check,
        task_name=location.full_name,
        county=location.name,
        state=location.state,
    )


async def _down_select_docs_correct_content(docs, location, **kwargs):
    """Remove all documents that don't contain ordinance info"""
    return await filter_documents(
        docs,
        validation_coroutine=_contains_ords,
        task_name=location.full_name,
        **kwargs,
    )


async def _contains_ords(doc, **kwargs):
    """Helper coroutine that checks for ordinance info"""
    doc = await check_for_ordinance_info(doc, **kwargs)
    return doc.metadata.get("contains_ord_info", False)


def _sort_final_ord_docs(all_ord_docs):
    """Sort the list of documents by year, type, and text length"""
    if not all_ord_docs:
        return None

    return max(all_ord_docs, key=_ord_doc_sorting_key)


def _ord_doc_sorting_key(doc):
    """Sorting key for documents. The higher this value, the better"""
    latest_year, latest_month, latest_day = doc.metadata.get(
        "date", (-1, -1, -1)
    )
    prefer_pdf_files = isinstance(doc, PDFDocument)
    highest_name_score = doc.metadata.get(
        CountyNameValidator.META_SCORE_KEY, 0
    )
    highest_jurisdiction_score = doc.metadata.get(
        CountyJurisdictionValidator.META_SCORE_KEY, 0
    )
    shortest_text_length = -1 * len(doc.text)
    return (
        latest_year,
        prefer_pdf_files,
        highest_name_score,
        highest_jurisdiction_score,
        shortest_text_length,
        latest_month,
        latest_day,
    )
