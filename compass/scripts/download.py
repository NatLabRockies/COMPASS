"""Ordinance file downloading logic"""

import logging

from elm.web.document import PDFDocument
from elm.web.search.run import web_search_links_as_docs
from elm.web.utilities import filter_documents

from compass.extraction import check_for_ordinance_info, extract_date
from compass.services.threaded import TempFileCachePB
from compass.validation.location import (
    OneShotCountyJurisdictionValidator,
    OneShotCountyNameValidator,
    JurisdictionValidator,
)
from compass.utilities.enums import LLMTasks
from compass.pb import COMPASS_PB


logger = logging.getLogger(__name__)


async def download_jurisdiction_ordinance(  # noqa: PLR0913, PLR0917
    question_templates,
    jurisdiction,
    model_configs,
    heuristic,
    ordinance_text_collector_class,
    permitted_use_text_collector_class,
    num_urls=5,
    file_loader_kwargs=None,
    browser_semaphore=None,
    url_ignore_substrings=None,
    usage_tracker=None,
):
    """Download the ordinance document(s) for a single jurisdiction

    Parameters
    ----------
    jurisdiction : :class:`~compass.utilities.location.Jurisdiction`
        Location objects representing the jurisdiction.
    model_configs : dict
        Dictionary of :class:`~compass.llm.config.LLMConfig` instances.
        Should have at minium a "default" key that is used as a fallback
        for all tasks.
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
    usage_tracker : compass.services.usage.UsageTracker, optional
        Optional tracker instance to monitor token usage during
        LLM calls. By default, ``None``.

    Returns
    -------
    list or None
        List of :obj:`~elm.web.document.BaseDocument` instances possibly
        containing ordinance information, or ``None`` if no ordinance
        document was found.
    """
    COMPASS_PB.update_jurisdiction_task(
        jurisdiction.full_name, description="Downloading files..."
    )
    async with COMPASS_PB.file_download_prog_bar(
        jurisdiction.full_name, num_urls
    ):
        docs = await _docs_from_web_search(
            question_templates,
            jurisdiction,
            num_urls,
            browser_semaphore,
            url_ignore_substrings,
            **(file_loader_kwargs or {}),
        )

    COMPASS_PB.update_jurisdiction_task(
        jurisdiction.full_name,
        description="Checking files for correct jurisdiction...",
    )
    docs = await _down_select_docs_correct_jurisdiction(
        docs,
        jurisdiction=jurisdiction,
        usage_tracker=usage_tracker,
        model_config=model_configs.get(
            LLMTasks.DOCUMENT_JURISDICTION_VALIDATION,
            model_configs[LLMTasks.DEFAULT],
        ),
    )
    logger.info(
        "%d document(s) remaining after jurisdiction filter for %s\n\t- %s",
        len(docs),
        jurisdiction.full_name,
        "\n\t- ".join(
            [doc.attrs.get("source", "Unknown source") for doc in docs]
        ),
    )
    COMPASS_PB.update_jurisdiction_task(
        jurisdiction.full_name, description="Checking files for legal text..."
    )
    docs = await _down_select_docs_correct_content(
        docs,
        jurisdiction=jurisdiction,
        model_configs=model_configs,
        heuristic=heuristic,
        ordinance_text_collector_class=ordinance_text_collector_class,
        permitted_use_text_collector_class=permitted_use_text_collector_class,
        usage_tracker=usage_tracker,
    )
    if not docs:
        logger.info(
            "Did not find any potential ordinance documents for %s",
            jurisdiction.full_name,
        )
    else:
        logger.info(
            "Found %d potential ordinance documents for %s\n\t- %s",
            len(docs),
            jurisdiction.full_name,
            "\n\t- ".join(
                [doc.attrs.get("source", "Unknown source") for doc in docs]
            ),
        )
    return _sort_final_ord_docs(docs)


async def _docs_from_web_search(
    question_templates,
    jurisdiction,
    num_urls,
    browser_semaphore,
    url_ignore_substrings,
    **file_loader_kwargs,
):
    """Download docs from web using jurisdiction queries"""
    queries = [
        question.format(jurisdiction=jurisdiction.full_name)
        for question in question_templates
    ]
    file_loader_kwargs.update({"file_cache_coroutine": TempFileCachePB.call})
    return await web_search_links_as_docs(
        queries,
        num_urls=num_urls,
        browser_semaphore=browser_semaphore,
        ignore_url_parts=url_ignore_substrings,
        task_name=jurisdiction.full_name,
        **file_loader_kwargs,
    )


async def _down_select_docs_correct_jurisdiction(
    docs, jurisdiction, usage_tracker, model_config
):
    """Remove all documents not pertaining to the jurisdiction"""
    jurisdiction_validator = JurisdictionValidator(
        text_splitter=model_config.text_splitter,
        llm_service=model_config.llm_service,
        usage_tracker=usage_tracker,
        **model_config.llm_call_kwargs,
    )
    logger.debug("Validating documents for %r", jurisdiction)
    return await filter_documents(
        docs,
        validation_coroutine=jurisdiction_validator.check,
        jurisdiction=jurisdiction,
        task_name=jurisdiction.full_name,
    )


async def _down_select_docs_correct_content(
    docs,
    jurisdiction,
    model_configs,
    heuristic,
    ordinance_text_collector_class,
    permitted_use_text_collector_class,
    usage_tracker,
):
    """Remove all documents that don't contain ordinance info"""
    return await filter_documents(
        docs,
        validation_coroutine=_contains_ordinances,
        task_name=jurisdiction.full_name,
        model_configs=model_configs,
        heuristic=heuristic,
        ordinance_text_collector_class=ordinance_text_collector_class,
        permitted_use_text_collector_class=permitted_use_text_collector_class,
        usage_tracker=usage_tracker,
    )


async def _contains_ordinances(
    doc, model_configs, usage_tracker=None, **kwargs
):
    """Helper coroutine that checks for ordinance and date info"""
    model_config = model_configs.get(
        LLMTasks.DOCUMENT_CONTENT_VALIDATION,
        model_configs[LLMTasks.DEFAULT],
    )
    doc = await check_for_ordinance_info(
        doc,
        model_config=model_config,
        usage_tracker=usage_tracker,
        **kwargs,
    )
    contains_ordinances = doc.attrs.get("contains_ord_info", False)
    if contains_ordinances:
        logger.debug("Detected ordinance info; parsing date...")
        date_model_config = model_configs.get(
            LLMTasks.DATE_EXTRACTION, model_configs[LLMTasks.DEFAULT]
        )
        doc = await extract_date(
            doc, date_model_config, usage_tracker=usage_tracker
        )
    return contains_ordinances


def _sort_final_ord_docs(all_ord_docs):
    """Sort the list of documents by year, type, and text length"""
    if not all_ord_docs:
        return None

    return sorted(all_ord_docs, key=_ord_doc_sorting_key, reverse=True)


def _ord_doc_sorting_key(doc):
    """Sorting key for documents. The higher this value, the better"""
    latest_year, latest_month, latest_day = doc.attrs.get("date", (-1, -1, -1))
    prefer_pdf_files = isinstance(doc, PDFDocument)
    highest_name_score = doc.attrs.get(
        # missing key means we were so confident that check wasn't
        # even applied, so we default to 1 here
        OneShotCountyNameValidator.META_SCORE_KEY,
        1,
    )
    highest_jurisdiction_score = doc.attrs.get(
        OneShotCountyJurisdictionValidator.META_SCORE_KEY, 0
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
