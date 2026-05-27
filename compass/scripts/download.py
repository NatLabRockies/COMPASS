"""Ordinance file downloading logic"""

import pprint
import logging
from contextlib import AsyncExitStack

from elm.web.document import PDFDocument
from elm.web.search.run import load_docs, search_with_fallback
from elm.web.website_crawl import (
    _SCORE_KEY,  # noqa: PLC2701
    ELMWebsiteCrawler,
    ELMLinkScorer,
)
from elm.web.utilities import filter_documents

from compass.extraction import check_for_relevant_text, extract_date
from compass.services.threaded import TempFileCache, TempFileCachePB
from compass.validation.location import (
    DTreeJurisdictionValidator,
    JurisdictionValidator,
    JurisdictionWebsiteValidator,
)
from compass.web.file_loader import (
    COMPASSWebFileLoader,
    COMPASSLocalFileLoader,
)
from compass.web.website_crawl import COMPASSCrawler, COMPASSLinkScorer
from compass.web.url_utils import sanitize_url
from compass.utilities.enums import LLMTasks
from compass.pb import COMPASS_PB


logger = logging.getLogger(__name__)
_NEG_INF = -1 * float("infinity")


async def download_known_urls(
    jurisdiction, urls, browser_semaphore=None, file_loader_kwargs=None
):
    """Download documents from known URLs

    Parameters
    ----------
    jurisdiction : Jurisdiction
        Jurisdiction instance representing the jurisdiction
        corresponding to the documents.
    urls : iterable of str
        Collection of URLs to download documents from.
    browser_semaphore : :class:`asyncio.Semaphore`, optional
        Semaphore instance that can be used to limit the number of
        downloads happening concurrently. If ``None``, no limits
        are applied. By default, ``None``.
    file_loader_kwargs : dict, optional
        Dictionary of keyword arguments pairs to initialize
        :class:`elm.web.file_loader.AsyncWebFileLoader`.
        By default, ``None``.

    Returns
    -------
    out_docs : list
        List of BaseDocument instances containing documents from the
        URL's, or an empty list if something went wrong during the
        retrieval process.

    Notes
    -----
    Requires :class:`~compass.services.threaded.TempFileCachePB`
    service to be running.
    """

    COMPASS_PB.update_jurisdiction_task(
        jurisdiction.full_name,
        description="Downloading known URL(s)...",
    )

    file_loader_kwargs = file_loader_kwargs or {}
    file_loader_kwargs.update({"file_cache_coroutine": TempFileCachePB.call})
    logger.trace(
        "kwargs for COMPASSWebFileLoader:\n%s",
        pprint.PrettyPrinter().pformat(file_loader_kwargs),
    )
    file_loader = COMPASSWebFileLoader(
        browser_semaphore=browser_semaphore, **file_loader_kwargs
    )

    async with COMPASS_PB.file_download_prog_bar(
        jurisdiction.full_name, len(urls)
    ):
        try:
            out_docs = await load_docs(urls, file_loader)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            msg = (
                "Encountered error of type %r while downloading known URLs: %r"
            )
            err_type = type(e)
            logger.exception(msg, err_type, urls)
            out_docs = []

    return out_docs


async def load_known_docs(jurisdiction, fps, local_file_loader_kwargs=None):
    """Load documents from known local paths

    Parameters
    ----------
    jurisdiction : Jurisdiction
        Jurisdiction instance representing the jurisdiction
        corresponding to the documents.
    fps : iterable of path-like
        Collection of paths to load documents from.
    local_file_loader_kwargs : dict, optional
        Dictionary of keyword arguments pairs to initialize
        :class:`~elm.web.file_loader.AsyncLocalFileLoader` (for "elm"
        file loader backend) or
        :class:`~compass.web.file_loader.AsyncLocalDoclingFileLoader`
        (for "docling" file loader backend). By default, ``None``.

    Returns
    -------
    out_docs : list
        List of BaseDocument instances containing documents from the
        paths, or an empty list if something went wrong during the
        retrieval process.

    Notes
    -----
    Requires :class:`~compass.services.threaded.TempFileCachePB`
    service to be running.
    """

    COMPASS_PB.update_jurisdiction_task(
        jurisdiction.full_name, description="Loading known document(s)..."
    )

    local_file_loader_kwargs = local_file_loader_kwargs or {}
    local_file_loader_kwargs.update(
        {"file_cache_coroutine": TempFileCachePB.call}
    )
    logger.trace(
        "kwargs for COMPASSLocalFileLoader:\n%s",
        pprint.PrettyPrinter().pformat(local_file_loader_kwargs),
    )
    fl = COMPASSLocalFileLoader(**local_file_loader_kwargs)
    async with COMPASS_PB.file_download_prog_bar(
        jurisdiction.full_name, len(fps)
    ):
        try:
            out_docs = await load_docs(fps, fl)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            msg = (
                "Encountered error of type %r while loading known documents: "
                "%r"
            )
            err_type = type(e)
            logger.exception(msg, err_type, fps)
            out_docs = []

    return out_docs


async def find_jurisdiction_website(
    jurisdiction,
    model_configs,
    file_loader_kwargs=None,
    search_semaphore=None,
    browser_semaphore=None,
    usage_tracker=None,
    url_ignore_substrings=None,
    **kwargs,
):
    """Search for the main landing page of a given jurisdiction

    This function submits two pre-determined queries based on the
    jurisdiction name, prioritizing official landing pages. Additional
    ``kwargs`` (for example, alternate search engines) can be supplied
    to fine-tune behavior.

    Parameters
    ----------
    jurisdiction : Jurisdiction
        Jurisdiction instance representing the jurisdiction to find the
        main webpage for.
    model_configs : dict
        Dictionary of :class:`~compass.llm.config.LLMConfig` instances.
        Should have at minium a "default" key that is used as a fallback
        for all tasks.
    file_loader_kwargs : dict, optional
        Dictionary of keyword arguments pairs to initialize
        :class:`elm.web.file_loader.AsyncWebFileLoader`. If found, the
        "pw_launch_kwargs" key in these will also be used to initialize
        the :class:`elm.web.search.google.PlaywrightGoogleLinkSearch`
        used for the Google URL search. By default, ``None``.
    search_semaphore : :class:`asyncio.Semaphore`, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers used to submit search engine queries open
        concurrently.  If ``None``, no limits are applied.
        By default, ``None``.
    browser_semaphore : :class:`asyncio.Semaphore`, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers open concurrently. If ``None``, no limits
        are applied. By default, ``None``.
    usage_tracker : UsageTracker, optional
        Optional tracker instance to monitor token usage during
        LLM calls. By default, ``None``.
    url_ignore_substrings : list of str, optional
        URL substrings that should be excluded from search results.
        Substrings are applied case-insensitively. By default, ``None``.
    **kwargs
        Additional arguments forwarded to
        :func:`elm.web.search.run.search_with_fallback`.

    Returns
    -------
    str or None
        URL for the jurisdiction website, if found; ``None`` otherwise.
    """
    kwargs.update(file_loader_kwargs or {})

    name = jurisdiction.full_name_the_prefixed
    name_no_the = name.removeprefix("the ")
    query_1 = f"{name_no_the} website".casefold().replace(",", "")
    query_2 = f"main website {name}".casefold().replace(",", "")

    potential_website_links = await search_with_fallback(
        queries=[query_1, query_2],
        num_urls=3,
        ignore_url_parts=url_ignore_substrings,
        browser_semaphore=search_semaphore,
        task_name=jurisdiction.full_name,
        **kwargs,
    )

    if not potential_website_links:
        return None

    model_config = model_configs.get(
        LLMTasks.JURISDICTION_MAIN_WEBSITE_VALIDATION,
        model_configs[LLMTasks.DEFAULT],
    )

    validator = JurisdictionWebsiteValidator(
        browser_semaphore=browser_semaphore,
        file_loader_kwargs=file_loader_kwargs,
        usage_tracker=usage_tracker,
        llm_service=model_config.llm_service,
        **model_config.llm_call_kwargs,
    )

    for url in potential_website_links:
        if await validator.check(url, jurisdiction):
            return url

    return None


async def download_jurisdiction_ordinances_from_website(
    website,
    heuristic,
    keyword_points,
    file_loader_kwargs=None,
    browser_config_kwargs=None,
    crawler_config_kwargs=None,
    max_urls=100,
    crawl_semaphore=None,
    pb_jurisdiction_name=None,
    return_c4ai_results=False,
):
    """Download ordinance documents from a jurisdiction website

    Parameters
    ----------
    website : str
        URL of the jurisdiction website to search.
    heuristic : callable
        Callable taking an BaseDocument and returning ``True`` when the
        document should be kept.
    keyword_points : dict
        Dictionary of keyword points to use for scoring links.
        Keys are keywords, values are points to assign to links
        containing the keyword. If a link contains multiple keywords,
        the points are summed up.
    file_loader_kwargs : dict, optional
        Dictionary of keyword arguments pairs to initialize
        :class:`elm.web.file_loader.AsyncWebFileLoader`. If found, the
        "pw_launch_kwargs" key in these will also be used to initialize
        the :class:`elm.web.search.google.PlaywrightGoogleLinkSearch`
        used for the Google URL search. By default, ``None``.
    browser_config_kwargs : dict, optional
        Dictionary of keyword arguments pairs to initialize the
        ``crawl4ai.async_configs.BrowserConfig`` class used for the
        web crawl. By default, ``None``.
    crawler_config_kwargs : dict, optional
        Dictionary of keyword arguments pairs to initialize the
        ``crawl4ai.async_configs.CrawlerConfig`` class used for the
        web crawl. By default, ``None``.
    max_urls : int, optional
        Max number of URLs to check from the website before terminating
        the search. By default, ``100``.
    crawl_semaphore : :class:`asyncio.Semaphore`, optional
        Semaphore instance that can be used to limit the number of
        website searches happening concurrently. If ``None``, no limits
        are applied. By default, ``None``.
    pb_jurisdiction_name : str, optional
        Optional jurisdiction name to use to update progress bar, if
        it's being used. By default, ``None``.
    return_c4ai_results : bool, default=False
        If ``True``, the crawl4ai results will be returned as a second
        return value. This is useful for debugging and examining the
        crawled URLs. If ``False``, only the documents will be returned.
        By default, ``False``.

    Returns
    -------
    out_docs : list
        List of BaseDocument instances containing potential ordinance
        information, or an empty list if no ordinance document was
        found.
    results : list, optional
        List of crawl4ai results containing metadata about the crawled
        pages. Only returned when ``return_c4ai_results`` evaluates to
        ``True``.

    Notes
    -----
    Requires :class:`~compass.services.threaded.TempFileCache` service
    to be running.
    """

    if crawl_semaphore is None:
        crawl_semaphore = AsyncExitStack()

    async def _doc_heuristic(doc):  # noqa: RUF029
        """Heuristic check for wind ordinance documents"""
        is_valid_document = heuristic.check(doc.text.lower())
        if is_valid_document and pb_jurisdiction_name:
            COMPASS_PB.update_website_crawl_doc_found(pb_jurisdiction_name)

        return is_valid_document

    async def _crawl_hook(*__, **___):  # noqa: RUF029
        """Update progress bar as pages are searched"""
        COMPASS_PB.update_website_crawl_task(pb_jurisdiction_name, advance=1)

    flk = {"verify_ssl": False}
    flk.update(file_loader_kwargs or {})
    flk.update({"file_cache_coroutine": TempFileCache.call})

    browser_config_kwargs = browser_config_kwargs or {}
    pw_launch_kwargs = flk.get("pw_launch_kwargs", {})
    browser_config_kwargs["headless"] = pw_launch_kwargs.get("headless", True)

    logger.trace(
        "kwargs for COMPASSWebFileLoader:\n%s",
        pprint.PrettyPrinter().pformat(flk),
    )
    afl = COMPASSWebFileLoader(**flk)
    crawler = ELMWebsiteCrawler(
        validator=_doc_heuristic,
        async_file_loader=afl,
        url_scorer=ELMLinkScorer(keyword_points).score,
        browser_config_kwargs=browser_config_kwargs,
        crawler_config_kwargs=crawler_config_kwargs,
        include_external=True,
        max_pages=max_urls,
        page_limit=int(max_urls * 3),
    )

    if pb_jurisdiction_name:
        COMPASS_PB.update_jurisdiction_task(
            pb_jurisdiction_name,
            description=f"Searching for documents from {website} ...",
        )
        cpb = COMPASS_PB.website_crawl_prog_bar(pb_jurisdiction_name, max_urls)
        ch = _crawl_hook
    else:
        cpb = AsyncExitStack()
        ch = None

    async with crawl_semaphore, cpb:
        docs_or_pair = await crawler.run(
            website,
            on_result_hook=ch,
            return_c4ai_results=return_c4ai_results,
        )

    if return_c4ai_results:
        docs, c4ai_results = docs_or_pair
        _sanitize_doc_sources(docs)
        return docs, c4ai_results

    _sanitize_doc_sources(docs_or_pair)
    return docs_or_pair


async def download_jurisdiction_ordinances_from_website_compass_crawl(
    website,
    heuristic,
    keyword_points,
    file_loader_kwargs=None,
    already_visited=None,
    num_link_scores_to_check_per_page=4,
    max_urls=100,
    crawl_semaphore=None,
    pb_jurisdiction_name=None,
):
    """Download ord documents from a website using the COMPASS crawler

    The COMPASS crawler is much more simplistic than the Crawl4AI
    crawler, but is designed to access some links that Crawl4AI cannot
    (such as those behind a button interface).

    Parameters
    ----------
    website : str
        URL of the jurisdiction website to search.
    heuristic : callable
        Callable taking an BaseDocument and returning ``True`` when the
        document should be kept.
    keyword_points : dict
        Dictionary of keyword points to use for scoring links.
        Keys are keywords, values are points to assign to links
        containing the keyword. If a link contains multiple keywords,
        the points are summed up.
    file_loader_kwargs : dict, optional
        Dictionary of keyword arguments pairs to initialize
        :class:`elm.web.file_loader.AsyncWebFileLoader`. If found, the
        "pw_launch_kwargs" key in these will also be used to initialize
        the :class:`elm.web.search.google.PlaywrightGoogleLinkSearch`
        used for the Google URL search. By default, ``None``.
    already_visited : set of str, optional
        URLs that have already been crawled and should be skipped.
        By default, ``None``.
    num_link_scores_to_check_per_page : int, default=4
        Number of top-scoring links to visit per page.
        By default, ``4``.
    max_urls : int, default=100
        Max number of URLs to check from the website before terminating
        the search. By default, ``100``.
    crawl_semaphore : :class:`asyncio.Semaphore`, optional
        Semaphore instance that can be used to limit the number of
        website crawls happening concurrently. If ``None``, no limits
        are applied. By default, ``None``.
    pb_jurisdiction_name : str, optional
        Optional jurisdiction name to use to update progress bar, if
        it's being used. By default, ``None``.

    Returns
    -------
    out_docs : list
        List of BaseDocument instances containing potential ordinance
        information, or an empty list if no ordinance document was
        found.

    Notes
    -----
    Requires :class:`~compass.services.threaded.TempFileCache` service
    to be running.
    """
    if crawl_semaphore is None:
        crawl_semaphore = AsyncExitStack()

    async def _doc_heuristic(doc):  # noqa: RUF029
        """Heuristic check for wind ordinance documents"""
        is_valid_document = heuristic.check(doc.text.lower())
        if is_valid_document and pb_jurisdiction_name:
            COMPASS_PB.update_compass_website_crawl_doc_found(
                pb_jurisdiction_name
            )
        return is_valid_document

    async def _crawl_hook(*__, **___):  # noqa: RUF029
        """Update progress bar as pages are searched"""
        COMPASS_PB.update_compass_website_crawl_task(
            pb_jurisdiction_name, advance=1
        )

    file_loader_kwargs = file_loader_kwargs or {}
    file_loader_kwargs.update({"file_cache_coroutine": TempFileCache.call})

    crawler = COMPASSCrawler(
        validator=_doc_heuristic,
        url_scorer=COMPASSLinkScorer(keyword_points).score,
        file_loader_kwargs=file_loader_kwargs,
        num_link_scores_to_check_per_page=num_link_scores_to_check_per_page,
        already_visited=already_visited,
        max_pages=max_urls,
    )

    if pb_jurisdiction_name:
        COMPASS_PB.update_jurisdiction_task(
            pb_jurisdiction_name,
            description=f"Double-checking {website} for documents ...",
        )
        cpb = COMPASS_PB.compass_website_crawl_prog_bar(
            pb_jurisdiction_name, max_urls
        )
        ch = _crawl_hook
    else:
        cpb = AsyncExitStack()
        ch = None

    async with crawl_semaphore, cpb:
        return await crawler.run(website, on_new_page_visit_hook=ch)


async def download_jurisdiction_ordinance_using_search_engine(
    query_templates,
    jurisdiction,
    num_urls=5,
    file_loader_kwargs=None,
    search_semaphore=None,
    browser_semaphore=None,
    url_ignore_substrings=None,
    **kwargs,
):
    """Download the ordinance document(s) for a single jurisdiction

    Parameters
    ----------
    query_templates : sequence of str
        Query templates that will be formatted with the jurisdiction
        name before submission to the search engine.
    jurisdiction : Jurisdiction
        Location objects representing the jurisdiction.
    num_urls : int, optional
        Number of unique Google search result URL's to check for
        ordinance document. By default, ``5``.
    file_loader_kwargs : dict, optional
        Dictionary of keyword-argument pairs to initialize
        :class:`elm.web.file_loader.AsyncWebFileLoader` with. If found,
        the "pw_launch_kwargs" key in these will also be used to
        initialize the
        :class:`elm.web.search.google.PlaywrightGoogleLinkSearch`
        used for the google URL search. By default, ``None``.
    search_semaphore : :class:`asyncio.Semaphore`, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers used to submit search engine queries open
        concurrently. If this input is ``None``, the input from
        `browser_semaphore` will be used in its place (i.e. the searches
        and file downloads will be limited using the same semaphore).
        By default, ``None``.
    browser_semaphore : :class:`asyncio.Semaphore`, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers used to download content from the web open
        concurrently. If ``None``, no limits are applied.
        By default, ``None``.
    url_ignore_substrings : list of str, optional
        URL substrings that should be excluded from search results.
        Substrings are applied case-insensitively. By default, ``None``.
    **kwargs
        Additional keyword arguments forwarded to
        :func:`elm.web.search.run.web_search_links_as_docs`. Common
        entries include ``usage_tracker`` for logging LLM usage and
        extra Playwright configuration.

    Returns
    -------
    list or None
        List of BaseDocument instances possibly containing ordinance
        information, or ``None`` if no ordinance document was found.

    Notes
    -----
    Requires :class:`~compass.services.threaded.TempFileCachePB`
    service to be running.
    """
    COMPASS_PB.update_jurisdiction_task(
        jurisdiction.full_name, description="Searching web..."
    )

    kwargs.update(file_loader_kwargs or {})
    kwargs.update({"file_cache_coroutine": TempFileCachePB.call})
    try:
        docs = await _docs_from_web_search(
            query_templates,
            num_urls=num_urls,
            search_semaphore=search_semaphore,
            browser_semaphore=browser_semaphore,
            ignore_url_parts=url_ignore_substrings,
            jurisdiction_full_name=jurisdiction.full_name,
            **kwargs,
        )
    except KeyboardInterrupt:
        raise
    except Exception as e:
        msg = (
            "Encountered error of type %r while searching web for docs for %s:"
        )
        err_type = type(e)
        logger.exception(msg, err_type, jurisdiction.full_name)
        docs = []

    return docs


async def filter_ordinance_docs(
    docs,
    jurisdiction,
    model_configs,
    heuristic,
    tech,
    text_collectors,
    usage_tracker=None,
    check_for_correct_jurisdiction=True,
):
    """Filter a list of documents to only those that contain ordinances

    Parameters
    ----------
    docs : sequence of BaseDocument
        Documents to screen for ordinance content.
    jurisdiction : Jurisdiction
        Location objects representing the jurisdiction.
    model_configs : dict
        Dictionary of LLMConfig instances. Should have at minium a
        "default" key that is used as a fallback for all tasks.
    heuristic : object
        Domain-specific heuristic implementing a ``check`` method to
        qualify ordinance content.
    tech : str
        Technology of interest (e.g. "solar", "wind", etc). This is
        used to set up some document validation decision trees.
    text_collectors : iterable
        Iterable of text collector classes to run during document
        parsing. Each class must implement the
        :class:`compass.plugin.interface.BaseTextCollector` interface.
        If the document already contains text collected by a given
        collector (i.e. the collector's ``OUT_LABEL`` is found in
        ``doc.attrs``), that collector will be skipped.
    usage_tracker : UsageTracker, optional
        Optional tracker instance to monitor token usage during
        LLM calls. By default, ``None``.
    check_for_correct_jurisdiction : bool, default=True
        If ``True`` run jurisdiction validation before, content checks.
        By default, ``True``.

    Returns
    -------
    list or None
        List of BaseDocument instances possibly containing ordinance
        information, or ``None`` if no ordinance document was found.

    Notes
    -----
    The function updates CLI progress bars to reflect each filtering
    phase and returns documents sorted by quality heuristics.
    """
    logger.info(
        "%d document(s) passed in to COMPASS filter for %s\n\t- %s",
        len(docs),
        jurisdiction.full_name,
        "\n\t- ".join(
            [doc.attrs.get("source", "Unknown source") for doc in docs]
        ),
    )

    if check_for_correct_jurisdiction:
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
            "%d document(s) remaining after jurisdiction filter for %s"
            "\n\t- %s",
            len(docs),
            jurisdiction.full_name,
            "\n\t- ".join(
                [doc.attrs.get("source", "Unknown source") for doc in docs]
            ),
        )

    COMPASS_PB.update_jurisdiction_task(
        jurisdiction.full_name, description="Checking files for legal text..."
    )
    docs = await filter_documents(
        docs,
        validation_coroutine=_contains_relevant_text,
        task_name=jurisdiction.full_name,
        model_configs=model_configs,
        heuristic=heuristic,
        tech=tech,
        text_collectors=text_collectors,
        usage_tracker=usage_tracker,
    )
    if not docs:
        logger.info(
            "Did not find any potential ordinance documents for %s",
            jurisdiction.full_name,
        )
        return docs

    docs = _sort_final_ord_docs(docs)
    logger.info(
        "Found %d potential ordinance documents for %s\n\t- %s",
        len(docs),
        jurisdiction.full_name,
        "\n\t- ".join([str(doc) for doc in docs]),
    )
    return docs


async def _docs_from_web_search(
    query_templates,
    num_urls,
    search_semaphore,
    browser_semaphore,
    ignore_url_parts,
    jurisdiction_full_name,
    **kwargs,
):
    """Retrieve top ``N`` search results as document instances"""

    queries = [
        query.format(jurisdiction=jurisdiction_full_name)
        for query in query_templates
    ]
    urls = await search_with_fallback(
        queries,
        num_urls=num_urls,
        ignore_url_parts=ignore_url_parts,
        browser_semaphore=search_semaphore,
        task_name=jurisdiction_full_name,
        **kwargs,
    )
    if not urls:
        return []

    return await _docs_from_urls(
        urls, jurisdiction_full_name, browser_semaphore, **kwargs
    )


async def _docs_from_urls(
    urls, jurisdiction_full_name, browser_semaphore, **kwargs
):
    """Load documents from a list of URLs using AsyncWebFileLoader"""
    logger.debug("Downloading documents for URLS: \n\t-%s", "\n\t-".join(urls))
    logger.trace(
        "kwargs for COMPASSWebFileLoader:\n%s",
        pprint.PrettyPrinter().pformat(kwargs),
    )
    file_loader = COMPASSWebFileLoader(
        browser_semaphore=browser_semaphore, **kwargs
    )

    COMPASS_PB.update_jurisdiction_task(
        jurisdiction_full_name, description="Downloading files..."
    )
    async with COMPASS_PB.file_download_prog_bar(
        jurisdiction_full_name, len(urls)
    ):
        return await load_docs(urls, file_loader)


async def _down_select_docs_correct_jurisdiction(
    docs, jurisdiction, usage_tracker, model_config
):
    """Remove documents that do not match the target jurisdiction"""
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


async def _contains_relevant_text(
    doc, model_configs, usage_tracker=None, **kwargs
):
    """Determine whether a document contains ordinance information"""
    model_config = model_configs.get(
        LLMTasks.DOCUMENT_CONTENT_VALIDATION,
        model_configs[LLMTasks.DEFAULT],
    )
    logger.debug(
        "Checking doc for ordinance info (source: %r)...",
        doc.attrs.get("source", "unknown"),
    )
    found_text = await check_for_relevant_text(
        doc,
        model_config=model_config,
        usage_tracker=usage_tracker,
        **kwargs,
    )
    if found_text:
        logger.debug("Detected relevant text; parsing date...")
        date_model_config = model_configs.get(
            LLMTasks.DATE_EXTRACTION, model_configs[LLMTasks.DEFAULT]
        )
        doc = await extract_date(
            doc, date_model_config, usage_tracker=usage_tracker
        )
    return found_text


def _sanitize_doc_sources(docs):
    """Rewrite source attrs on documents returned by ELMWebsiteCrawler

    crawl4ai can surface PDF URLs containing raw spaces (e.g. filenames
    like "Land Use Code.pdf").  These fail when the file loader issues
    an HTTP request because spaces are invalid in a URL path.  This
    function percent-encodes each document's ``source`` attribute
    in-place so that all downstream consumers receive a valid URL.
    """
    for doc in docs:
        source = doc.attrs.get("source")
        if source and " " in source:
            doc.attrs["source"] = sanitize_url(source)


def _sort_final_ord_docs(all_ord_docs):
    """Sort ordinance documents by desirability heuristics"""
    if not all_ord_docs:
        return None

    return sorted(all_ord_docs, key=_ord_doc_sorting_key, reverse=True)


def _ord_doc_sorting_key(doc):
    """Compute a composite sorting score for ordinance documents"""
    no_date = (_NEG_INF, _NEG_INF, _NEG_INF)
    latest_year, latest_month, latest_day = doc.attrs.get("date") or no_date
    best_docs_from_website = doc.attrs.get(_SCORE_KEY, 0)
    prefer_pdf_files = isinstance(doc, PDFDocument)
    highest_jurisdiction_score = doc.attrs.get(
        # If not present, URL check passed with confidence so we set
        # score to 1
        DTreeJurisdictionValidator.META_SCORE_KEY,
        1,
    )
    shortest_text_length = -1 * len(doc.text)
    return (
        best_docs_from_website,
        latest_year or _NEG_INF,
        prefer_pdf_files,
        highest_jurisdiction_score,
        shortest_text_length,
        latest_month or _NEG_INF,
        latest_day or _NEG_INF,
    )
