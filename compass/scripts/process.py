"""Ordinance full processing logic"""

import time
import asyncio
import logging
from copy import deepcopy
from functools import cached_property
from contextlib import AsyncExitStack, contextmanager
from datetime import datetime, UTC

import pandas as pd
from elm.web.utilities import get_redirected_url

from compass import __version__
from compass.scripts.download import (
    find_jurisdiction_website,
    download_known_urls,
    load_known_docs,
    download_jurisdiction_ordinance_using_search_engine,
    download_jurisdiction_ordinances_from_website,
    download_jurisdiction_ordinances_from_website_compass_crawl,
    filter_ordinance_docs,
)
from compass.exceptions import COMPASSValueError
from compass.extraction import (
    extract_ordinance_values,
    extract_ordinance_text_with_ngram_validation,
)
from compass.extraction.solar import (
    SolarHeuristic,
    SolarOrdinanceTextCollector,
    SolarOrdinanceTextExtractor,
    SolarPermittedUseDistrictsTextCollector,
    SolarPermittedUseDistrictsTextExtractor,
    StructuredSolarOrdinanceParser,
    StructuredSolarPermittedUseDistrictsParser,
    SOLAR_QUESTION_TEMPLATES,
    BEST_SOLAR_ORDINANCE_WEBSITE_URL_KEYWORDS,
)
from compass.extraction.wind import (
    WindHeuristic,
    WindOrdinanceTextCollector,
    WindOrdinanceTextExtractor,
    WindPermittedUseDistrictsTextCollector,
    WindPermittedUseDistrictsTextExtractor,
    StructuredWindOrdinanceParser,
    StructuredWindPermittedUseDistrictsParser,
    WIND_QUESTION_TEMPLATES,
    BEST_WIND_ORDINANCE_WEBSITE_URL_KEYWORDS,
)
from compass.extraction.small_wind import (
    SmallWindHeuristic,
    SmallWindOrdinanceTextCollector,
    SmallWindOrdinanceTextExtractor,
    SmallWindPermittedUseDistrictsTextCollector,
    SmallWindPermittedUseDistrictsTextExtractor,
    StructuredSmallWindOrdinanceParser,
    StructuredSmallWindPermittedUseDistrictsParser,
    SMALL_WIND_QUESTION_TEMPLATES,
    BEST_SMALL_WIND_ORDINANCE_WEBSITE_URL_KEYWORDS,
)
from compass.validation.location import JurisdictionWebsiteValidator
from compass.llm import LLMCaller, OpenAIConfig
from compass.services.cpu import (
    PDFLoader,
    OCRPDFLoader,
    read_pdf_doc,
    read_pdf_doc_ocr,
    read_pdf_file,
    read_pdf_file_ocr,
)
from compass.services.usage import UsageTracker
from compass.services.openai import usage_from_response
from compass.services.provider import RunningAsyncServices
from compass.services.threaded import (
    TempFileCachePB,
    TempFileCache,
    FileMover,
    CleanedFileWriter,
    OrdDBFileWriter,
    UsageUpdater,
    JurisdictionUpdater,
    HTMLFileLoader,
)
from compass.utilities import (
    LLM_COST_REGISTRY,
    compile_run_summary_message,
    doc_infos_to_db,
    load_all_jurisdiction_info,
    load_jurisdictions_from_fp,
    num_ordinances_in_doc,
    save_db,
    save_run_meta,
    Directories,
    ProcessKwargs,
    TechSpec,
)
from compass.utilities.enums import LLMTasks
from compass.utilities.location import Jurisdiction
from compass.utilities.logs import (
    LocationFileLog,
    LogListener,
    NoLocationFilter,
)
from compass.utilities.base import WebSearchParams
from compass.utilities.parsing import load_config
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
_TEXT_EXTRACTION_TASKS = {
    WindOrdinanceTextExtractor: "Extracting wind ordinance text",
    WindPermittedUseDistrictsTextExtractor: (
        "Extracting wind permitted use text"
    ),
    SolarOrdinanceTextExtractor: "Extracting solar ordinance text",
    SolarPermittedUseDistrictsTextExtractor: (
        "Extracting solar permitted use text"
    ),
    SmallWindOrdinanceTextExtractor: ("Extracting small wind ordinance text"),
    SmallWindPermittedUseDistrictsTextExtractor: (
        "Extracting small wind permitted use text"
    ),
}
_JUR_COLS = [
    "Jurisdiction Type",
    "State",
    "County",
    "Subdivision",
    "FIPS",
    "Website",
]
MAX_CONCURRENT_SEARCH_ENGINE_QUERIES = 10


async def process_jurisdictions_with_openai(  # noqa: PLR0917, PLR0913
    out_dir,
    tech,
    jurisdiction_fp,
    model="gpt-4o-mini",
    num_urls_to_check_per_jurisdiction=5,
    max_num_concurrent_browsers=10,
    max_num_concurrent_website_searches=10,
    max_num_concurrent_jurisdictions=25,
    url_ignore_substrings=None,
    known_local_docs=None,
    known_doc_urls=None,
    file_loader_kwargs=None,
    search_engines=None,
    pytesseract_exe_fp=None,
    td_kwargs=None,
    tpe_kwargs=None,
    ppe_kwargs=None,
    log_dir=None,
    clean_dir=None,
    ordinance_file_dir=None,
    jurisdiction_dbs_dir=None,
    perform_se_search=True,
    perform_website_search=True,
    llm_costs=None,
    log_level="INFO",
    keep_async_logs=False,
):
    """Extract ordinances for one or more jurisdiction(s)

    This function scrapes ordinance documents (PDFs or HTML text) for a
    given set of jurisdictions and processes them using one or more
    LLM models. Output files, logs, and intermediate artifacts are
    stored in configurable directories.

    The processing has a well-defined order:

        1. Process any/all known local documents
        2. Process any/all known document URLs
        3. Search engine-based search for ordinance documents
        4. Jurisdiction website crawl-based search for ordinance
           documents

    Users can disable any of these steps via inputs to this function. If
    any step returns a document with extractable ordinance information,
    subsequent steps are skipped for that jurisdiction.

    Parameters
    ----------
    out_dir : path-like
        Path to the output directory. If it does not exist, it will be
        created. This directory will contain the structured ordinance
        CSV file, all downloaded ordinance documents (PDFs and HTML),
        usage metadata, and default subdirectories for logs and
        intermediate outputs (unless otherwise specified).
    tech : {"wind", "solar", "small wind"}
        Label indicating which technology type is being processed.
    jurisdiction_fp : path-like
        Path to a CSV file specifying the jurisdictions to process.
        The CSV must contain at least two columns: "County" and "State",
        which specify the county and state names, respectively. If you
        would like to process a subdivision with a county, you must also
        include "Subdivision" and "Jurisdiction Type" columns. The
        "Subdivision" should be the name of the subdivision, and the
        "Jurisdiction Type" should be a string identifying the type of
        subdivision (e.g., "City", "Township", etc.)
    model : str or list of dict, optional
        LLM model(s) to use for scraping and parsing ordinance
        documents. If a string is provided, it is assumed to be the name
        of the default model (e.g., "gpt-4o"), and environment variables
        are used for authentication.

        If a list is provided, it should contain dictionaries of
        arguments that can initialize instances of
        :class:`~compass.llm.config.OpenAIConfig`. Each dictionary can
        specify the model name, client type, and initialization
        arguments.

        Each dictionary must also include a ``tasks`` key, which maps to
        a string or list of strings indicating the tasks that instance
        should handle. Exactly one of the instances **must** include
        "default" as a task, which will be used when no specific task is
        matched. For example::

            "model": [
                {
                    "model": "gpt-4o-mini",
                    "llm_call_kwargs": {
                        "temperature": 0,
                        "timeout": 300,
                    },
                    "client_kwargs": {
                        "api_key": "<your_api_key>",
                        "api_version": "<your_api_version>",
                        "azure_endpoint": "<your_azure_endpoint>",
                    },
                    "tasks": ["default", "date_extraction"],
                },
                {
                    "model": "gpt-4o",
                    "client_type": "openai",
                    "tasks": ["ordinance_text_extraction"],
                }
            ]

        By default, ``"gpt-4o"``.
    num_urls_to_check_per_jurisdiction : int, optional
        Number of unique Google search result URLs to check for each
        jurisdiction when attempting to locate ordinance documents.
        By default, ``5``.
    max_num_concurrent_browsers : int, optional
        Maximum number of browser instances to launch concurrently for
        retrieving information from the web. Increasing this value too
        much may lead to timeouts or performance issues on machines with
        limited resources. By default, ``10``.
    max_num_concurrent_website_searches : int, optional
        Maximum number of website searches allowed to run
        simultaneously. Increasing this value can speed up searches, but
        may lead to timeouts or performance issues on machines with
        limited resources. By default, ``10``.
    max_num_concurrent_jurisdictions : int, default=25
        Maximum number of jurisdictions to process concurrently.
        Limiting this can help manage memory usage when dealing with a
        large number of documents. By default ``25``.
    url_ignore_substrings : list of str, optional
        A list of substrings that, if found in any URL, will cause the
        URL to be excluded from consideration. This can be used to
        specify particular websites or entire domains to ignore. For
        example::

            url_ignore_substrings = [
                "wikipedia",
                "nrel.gov",
                "www.co.delaware.in.us/documents/1649699794_0382.pdf",
            ]

        The above configuration would ignore all `wikipedia` articles,
        all websites on the NREL domain, and the specific file located
        at `www.co.delaware.in.us/documents/1649699794_0382.pdf`.
        By default, ``None``.
    known_local_docs : dict or path-like, optional
        A dictionary where keys are the jurisdiction codes (as strings)
        and values are lists of dictionaries containing information
        about each document. The latter dictionaries should contain at
        least the key ``"source_fp"`` pointing to the **full** path of
        the local document file. All other keys will be added as
        attributes to the loaded document instance. You can include the
        key ``"is_legal_doc"`` to skip the legal document check for
        known documents. Similarly, you can provide the ``"date"`` key,
        which is a list of ``[year, month, day]``, some or all of which
        can be null, to skip the date extraction step of the processing
        pipeline. If this input is provided, local documents will be
        checked first. See the top-level documentation of this function
        for the full processing of the pipeline. This input can also be
        a path to a JSON file containing the dictionary of
        code-to-document-info mappings. By default, ``None``.
    known_doc_urls : dict or path-like, optional
        A dictionary where keys are the jurisdiction codes (as strings)
        and values are lists of dictionaries containing information
        about each document. The latter dictionaries should contain at
        least the key ``"source"`` representing the known URL to check
        for that document. All other keys will be added as attributes
        to the loaded document instance. You can include the key
        ``"is_legal_doc"`` to skip the legal document check for known
        documents. Similarly, you can provide the ``"date"`` key, which
        is a list of ``[year, month, day]``, some or all of which can
        be null, to skip the date extraction step of the processing
        pipeline. If this input is provided, the known URLs will be
        checked before applying the search engine search. See the
        top-level documentation of this function for the full processing
        order of the pipeline. This input can also be a path to a JSON
        file containing the dictionary of code-to-document-info
        mappings.

        .. Note:: The same input can be used for both `known_local_docs`
                  and `known_doc_urls` as long as both ``"source_fp"``
                  and ``"source"`` keys are provided in each document
                  info dictionary.

        By default, ``None``.
    file_loader_kwargs : dict, optional
        Dictionary of keyword arguments pairs to initialize
        :class:`elm.web.file_loader.AsyncWebFileLoader`. If found, the
        "pw_launch_kwargs" key in these will also be used to initialize
        the :class:`elm.web.search.google.PlaywrightGoogleLinkSearch`
        used for the google URL search. By default, ``None``.
    search_engines : list, optional
        A list of dictionaries, where each dictionary contains
        information about a search engine class that should be used for
        the document retrieval process. Each dictionary should contain
        at least the key ``"se_name"``, which should correspond to one
        of the search engine class names from
        :obj:`elm.web.search.run.SEARCH_ENGINE_OPTIONS`. The rest of the
        keys in the dictionary should contain keyword-value pairs to be
        used as parameters to initialize the search engine class (things
        like API keys and configuration options; see the ELM
        documentation for details on search engine class parameters).
        The list should be ordered by search engine preference - the
        first search engine parameters will be used to submit the
        queries initially, then any subsequent search engine listings
        will be used as fallback (in order that they appear). Do not
        repeat search engines - only the last config dictionary will be
        used to initialize the search engine if you do. If ``None``,
        then all default configurations for the search engines
        (along with the fallback order) are used. By default, ``None``.
    pytesseract_exe_fp : path-like, optional
        Path to the `pytesseract` executable. If specified, OCR will be
        used to extract text from scanned PDFs using Google's Tesseract.
        By default ``None``.
    td_kwargs : dict, optional
        Additional keyword arguments to pass to
        :class:`tempfile.TemporaryDirectory`. The temporary directory is
        used to store documents which have not yet been confirmed to
        contain relevant information. By default, ``None``.
    tpe_kwargs : dict, optional
        Additional keyword arguments to pass to
        :class:`concurrent.futures.ThreadPoolExecutor`, used for
        I/O-bound tasks such as logging. By default, ``None``.
    ppe_kwargs : dict, optional
        Additional keyword arguments to pass to
        :class:`concurrent.futures.ProcessPoolExecutor`, used for
        CPU-bound tasks such as PDF loading and parsing.
        By default, ``None``.
    log_dir : path-like, optional
        Path to the directory for storing log files. If not provided, a
        ``logs`` subdirectory will be created inside `out_dir`.
        By default, ``None``.
    clean_dir : path-like, optional
        Path to the directory for storing cleaned ordinance text output.
        If not provided, a ``cleaned_text`` subdirectory will be created
        inside `out_dir`. By default, ``None``.
    ordinance_file_dir : path-like, optional
        Path to the directory where downloaded ordinance files (PDFs or
        HTML) for each jurisdiction are stored. If not provided, a
        ``ordinance_files`` subdirectory will be created inside
        `out_dir`. By default, ``None``.
    jurisdiction_dbs_dir : path-like, optional
        Path to the directory where parsed ordinance database files are
        stored for each jurisdiction. If not provided, a
        ``jurisdiction_dbs`` subdirectory will be created inside
        `out_dir`. By default, ``None``.
    perform_se_search : bool, default=True
        Option to perform a search engine-based search for ordinance
        documents. This is the standard way to collect ordinance
        documents, and it is recommended to leave this set to ``True``
        unless you are re-processing local documents. If ``True``, the
        search engine approach is used to locate ordinance documents
        before falling back to a website crawl-based search (if that has
        been selected). By default, ``True``.
    perform_website_search : bool, default=True
        Option to fallback to a jurisdiction website crawl-based search
        for ordinance documents if the search engine approach fails to
        recover any relevant documents. By default, ``True``.
    llm_costs : dict, optional
        Dictionary mapping model names to their token costs, used to
        track the estimated total cost of LLM usage during the run. The
        structure should be::

            {"model_name": {"prompt": float, "response": float}}

        Costs are specified in dollars per million tokens. For example::

            "llm_costs": {"my_gpt": {"prompt": 1.5, "response": 3.7}}

        registers a model named `"my_gpt"` with a cost of $1.5 per
        million input (prompt) tokens and $3.7 per million output
        (response) tokens for the current processing run.

        .. NOTE::

            The displayed total cost does not track cached tokens, so
            treat it like an estimate. Your final API costs may vary.

        If set to ``None``, no custom model costs are recorded, and
        cost tracking may be unavailable in the progress bar.
        By default, ``None``.
    log_level : str, optional
        Logging level for ordinance scraping and parsing (e.g., "TRACE",
        "DEBUG", "INFO", "WARNING", or "ERROR"). By default, ``"INFO"``.
    keep_async_logs : bool, default=False
        Option to store the full asynchronous log record to a file. This
        is only useful if you intend to monitor overall processing
        progress from a file instead of from the terminal. If ``True``,
        all of the unordered records are written to a "all.log" file in
        the `log_dir` directory. By default, ``False``.

    Returns
    -------
    str
        Message summarizing run results, including total processing
        time, total cost, output directory, and number of documents
        found. The message is formatted for easy reading in the terminal
        and may include color-coded cost information if the terminal
        supports it.
    """
    if log_level == "DEBUG":
        log_level = "DEBUG_TO_FILE"

    log_listener = LogListener(["compass", "elm"], level=log_level)
    LLM_COST_REGISTRY.update(llm_costs or {})
    dirs = _setup_folders(
        out_dir,
        log_dir=log_dir,
        clean_dir=clean_dir,
        ofd=ordinance_file_dir,
        jdd=jurisdiction_dbs_dir,
    )
    pk = ProcessKwargs(
        known_local_docs,
        known_doc_urls,
        file_loader_kwargs,
        td_kwargs,
        tpe_kwargs,
        ppe_kwargs,
        max_num_concurrent_jurisdictions,
    )
    wsp = WebSearchParams(
        num_urls_to_check_per_jurisdiction,
        max_num_concurrent_browsers,
        max_num_concurrent_website_searches,
        url_ignore_substrings,
        pytesseract_exe_fp,
        search_engines,
    )
    models = _initialize_model_params(model)
    runner = _COMPASSRunner(
        dirs=dirs,
        log_listener=log_listener,
        tech=tech,
        models=models,
        web_search_params=wsp,
        process_kwargs=pk,
        perform_se_search=perform_se_search,
        perform_website_search=perform_website_search,
        log_level=log_level,
    )
    async with log_listener as ll:
        _setup_main_logging(dirs.logs, log_level, ll, keep_async_logs)
        return await runner.run(jurisdiction_fp)


class _COMPASSRunner:
    """Helper class to run COMPASS"""

    def __init__(
        self,
        dirs,
        log_listener,
        tech,
        models,
        web_search_params=None,
        process_kwargs=None,
        perform_se_search=True,
        perform_website_search=True,
        log_level="INFO",
    ):
        self.dirs = dirs
        self.log_listener = log_listener
        self.tech = tech
        self.models = models
        self.web_search_params = web_search_params or WebSearchParams()
        self.process_kwargs = process_kwargs or ProcessKwargs()
        self.perform_se_search = perform_se_search
        self.perform_website_search = perform_website_search
        self.log_level = log_level

    @cached_property
    def browser_semaphore(self):
        """asyncio.Semaphore or None: Sem to limit # of browsers"""
        return (
            asyncio.Semaphore(
                self.web_search_params.max_num_concurrent_browsers
            )
            if self.web_search_params.max_num_concurrent_browsers
            else None
        )

    @cached_property
    def crawl_semaphore(self):
        """asyncio.Semaphore or None: Sem to limit # of crawls"""
        return (
            asyncio.Semaphore(
                self.web_search_params.max_num_concurrent_website_searches
            )
            if self.web_search_params.max_num_concurrent_website_searches
            else None
        )

    @cached_property
    def search_engine_semaphore(self):
        """asyncio.Semaphore or None: Sem to limit # of SE queries"""
        return asyncio.Semaphore(MAX_CONCURRENT_SEARCH_ENGINE_QUERIES)

    @cached_property
    def _jurisdiction_semaphore(self):
        """asyncio.Semaphore or None: Sem to limit # of processes"""
        return (
            asyncio.Semaphore(
                self.process_kwargs.max_num_concurrent_jurisdictions
            )
            if self.process_kwargs.max_num_concurrent_jurisdictions
            else None
        )

    @property
    def jurisdiction_semaphore(self):
        """asyncio.Semaphore or AsyncExitStack: Jurisdictions limit"""
        if self._jurisdiction_semaphore is None:
            return AsyncExitStack()
        return self._jurisdiction_semaphore

    @cached_property
    def file_loader_kwargs(self):
        """dict: Keyword arguments for `AsyncWebFileLoader`"""
        file_loader_kwargs = _configure_file_loader_kwargs(
            self.process_kwargs.file_loader_kwargs
        )
        if self.web_search_params.pytesseract_exe_fp is not None:
            _setup_pytesseract(self.web_search_params.pytesseract_exe_fp)
            file_loader_kwargs.update(
                {"pdf_ocr_read_coroutine": read_pdf_doc_ocr}
            )
        return file_loader_kwargs

    @cached_property
    def local_file_loader_kwargs(self):
        """dict: Keyword arguments for `AsyncLocalFileLoader`"""
        file_loader_kwargs = {
            "pdf_read_coroutine": read_pdf_file,
            "pdf_read_kwargs": (
                self.process_kwargs.file_loader_kwargs.get("pdf_read_kwargs")
            ),
            "html_read_kwargs": (
                self.process_kwargs.file_loader_kwargs.get("html_read_kwargs")
            ),
        }

        if self.web_search_params.pytesseract_exe_fp is not None:
            _setup_pytesseract(self.web_search_params.pytesseract_exe_fp)
            file_loader_kwargs.update(
                {"pdf_ocr_read_coroutine": read_pdf_file_ocr}
            )
        return file_loader_kwargs

    @cached_property
    def known_local_docs(self):
        """dict: Known filepaths by jurisdiction code"""
        known_local_docs = self.process_kwargs.known_local_docs or {}
        if isinstance(known_local_docs, str):
            known_local_docs = load_config(known_local_docs)
        return {int(key): val for key, val in known_local_docs.items()}

    @cached_property
    def known_doc_urls(self):
        """dict: Known URL's keyed by jurisdiction code"""
        known_doc_urls = self.process_kwargs.known_doc_urls or {}
        if isinstance(known_doc_urls, str):
            known_doc_urls = load_config(known_doc_urls)
        return {int(key): val for key, val in known_doc_urls.items()}

    @cached_property
    def tpe_kwargs(self):
        """dict: Keyword arguments for `ThreadPoolExecutor`"""
        return _configure_thread_pool_kwargs(self.process_kwargs.tpe_kwargs)

    @cached_property
    def _base_services(self):
        """list: List of required services to run for processing"""
        base_services = [
            TempFileCachePB(
                td_kwargs=self.process_kwargs.td_kwargs,
                tpe_kwargs=self.tpe_kwargs,
            ),
            TempFileCache(
                td_kwargs=self.process_kwargs.td_kwargs,
                tpe_kwargs=self.tpe_kwargs,
            ),
            FileMover(self.dirs.ordinance_files, tpe_kwargs=self.tpe_kwargs),
            CleanedFileWriter(
                self.dirs.clean_files, tpe_kwargs=self.tpe_kwargs
            ),
            OrdDBFileWriter(
                self.dirs.jurisdiction_dbs, tpe_kwargs=self.tpe_kwargs
            ),
            UsageUpdater(
                self.dirs.out / "usage.json", tpe_kwargs=self.tpe_kwargs
            ),
            JurisdictionUpdater(
                self.dirs.out / "jurisdictions.json",
                tpe_kwargs=self.tpe_kwargs,
            ),
            PDFLoader(**(self.process_kwargs.ppe_kwargs or {})),
            HTMLFileLoader(**self.tpe_kwargs),
        ]

        if self.web_search_params.pytesseract_exe_fp is not None:
            base_services.append(
                # pytesseract locks up with multiple processes, so
                # hardcode to only use 1 for now
                OCRPDFLoader(max_workers=1),
            )
        return base_services

    async def run(self, jurisdiction_fp):
        """Run COMPASS for a set of jurisdictions

        Parameters
        ----------
        jurisdiction_fp : path-like
            Path to CSV file containing the jurisdictions to search.

        Returns
        -------
        str
            Message summarizing run results, including total processing
            time, total cost, output directory, and number of documents
            found. The message is formatted for easy reading in the
            terminal and may include color-coded cost information if
            the terminal supports it.
        """
        logger.info("Running COMPASS version %s", __version__)
        jurisdictions = _load_jurisdictions_to_process(jurisdiction_fp)

        num_jurisdictions = len(jurisdictions)
        COMPASS_PB.create_main_task(num_jurisdictions=num_jurisdictions)
        start_date = datetime.now(UTC)

        doc_infos, total_cost = await self._run_all(jurisdictions)

        db, num_docs_found = doc_infos_to_db(doc_infos)
        save_db(db, self.dirs.out)
        total_time = save_run_meta(
            self.dirs,
            self.tech,
            start_date=start_date,
            end_date=datetime.now(UTC),
            num_jurisdictions_searched=num_jurisdictions,
            num_jurisdictions_found=num_docs_found,
            total_cost=total_cost,
            models=self.models,
        )
        run_msg = compile_run_summary_message(
            total_seconds=total_time,
            total_cost=total_cost,
            out_dir=self.dirs.out,
            document_count=num_docs_found,
        )
        for sub_msg in run_msg.split("\n"):
            logger.info(
                sub_msg.replace("[#71906e]", "").replace("[/#71906e]", "")
            )
        return run_msg

    async def _run_all(self, jurisdictions):
        """Process all jurisdictions with running services"""
        services = [model.llm_service for model in set(self.models.values())]
        services += self._base_services
        _ = self.file_loader_kwargs  # init loader kwargs once
        _ = self.local_file_loader_kwargs  # init local loader kwargs once
        logger.info("Processing %d jurisdiction(s)", len(jurisdictions))
        async with RunningAsyncServices(services):
            tasks = []
            for __, row in jurisdictions.iterrows():
                jur_type, state, county, sub, fips, website = row[_JUR_COLS]
                jurisdiction = Jurisdiction(
                    subdivision_type=jur_type,
                    state=state,
                    county=county,
                    subdivision_name=sub,
                    code=fips,
                )
                usage_tracker = UsageTracker(
                    jurisdiction.full_name, usage_from_response
                )
                task = asyncio.create_task(
                    self._processed_jurisdiction_info_with_pb(
                        jurisdiction,
                        website,
                        self.known_local_docs.get(fips),
                        self.known_doc_urls.get(fips),
                        usage_tracker=usage_tracker,
                    ),
                    name=jurisdiction.full_name,
                )
                tasks.append(task)
            doc_infos = await asyncio.gather(*tasks)
            total_cost = await _compute_total_cost()

        return doc_infos, total_cost

    async def _processed_jurisdiction_info_with_pb(
        self, jurisdiction, *args, **kwargs
    ):
        """Process jurisdiction and update progress bar"""
        async with self.jurisdiction_semaphore:
            with COMPASS_PB.jurisdiction_prog_bar(jurisdiction.full_name):
                return await self._processed_jurisdiction_info(
                    jurisdiction, *args, **kwargs
                )

    async def _processed_jurisdiction_info(self, *args, **kwargs):
        """Drop `doc` from RAM and only keep enough info to re-build"""

        doc = await self._process_jurisdiction_with_logging(*args, **kwargs)

        if doc is None or isinstance(doc, Exception):
            return None

        keys = ["source", "date", "jurisdiction", "ord_db_fp"]
        doc_info = {key: doc.attrs.get(key) for key in keys}
        logger.debug("Saving the following doc info:\n%s", doc_info)
        return doc_info

    async def _process_jurisdiction_with_logging(
        self,
        jurisdiction,
        jurisdiction_website,
        known_local_docs=None,
        known_doc_urls=None,
        usage_tracker=None,
    ):
        """Retrieve ordinance document with async logs"""
        async with LocationFileLog(
            self.log_listener,
            self.dirs.logs,
            location=jurisdiction.full_name,
            level=self.log_level,
        ):
            task = asyncio.create_task(
                _SingleJurisdictionRunner(
                    self.tech,
                    jurisdiction,
                    self.models,
                    self.web_search_params,
                    self.file_loader_kwargs,
                    local_file_loader_kwargs=self.local_file_loader_kwargs,
                    known_local_docs=known_local_docs,
                    known_doc_urls=known_doc_urls,
                    browser_semaphore=self.browser_semaphore,
                    crawl_semaphore=self.crawl_semaphore,
                    search_engine_semaphore=self.search_engine_semaphore,
                    jurisdiction_website=jurisdiction_website,
                    perform_se_search=self.perform_se_search,
                    perform_website_search=self.perform_website_search,
                    usage_tracker=usage_tracker,
                ).run(),
                name=jurisdiction.full_name,
            )
            try:
                doc, *__ = await asyncio.gather(task)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                msg = "Encountered error of type %r while processing %s:"
                err_type = type(e)
                logger.exception(msg, err_type, jurisdiction.full_name)
                doc = None

            return doc


class _SingleJurisdictionRunner:
    """Helper class to process a single jurisdiction"""

    def __init__(  # noqa: PLR0913
        self,
        tech,
        jurisdiction,
        models,
        web_search_params,
        file_loader_kwargs,
        *,
        local_file_loader_kwargs=None,
        known_local_docs=None,
        known_doc_urls=None,
        browser_semaphore=None,
        crawl_semaphore=None,
        search_engine_semaphore=None,
        jurisdiction_website=None,
        perform_se_search=True,
        perform_website_search=True,
        usage_tracker=None,
    ):
        self.tech_specs = _compile_tech_specs(tech)
        self.jurisdiction = jurisdiction
        self.models = models
        self.web_search_params = web_search_params
        self.file_loader_kwargs = file_loader_kwargs
        self.local_file_loader_kwargs = local_file_loader_kwargs
        self.known_local_docs = known_local_docs
        self.known_doc_urls = known_doc_urls
        self.browser_semaphore = browser_semaphore
        self.crawl_semaphore = crawl_semaphore
        self.search_engine_semaphore = search_engine_semaphore
        self.usage_tracker = usage_tracker
        self.jurisdiction_website = jurisdiction_website
        self.perform_se_search = perform_se_search
        self.perform_website_search = perform_website_search
        self.validate_user_website_input = True
        self._jsp = None

    @cached_property
    def file_loader_kwargs_no_ocr(self):
        """dict: Keyword arguments for `AsyncWebFileLoader` (no OCR)"""
        flk = deepcopy(self.file_loader_kwargs)
        flk.pop("pdf_ocr_read_coroutine", None)
        return flk

    @contextmanager
    def _tracked_progress(self):
        """Context manager to set up jurisdiction sub-progress bar"""
        loc = self.jurisdiction.full_name
        with COMPASS_PB.jurisdiction_sub_prog(loc) as self._jsp:
            yield

        self._jsp = None

    async def run(self):
        """Download and parse document for a single jurisdiction"""
        start_time = time.monotonic()
        doc = None
        try:
            doc = await self._run()
        finally:
            await self._record_usage()
            await _record_jurisdiction_info(
                self.jurisdiction, doc, start_time, self.usage_tracker
            )

        return doc

    async def _run(self):
        """Search for docs and parse them for ordinances"""
        if self.known_local_docs:
            doc = await self._try_find_ordinances(
                method=self._load_known_local_documents,
            )
            if doc is not None:
                return doc

        if self.known_doc_urls:
            doc = await self._try_find_ordinances(
                method=self._download_known_url_documents,
            )
            if doc is not None:
                return doc

        if self.perform_se_search:
            doc = await self._try_find_ordinances(
                method=self._find_documents_using_search_engine,
            )
            if doc is not None:
                return doc

        if self.perform_website_search:
            doc = await self._try_find_ordinances(
                method=self._find_documents_from_website,
            )
            if doc is not None:
                return doc

        return None

    async def _try_find_ordinances(self, method, *args, **kwargs):
        """Try to find ordinances using specified method"""
        docs = await method(*args, **kwargs)
        if docs is None:
            return None

        COMPASS_PB.update_jurisdiction_task(
            self.jurisdiction.full_name,
            description="Extracting structured data...",
        )
        return await self._parse_docs_for_ordinances(docs)

    async def _load_known_local_documents(self):
        """Load local ordinance documents"""

        docs = await load_known_docs(
            self.jurisdiction,
            [info["source_fp"] for info in self.known_local_docs],
            local_file_loader_kwargs=self.local_file_loader_kwargs,
        )

        if not docs:
            return None

        _add_known_doc_attrs_to_all_docs(
            docs, self.known_local_docs, key="source_fp"
        )
        docs = await filter_ordinance_docs(
            docs,
            self.jurisdiction,
            self.models,
            heuristic=self.tech_specs.heuristic,
            tech=self.tech_specs.name,
            ordinance_text_collector_class=(
                self.tech_specs.ordinance_text_collector
            ),
            permitted_use_text_collector_class=(
                self.tech_specs.permitted_use_text_collector
            ),
            usage_tracker=self.usage_tracker,
            check_for_correct_jurisdiction=False,
        )
        if not docs:
            return None

        for doc in docs:
            doc.attrs["jurisdiction"] = self.jurisdiction
            doc.attrs["jurisdiction_name"] = self.jurisdiction.full_name
            doc.attrs["jurisdiction_website"] = None
            doc.attrs["compass_crawl"] = False

        await self._record_usage()
        return docs

    async def _download_known_url_documents(self):
        """Download ordinance documents from known URLs"""

        docs = await download_known_urls(
            self.jurisdiction,
            [info["source"] for info in self.known_doc_urls],
            browser_semaphore=self.browser_semaphore,
            file_loader_kwargs=self.file_loader_kwargs,
        )

        if not docs:
            return None

        _add_known_doc_attrs_to_all_docs(
            docs, self.known_doc_urls, key="source"
        )
        docs = await filter_ordinance_docs(
            docs,
            self.jurisdiction,
            self.models,
            heuristic=self.tech_specs.heuristic,
            tech=self.tech_specs.name,
            ordinance_text_collector_class=(
                self.tech_specs.ordinance_text_collector
            ),
            permitted_use_text_collector_class=(
                self.tech_specs.permitted_use_text_collector
            ),
            usage_tracker=self.usage_tracker,
            check_for_correct_jurisdiction=False,
        )
        if not docs:
            return None

        for doc in docs:
            doc.attrs["jurisdiction"] = self.jurisdiction
            doc.attrs["jurisdiction_name"] = self.jurisdiction.full_name
            doc.attrs["jurisdiction_website"] = None
            doc.attrs["compass_crawl"] = False

        await self._record_usage()
        return docs

    async def _find_documents_using_search_engine(self):
        """Search the web for an ordinance document and construct it"""
        docs = await download_jurisdiction_ordinance_using_search_engine(
            self.tech_specs.questions,
            self.jurisdiction,
            num_urls=self.web_search_params.num_urls_to_check_per_jurisdiction,
            file_loader_kwargs=self.file_loader_kwargs,
            search_semaphore=self.search_engine_semaphore,
            browser_semaphore=self.browser_semaphore,
            url_ignore_substrings=self.web_search_params.url_ignore_substrings,
            **self.web_search_params.se_kwargs,
        )
        docs = await filter_ordinance_docs(
            docs,
            self.jurisdiction,
            self.models,
            heuristic=self.tech_specs.heuristic,
            tech=self.tech_specs.name,
            ordinance_text_collector_class=(
                self.tech_specs.ordinance_text_collector
            ),
            permitted_use_text_collector_class=(
                self.tech_specs.permitted_use_text_collector
            ),
            usage_tracker=self.usage_tracker,
            check_for_correct_jurisdiction=True,
        )
        if not docs:
            return None

        for doc in docs:
            doc.attrs["jurisdiction"] = self.jurisdiction
            doc.attrs["jurisdiction_name"] = self.jurisdiction.full_name
            doc.attrs["jurisdiction_website"] = None
            doc.attrs["compass_crawl"] = False

        await self._record_usage()
        return docs

    async def _find_documents_from_website(self):
        """Search the website for ordinance documents"""
        if self.jurisdiction_website and self.validate_user_website_input:
            await self._validate_jurisdiction_website()

        if not self.jurisdiction_website:
            website = await self._try_find_jurisdiction_website()
            if not website:
                return None
            self.jurisdiction_website = website

        docs, scrape_results = await self._try_elm_crawl()

        found_with_compass_crawl = False
        if not docs:
            docs = await self._try_compass_crawl(scrape_results)
            found_with_compass_crawl = True

        if not docs:
            return None

        for doc in docs:
            doc.attrs["jurisdiction"] = self.jurisdiction
            doc.attrs["jurisdiction_name"] = self.jurisdiction.full_name
            doc.attrs["jurisdiction_website"] = self.jurisdiction_website
            doc.attrs["compass_crawl"] = found_with_compass_crawl

        await self._record_usage()
        return docs

    async def _validate_jurisdiction_website(self):
        """Validate user input for jurisdiction website"""
        if self.jurisdiction_website is None:
            return

        self.jurisdiction_website = await get_redirected_url(
            self.jurisdiction_website, timeout=30
        )
        COMPASS_PB.update_jurisdiction_task(
            self.jurisdiction.full_name,
            description=(
                f"Validating user input website: {self.jurisdiction_website}"
            ),
        )
        model_config = self.models.get(
            LLMTasks.DOCUMENT_JURISDICTION_VALIDATION,
            self.models[LLMTasks.DEFAULT],
        )
        validator = JurisdictionWebsiteValidator(
            browser_semaphore=self.browser_semaphore,
            file_loader_kwargs=self.file_loader_kwargs_no_ocr,
            usage_tracker=self.usage_tracker,
            llm_service=model_config.llm_service,
            **model_config.llm_call_kwargs,
        )
        is_website_correct = await validator.check(
            self.jurisdiction_website, self.jurisdiction
        )
        if not is_website_correct:
            self.jurisdiction_website = None

    async def _try_find_jurisdiction_website(self):
        """Use web to try to find the main jurisdiction website"""
        COMPASS_PB.update_jurisdiction_task(
            self.jurisdiction.full_name,
            description="Searching for jurisdiction website...",
        )
        return await find_jurisdiction_website(
            self.jurisdiction,
            self.models,
            file_loader_kwargs=self.file_loader_kwargs_no_ocr,
            search_semaphore=self.search_engine_semaphore,
            browser_semaphore=self.browser_semaphore,
            usage_tracker=self.usage_tracker,
            url_ignore_substrings=(
                self.web_search_params.url_ignore_substrings
            ),
            **self.web_search_params.se_kwargs,
        )

    async def _try_elm_crawl(self):
        """Try crawling website using ELM crawler"""
        self.jurisdiction_website = await get_redirected_url(
            self.jurisdiction_website, timeout=30
        )
        out = await download_jurisdiction_ordinances_from_website(
            self.jurisdiction_website,
            heuristic=self.tech_specs.heuristic,
            keyword_points=self.tech_specs.website_url_keyword_points,
            file_loader_kwargs=self.file_loader_kwargs_no_ocr,
            crawl_semaphore=self.crawl_semaphore,
            pb_jurisdiction_name=self.jurisdiction.full_name,
            return_c4ai_results=True,
        )
        docs, scrape_results = out
        docs = await filter_ordinance_docs(
            docs,
            self.jurisdiction,
            self.models,
            heuristic=self.tech_specs.heuristic,
            tech=self.tech_specs.name,
            ordinance_text_collector_class=(
                self.tech_specs.ordinance_text_collector
            ),
            permitted_use_text_collector_class=(
                self.tech_specs.permitted_use_text_collector
            ),
            usage_tracker=self.usage_tracker,
            check_for_correct_jurisdiction=True,
        )
        return docs, scrape_results

    async def _try_compass_crawl(self, scrape_results):
        """Try to crawl the website with compass-style crawling"""
        checked_urls = set()
        for scrape_result in scrape_results:
            checked_urls.update({sub_res.url for sub_res in scrape_result})
        docs = (
            await download_jurisdiction_ordinances_from_website_compass_crawl(
                self.jurisdiction_website,
                heuristic=self.tech_specs.heuristic,
                keyword_points=self.tech_specs.website_url_keyword_points,
                file_loader_kwargs=self.file_loader_kwargs_no_ocr,
                already_visited=checked_urls,
                crawl_semaphore=self.crawl_semaphore,
                pb_jurisdiction_name=self.jurisdiction.full_name,
            )
        )
        return await filter_ordinance_docs(
            docs,
            self.jurisdiction,
            self.models,
            heuristic=self.tech_specs.heuristic,
            tech=self.tech_specs.name,
            ordinance_text_collector_class=(
                self.tech_specs.ordinance_text_collector
            ),
            permitted_use_text_collector_class=(
                self.tech_specs.permitted_use_text_collector
            ),
            usage_tracker=self.usage_tracker,
            check_for_correct_jurisdiction=True,
        )

    async def _parse_docs_for_ordinances(self, docs):
        """Parse docs (in order) for ordinances"""
        for possible_ord_doc in docs:
            doc = await self._try_extract_all_ordinances(possible_ord_doc)
            ord_count = num_ordinances_in_doc(
                doc, exclude_features=EXCLUDE_FROM_ORD_DOC_CHECK
            )
            if ord_count > 0:
                logger.debug(
                    "Found ordinances in doc from %s",
                    possible_ord_doc.attrs.get("source", "unknown source"),
                )
                return await _move_files(doc, self.jurisdiction)

        logger.debug("No ordinances found; searched %d docs", len(docs))
        return None

    async def _try_extract_all_ordinances(self, possible_ord_doc):
        """Try to extract ordinance values and permitted districts"""
        with self._tracked_progress():
            tasks = [
                asyncio.create_task(
                    self._try_extract_ordinances(possible_ord_doc, **kwargs),
                    name=self.jurisdiction.full_name,
                )
                for kwargs in self._extraction_task_kwargs
            ]

            docs = await asyncio.gather(*tasks)

        return _concat_scrape_results(docs[0])

    @property
    def _extraction_task_kwargs(self):
        """Keyword-argument pairs to pass to _try_extract_ordinances"""
        return [
            {
                "extractor_class": self.tech_specs.ordinance_text_extractor,
                "original_text_key": "ordinance_text",
                "cleaned_text_key": "cleaned_ordinance_text",
                "text_model": self.models.get(
                    LLMTasks.ORDINANCE_TEXT_EXTRACTION,
                    self.models[LLMTasks.DEFAULT],
                ),
                "parser_class": self.tech_specs.structured_ordinance_parser,
                "out_key": "ordinance_values",
                "value_model": self.models.get(
                    LLMTasks.ORDINANCE_VALUE_EXTRACTION,
                    self.models[LLMTasks.DEFAULT],
                ),
            },
            {
                "extractor_class": (
                    self.tech_specs.permitted_use_text_extractor
                ),
                "original_text_key": "permitted_use_text",
                "cleaned_text_key": "districts_text",
                "text_model": self.models.get(
                    LLMTasks.PERMITTED_USE_TEXT_EXTRACTION,
                    self.models[LLMTasks.DEFAULT],
                ),
                "parser_class": (
                    self.tech_specs.structured_permitted_use_parser
                ),
                "out_key": "permitted_district_values",
                "value_model": self.models.get(
                    LLMTasks.PERMITTED_USE_VALUE_EXTRACTION,
                    self.models[LLMTasks.DEFAULT],
                ),
            },
        ]

    async def _try_extract_ordinances(
        self,
        possible_ord_doc,
        extractor_class,
        original_text_key,
        cleaned_text_key,
        parser_class,
        out_key,
        text_model,
        value_model,
    ):
        """Try applying a single extractor to the relevant legal text"""
        logger.debug(
            "Checking for ordinances in doc from %s",
            possible_ord_doc.attrs.get("source", "unknown source"),
        )
        assert self._jsp is not None, "No progress bar set!"
        task_id = self._jsp.add_task(_TEXT_EXTRACTION_TASKS[extractor_class])
        doc = await _extract_ordinance_text(
            possible_ord_doc,
            extractor_class=extractor_class,
            original_text_key=original_text_key,
            usage_tracker=self.usage_tracker,
            model_config=text_model,
        )
        await self._record_usage()
        self._jsp.remove_task(task_id)
        out = await _extract_ordinances_from_text(
            doc,
            parser_class=parser_class,
            text_key=cleaned_text_key,
            out_key=out_key,
            usage_tracker=self.usage_tracker,
            model_config=value_model,
        )
        await self._record_usage()
        return out

    async def _record_usage(self):
        """Dump usage to file if tracker given"""
        if self.usage_tracker is None:
            return

        total_usage = await UsageUpdater.call(self.usage_tracker)
        total_cost = _compute_total_cost_from_usage(total_usage)
        COMPASS_PB.update_total_cost(total_cost, replace=True)


def _compile_tech_specs(tech):
    """Compile `TechSpec` tuple based on the user `tech` input"""
    if tech.casefold() == "wind":
        return TechSpec(
            "wind",
            WIND_QUESTION_TEMPLATES,
            WindHeuristic(),
            WindOrdinanceTextCollector,
            WindOrdinanceTextExtractor,
            WindPermittedUseDistrictsTextCollector,
            WindPermittedUseDistrictsTextExtractor,
            StructuredWindOrdinanceParser,
            StructuredWindPermittedUseDistrictsParser,
            BEST_WIND_ORDINANCE_WEBSITE_URL_KEYWORDS,
        )
    if tech.casefold() == "solar":
        return TechSpec(
            "solar",
            SOLAR_QUESTION_TEMPLATES,
            SolarHeuristic(),
            SolarOrdinanceTextCollector,
            SolarOrdinanceTextExtractor,
            SolarPermittedUseDistrictsTextCollector,
            SolarPermittedUseDistrictsTextExtractor,
            StructuredSolarOrdinanceParser,
            StructuredSolarPermittedUseDistrictsParser,
            BEST_SOLAR_ORDINANCE_WEBSITE_URL_KEYWORDS,
        )
    if tech.casefold() == "small wind":
        return TechSpec(
            "small wind",
            SMALL_WIND_QUESTION_TEMPLATES,
            SmallWindHeuristic(),
            SmallWindOrdinanceTextCollector,
            SmallWindOrdinanceTextExtractor,
            SmallWindPermittedUseDistrictsTextCollector,
            SmallWindPermittedUseDistrictsTextExtractor,
            StructuredSmallWindOrdinanceParser,
            StructuredSmallWindPermittedUseDistrictsParser,
            BEST_SMALL_WIND_ORDINANCE_WEBSITE_URL_KEYWORDS,
        )

    msg = f"Unknown tech input: {tech}"
    raise COMPASSValueError(msg)


def _setup_main_logging(log_dir, level, listener, keep_async_logs):
    """Setup main logger for catching exceptions during execution"""
    fmt = logging.Formatter(fmt="[%(asctime)s] %(levelname)s: %(message)s")
    handler = logging.FileHandler(log_dir / "main.log", encoding="utf-8")
    handler.setFormatter(fmt)
    handler.setLevel(level)
    handler.addFilter(NoLocationFilter())
    listener.addHandler(handler)

    if keep_async_logs:
        handler = logging.FileHandler(log_dir / "all.log", encoding="utf-8")
        fmt = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s - %(taskName)s: %(message)s",
        )
        handler.setFormatter(fmt)
        handler.setLevel(level)
        listener.addHandler(handler)


def _setup_folders(out_dir, log_dir=None, clean_dir=None, ofd=None, jdd=None):
    """Setup output directory folders"""
    dirs = Directories(out_dir, log_dir, clean_dir, ofd, jdd)

    if dirs.out.exists():
        msg = (
            f"Output directory '{out_dir!s}' already exists! Please specify a "
            "new directory for every COMPASS run."
        )
        raise COMPASSValueError(msg)

    dirs.make_dirs()
    return dirs


def _initialize_model_params(user_input):
    """Initialize llm caller args for models from user input"""
    if isinstance(user_input, str):
        return {LLMTasks.DEFAULT: OpenAIConfig(name=user_input)}

    caller_instances = {}
    for kwargs in user_input:
        tasks = kwargs.pop("tasks", LLMTasks.DEFAULT)
        if isinstance(tasks, str):
            tasks = [tasks]

        model_config = OpenAIConfig(**kwargs)
        for task in tasks:
            if task in caller_instances:
                msg = (
                    f"Found duplicated task: {task!r}. Please ensure each "
                    "LLM caller definition has uniquely-assigned tasks."
                )
                raise COMPASSValueError(msg)
            caller_instances[task] = model_config

    return caller_instances


def _load_jurisdictions_to_process(jurisdiction_fp):
    """Load the jurisdictions to retrieve documents for"""
    if jurisdiction_fp is None:
        logger.info("No `jurisdiction_fp` input! Loading all jurisdictions")
        return load_all_jurisdiction_info()
    return load_jurisdictions_from_fp(jurisdiction_fp)


def _configure_thread_pool_kwargs(tpe_kwargs):
    """Set thread pool workers to 5 if user didn't specify"""
    tpe_kwargs = tpe_kwargs or {}
    tpe_kwargs.setdefault("max_workers", 5)
    return tpe_kwargs


def _configure_file_loader_kwargs(file_loader_kwargs):
    """Add PDF reading coroutine to kwargs"""
    file_loader_kwargs = file_loader_kwargs or {}
    file_loader_kwargs.update({"pdf_read_coroutine": read_pdf_doc})
    return file_loader_kwargs


async def _extract_ordinance_text(
    doc, extractor_class, original_text_key, usage_tracker, model_config
):
    """Extract text pertaining to ordinance of interest"""
    llm_caller = LLMCaller(
        llm_service=model_config.llm_service,
        usage_tracker=usage_tracker,
        **model_config.llm_call_kwargs,
    )
    extractor = extractor_class(llm_caller)
    doc = await extract_ordinance_text_with_ngram_validation(
        doc,
        model_config.text_splitter,
        extractor,
        original_text_key=original_text_key,
    )
    return await _write_cleaned_text(doc)


async def _extract_ordinances_from_text(
    doc, parser_class, text_key, out_key, usage_tracker, model_config
):
    """Extract values from ordinance text"""
    parser = parser_class(
        llm_service=model_config.llm_service,
        usage_tracker=usage_tracker,
        **model_config.llm_call_kwargs,
    )
    logger.info("Extracting %s...", out_key.replace("_", " "))
    return await extract_ordinance_values(
        doc, parser, text_key=text_key, out_key=out_key
    )


async def _move_files(doc, jurisdiction):
    """Move files to output folders, if applicable"""
    ord_count = num_ordinances_in_doc(doc)
    if ord_count == 0:
        logger.info("No ordinances found for %s.", jurisdiction.full_name)
        return doc

    doc = await _move_file_to_out_dir(doc)
    doc = await _write_ord_db(doc)
    logger.info(
        "%d ordinance value(s) found for %s. Outputs are here: '%s'",
        ord_count,
        jurisdiction.full_name,
        doc.attrs["ord_db_fp"],
    )
    return doc


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


async def _record_jurisdiction_info(loc, doc, start_time, usage_tracker):
    """Record info about jurisdiction"""
    seconds_elapsed = time.monotonic() - start_time
    await JurisdictionUpdater.call(loc, doc, seconds_elapsed, usage_tracker)


def _setup_pytesseract(exe_fp):
    """Set the pytesseract command"""
    import pytesseract  # noqa: PLC0415

    logger.debug("Setting `tesseract_cmd` to %s", exe_fp)
    pytesseract.pytesseract.tesseract_cmd = exe_fp


def _concat_scrape_results(doc):
    data = [
        doc.attrs.get(key, None)
        for key in ["ordinance_values", "permitted_district_values"]
    ]
    data = [df for df in data if df is not None and not df.empty]
    if len(data) == 0:
        return doc

    if len(data) == 1:
        doc.attrs["scraped_values"] = data[0]
        return doc

    doc.attrs["scraped_values"] = pd.concat(data)
    return doc


async def _compute_total_cost():
    """Compute total cost from tracked usage"""
    total_usage = await UsageUpdater.call(None)
    if not total_usage:
        return 0

    return _compute_total_cost_from_usage(total_usage)


def _compute_total_cost_from_usage(tracked_usage):
    """Compute total cost from total tracked usage"""

    total_cost = 0
    for usage in tracked_usage.values():
        totals = usage.get("tracker_totals", {})
        for model, total_usage in totals.items():
            model_costs = LLM_COST_REGISTRY.get(model, {})
            total_cost += (
                total_usage.get("prompt_tokens", 0)
                / 1e6
                * model_costs.get("prompt", 0)
            )
            total_cost += (
                total_usage.get("response_tokens", 0)
                / 1e6
                * model_costs.get("response", 0)
            )

    return total_cost


def _add_known_doc_attrs_to_all_docs(docs, doc_infos, key):
    """Add user-defined doc attributes to all loaded docs"""
    for doc in docs:
        source_fp = doc.attrs.get(key)
        if not source_fp:
            continue

        _add_known_doc_attrs(doc, source_fp, doc_infos, key)


def _add_known_doc_attrs(doc, source_fp, doc_infos, key):
    """Add user-defined doc attributes to a loaded doc"""
    for info in doc_infos:
        if str(info[key]) == str(source_fp):
            doc.attrs.update(info)
            return
