"""Ordinance full processing logic"""

import time
import json
import asyncio
import logging
import getpass
from pathlib import Path
from functools import cached_property
from collections import namedtuple
from contextlib import AsyncExitStack, contextmanager
from datetime import datetime, timedelta, UTC

import pandas as pd
from elm.version import __version__ as elm_version

from compass import __version__ as compass_version
from compass.scripts.download import download_county_ordinance
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
)
from compass.llm import LLMCaller, OpenAIConfig
from compass.services.cpu import PDFLoader, read_pdf_doc, read_pdf_doc_ocr
from compass.services.usage import UsageTracker
from compass.services.openai import usage_from_response
from compass.services.provider import RunningAsyncServices
from compass.services.threaded import (
    TempFileCachePB,
    FileMover,
    CleanedFileWriter,
    OrdDBFileWriter,
    UsageUpdater,
    JurisdictionUpdater,
)
from compass.utilities import (
    LLM_COST_REGISTRY,
    load_all_county_info,
    load_counties_from_fp,
    extract_ord_year_from_doc_attrs,
    num_ordinances_in_doc,
    num_ordinances_dataframe,
)
from compass.utilities.enums import LLMTasks
from compass.utilities.location import County
from compass.utilities.logs import (
    LocationFileLog,
    LogListener,
    NoLocationFilter,
)
from compass.pb import COMPASS_PB


logger = logging.getLogger(__name__)
TechSpec = namedtuple(
    "TechSpec",
    [
        "questions",
        "heuristic",
        "ordinance_text_collector",
        "ordinance_text_extractor",
        "permitted_use_text_collector",
        "permitted_use_text_extractor",
        "structured_ordinance_parser",
        "structured_permitted_use_parser",
    ],
)
ProcessKwargs = namedtuple(
    "ProcessKwargs",
    [
        "file_loader_kwargs",
        "td_kwargs",
        "tpe_kwargs",
        "ppe_kwargs",
        "max_num_concurrent_jurisdictions",
    ],
    defaults=[None, None, None, None],
)
Directories = namedtuple(
    "Directories",
    ["out", "logs", "clean_files", "ordinance_files", "jurisdiction_dbs"],
)
AzureParams = namedtuple(
    "AzureParams",
    ["azure_api_key", "azure_version", "azure_endpoint"],
    defaults=[None, None, None],
)
WebSearchParams = namedtuple(
    "WebSearchParams",
    [
        "num_urls_to_check_per_jurisdiction",
        "max_num_concurrent_browsers",
        "url_ignore_substrings",
        "pytesseract_exe_fp",
    ],
    defaults=[5, 10, None, None],
)
PARSED_COLS = [
    "county",
    "state",
    "subdivision",
    "jurisdiction_type",
    "FIPS",
    "feature",
    "value",
    "units",
    "adder",
    "min_dist",
    "max_dist",
    "summary",
    "ord_year",
    "section",
    "source",
    "quantitative",
]
QUANT_OUT_COLS = PARSED_COLS[:-1]
QUAL_OUT_COLS = PARSED_COLS[:6] + PARSED_COLS[-5:-1]
_TEXT_EXTRACTION_TASKS = {
    WindOrdinanceTextExtractor: "Extracting wind ordinance text",
    WindPermittedUseDistrictsTextExtractor: (
        "Extracting wind permitted use text"
    ),
    SolarOrdinanceTextExtractor: "Extracting solar ordinance text",
    SolarPermittedUseDistrictsTextExtractor: (
        "Extracting solar permitted use text"
    ),
}


async def process_counties_with_openai(  # noqa: PLR0917, PLR0913
    out_dir,
    tech,
    jurisdiction_fp,
    model="gpt-4o",
    num_urls_to_check_per_jurisdiction=5,
    max_num_concurrent_browsers=10,
    max_num_concurrent_jurisdictions=None,
    url_ignore_substrings=None,
    file_loader_kwargs=None,
    pytesseract_exe_fp=None,
    td_kwargs=None,
    tpe_kwargs=None,
    ppe_kwargs=None,
    log_dir=None,
    clean_dir=None,
    ordinance_file_dir=None,
    jurisdiction_dbs_dir=None,
    llm_costs=None,
    log_level="INFO",
):
    """Download and extract ordinances for a list of counties

    This function scrapes ordinance documents (PDFs or HTML text) for a
    list of specified counties and processes them using one or more LLM
    models. Output files, logs, and intermediate artifacts are stored in
    configurable directories.

    Parameters
    ----------
    out_dir : path-like
        Path to the output directory. If it does not exist, it will be
        created. This directory will contain the structured ordinance
        CSV file, all downloaded ordinance documents (PDFs and HTML),
        usage metadata, and default subdirectories for logs and
        intermediate outputs (unless otherwise specified).
    tech : {"wind", "solar"}
        Label indicating which technology type is being processed.
    jurisdiction_fp : path-like
        Path to a CSV file specifying the jurisdictions to process.
        The CSV must contain two columns: "County" and "State", which
        specify the county and state names, respectively.
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
        performing Google searches. Increasing this value can speed up
        searches, but may lead to timeouts or performance issues on
        machines with limited resources. By default, ``10``.
    max_num_concurrent_jurisdictions : int, optional
        Maximum number of jurisdictions to process in parallel. Limiting
        this can help manage memory usage when dealing with a large
        number of documents. By default ``None`` (no limit).
    url_ignore_substrings : list of str, optional
        A list of substrings that, if found in any URL, will cause the
        URL to be excluded from consideration. This can be used to
        specify particular websites or entire domains to ignore. For
        example:

            url_ignore_substrings = [
                "wikipedia",
                "nrel.gov",
                "www.co.delaware.in.us/egov/documents/1649699794_0382.pdf",
            ]

        The above configuration would ignore all `wikipedia` articles,
        all websites on the NREL domain, and the specific file located
        at `www.co.delaware.in.us/egov/documents/1649699794_0382.pdf`.
        By default, ``None``.
    file_loader_kwargs : dict, optional
        Dictionary of keyword arguments pairs to initialize
        :class:`elm.web.file_loader.AsyncFileLoader`. If found, the
        "pw_launch_kwargs" key in these will also be used to initialize
        the :class:`elm.web.search.google.PlaywrightGoogleLinkSearch`
        used for the google URL search. By default, ``None``.
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

    Returns
    -------
    seconds_elapsed : float
        Total time taken to complete the processing, in seconds.
    cost : float
        Total estimated cost for the LLM calls made during this run. If
        no cost info is provided, this will be 0.
    output_directory : path-like
        Path to output directory containing data.
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
        file_loader_kwargs,
        td_kwargs,
        tpe_kwargs,
        ppe_kwargs,
        max_num_concurrent_jurisdictions,
    )
    wsp = WebSearchParams(
        num_urls_to_check_per_jurisdiction,
        max_num_concurrent_browsers,
        url_ignore_substrings,
        pytesseract_exe_fp,
    )
    models = _initialize_model_params(model)
    runner = _COMPASSRunner(
        dirs=dirs,
        log_listener=log_listener,
        tech=tech,
        models=models,
        web_search_params=wsp,
        process_kwargs=pk,
        log_level=log_level,
    )
    async with log_listener as ll:
        _setup_main_logging(dirs.logs, log_level, ll)
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
        log_level="INFO",
    ):
        self.dirs = dirs
        self.log_listener = log_listener
        self.tech = tech
        self.models = models
        self.web_search_params = web_search_params or WebSearchParams()
        self.process_kwargs = process_kwargs or ProcessKwargs()
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
        """dict: Keyword arguments for `AsyncFileLoader`"""
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
    def tpe_kwargs(self):
        """dict: Keyword arguments for `ThreadPoolExecutor`"""
        return _configure_thread_pool_kwargs(self.process_kwargs.tpe_kwargs)

    @cached_property
    def _base_services(self):
        """list: List of required services to run for processing"""
        return [
            TempFileCachePB(
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
        ]

    async def run(self, jurisdiction_fp):
        """Run COMPASS for a set of jurisdictions

        Parameters
        ----------
        jurisdiction_fp : path-like
            Path to CSV file containing the jurisdictions to search.

        Returns
        -------
        pd.DataFrame
            DataFrame containing scraped ordinance values (could be
            empty if no ordinances found).
        """
        jurisdictions = _load_counties_to_process(jurisdiction_fp)

        num_jurisdictions = len(jurisdictions)
        COMPASS_PB.create_main_task(num_jurisdictions=num_jurisdictions)
        start_date = datetime.now(UTC).isoformat()
        start_time = time.monotonic()

        doc_infos, total_cost = await self._run_all(jurisdictions)

        db, num_docs_found = _doc_infos_to_db(doc_infos)
        _save_db(db, self.dirs.out)
        total_time = _save_run_meta(
            self.dirs,
            self.tech,
            start_time,
            start_date,
            num_jurisdictions_searched=num_jurisdictions,
            num_jurisdictions_found=num_docs_found,
            total_cost=total_cost,
            models=self.models,
        )
        return total_time, total_cost, self.dirs.out

    async def _run_all(self, jurisdictions):
        """Process all counties with running services"""
        services = [model.llm_service for model in set(self.models.values())]
        services += self._base_services
        _ = self.file_loader_kwargs  # init loader kwargs once
        logger.info("Processing %d jurisdiction(s)", len(jurisdictions))
        async with RunningAsyncServices(services):
            tasks = []
            for __, row in jurisdictions.iterrows():
                county, state, fips = row[["County", "State", "FIPS"]]
                location = County(
                    county.strip(), state=state.strip(), fips=fips
                )
                usage_tracker = UsageTracker(
                    location.full_name, usage_from_response
                )
                task = asyncio.create_task(
                    self._processed_jurisdiction_info_with_pb(
                        location, usage_tracker=usage_tracker
                    ),
                    name=location.full_name,
                )
                tasks.append(task)
            doc_infos = await asyncio.gather(*tasks)
            total_cost = await _compute_total_cost()

        return doc_infos, total_cost

    async def _processed_jurisdiction_info_with_pb(
        self, county, *args, **kwargs
    ):
        """Process county and update progress bar"""
        async with self.jurisdiction_semaphore:
            with COMPASS_PB.jurisdiction_prog_bar(county.full_name):
                return await self._processed_jurisdiction_info(
                    county, *args, **kwargs
                )

    async def _processed_jurisdiction_info(self, *args, **kwargs):
        """Drop `doc` from RAM and only keep enough info to re-build"""

        doc = await self._process_jurisdiction_with_logging(*args, **kwargs)

        if doc is None or isinstance(doc, Exception):
            return None

        keys = ["source", "date", "location", "ord_db_fp"]
        doc_info = {key: doc.attrs.get(key) for key in keys}
        logger.debug("Saving the following doc info:\n%s", str(doc_info))
        return doc_info

    async def _process_jurisdiction_with_logging(
        self, county, usage_tracker=None
    ):
        """Retrieve ordinance document with async logs"""
        with LocationFileLog(
            self.log_listener,
            self.dirs.logs,
            location=county.full_name,
            level=self.log_level,
        ):
            task = asyncio.create_task(
                _SingleJurisdictionRunner(
                    self.tech,
                    county,
                    self.models,
                    self.web_search_params,
                    self.file_loader_kwargs,
                    self.browser_semaphore,
                    usage_tracker=usage_tracker,
                ).run(),
                name=county.full_name,
            )
            try:
                doc, *__ = await asyncio.gather(task)
            except KeyboardInterrupt:
                raise
            except Exception:
                msg = "Encountered error while processing %s:"
                logger.exception(msg, county.full_name)
                doc = None

            return doc


class _SingleJurisdictionRunner:
    """Helper class to process a single jurisdiction"""

    def __init__(
        self,
        tech,
        jurisdiction,
        models,
        web_search_params,
        file_loader_kwargs,
        browser_semaphore,
        usage_tracker=None,
    ):
        self.tech_specs = _compile_tech_specs(tech)
        self.jurisdiction = jurisdiction
        self.models = models
        self.web_search_params = web_search_params
        self.file_loader_kwargs = file_loader_kwargs
        self.browser_semaphore = browser_semaphore
        self.usage_tracker = usage_tracker
        self._jsp = None

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
        doc = await self._run()
        await self._record_usage()
        await _record_jurisdiction_info(
            self.jurisdiction, doc, start_time, self.usage_tracker
        )
        return doc

    async def _run(self):
        """Search for docs and parse them for ordinances"""
        docs = await self._find_documents_with_location_attr()
        if docs is None:
            return None

        COMPASS_PB.update_jurisdiction_task(
            self.jurisdiction.full_name,
            description="Extracting structured data...",
        )
        return await self._parse_docs_for_ordinances(docs)

    async def _find_documents_with_location_attr(self):
        """Search the web for an ordinance document and construct it"""
        docs = await download_county_ordinance(
            self.tech_specs.questions,
            self.jurisdiction,
            self.models,
            heuristic=self.tech_specs.heuristic,
            ordinance_text_collector_class=(
                self.tech_specs.ordinance_text_collector
            ),
            permitted_use_text_collector_class=(
                self.tech_specs.permitted_use_text_collector
            ),
            num_urls=self.web_search_params.num_urls_to_check_per_jurisdiction,
            file_loader_kwargs=self.file_loader_kwargs,
            browser_semaphore=self.browser_semaphore,
            url_ignore_substrings=self.web_search_params.url_ignore_substrings,
            usage_tracker=self.usage_tracker,
        )
        if docs is None:
            return None

        for doc in docs:
            doc.attrs["location"] = self.jurisdiction
            doc.attrs["location_name"] = self.jurisdiction.full_name

        await self._record_usage()
        return docs

    async def _parse_docs_for_ordinances(self, docs):
        """Parse docs (in order) for ordinances"""
        for possible_ord_doc in docs:
            doc = await self._try_extract_all_ordinances(possible_ord_doc)
            if num_ordinances_in_doc(doc) > 0:
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
                for kwargs in self.extraction_task_kwargs
            ]

            docs = await asyncio.gather(*tasks)

        return _concat_scrape_results(docs[0])

    @property
    def extraction_task_kwargs(self):
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
            WIND_QUESTION_TEMPLATES,
            WindHeuristic(),
            WindOrdinanceTextCollector,
            WindOrdinanceTextExtractor,
            WindPermittedUseDistrictsTextCollector,
            WindPermittedUseDistrictsTextExtractor,
            StructuredWindOrdinanceParser,
            StructuredWindPermittedUseDistrictsParser,
        )
    if tech.casefold() == "solar":
        return TechSpec(
            SOLAR_QUESTION_TEMPLATES,
            SolarHeuristic(),
            SolarOrdinanceTextCollector,
            SolarOrdinanceTextExtractor,
            SolarPermittedUseDistrictsTextCollector,
            SolarPermittedUseDistrictsTextExtractor,
            StructuredSolarOrdinanceParser,
            StructuredSolarPermittedUseDistrictsParser,
        )

    msg = f"Unknown tech input: {tech}"
    raise COMPASSValueError(msg)


def _setup_main_logging(log_dir, level, listener):
    """Setup main logger for catching exceptions during execution"""
    handler = logging.FileHandler(log_dir / "main.log", encoding="utf-8")
    handler.setLevel(level)
    handler.addFilter(NoLocationFilter())
    listener.addHandler(handler)


def _setup_folders(out_dir, log_dir=None, clean_dir=None, ofd=None, jdd=None):
    """Setup output directory folders"""
    out_dir = _full_path(out_dir)
    out_folders = Directories(
        out_dir,
        _full_path(log_dir) if log_dir else out_dir / "logs",
        _full_path(clean_dir) if clean_dir else out_dir / "cleaned_text",
        _full_path(ofd) if ofd else out_dir / "ordinance_files",
        _full_path(jdd) if jdd else out_dir / "jurisdiction_dbs",
    )
    for folder in out_folders:
        folder.mkdir(exist_ok=True, parents=True)
    return out_folders


def _full_path(in_path):
    """Expand and resolve input path"""
    return Path(in_path).expanduser().resolve()


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


def _load_counties_to_process(county_fp):
    """Load the counties to retrieve documents for"""
    if county_fp is None:
        logger.info("No `county_fp` input! Loading all counties")
        return load_all_county_info()
    return load_counties_from_fp(county_fp)


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


async def _move_files(doc, county):
    """Move files to output folders, if applicable"""
    ord_count = num_ordinances_in_doc(doc)
    if ord_count == 0:
        logger.info("No ordinances found for %s.", county.full_name)
        return doc

    doc = await _move_file_to_out_dir(doc)
    doc = await _write_ord_db(doc)
    logger.info(
        "%d ordinance value(s) found for %s. Outputs are here: '%s'",
        ord_count,
        county.full_name,
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


async def _record_jurisdiction_info(county, doc, start_time, usage_tracker):
    """Record info about jurisdiction"""
    seconds_elapsed = time.monotonic() - start_time
    await JurisdictionUpdater.call(county, doc, seconds_elapsed, usage_tracker)


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


def _doc_infos_to_db(doc_infos):
    """Convert list of docs to output database"""
    db = []
    for doc_info in doc_infos:
        if doc_info is None:
            continue

        ord_db_fp = doc_info.get("ord_db_fp")
        if ord_db_fp is None:
            continue

        ord_db = pd.read_csv(ord_db_fp)

        if num_ordinances_dataframe(ord_db) == 0:
            continue

        results = _db_results(ord_db, doc_info)
        results = _formatted_db(results)
        db.append(results)

    if not db:
        return pd.DataFrame(columns=PARSED_COLS), 0

    logger.info("Compiling final database for %d jurisdiction(s)", len(db))
    num_jurisdictions_found = len(db)
    db = pd.concat([df.dropna(axis=1, how="all") for df in db], axis=0)
    db = _empirical_adjustments(db)
    return _formatted_db(db), num_jurisdictions_found


def _db_results(results, doc_info):
    """Extract results from doc attrs to DataFrame"""

    results["source"] = doc_info.get("source")
    results["ord_year"] = extract_ord_year_from_doc_attrs(doc_info)

    location = doc_info["location"]
    results["FIPS"] = location.fips
    results["county"] = location.name
    results["state"] = location.state
    return results


def _empirical_adjustments(db):
    """Post-processing adjustments based on empirical observations

    Current adjustments include:

        - Limit adder to max of 250 ft.
            - Chat GPT likes to report large values here, but in
            practice all values manually observed in ordinance documents
            are below 250 ft. If large value is detected, assume it's an
            error on Chat GPT's part and remove it.

    """
    if "adder" in db.columns:
        db.loc[db["adder"] > 250, "adder"] = None  # noqa: PLR2004
    return db


def _formatted_db(db):
    """Format DataFrame for output"""
    for col in PARSED_COLS:
        if col not in db.columns:
            db[col] = None

    db["quantitative"] = db["quantitative"].astype("boolean").fillna(True)
    return db[PARSED_COLS]


def _save_db(db, out_dir):
    """Split DB into qualitative vs quantitative and save to disk"""
    if db.empty:
        return
    qual_db = db[~db["quantitative"]][QUAL_OUT_COLS]
    quant_db = db[db["quantitative"]][QUANT_OUT_COLS]
    qual_db.to_csv(out_dir / "qualitative_ordinances.csv", index=False)
    quant_db.to_csv(out_dir / "quantitative_ordinances.csv", index=False)


def _save_run_meta(
    dirs,
    tech,
    start_time,
    start_date,
    num_jurisdictions_searched,
    num_jurisdictions_found,
    total_cost,
    models,
):
    """Write out meta information about ordinance collection run"""
    end_date = datetime.now(UTC).isoformat()
    end_time = time.monotonic()
    seconds_elapsed = end_time - start_time

    try:
        username = getpass.getuser()
    except OSError:
        username = "Unknown"

    meta_data = {
        "username": username,
        "versions": {"elm": elm_version, "compass": compass_version},
        "technology": tech,
        "models": _extract_model_info_from_all_models(models),
        "time_start_utc": start_date,
        "time_end_utc": end_date,
        "total_time": seconds_elapsed,
        "total_time_string": str(timedelta(seconds=seconds_elapsed)),
        "num_jurisdictions_searched": num_jurisdictions_searched,
        "num_jurisdictions_found": num_jurisdictions_found,
        "cost": total_cost or None,
        "manifest": {},
    }
    manifest = {
        "LOG_DIR": dirs.logs,
        "CLEAN_FILE_DIR": dirs.clean_files,
        "JURISDICTION_DBS_DIR": dirs.jurisdiction_dbs,
        "ORDINANCE_FILES_DIR": dirs.ordinance_files,
        "USAGE_FILE": dirs.out / "usage.json",
        "JURISDICTION_FILE": dirs.out / "jurisdictions.json",
        "QUANT_DATA_FILE": dirs.out / "quantitative_ordinances.csv",
        "QUAL_DATA_FILE": dirs.out / "quantitative_ordinances.csv",
    }
    for name, file_path in manifest.items():
        if file_path.exists():
            meta_data["manifest"][name] = str(file_path.relative_to(dirs.out))
        else:
            meta_data["manifest"][name] = None

    meta_data["manifest"]["META_FILE"] = "meta.json"
    with (dirs.out / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta_data, fh, indent=4)

    return seconds_elapsed


def _extract_model_info_from_all_models(models):
    """Group model info together"""
    models_to_tasks = {}
    for task, caller_args in models.items():
        models_to_tasks.setdefault(caller_args, []).append(task)

    return [
        {
            "name": caller_args.name,
            "llm_call_kwargs": caller_args.llm_call_kwargs,
            "llm_service_rate_limit": caller_args.llm_service_rate_limit,
            "text_splitter_chunk_size": caller_args.text_splitter_chunk_size,
            "text_splitter_chunk_overlap": (
                caller_args.text_splitter_chunk_overlap
            ),
            "client_type": caller_args.client_type,
            "tasks": tasks,
        }
        for caller_args, tasks in models_to_tasks.items()
    ]


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
