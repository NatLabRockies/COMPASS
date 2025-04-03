"""Ordinance full processing logic"""

import time
import json
import asyncio
import logging
import getpass
from pathlib import Path
from functools import partial
from collections import namedtuple
from contextlib import AsyncExitStack
from datetime import datetime, timedelta, UTC

import openai
import pandas as pd
from elm import ApiBase
from elm.version import __version__ as elm_version
from elm.utilities import validate_azure_api_params
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
from compass.llm import LLMCaller
from compass.services.cpu import PDFLoader, read_pdf_doc, read_pdf_doc_ocr
from compass.services.usage import UsageTracker
from compass.services.openai import OpenAIService, usage_from_response
from compass.services.provider import RunningAsyncServices
from compass.services.threaded import (
    TempFileCache,
    FileMover,
    CleanedFileWriter,
    OrdDBFileWriter,
    UsageUpdater,
    JurisdictionUpdater,
)
from compass.utilities import (
    RTS_SEPARATORS,
    load_all_county_info,
    load_counties_from_fp,
    extract_ord_year_from_doc_attrs,
    num_ordinances_in_doc,
    num_ordinances_dataframe,
)
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
LLMParseArgs = namedtuple(
    "LLMParseArgs",
    [
        "model",
        "llm_call_kwargs",
        "llm_service_rate_limit",
        "text_splitter_chunk_size",
        "text_splitter_chunk_overlap",
    ],
    defaults=["gpt-4", None, 4000, 10_000, 1000],
)
WebSearchParams = namedtuple(
    "WebSearchParams",
    [
        "num_urls_to_check_per_county",
        "max_num_concurrent_browsers",
        "pytesseract_exe_fp",
    ],
    defaults=[5, 10, None],
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
    jurisdiction_fp=None,
    model="gpt-4o",
    azure_api_key=None,
    azure_version=None,
    azure_endpoint=None,
    llm_call_kwargs=None,
    llm_service_rate_limit=10_000,
    text_splitter_chunk_size=10_000,
    text_splitter_chunk_overlap=500,
    num_urls_to_check_per_county=5,
    max_num_concurrent_browsers=10,
    max_num_concurrent_jurisdictions=None,
    file_loader_kwargs=None,
    pytesseract_exe_fp=None,
    td_kwargs=None,
    tpe_kwargs=None,
    ppe_kwargs=None,
    log_dir=None,
    clean_dir=None,
    ordinance_file_dir=None,
    county_dbs_dir=None,
    log_level="INFO",
):
    """Download and extract ordinances for a list of counties

    Parameters
    ----------
    out_dir : path-like
        Path to output directory. This directory will be created if it
        does not exist. This directory will contain the structured
        ordinance output CSV as well as all of the scraped ordinance
        documents (PDFs and HTML text files). Usage information and
        default options for log/clean directories will also be stored
        here.
    jurisdiction_fp : path-like, optional
        Path to CSV file containing a list of jurisdictions to extract
        ordinance information for. This CSV should have "County" and
        "State" columns that contains the county and state names.
        By default, ``None``, which runs the extraction for all known
        jurisdictions (this is untested and not currently recommended).
    model : str, optional
        Name of LLM model to perform scraping. By default, ``"gpt-4"``.
    azure_api_key : str, optional
        Azure OpenAI API key. By default, ``None``, which pulls the key
        from the environment variable ``AZURE_OPENAI_API_KEY`` instead.
    azure_version : str, optional
        Azure OpenAI API version. By default, ``None``, which pulls the
        version from the environment variable ``AZURE_OPENAI_VERSION``
        instead.
    azure_endpoint : str, optional
        Azure OpenAI API endpoint. By default, ``None``, which pulls the
        endpoint from the environment variable ``AZURE_OPENAI_ENDPOINT``
        instead.
    llm_call_kwargs : dict, optional
        Keyword-value pairs used to initialize an
        `compass.llm.LLMCaller` instance. By default, ``None``.
    llm_service_rate_limit : int, optional
        Token rate limit (i.e. tokens per minute) of LLM service being
        used (OpenAI). By default, ``10_000``.
    text_splitter_chunk_size : int, optional
        Chunk size used to split the ordinance text. Parsing is
        performed on each individual chunk. Units are in token count of
        the model in charge of parsing ordinance text. Keeping this
        value low can help reduce token usage since (free) heuristics
        checks may be able to throw away irrelevant chunks of text
        before passing to the LLM. By default, ``10000``.
    text_splitter_chunk_overlap : int, optional
        Overlap of consecutive chunks of the ordinance text. Parsing is
        performed on each individual chunk. Units are in token count of
        the model in charge of parsing ordinance text.
        By default, ``1000``.
    num_urls_to_check_per_county : int, optional
        Number of unique Google search result URL's to check for
        ordinance document. By default, ``5``.
    max_num_concurrent_browsers : int, optional
        Number of unique concurrent browser instances to open when
        performing Google search. Setting this number too high on a
        machine with limited processing can lead to increased timeouts
        and therefore decreased quality of Google search results.
        By default, ``10``.
    max_num_concurrent_jurisdictions : int, optional
        Number of unique jurisdictions to process concurrently. Setting
        this value limits the number of documents stored in RAM at a
        time and can therefore help avoid memory issues.
        By default, ``None``, which does not limit the number of
        jurisdictions processed concurrently.
    pytesseract_exe_fp : path-like, optional
        Path to pytesseract executable. If this option is specified, OCR
        parsing for PDf files will be enabled via pytesseract.
        By default, ``None``.
    td_kwargs : dict, optional
        Keyword-value argument pairs to pass to
        :class:`tempfile.TemporaryDirectory`. The temporary directory is
        used to store files downloaded from the web that are still being
        parsed for ordinance information. By default, ``None``.
    tpe_kwargs : dict, optional
        Keyword-value argument pairs to pass to
        :class:`concurrent.futures.ThreadPoolExecutor`. The thread pool
        executor is used to run I/O intensive tasks like writing to a
        log file. By default, ``None``.
    ppe_kwargs : dict, optional
        Keyword-value argument pairs to pass to
        :class:`concurrent.futures.ProcessPoolExecutor`. The process
        pool executor is used to run CPU intensive tasks like loading
        a PDF file. By default, ``None``.
    log_dir : path-like, optional
        Path to directory for log files. This directory will be created
        if it does not exist. By default, ``None``, which
        creates a ``logs`` folder in the output directory for the
        county-specific log files.
    clean_dir : path-like, optional
        Path to directory for cleaned ordinance text output. This
        directory will be created if it does not exist. By default,
        ``None``, which creates a ``clean`` folder in the output
        directory for the cleaned ordinance text files.
    ordinance_file_dir : path-like, optional
        Path to directory for individual county ordinance file outputs.
        This directory will be created if it does not exist.
        By default, ``None``, which creates a ``county_ord_files``
        folder in the output directory.
    county_dbs_dir : path-like, optional
        Path to directory for individual county ordinance database
        outputs. This directory will be created if it does not exist.
        By default, ``None``, which creates a ``county_dbs`` folder in
        the output directory.
    log_level : str, optional
        Log level to set for county retrieval and parsing loggers.
        By default, ``"INFO"``.

    Returns
    -------
    pd.DataFrame
        DataFrame of parsed ordinance information. This file will also
        be stored in the output directory under "wind_db.csv".
    """
    log_listener = LogListener(["compass", "elm"], level=log_level)
    dirs = _setup_folders(
        out_dir,
        log_dir=log_dir,
        clean_dir=clean_dir,
        ofd=ordinance_file_dir,
        cdd=county_dbs_dir,
    )
    ap = AzureParams(
        *validate_azure_api_params(
            azure_api_key, azure_version, azure_endpoint
        )
    )
    pk = ProcessKwargs(
        file_loader_kwargs,
        td_kwargs,
        tpe_kwargs,
        ppe_kwargs,
        max_num_concurrent_jurisdictions,
    )
    wsp = WebSearchParams(
        num_urls_to_check_per_county,
        max_num_concurrent_browsers,
        pytesseract_exe_fp,
    )
    lpa = LLMParseArgs(
        model,
        llm_call_kwargs,
        llm_service_rate_limit,
        text_splitter_chunk_size,
        text_splitter_chunk_overlap,
    )

    async with log_listener as ll:
        _setup_main_logging(dirs.logs, log_level, ll)
        return await _process_with_logs(
            dirs=dirs,
            log_listener=ll,
            azure_params=ap,
            tech=tech,
            jurisdiction_fp=jurisdiction_fp,
            llm_parse_args=lpa,
            web_search_params=wsp,
            process_kwargs=pk,
            log_level=log_level,
        )


async def _process_with_logs(  # noqa: PLR0914
    dirs,
    log_listener,
    azure_params,
    tech,
    jurisdiction_fp=None,
    llm_parse_args=None,
    web_search_params=None,
    process_kwargs=None,
    log_level="INFO",
):
    """Process counties with logging enabled."""
    counties = _load_counties_to_process(jurisdiction_fp)
    lpa = llm_parse_args or LLMParseArgs()
    wsp = web_search_params or WebSearchParams()
    process_kwargs = process_kwargs or ProcessKwargs()

    tpe_kwargs = _configure_thread_pool_kwargs(process_kwargs.tpe_kwargs)
    file_loader_kwargs = _configure_file_loader_kwargs(
        process_kwargs.file_loader_kwargs
    )
    if wsp.pytesseract_exe_fp is not None:
        _setup_pytesseract(wsp.pytesseract_exe_fp)
        file_loader_kwargs.update({"pdf_ocr_read_coroutine": read_pdf_doc_ocr})

    text_splitter = RecursiveCharacterTextSplitter(
        RTS_SEPARATORS,
        chunk_size=lpa.text_splitter_chunk_size,
        chunk_overlap=lpa.text_splitter_chunk_overlap,
        length_function=partial(ApiBase.count_tokens, model=lpa.model),
        is_separator_regex=True,
    )
    client = openai.AsyncAzureOpenAI(
        api_key=azure_params.azure_api_key,
        api_version=azure_params.azure_version,
        azure_endpoint=azure_params.azure_endpoint,
    )
    if tech.casefold() == "wind":
        tech_specs = TechSpec(
            WIND_QUESTION_TEMPLATES,
            WindHeuristic(),
            WindOrdinanceTextCollector,
            WindOrdinanceTextExtractor,
            WindPermittedUseDistrictsTextCollector,
            WindPermittedUseDistrictsTextExtractor,
            StructuredWindOrdinanceParser,
            StructuredWindPermittedUseDistrictsParser,
        )
    elif tech.casefold() == "solar":
        tech_specs = TechSpec(
            SOLAR_QUESTION_TEMPLATES,
            SolarHeuristic(),
            SolarOrdinanceTextCollector,
            SolarOrdinanceTextExtractor,
            SolarPermittedUseDistrictsTextCollector,
            SolarPermittedUseDistrictsTextExtractor,
            StructuredSolarOrdinanceParser,
            StructuredSolarPermittedUseDistrictsParser,
        )
    else:
        msg = f"Unknown tech input: {tech}"
        raise COMPASSValueError(msg)

    llm_service = OpenAIService(
        client, lpa.model, rate_limit=lpa.llm_service_rate_limit
    )

    services = [
        llm_service,
        TempFileCache(
            td_kwargs=process_kwargs.td_kwargs, tpe_kwargs=tpe_kwargs
        ),
        FileMover(dirs.ordinance_files, tpe_kwargs=tpe_kwargs),
        CleanedFileWriter(dirs.clean_files, tpe_kwargs=tpe_kwargs),
        OrdDBFileWriter(dirs.jurisdiction_dbs, tpe_kwargs=tpe_kwargs),
        UsageUpdater(dirs.out / "usage.json", tpe_kwargs=tpe_kwargs),
        JurisdictionUpdater(
            dirs.out / "jurisdictions.json", tpe_kwargs=tpe_kwargs
        ),
        PDFLoader(**(process_kwargs.ppe_kwargs or {})),
    ]

    browser_semaphore = (
        asyncio.Semaphore(wsp.max_num_concurrent_browsers)
        if wsp.max_num_concurrent_browsers
        else None
    )
    jurisdiction_semaphore = (
        asyncio.Semaphore(process_kwargs.max_num_concurrent_jurisdictions)
        if process_kwargs.max_num_concurrent_jurisdictions
        else None
    )

    COMPASS_PB.create_main_task(num_jurisdictions=len(counties))
    start_date = datetime.now(UTC).isoformat()
    start_time = time.monotonic()
    async with RunningAsyncServices(services):
        tasks = []
        trackers = []
        for __, row in counties.iterrows():
            county, state, fips = row[["County", "State", "FIPS"]]
            location = County(county.strip(), state=state.strip(), fips=fips)
            usage_tracker = UsageTracker(
                location.full_name, usage_from_response
            )
            trackers.append(usage_tracker)
            task = asyncio.create_task(
                _processed_county_info_with_pb(
                    log_listener,
                    dirs.logs,
                    location,
                    text_splitter,
                    tech_specs,
                    num_urls=wsp.num_urls_to_check_per_county,
                    file_loader_kwargs=file_loader_kwargs,
                    browser_semaphore=browser_semaphore,
                    jurisdiction_semaphore=jurisdiction_semaphore,
                    level=log_level,
                    llm_service=llm_service,
                    usage_tracker=usage_tracker,
                    **(lpa.llm_call_kwargs or {}),
                ),
                name=location.full_name,
            )
            tasks.append(task)
        doc_infos = await asyncio.gather(*tasks)

    db, num_docs_found = _doc_infos_to_db(doc_infos)
    _save_db(db, dirs.out)
    _save_run_meta(
        dirs,
        tech,
        start_time,
        start_date,
        num_jurisdictions_searched=len(counties),
        num_jurisdictions_found=num_docs_found,
        llm_parse_args=lpa,
    )
    return db


def _setup_main_logging(log_dir, level, listener):
    """Setup main logger for catching exceptions during execution"""
    handler = logging.FileHandler(log_dir / "main.log", encoding="utf-8")
    handler.setLevel(level)
    handler.addFilter(NoLocationFilter())
    listener.addHandler(handler)


def _setup_folders(out_dir, log_dir=None, clean_dir=None, ofd=None, cdd=None):
    """Setup output directory folders"""
    out_dir = Path(out_dir)
    out_folders = Directories(
        out_dir,
        Path(log_dir) if log_dir else out_dir / "logs",
        Path(clean_dir) if clean_dir else out_dir / "cleaned_text",
        Path(ofd) if ofd else out_dir / "ordinance_files",
        Path(cdd) if cdd else out_dir / "jurisdiction_dbs",
    )
    for folder in out_folders:
        folder.mkdir(exist_ok=True, parents=True)
    return out_folders


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


async def _processed_county_info_with_pb(
    listener, log_dir, county, *args, **kwargs
):
    """Process county and update progress bar"""
    with COMPASS_PB.jurisdiction_prog_bar(county.full_name):
        return await _processed_county_info(
            listener, log_dir, county, *args, **kwargs
        )


async def _processed_county_info(
    listener,
    log_dir,
    county,
    text_splitter,
    tech_specs,
    num_urls=5,
    file_loader_kwargs=None,
    browser_semaphore=None,
    jurisdiction_semaphore=None,
    level="INFO",
    **kwargs,
):
    """Drop `doc` from RAM and only keep enough info to re-build doc"""
    if jurisdiction_semaphore is None:
        jurisdiction_semaphore = AsyncExitStack()

    async with jurisdiction_semaphore:
        doc = await process_county_with_logging(
            listener,
            log_dir,
            county,
            text_splitter,
            tech_specs,
            num_urls=num_urls,
            file_loader_kwargs=file_loader_kwargs,
            browser_semaphore=browser_semaphore,
            level=level,
            **kwargs,
        )

    if doc is None or isinstance(doc, Exception):
        return None

    keys = ["source", "date", "location", "ord_db_fp"]
    doc_info = {key: doc.attrs.get(key) for key in keys}
    logger.debug("Saving the following doc info:\n%s", str(doc_info))
    return doc_info


async def process_county_with_logging(
    listener,
    log_dir,
    county,
    text_splitter,
    tech_specs,
    num_urls=5,
    file_loader_kwargs=None,
    browser_semaphore=None,
    level="INFO",
    **kwargs,
):
    """Retrieve ordinance document for a single county with async logs

    Parameters
    ----------
    listener : compass.utilities.logs.LogListener
        Active ``LogListener`` instance that can be passed to
        :class:`compass.utilities.logs.LocationFileLog`.
    log_dir : path-like
        Path to output directory to contain log file.
    county : compass.utilities.location.Location
        County to retrieve ordinance document for.
    text_splitter : obj, optional
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    num_urls : int, optional
        Number of unique Google search result URL's to check for
        ordinance document. By default, ``5``.
    file_loader_kwargs : dict, optional
        Dictionary of keyword-argument pairs to initialize
        :class:`elm.web.file_loader.AsyncFileLoader` with. The
        "pw_launch_kwargs" key in these will also be used to initialize
        the :class:`elm.web.search.google.PlaywrightGoogleLinkSearch`
        used for the google URL search. By default, ``None``.
    browser_semaphore : asyncio.Semaphore, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers open concurrently. If ``None``, no limits
        are applied. By default, ``None``.
    level : str, optional
        Log level to set for retrieval logger. By default, ``"INFO"``.
    **kwargs
        Keyword-value pairs used to initialize an
        `compass.llm.LLMCaller` instance.

    Returns
    -------
    elm.web.document.BaseDocument | None
        Document instance for the ordinance document, or ``None`` if no
        document was found. Extracted ordinance information is stored in
        the document's ``attrs`` attribute.
    """
    with LocationFileLog(
        listener, log_dir, location=county.full_name, level=level
    ):
        task = asyncio.create_task(
            process_county(
                tech_specs,
                county,
                text_splitter,
                num_urls=num_urls,
                file_loader_kwargs=file_loader_kwargs,
                browser_semaphore=browser_semaphore,
                **kwargs,
            ),
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


async def process_county(
    tech_specs,
    county,
    text_splitter,
    num_urls=5,
    file_loader_kwargs=None,
    browser_semaphore=None,
    **kwargs,
):
    """Download and parse ordinance document for a single county.

    Parameters
    ----------
    county : compass.utilities.location.Location
        County to retrieve ordinance document for.
    text_splitter : obj, optional
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    num_urls : int, optional
        Number of unique Google search result URL's to check for
        ordinance document. By default, ``5``.
    file_loader_kwargs : dict, optional
        Dictionary of keyword-argument pairs to initialize
        :class:`elm.web.file_loader.AsyncFileLoader` with. The
        "pw_launch_kwargs" key in these will also be used to initialize
        the :class:`elm.web.search.google.PlaywrightGoogleLinkSearch`
        used for the google URL search. By default, ``None``.
    browser_semaphore : asyncio.Semaphore, optional
        Semaphore instance that can be used to limit the number of
        playwright browsers open concurrently. If ``None``, no limits
        are applied. By default, ``None``.
    **kwargs
        Keyword-value pairs used to initialize an
        `compass.llm.LLMCaller` instance.

    Returns
    -------
    elm.web.document.BaseDocument | None
        Document instance for the ordinance document, or ``None`` if no
        document was found. Extracted ordinance information is stored in
        the document's ``attrs`` attribute.
    """
    start_time = time.monotonic()
    docs = await _find_documents_with_location_attr(
        tech_specs,
        county,
        text_splitter,
        num_urls=num_urls,
        file_loader_kwargs=file_loader_kwargs,
        browser_semaphore=browser_semaphore,
        **kwargs,
    )
    if docs is None:
        await _record_usage(**kwargs)
        await _record_jurisdiction_info(
            county, doc=None, start_time=start_time
        )
        return None

    COMPASS_PB.update_jurisdiction_task(
        county.full_name, description="Extracting structured data..."
    )
    for possible_ord_doc in docs:
        doc = await _try_extract_all_ordinances(
            possible_ord_doc, text_splitter, tech_specs, county, **kwargs
        )
        if num_ordinances_in_doc(doc) > 0:
            logger.debug(
                "Found ordinances in doc from %s",
                possible_ord_doc.attrs.get("source", "unknown source"),
            )
            break

    doc = await _move_files(doc, county)
    await _record_usage(**kwargs)
    await _record_jurisdiction_info(county, doc, start_time)
    return doc


async def _find_documents_with_location_attr(
    tech_specs,
    county,
    text_splitter,
    num_urls=5,
    file_loader_kwargs=None,
    browser_semaphore=None,
    **kwargs,
):
    """Search the web for an ordinance document and construct it"""
    docs = await download_county_ordinance(
        tech_specs.questions,
        county,
        text_splitter,
        heuristic=tech_specs.heuristic,
        ordinance_text_collector_class=tech_specs.ordinance_text_collector,
        permitted_use_text_collector_class=(
            tech_specs.permitted_use_text_collector
        ),
        num_urls=num_urls,
        file_loader_kwargs=file_loader_kwargs,
        browser_semaphore=browser_semaphore,
        **kwargs,
    )
    if docs is None:
        return None

    for doc in docs:
        doc.attrs["location"] = county
        doc.attrs["location_name"] = county.full_name

    await _record_usage(**kwargs)
    return docs


async def _try_extract_all_ordinances(
    possible_ord_doc, text_splitter, tech_specs, county, **kwargs
):
    """Try to extract ordinance values and permitted districts"""
    loc = county.full_name
    with COMPASS_PB.jurisdiction_sub_prog(loc) as jsp:
        extraction_info = [
            (
                tech_specs.ordinance_text_extractor,
                "ordinance_text",
                "cleaned_ordinance_text",
                tech_specs.structured_ordinance_parser,
                "ordinance_values",
            ),
            (
                tech_specs.permitted_use_text_extractor,
                "permitted_use_text",
                "districts_text",
                tech_specs.structured_permitted_use_parser,
                "permitted_district_values",
            ),
        ]
        tasks = [
            asyncio.create_task(
                _try_extract_ordinances(
                    jsp,
                    possible_ord_doc,
                    text_splitter,
                    extractor_class=extractor,
                    original_text_key=o_key,
                    cleaned_text_key=c_key,
                    parser_class=parser,
                    out_key=out_key,
                    **kwargs,
                ),
                name=county.full_name,
            )
            for extractor, o_key, c_key, parser, out_key in extraction_info
        ]

        docs = await asyncio.gather(*tasks)

    return _concat_scrape_results(docs[0])


async def _try_extract_ordinances(
    jsp,
    possible_ord_doc,
    text_splitter,
    extractor_class,
    original_text_key,
    cleaned_text_key,
    parser_class,
    out_key,
    **kwargs,
):
    """Try applying a single extractor to the relevant legal text"""
    logger.debug(
        "Checking for ordinances in doc from %s",
        possible_ord_doc.attrs.get("source", "unknown source"),
    )
    task_id = jsp.add_task(_TEXT_EXTRACTION_TASKS[extractor_class])
    doc = await _extract_ordinance_text(
        possible_ord_doc,
        text_splitter,
        extractor_class=extractor_class,
        original_text_key=original_text_key,
        **kwargs,
    )
    jsp.remove_task(task_id)
    return await _extract_ordinances_from_text(
        doc,
        parser_class=parser_class,
        text_key=cleaned_text_key,
        out_key=out_key,
        **kwargs,
    )


async def _extract_ordinance_text(
    doc, text_splitter, extractor_class, original_text_key, **kwargs
):
    """Extract text pertaining to ordinance of interest"""
    llm_caller = LLMCaller(**kwargs)
    extractor = extractor_class(llm_caller)
    doc = await extract_ordinance_text_with_ngram_validation(
        doc, text_splitter, extractor, original_text_key=original_text_key
    )
    await _record_usage(**kwargs)

    return await _write_cleaned_text(doc)


async def _extract_ordinances_from_text(
    doc, parser_class, text_key, out_key, **kwargs
):
    """Extract values from ordinance text"""
    parser = parser_class(**kwargs)
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


async def _record_usage(**kwargs):
    """Dump usage to file if tracker found in kwargs"""
    usage_tracker = kwargs.get("usage_tracker")
    if usage_tracker:
        await UsageUpdater.call(usage_tracker)


async def _record_jurisdiction_info(county, doc, start_time):
    """Record info about jurisdiction"""
    seconds_elapsed = time.monotonic() - start_time
    await JurisdictionUpdater.call(county, doc, seconds_elapsed)


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


def _docs_to_db(docs):
    """Convert list of docs to output database"""
    db = []
    for doc in docs:
        if doc is None or isinstance(doc, Exception):
            continue

        if num_ordinances_in_doc(doc) == 0:
            continue

        results = _db_results(doc)
        results = _formatted_db(results)
        db.append(results)

    if not db:
        return pd.DataFrame(columns=PARSED_COLS), 0

    num_jurisdictions_found = len(db)
    db = pd.concat(db)
    db = _empirical_adjustments(db)
    return _formatted_db(db), num_jurisdictions_found


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

    logger.info("Compiling final database for %d jurisdictions", len(db))
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
    llm_parse_args,
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
        "llm_parse_args": {
            "llm_call_kwargs": llm_parse_args.llm_call_kwargs,
            "text_splitter_chunk_size": (
                llm_parse_args.text_splitter_chunk_size
            ),
            "text_splitter_chunk_overlap": (
                llm_parse_args.text_splitter_chunk_overlap
            ),
        },
        "time_start_utc": start_date,
        "time_end_utc": end_date,
        "total_time": seconds_elapsed,
        "total_time_string": str(timedelta(seconds=seconds_elapsed)),
        "num_jurisdictions_searched": num_jurisdictions_searched,
        "num_jurisdictions_found": num_jurisdictions_found,
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
