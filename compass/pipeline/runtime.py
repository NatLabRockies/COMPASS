"""Runtime state for the COMPASS pipeline"""

import asyncio
import logging
from copy import deepcopy
from functools import cached_property
from contextlib import AsyncExitStack

from compass.plugin.registry import resolve_plugin
from compass.exceptions import COMPASSValueError
from compass.services.cpu import (
    FileLoader,
    OCRPDFLoader,
    read_pdf_doc,
    read_pdf_doc_ocr,
    read_pdf_file,
    read_pdf_file_ocr,
)
from compass.services.provider import RunningAsyncServices
from compass.services.threaded import (
    CleanedFileWriter,
    FileMover,
    HTMLFileLoader,
    JurisdictionUpdater,
    OrdDBFileWriter,
    ParsedFileWriter,
    TempFileCache,
    TempFileCacheCopier,
    TempFileCachePB,
    UsageUpdater,
    GenericFuncRunner,
    read_html_file,
)
from compass.utilities import LLM_COST_REGISTRY, Directories
from compass.utilities.jurisdictions import fips_to_str
from compass.utilities.io import load_config
from compass.utilities.logs import NoLocationFilter, LogListener


logger = logging.getLogger(__name__)
MAX_CONCURRENT_SEARCH_ENGINE_QUERIES = 10


class PipelineRuntime:
    """Context Object for runtime dependencies in one pipeline run"""

    def __init__(self, request):
        """

        Parameters
        ----------
        request : compass.pipeline.data_classes.BaseRequest
            Request object containing all user inputs and settings for
            this run.
        """
        self.request = request
        self.mode = request.MODE
        self.tech = request.tech
        self.models = request.models
        self.search_params = request.search_settings
        self.log_level = _normalize_log_level(
            request.runtime_settings.log_level
        )
        self.keep_async_logs = request.runtime_settings.keep_async_logs
        self.log_listener = LogListener(
            ["compass", "elm"], level=self.log_level
        )
        self.known_local_docs = _load_known_source(
            request.known_sources.known_local_docs
        )
        self.known_doc_urls = _load_known_source(
            request.known_sources.known_doc_urls
        )
        self._pytesseract_was_set_up = False
        LLM_COST_REGISTRY.update(request.llm_costs or {})

    async def __aenter__(self):
        self._listener_ctx = self.log_listener
        listener = await self._listener_ctx.__aenter__()
        _configure_main_logging(
            self.dirs.logs, self.log_level, listener, self.keep_async_logs
        )
        await self._running_services.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._running_services.__aexit__(exc_type, exc, tb)
        await self._listener_ctx.__aexit__(exc_type, exc, tb)
        (self.dirs.logs / "all.log").unlink(missing_ok=True)

    @cached_property
    def dirs(self):
        """Directories object for this run"""
        return _setup_folders(
            self.request.output_settings,
            collect_only=(self.mode == self.mode.COLLECT),
        )

    @cached_property
    def tpe_kwargs(self):
        """Thread pool kwargs for this run"""
        return _build_tpe_kwargs(self.request.runtime_settings)

    @cached_property
    def extractor_class(self):
        """Return the extractor class for the configured tech"""
        return resolve_plugin(self.tech)

    @cached_property
    def browser_semaphore(self):
        """Browser concurrency limiter"""
        if not self.search_params.max_num_concurrent_browsers:
            return None
        return asyncio.Semaphore(
            self.search_params.max_num_concurrent_browsers
        )

    @cached_property
    def crawl_semaphore(self):
        """Crawl concurrency limiter"""
        if not self.search_params.max_num_concurrent_website_searches:
            return None
        return asyncio.Semaphore(
            self.search_params.max_num_concurrent_website_searches
        )

    @cached_property
    def search_engine_semaphore(self):
        """Search engine concurrency limiter"""
        return asyncio.Semaphore(MAX_CONCURRENT_SEARCH_ENGINE_QUERIES)

    @cached_property
    def _jurisdiction_semaphore(self):
        """Jurisdiction concurrency limiter"""
        max_num = (
            self.request.runtime_settings.max_num_concurrent_jurisdictions
        )
        if not max_num:
            return None
        return asyncio.Semaphore(max_num)

    @property
    def jurisdiction_semaphore(self):
        """Jurisdiction semaphore or inert context manager"""
        if self._jurisdiction_semaphore is None:
            return AsyncExitStack()
        return self._jurisdiction_semaphore

    @cached_property
    def file_loader_kwargs(self):
        """dict: Loader kwargs for remote documents"""
        kwargs = _build_file_loader_kwargs(self.request.file_loader_kwargs)
        if self.search_params.pytesseract_exe_fp is not None:
            self._setup_pytesseract()
            kwargs.update(
                {
                    "pdf_ocr_read_coroutine": read_pdf_doc_ocr,
                    "pytesseract_exe_fp": (
                        self.search_params.pytesseract_exe_fp
                    ),
                }
            )
        return kwargs

    @cached_property
    def local_file_loader_kwargs(self):
        """dict: Loader kwargs for local documents"""
        kwargs = {
            "pdf_read_coroutine": read_pdf_file,
            "html_read_coroutine": read_html_file,
            "pdf_read_kwargs": self.file_loader_kwargs.get("pdf_read_kwargs"),
            "html_read_kwargs": self.file_loader_kwargs.get(
                "html_read_kwargs"
            ),
        }
        if self.search_params.pytesseract_exe_fp is not None:
            self._setup_pytesseract()
            kwargs.update(
                {
                    "pdf_ocr_read_coroutine": read_pdf_file_ocr,
                    "pytesseract_exe_fp": (
                        self.search_params.pytesseract_exe_fp
                    ),
                }
            )
        return kwargs

    @cached_property
    def file_loader_kwargs_no_ocr(self):
        """dict: Loader kwargs without OCR for website validation"""
        kwargs = deepcopy(self.file_loader_kwargs)
        kwargs.pop("pdf_ocr_read_coroutine", None)
        return kwargs

    @cached_property
    def _base_services(self):
        """Base services required for this run"""
        runtime_settings = self.request.runtime_settings
        services = [
            TempFileCachePB(
                td_kwargs=runtime_settings.td_kwargs,
                tpe_kwargs=self.tpe_kwargs,
            ),
            TempFileCache(
                td_kwargs=runtime_settings.td_kwargs,
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
            FileLoader(**(runtime_settings.ppe_kwargs or {})),
            HTMLFileLoader(**self.tpe_kwargs),
            GenericFuncRunner(**self.tpe_kwargs),
        ]

        if self.mode == self.mode.COLLECT:
            services.append(
                ParsedFileWriter(
                    self.dirs.clean_files,
                    tpe_kwargs=self.tpe_kwargs,
                )
            )
        elif self.mode == self.mode.EXTRACT:
            services.append(
                TempFileCacheCopier(
                    td_kwargs=runtime_settings.td_kwargs,
                    tpe_kwargs=self.tpe_kwargs,
                )
            )

        if self.search_params.pytesseract_exe_fp is not None:
            services.append(OCRPDFLoader(max_workers=1))
        return services

    @cached_property
    def _llm_services(self):
        """LLM services for modes that require them"""
        if self.mode == self.mode.COLLECT:
            return []
        return [model.llm_service for model in set(self.models.values())]

    @property
    def _services(self):
        """All running services for this runtime"""
        return self._base_services + self._llm_services

    @cached_property
    def _running_services(self):
        """Context manager for active async services"""
        return RunningAsyncServices(self._services)

    def _setup_pytesseract(self):
        """Set the pytesseract command"""
        if self._pytesseract_was_set_up:
            return

        import pytesseract  # noqa: PLC0415

        logger.debug(
            "Setting `tesseract_cmd` to %s",
            self.search_params.pytesseract_exe_fp,
        )
        pytesseract.pytesseract.tesseract_cmd = (
            self.search_params.pytesseract_exe_fp
        )
        self._pytesseract_was_set_up = True


def _normalize_log_level(log_level):
    """Normalize log level for file logging"""
    if log_level == "DEBUG":
        return "DEBUG_TO_FILE"
    return log_level


def _build_tpe_kwargs(runtime_settings):
    """Set thread pool workers to 5 if user did not specify them"""
    tpe_kwargs = runtime_settings.tpe_kwargs or {}
    tpe_kwargs.setdefault("max_workers", 5)
    return tpe_kwargs


def _build_file_loader_kwargs(file_loader_kwargs):
    """Add PDF reading coroutine to file loader kwargs"""

    kwargs = file_loader_kwargs or {}
    kwargs.update({"pdf_read_coroutine": read_pdf_doc})
    return kwargs


def _configure_main_logging(log_dir, level, listener, keep_async_logs):
    """Configure top-level run logging"""
    fmt = logging.Formatter(fmt="[%(asctime)s] %(levelname)s: %(message)s")
    handler = logging.FileHandler(log_dir / "main.log", encoding="utf-8")
    handler.setFormatter(fmt)
    handler.setLevel(level)
    handler.addFilter(NoLocationFilter())
    listener.addHandler(handler)

    if keep_async_logs:
        handler = logging.FileHandler(log_dir / "all.log", encoding="utf-8")
        log_fmt = "[%(asctime)s] %(levelname)s - %(taskName)s: %(message)s"
        fmt = logging.Formatter(fmt=log_fmt)
        handler.setFormatter(fmt)
        handler.setLevel(level)
        listener.addHandler(handler)
        logger.debug_to_file("Using async log format: %s", log_fmt)


def _setup_folders(output_settings, collect_only=False):
    """Create output folders for the run"""
    dirs = Directories(
        output_settings.out_dir,
        output_settings.log_dir,
        output_settings.clean_dir,
        output_settings.ordinance_file_dir,
        output_settings.jurisdiction_dbs_dir,
        collect_only,
    )

    if dirs.out.exists():
        msg = (
            f"Output directory '{output_settings.out_dir!s}' already "
            "exists! Please specify a new directory for every COMPASS run."
        )
        raise COMPASSValueError(msg)

    dirs.make_dirs()
    return dirs


def _load_known_source(known_source):
    """Load configured known sources as int-keyed dictionaries"""
    known_source = known_source or {}
    if isinstance(known_source, str):
        known_source = load_config(known_source)
    return {fips_to_str(k): v for k, v in known_source.items()}
