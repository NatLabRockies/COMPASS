"""COMPASS Ordinance CPU-bound services"""

import ast
import os
import sys
import time
import pprint
import asyncio
import logging
import warnings
import platform
import contextlib
import multiprocessing
from glob import iglob
from io import BytesIO
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from logging.handlers import QueueHandler

import numpy as np
import pandas as pd
from elm.web.document import PDFDocument, MDDocument
from elm.utilities.parse import read_pdf, read_pdf_ocr
from docling.datamodel.backend_options import HTMLBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.document_converter import (
    DocumentConverter,
    HTMLFormatOption,
    PdfFormatOption,
)
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TesseractCliOcrOptions,
)
from docling.exceptions import ConversionError

from compass.services.base import Service
from compass.utilities.logs import AddLocationFilter, LQ


logger = logging.getLogger(__name__)


class ProcessPoolService(Service):
    """Service that contains a ProcessPoolExecutor instance"""

    _DEFAULT_MAX_TASKS_PER_CHILD = 100
    _SHUTDOWN_TIMEOUT = 5
    _FORCE_SHUTDOWN_TIMEOUT = 1

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs
            Keyword-value argument pairs to pass to
            :class:`concurrent.futures.ProcessPoolExecutor`.
            By default, ``None``.
        """
        self._ppe_kwargs = kwargs or {}
        self.pool = None

    def acquire_resources(self):
        """Open thread pool and temp directory"""
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        ppe_kwargs = dict(self._ppe_kwargs)
        ppe_kwargs = self._set_tasks_per_child(ppe_kwargs)
        ppe_kwargs = self._set_ppe_initializer(ppe_kwargs)
        logger.debug(
            "  - Setting up ProcessPoolExecutor with kwargs:\n%s",
            pprint.PrettyPrinter().pformat(ppe_kwargs),
        )
        self.pool = ProcessPoolExecutor(**ppe_kwargs)

    def _set_tasks_per_child(self, ppe_kwargs):
        """Set default ``max_tasks_per_child``"""
        ppe_kwargs.setdefault(
            "max_tasks_per_child", self._DEFAULT_MAX_TASKS_PER_CHILD
        )
        ppe_kwargs.setdefault(
            "mp_context", multiprocessing.get_context("spawn")
        )
        return ppe_kwargs

    def _set_ppe_initializer(self, ppe_kwargs):  # ruff:ignore[no-self-use]
        """Set initializer to configure subprocess logging"""
        user_initializer = ppe_kwargs.pop("initializer", None)
        initargs = tuple(ppe_kwargs.pop("initargs", ()))
        ppe_kwargs["initializer"] = _configure_subprocess_logging
        ppe_kwargs["initargs"] = (LQ.QUEUE, user_initializer, initargs)
        return ppe_kwargs

    def release_resources(self):
        """Shutdown thread pool and cleanup temp directory"""
        pool = self.pool
        self.pool = None
        if pool is None:
            return

        manager_thread = getattr(pool, "_executor_manager_thread", None)
        processes = list(getattr(pool, "_processes", {}).values())
        pool.shutdown(wait=False, cancel_futures=True)

        if not _needs_forced_shutdown(
            manager_thread, processes, self._SHUTDOWN_TIMEOUT
        ):
            return

        logger.warning(
            "Process pool did not shut down within %.1f seconds; "
            "terminating lingering workers",
            self._SHUTDOWN_TIMEOUT,
        )
        _force_shutdown_processes(
            processes, timeout=self._FORCE_SHUTDOWN_TIMEOUT
        )
        _join_manager_thread(
            manager_thread, timeout=self._FORCE_SHUTDOWN_TIMEOUT
        )


class FileLoader(ProcessPoolService):
    """Class to load files in a ProcessPoolExecutor"""

    @property
    def can_process(self):
        """bool: Always ``True`` (limiting is handled by asyncio)"""
        return True

    async def process(self, fn, source, **kwargs):
        """Execute a file parsing function in the process pool

        Parameters
        ----------
        fn : callable
            Callable executed inside the process pool. Receives
            ``pdf_bytes`` as the first argument.
        source : bytes
            Raw document payload or path to file on disk. Argument
            forwarded to ``read_fn``.
        **kwargs
            Additional keyword arguments passed to ``fn``.

        Returns
        -------
        Any
            Result returned by ``fn`` after execution.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.pool, partial(fn, source, **kwargs)
        )


class OCRPDFLoader(FileLoader):
    """Loader service for OCR"""


async def read_pdf_doc(pdf_bytes, **kwargs):
    """Read PDF file from bytes in a Process Pool

    Parameters
    ----------
    pdf_bytes : bytes
        Bytes containing PDF file.
    **kwargs
        Keyword-value arguments to pass to
        :class:`elm.web.document.PDFDocument` initializer.

    Returns
    -------
    elm.web.document.PDFDocument
        PDFDocument instances with pages loaded as text.
    """
    return await FileLoader.call(_read_pdf, pdf_bytes, **kwargs)


async def read_pdf_file(pdf_fp, **kwargs):
    """Read local PDF file in a Process Pool

    Parameters
    ----------
    pdf_fp : path-like
        Path to PDF file (non-OCR).
    **kwargs
        Keyword-value arguments to pass to
        :class:`elm.web.document.PDFDocument` initializer.

    Returns
    -------
    elm.web.document.PDFDocument
        PDFDocument instances with pages loaded as text.
    bytes
        Raw bytes of the PDF file.
    """
    return await FileLoader.call(_read_pdf_file, pdf_fp, **kwargs)


async def read_pdf_doc_ocr(pdf_bytes, **kwargs):
    """Read PDF file using OCR (pytesseract)

    Note that Pytesseract must be set up properly for this method to
    work. In particular, the `pytesseract.pytesseract.tesseract_cmd`
    attribute must be set to point to the pytesseract exe.

    Parameters
    ----------
    pdf_bytes : bytes
        Bytes containing PDF file.
    **kwargs
        Keyword-value arguments to pass to
        :class:`elm.web.document.PDFDocument` initializer.

    Returns
    -------
    elm.web.document.PDFDocument
        PDFDocument instances with pages loaded as text.
    """
    import pytesseract  # ruff:ignore[import-outside-top-level]

    return await OCRPDFLoader.call(
        _read_pdf_ocr,
        pdf_bytes,
        tesseract_cmd=pytesseract.pytesseract.tesseract_cmd,
        **kwargs,
    )


async def read_pdf_file_ocr(pdf_fp, **kwargs):
    """Read local PDF file using OCR (pytesseract)

    Note that Pytesseract must be set up properly for this method to
    work. In particular, the `pytesseract.pytesseract.tesseract_cmd`
    attribute must be set to point to the pytesseract exe.

    Parameters
    ----------
    pdf_fp : path-like
        Path to PDF file (OCR).
    **kwargs
        Keyword-value arguments to pass to
        :class:`elm.web.document.PDFDocument` initializer.

    Returns
    -------
    elm.web.document.PDFDocument
        PDFDocument instances with pages loaded as text.
    bytes
        Raw bytes of the PDF file.
    """
    import pytesseract  # ruff:ignore[import-outside-top-level]

    return await OCRPDFLoader.call(
        _read_pdf_file_ocr,
        pdf_fp,
        tesseract_cmd=pytesseract.pytesseract.tesseract_cmd,
        **kwargs,
    )


async def read_docling_web_file(
    doc_bytes, url, source_uri=None, pdf_pipeline_options=None, **kwargs
):
    """Read a web file using Docling in a Process Pool

    Parameters
    ----------
    doc_bytes : bytes
        Raw document payload forwarded to the Docling parser.
    url : str
        Filename or URL of the file to read.
    source_uri : str, optional
        Original remote URL for the file. If specified, this is used
        as the HTML base URI while ``url`` is still used as the stream
        name for Docling format inference. By default, ``None``.
    pdf_pipeline_options : dict, optional
        Dictionary of keyword-value arguments to pass to
        :class:`docling.datamodel.pipeline_options.PdfPipelineOptions`
        initializer. By default, ``None``.
    **kwargs
        Additional keyword arguments passed to Docling's
        :func:`~docling_core.types.doc.DoclingDocument.export_to_markdown`
        method.

    Returns
    -------
    elm.web.document.MDDocument
        Parsed document.
    """
    return await FileLoader.call(
        _read_docling_catch_error,
        doc_bytes,
        file_source=url,
        source_uri=source_uri,
        pdf_pipeline_options=pdf_pipeline_options,
        **kwargs,
    )


async def read_docling_local_file(fp, **kwargs):
    """Read a web file using Docling in a Process Pool

    Parameters
    ----------
    fp : path-like
        Path to local file to read.
    **kwargs
        Additional keyword arguments passed to Docling's
        :func:`~docling_core.types.doc.DoclingDocument.export_to_markdown`
        method.

    Returns
    -------
    elm.web.document.MDDocument
        Parsed document.
    bytes
        Raw bytes of the PDF file.
    """
    return await FileLoader.call(_read_file_docling, fp, **kwargs)


def _read_pdf(pdf_bytes, **kwargs):
    """Utility func so that pdftotext.PDF doesn't have to be pickled"""
    pages = read_pdf(pdf_bytes, verbose=False)
    return PDFDocument(pages, **kwargs)


def _read_pdf_ocr(pdf_bytes, tesseract_cmd, **kwargs):
    """Utility function that mimics `_read_pdf`"""
    if tesseract_cmd:
        _configure_pytesseract(tesseract_cmd)

    pages = read_pdf_ocr(pdf_bytes, verbose=False)
    doc = PDFDocument(_try_decode_ocr_pages(pages), **kwargs)
    doc.attrs["from_ocr"] = True
    return doc


def _read_pdf_file(pdf_fp, **kwargs):
    """Utility func so that pdftotext.PDF doesn't have to be pickled"""
    kwargs.pop("image_to_string_kwargs", None)
    kwargs.pop("convert_from_bytes_kwargs", None)
    pdf_bytes = Path(pdf_fp).read_bytes()
    pages = read_pdf(pdf_bytes, verbose=False)
    return PDFDocument(pages, **kwargs), pdf_bytes


def _read_pdf_file_ocr(pdf_fp, tesseract_cmd, **kwargs):
    """Utility function that mimics `_read_pdf_file`"""
    if tesseract_cmd:
        _configure_pytesseract(tesseract_cmd)

    image_to_string_kwargs = kwargs.pop("image_to_string_kwargs", None)
    convert_from_bytes_kwargs = kwargs.pop("convert_from_bytes_kwargs", None)

    pdf_bytes = Path(pdf_fp).read_bytes()
    pages = read_pdf_ocr(
        pdf_bytes,
        verbose=True,
        image_to_string_kwargs=image_to_string_kwargs,
        convert_from_bytes_kwargs=convert_from_bytes_kwargs,
    )
    doc = PDFDocument(_try_decode_ocr_pages(pages), **kwargs)
    doc.attrs["from_ocr"] = True
    return doc, pdf_bytes


def _read_docling_catch_error(
    doc_bytes,
    file_source,
    headers=None,
    pytesseract_exe_fp=None,
    source_uri=None,
    pdf_pipeline_options=None,
    **kwargs,
):
    """Utility to return empty docs on Docling conversion errors"""
    try:
        return _read_docling(
            doc_bytes=doc_bytes,
            file_source=file_source,
            headers=headers,
            pytesseract_exe_fp=pytesseract_exe_fp,
            source_uri=source_uri,
            pdf_pipeline_options=pdf_pipeline_options,
            **kwargs,
        )
    except ConversionError:
        return MDDocument(pages=[], attrs={"doc_type": "unknown"})


def _read_docling(
    doc_bytes,
    file_source,
    headers=None,
    pytesseract_exe_fp=None,
    source_uri=None,
    pdf_pipeline_options=None,
    **kwargs,
):
    """Utility func to read documents using Docling"""

    file_source = str(file_source)
    source_uri = file_source if source_uri is None else str(source_uri)
    if headers is not None:
        headers = dict(headers)

    pipeline_options = PdfPipelineOptions(**(pdf_pipeline_options or {}))
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True
    )
    if pytesseract_exe_fp is None:
        pipeline_options.do_ocr = False
    else:
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = TesseractCliOcrOptions(
            tesseract_cmd=pytesseract_exe_fp
        )

    html_backend_options = HTMLBackendOptions(source_uri=source_uri)

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            ),
            InputFormat.HTML: HTMLFormatOption(
                backend_options=html_backend_options
            ),
        }
    )

    start_time = time.perf_counter()
    stream = DocumentStream(name=file_source, stream=BytesIO(doc_bytes))
    conv_result = doc_converter.convert(stream, headers=headers)
    conversion_time_seconds = time.perf_counter() - start_time

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_confidence = _none_if_missing(conv_result.confidence.mean_score)
        low_score_confidence = _none_if_missing(
            conv_result.confidence.low_score
        )

    attrs = {
        "doc_filename": conv_result.input.file.stem,
        "doc_type": conv_result.input.format.value,
        "conversion_time_seconds": conversion_time_seconds,
        "conversion_status": conv_result.status.value,
        "num_pages": len(conv_result.pages),
        "from_ocr": any(
            ~np.isnan(c.ocr_score)
            for c in conv_result.confidence.pages.values()
        ),
        "mean_confidence": mean_confidence,
        "low_score_confidence": low_score_confidence,
    }
    doc_text = conv_result.document.export_to_markdown(**kwargs)

    return MDDocument([doc_text], attrs=attrs, remove_comments=False)


def _read_file_docling(fp, **kwargs):
    """Read a local file using Docling"""

    fp = Path(fp)
    doc_bytes = fp.read_bytes()
    doc = _read_docling_catch_error(
        doc_bytes, str(fp).replace(".txt", ".md"), headers=None, **kwargs
    )
    return doc, doc_bytes


def _configure_pytesseract(tesseract_cmd):
    """Set the tesseract_cmd"""
    import pytesseract  # ruff:ignore[import-outside-top-level]

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    if platform.system() == "Windows":
        pytesseract.pytesseract.cleanup = _pytesseract_cleanup_win


def _pytesseract_cleanup_win(temp_name):
    """Suppress all OSErrors when cleaning up temp files on Windows

    On Windows, Tesseract may still hold the temp PPM file open when
    pytesseract's cleanup runs, causing WinError 32. This function
    patches cleanup to suppress all OSErrors so the OCR result is not
    lost.
    """
    for filename in iglob(f"{temp_name}*" if temp_name else temp_name):  # ruff:ignore[glob]
        with contextlib.suppress(OSError):
            os.remove(filename)  # ruff:ignore[os-remove]


def _none_if_missing(value):
    """Return ``None`` when a scalar confidence value is missing"""
    return None if pd.isna(value) else value


def _try_decode_ocr_pages(pages):
    """Try to decode pages into strings"""
    decoded_pages = []
    for page in pages:
        with contextlib.suppress(Exception):
            page = ast.literal_eval(page).decode("utf-8")  # ruff:ignore[redefined-loop-name]
        decoded_pages.append(page)
    return decoded_pages


def _needs_forced_shutdown(manager_thread, processes, shutdown_timeout):
    """Determine whether graceful shutdown exceeded the timeout"""
    if manager_thread is not None:
        return _join_manager_thread(manager_thread, timeout=shutdown_timeout)

    deadline = time.monotonic() + shutdown_timeout
    while time.monotonic() < deadline:
        if not any(_is_process_alive(process) for process in processes):
            return False
        time.sleep(0.05)

    return any(_is_process_alive(process) for process in processes)


def _join_manager_thread(manager_thread, timeout):
    """bool: Join a manager thread and report whether it lives"""
    if manager_thread is None:
        return False

    manager_thread.join(timeout=timeout)
    return manager_thread.is_alive()


def _force_shutdown_processes(processes, timeout=1):
    """Force lingering worker processes to exit"""
    for process in processes:
        if process is None or not _is_process_alive(process):
            continue

        with contextlib.suppress(Exception):
            process.terminate()
            process.join(timeout=timeout)

        if not _is_process_alive(process):
            continue

        with contextlib.suppress(Exception):
            process.kill()
            process.join(timeout=timeout)


def _is_process_alive(process):
    """bool: Check whether a worker process is still alive"""
    return process is not None and process.is_alive()


def _configure_subprocess_logging(logging_queue, user_initializer, initargs):
    """Route subprocess output through the main process log queue"""
    queue_handler = QueueHandler(logging_queue)
    queue_handler.addFilter(AddLocationFilter())
    queue_handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(queue_handler)  # root emits to queue handler
    root_logger.setLevel(logging.INFO)

    for lib in ("compass", "elm", "docling", "openai"):
        lib_logger = logging.getLogger(lib)
        lib_logger.handlers = []  # no handlers within subprocess
        lib_logger.propagate = True  # instead, propogate to root logger
        lib_logger.setLevel(logging.INFO)

    stdout_logger = logging.getLogger("compass.subprocess.stdout")
    stderr_logger = logging.getLogger("compass.subprocess.stderr")
    stdout_logger.setLevel(logging.INFO)
    stderr_logger.setLevel(logging.WARNING)
    sys.stdout = _LogStream(stdout_logger, logging.INFO)
    sys.stderr = _LogStream(stderr_logger, logging.WARNING)

    logging.getLogger("compass").info("Subprocess logging initialized")
    if user_initializer is not None:
        user_initializer(*initargs)


class _LogStream:
    """File-like object that forwards writes into a logger"""

    def __init__(self, logger, level):
        """

        Parameters
        ----------
        logger : logging.Logger
            Logger to emit redirected stream output to.
        level : int
            Logging level used for forwarded messages.
        """
        self.logger = logger
        self.level = level
        self._buffer = ""
        self.encoding = "utf-8"

    def write(self, message):
        """Forward complete lines to the configured logger"""
        if not message:
            return 0

        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                self.logger.log(self.level, line)
        return len(message)

    def flush(self):
        """Flush any partial line buffered from the stream"""
        if self._buffer:
            self.logger.log(self.level, self._buffer)
            self._buffer = ""

    def isatty(self):  # ruff:ignore[no-self-use]
        """bool: Redirected subprocess streams are never TTYs"""
        return False
