"""COMPASS Ordinance CPU-bound services"""

import ast
import os
import sys
import time
import asyncio
import logging
import contextlib
import warnings
from io import BytesIO
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from logging.handlers import QueueHandler

import numpy as np
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

from compass.services.base import Service
from compass.utilities.logs import AddLocationFilter, LQ


class ProcessPoolService(Service):
    """Service that contains a ProcessPoolExecutor instance"""

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
        user_initializer = ppe_kwargs.pop("initializer", None)
        initargs = tuple(ppe_kwargs.pop("initargs", ()))
        ppe_kwargs["initializer"] = _configure_subprocess_logging
        ppe_kwargs["initargs"] = (
            LQ.QUEUE,
            user_initializer,
            initargs,
        )
        self.pool = ProcessPoolExecutor(**ppe_kwargs)

    def release_resources(self):
        """Shutdown thread pool and cleanup temp directory"""
        self.pool.shutdown(wait=True, cancel_futures=True)


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
    import pytesseract  # noqa: PLC0415

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
    import pytesseract  # noqa: PLC0415

    return await OCRPDFLoader.call(
        _read_pdf_file_ocr,
        pdf_fp,
        tesseract_cmd=pytesseract.pytesseract.tesseract_cmd,
        **kwargs,
    )


async def read_docling_web_file(doc_bytes, url, **kwargs):
    """Read a web file using Docling in a Process Pool

    Parameters
    ----------
    doc_bytes : bytes
        Raw document payload forwarded to the Docling parser.
    url : str
        URL of the file to read.
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
        _read_docling, doc_bytes, file_source=url, **kwargs
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
    pdf_bytes = Path(pdf_fp).read_bytes()
    pages = read_pdf(pdf_bytes, verbose=False)
    return PDFDocument(pages, **kwargs), pdf_bytes


def _read_pdf_file_ocr(pdf_fp, tesseract_cmd, **kwargs):
    """Utility function that mimics `_read_pdf_file`"""
    if tesseract_cmd:
        _configure_pytesseract(tesseract_cmd)

    pdf_bytes = Path(pdf_fp).read_bytes()
    pages = read_pdf_ocr(pdf_bytes, verbose=False)
    doc = PDFDocument(_try_decode_ocr_pages(pages), **kwargs)
    doc.attrs["from_ocr"] = True
    return doc, pdf_bytes


def _read_docling(
    doc_bytes, file_source, headers=None, pytesseract_exe_fp=None, **kwargs
):
    """Utility func to read documents using Docling"""

    file_source = str(file_source)
    if headers is not None:
        headers = dict(headers)

    pipeline_options = PdfPipelineOptions()
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

    html_backend_options = HTMLBackendOptions(source_uri=file_source)

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

    start_time = time.monotonic()
    stream = DocumentStream(name=file_source, stream=BytesIO(doc_bytes))
    conv_result = doc_converter.convert(stream, headers=headers)
    conversion_time_seconds = time.monotonic() - start_time

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_confidence = conv_result.confidence.mean_score
        low_score_confidence = conv_result.confidence.low_score

    attrs = {
        "doc_filename": conv_result.input.file.stem,
        "doc_type": conv_result.input.format.value,
        "conversion_time_seconds": conversion_time_seconds,
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
    doc = _read_docling(
        doc_bytes, str(fp).replace(".txt", ".md"), headers=None, **kwargs
    )
    return doc, doc_bytes


def _configure_pytesseract(tesseract_cmd):
    """Set the tesseract_cmd"""
    import pytesseract  # noqa: PLC0415

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def _try_decode_ocr_pages(pages):
    """Try to decode pages into strings"""
    decoded_pages = []
    for page in pages:
        with contextlib.suppress(Exception):
            page = ast.literal_eval(page).decode("utf-8")  # noqa: PLW2901
        decoded_pages.append(page)
    return decoded_pages


def _configure_subprocess_logging(logging_queue, user_initializer, initargs):
    """Route subprocess output through the main process log queue"""
    queue_handler = QueueHandler(logging_queue)
    queue_handler.addFilter(AddLocationFilter())

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

    def isatty(self):  # noqa: PLR6301
        """bool: Redirected subprocess streams are never TTYs"""
        return False
