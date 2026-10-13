"""COMPASS Ordinance CPU-bound services"""

import ast
import os
import time
import math
import pprint
import asyncio
import logging
import platform
import contextlib
import multiprocessing
from glob import iglob
from io import BytesIO
from pathlib import Path
from functools import partial
from urllib.parse import urljoin, urlparse
from concurrent.futures import ProcessPoolExecutor

from bs4 import BeautifulSoup
from elm.web.document import PDFDocument, MDDocument
from elm.utilities.parse import read_pdf, read_pdf_ocr
from pdf2image import convert_from_bytes
from docling_core.types.doc import DoclingDocument
from docling_rs.datamodel.base_models import (
    ConversionStatus,
    DocumentStream,
    InputFormat,
)
from docling_rs.document_converter import (
    ConversionResult,
    DocumentConverter,
    PdfFormatOption,
)
from docling_rs.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
)
from docling_rs.exceptions import ConversionError

from compass.exceptions import COMPASSValueError
from compass.services.base import Service
from compass.utilities.logs import (
    configure_subprocess_logging,
    configure_docling_subprocess_logging,
    LQ,
)


logger = logging.getLogger(__name__)
TIMEOUT_PARAMS = {
    "shutdown_timeout": 5,
    "force_shutdown_timeout": 1,
}


class ProcessPoolService(Service):
    """Service that contains a ProcessPoolExecutor instance"""

    _DEFAULT_MAX_TASKS_PER_CHILD = 100

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
        # ppe_kwargs = self._set_tasks_per_child(ppe_kwargs)
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
        ppe_kwargs["initializer"] = configure_subprocess_logging
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
            manager_thread, processes, TIMEOUT_PARAMS["shutdown_timeout"]
        ):
            return

        logger.warning(
            "Process pool did not shut down within %.1f seconds; "
            "terminating lingering workers",
            TIMEOUT_PARAMS["shutdown_timeout"],
        )
        _force_shutdown_processes(
            processes, timeout=TIMEOUT_PARAMS["force_shutdown_timeout"]
        )
        _join_manager_thread(
            manager_thread, timeout=TIMEOUT_PARAMS["force_shutdown_timeout"]
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
        to resolve relative HTML links while ``url`` is still used as
        the stream name for format inference. By default, ``None``.
    pdf_pipeline_options : dict, optional
        Dictionary of keyword-value arguments to pass to
        ``docling_rs.PdfPipelineOptions`` initializer. Table structure
        and native OCR default to enabled. ``document_timeout`` sets
        both the native conversion budget and the subprocess deadline.
        By default, ``None``.
    **kwargs
        Additional keyword arguments passed to Docling's
        :meth:`~docling_core.types.doc.DoclingDocument.export_to_markdown`
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
        :meth:`~docling_core.types.doc.DoclingDocument.export_to_markdown`
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
    """Utility to return empty docs on error"""
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
    except Exception:  # ruff:ignore[blind-except]
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
    """Read documents using Docling with an optional hard deadline"""

    pdf_pipeline_options = dict(pdf_pipeline_options or {})
    docling_timeout = pdf_pipeline_options.get(
        "document_timeout", 60 * 60 / 1.1
    )
    _validate_docling_timeout(docling_timeout)
    return _run_docling_in_subprocess(
        _read_docling_without_timeout,
        args=(doc_bytes, file_source),
        kwargs={
            "headers": headers,
            "pytesseract_exe_fp": pytesseract_exe_fp,
            "source_uri": source_uri,
            "pdf_pipeline_options": pdf_pipeline_options,
            **kwargs,
        },
        timeout=docling_timeout * 1.1,
    )


def _read_docling_without_timeout(
    doc_bytes,
    file_source,
    headers=None,
    source_uri=None,
    pdf_pipeline_options=None,
    **kwargs,
):
    """Read documents using Docling without an in-process deadline"""

    file_source = str(file_source)
    source_uri = file_source if source_uri is None else str(source_uri)
    if headers is not None:
        headers = dict(headers)

    kwargs.pop("pytesseract_exe_fp", None)
    configured_options = dict(pdf_pipeline_options or {})
    document_timeout = configured_options.pop("document_timeout", None)
    configured_options.setdefault("do_table_structure", True)
    configured_options.setdefault("do_ocr", True)
    configured_options.setdefault(
        "table_structure_options", TableStructureOptions(do_cell_matching=True)
    )
    if isinstance(configured_options["table_structure_options"], dict):
        configured_options["table_structure_options"] = TableStructureOptions(
            **configured_options["table_structure_options"]
        )
    pipeline_options = PdfPipelineOptions(**configured_options)

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            ),
        },
        document_timeout=document_timeout,
    )

    start_time = time.perf_counter()
    input_format = _docling_input_format(file_source, headers)
    if input_format == InputFormat.HTML:
        doc_bytes = _resolve_docling_html_links(doc_bytes, source_uri)

    stream = DocumentStream(
        name=file_source, stream=BytesIO(doc_bytes), format=input_format
    )
    conv_result = doc_converter.convert(stream)
    doc_text = conv_result.document.export_to_markdown(**kwargs)
    if (
        input_format == InputFormat.PDF
        and pipeline_options.do_ocr
        and conv_result.status.value == "success"
        and not doc_text.strip()
    ):
        logger.info(
            "Native PDF conversion returned no text; rendering pages "
            "for Docling-RS OCR"
        )
        conv_result = _read_docling_pdf_images(
            doc_converter, doc_bytes, file_source
        )
        doc_text = conv_result.document.export_to_markdown(**kwargs)
    conversion_time_seconds = time.perf_counter() - start_time

    attrs = {
        "doc_filename": conv_result.input.file.stem,
        "doc_type": input_format.value if input_format else "unknown",
        "conversion_time_seconds": conversion_time_seconds,
        "conversion_status": conv_result.status.value,
        "num_pages": len(conv_result.document.pages),
        "from_ocr": None,
        "mean_confidence": None,
        "low_score_confidence": None,
    }
    return MDDocument([doc_text], attrs=attrs, remove_comments=False)


def _read_docling_pdf_images(converter, doc_bytes, file_source):
    """Retry empty PDFs with Poppler renders and native OCR"""
    images = convert_from_bytes(doc_bytes)
    documents = []
    conversion_status = ConversionStatus.SUCCESS
    try:
        for page_number, image in enumerate(images, start=1):
            with BytesIO() as stream:
                image.save(stream, format="PNG")
                stream.seek(0)
                result = converter.convert(
                    DocumentStream(
                        name=f"{Path(file_source).stem}-{page_number}.png",
                        stream=stream,
                        format=InputFormat.IMAGE,
                    )
                )
            documents.append(result.document)
            if result.status != ConversionStatus.SUCCESS:
                conversion_status = ConversionStatus.PARTIAL_SUCCESS
    finally:
        for image in images:
            image.close()

    return ConversionResult(
        status=conversion_status,
        input_name=file_source,
        document=DoclingDocument.concatenate(documents),
    )


def _docling_input_format(file_source, headers):
    """Infer the native stream format from headers or its name"""
    content_type = next(
        (
            value.partition(";")[0].strip().lower()
            for key, value in (headers or {}).items()
            if key.lower() == "content-type"
        ),
        None,
    )
    content_formats = {
        "application/pdf": InputFormat.PDF,
        "text/html": InputFormat.HTML,
        "text/markdown": InputFormat.MD,
        "text/plain": InputFormat.MD,
    }
    if content_type in content_formats:
        return content_formats[content_type]

    if content_type and content_type.startswith("image/"):
        return InputFormat.IMAGE

    extension = Path(urlparse(file_source).path).suffix.lower().lstrip(".")
    aliases = {
        "htm": "html",
        "txt": "md",
        "jpg": "image",
        "jpeg": "image",
        "png": "image",
        "tif": "image",
        "tiff": "image",
        "bmp": "image",
    }
    extension = aliases.get(extension, extension)
    return next(
        (
            input_format
            for input_format in InputFormat
            if input_format.value == extension
        ),
        None,
    )


def _resolve_docling_html_links(doc_bytes, source_uri):
    """Resolve HTML references against their original source URL"""
    document = BeautifulSoup(doc_bytes, "html.parser")
    base = document.find("base", href=True)
    base_uri = urljoin(source_uri, base["href"]) if base else source_uri
    changed = False
    for element in document.find_all(["a", "img"]):
        attribute = "href" if element.name == "a" else "src"
        reference = element.get(attribute)
        if reference:
            resolved = urljoin(base_uri, reference)
            if resolved != reference:
                element[attribute] = resolved
                changed = True
    return str(document).encode("utf-8") if changed else doc_bytes


def _run_docling_in_subprocess(fn, *, args, kwargs, timeout):
    """Run one Docling conversion in a disposable child process"""
    mp_context = multiprocessing.get_context(
        "fork"
        if "fork" in multiprocessing.get_all_start_methods()
        else "spawn"
    )
    receiver, sender = mp_context.Pipe(duplex=False)
    process = mp_context.Process(
        target=_run_docling_subprocess, args=(sender, fn, args, kwargs)
    )
    process.start()
    sender.close()

    try:
        status, payload = _receive_docling_result(receiver, process, timeout)
    finally:
        receiver.close()
        _shutdown_docling_process(process)

    if status == "success":
        logger.info("Docling conversion ran successfully in subprocess")
        return payload

    msg = f"Docling conversion subprocess failed: {payload}"
    logger.error(msg)
    raise ConversionError(msg)


def _run_docling_subprocess(sender, fn, args, kwargs):
    """Execute a Docling conversion and send its result to the worker"""
    configure_docling_subprocess_logging(sender)
    try:
        sender.send(("success", fn(*args, **kwargs)))
    except Exception as error:  # ruff:ignore[blind-except]
        with contextlib.suppress(BrokenPipeError, EOFError, OSError):
            sender.send(("error", f"{type(error).__name__}: {error}"))
    finally:
        sender.close()


# complexipy: ignore
def _receive_docling_result(receiver, process, timeout):
    """Receive a child conversion result before its deadline expires"""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            msg = f"Docling conversion exceeded {timeout} seconds"
            raise TimeoutError(msg)

        if receiver.poll(min(remaining, 10)):
            try:
                status, payload = receiver.recv()
            except EOFError as error:
                msg = "Docling conversion subprocess exited without a result"
                raise ConversionError(msg) from error

            if status == "log":
                logger.handle(payload)
                continue
            return status, payload

        if not process.is_alive():
            if receiver.poll():
                continue
            msg = "Docling conversion subprocess exited without a result"
            raise ConversionError(msg)


def _shutdown_docling_process(process):
    """Join or force-stop a completed or timed-out Docling process"""
    process.join(timeout=TIMEOUT_PARAMS["shutdown_timeout"])
    if process.is_alive():
        _force_shutdown_processes(
            [process], timeout=TIMEOUT_PARAMS["force_shutdown_timeout"]
        )


def _validate_docling_timeout(timeout):
    """Validate the configured Docling deadline"""
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        msg = "`docling_timeout` must be a positive number of seconds"
        raise COMPASSValueError(msg)


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
    # ruff:ignore[glob]
    for filename in iglob(f"{temp_name}*" if temp_name else temp_name):
        with contextlib.suppress(OSError):
            os.remove(filename)  # ruff:ignore[os-remove]


def _try_decode_ocr_pages(pages):
    """Try to decode pages into strings"""
    decoded_pages = []
    for page in pages:
        with contextlib.suppress(Exception):
            # ruff:ignore[redefined-loop-name]
            page = ast.literal_eval(page).decode("utf-8")
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
