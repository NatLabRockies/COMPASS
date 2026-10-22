"""COMPASS Ordinance CPU-bound services"""

import ast
import os
import time
import pprint
import asyncio
import logging
import platform
import contextlib
import multiprocessing
from glob import iglob
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from elm.web.document import PDFDocument
from elm.utilities.parse import read_pdf, read_pdf_ocr

from compass.services.base import Service
from compass.utilities.logs import configure_subprocess_logging, LQ


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
