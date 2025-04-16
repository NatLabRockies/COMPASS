"""COMPASS Ordinance CPU-bound services"""

import asyncio
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from elm.web.document import PDFDocument
from elm.utilities.parse import read_pdf, read_pdf_ocr

from compass.services.base import Service


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
        self.pool = ProcessPoolExecutor(**self._ppe_kwargs)

    def release_resources(self):
        """Shutdown thread pool and cleanup temp directory"""
        self.pool.shutdown(wait=True, cancel_futures=True)


class PDFLoader(ProcessPoolService):
    """Class to load PDFs in a ProcessPoolExecutor"""

    @property
    def can_process(self):
        """bool: Always ``True`` (limiting is handled by asyncio)"""
        return True

    async def process(self, fn, pdf_bytes, **kwargs):
        """Write URL doc to file asynchronously

        Parameters
        ----------
        doc : elm.web.document.Document
            Document containing meta information about the file. Must
            have a "source" key in the ``attrs`` dict containing the
            URL, which will be converted to a file name using
            :func:`compute_fn_from_url`.
        file_content : str or bytes
            File content, typically string text for HTML files and bytes
            for PDF file.
        make_name_unique : bool, optional
            Option to make file name unique by adding a UUID at the end
            of the file name. By default, ``False``.

        Returns
        -------
        Path
            Path to output file.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.pool, partial(fn, pdf_bytes, **kwargs)
        )


def _read_pdf(pdf_bytes, **kwargs):
    """Utility func so that pdftotext.PDF doesn't have to be pickled"""
    pages = read_pdf(pdf_bytes, verbose=False)
    return PDFDocument(pages, **kwargs)


def _read_pdf_ocr(pdf_bytes, tesseract_cmd, **kwargs):
    """Utility function that mimics `_read_pdf`."""
    if tesseract_cmd:
        _configure_pytesseract(tesseract_cmd)

    pages = read_pdf_ocr(pdf_bytes, verbose=False)
    return PDFDocument(pages, **kwargs)


def _configure_pytesseract(tesseract_cmd):
    """Set the tesseract_cmd"""
    import pytesseract  # noqa: PLC0415

    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


async def read_pdf_doc(pdf_bytes, **kwargs):
    """Read PDF file from bytes in a Process Pool.

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
    return await PDFLoader.call(_read_pdf, pdf_bytes, **kwargs)


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

    return await PDFLoader.call(
        _read_pdf_ocr,
        pdf_bytes,
        tesseract_cmd=pytesseract.pytesseract.tesseract_cmd,
        **kwargs,
    )
