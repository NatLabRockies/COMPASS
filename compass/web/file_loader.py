"""COMPASS file loader for web files using Docling"""

import os
import asyncio
import logging

from elm.web.file_loader import (
    AsyncFetchWithRetry,
    AsyncHTMLLoader,
    BaseAsyncFileLoader,
    AsyncWebFileLoader,
    AsyncLocalFileLoader,
)
from elm.web.document import MDDocument
from docling_core.utils.file import resolve_remote_filename, AnyHttpUrl

from compass.services.cpu import read_docling_web_file, read_docling_local_file
from compass.services.threaded import TempFileCache


logger = logging.getLogger(__name__)


class _AsyncHTMLOnlyLoader(BaseAsyncFileLoader):
    """Class for loading HTML files using only the HTML loader"""

    def __init__(
        self,
        pw_launch_kwargs=None,
        html_read_kwargs=None,
        html_read_coroutine=None,
        browser_semaphore=None,
        use_scrapling_stealth=False,
        num_pw_html_retries=3,
        **__,  # consume any extra kwargs
    ):
        """

        Parameters
        ----------
        pw_launch_kwargs : dict, optional
            Keyword-value argument pairs to pass to
            ``async_playwright.chromium.launch`` (only used when
            reading HTML). By default, ``None``.
        html_read_kwargs : dict, optional
            Keyword-value argument pairs to pass to the
            `html_read_coroutine`. By default, ``None``.
        html_read_coroutine : callable, optional
            HTML file read coroutine. Must by an async function. Should
            accept HTML text as the first argument and kwargs as the
            rest. Must return a :obj:`elm.web.document.HTMLDocument`.
            If ``None``, a default function that runs in the main thread
            is used. By default, ``None``.
        browser_semaphore : asyncio.Semaphore, optional
            Semaphore instance that can be used to limit the number of
            playwright browsers open concurrently. If ``None``, no
            limits are applied. By default, ``None``.
        use_scrapling_stealth : bool, default=False
            Option to use scrapling stealth scripts instead of
            playwright-stealth. By default, ``False``.
        num_pw_html_retries : int, default=3
            Number of attempts to load HTML content. This is useful
            because the playwright parameters are stochastic, and
            sometimes a combination of them can fail to load HTML. The
            default value is likely a good balance between processing
            attempts and retrieval success. Note that the minimum number
            of attempts will always be 2, even if the user provides a
            value smaller than this. By default, ``3``.
        """
        super().__init__(file_cache_coroutine=TempFileCache.call)
        self._html_loader = AsyncHTMLLoader(
            pw_launch_kwargs=pw_launch_kwargs,
            html_read_kwargs=html_read_kwargs,
            html_read_coroutine=html_read_coroutine,
            browser_semaphore=browser_semaphore,
            use_scrapling_stealth=use_scrapling_stealth,
            num_pw_html_retries=num_pw_html_retries,
        )

    async def _fetch_doc(self, url):
        """Fetch a doc using Docling"""
        doc = await self._html_loader.fetch(url)
        return doc, doc.text


class AsyncDoclingWebFileLoader(BaseAsyncFileLoader):
    """Async web file loader using Docling"""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        header_template=None,
        verify_ssl=True,
        aget_kwargs=None,
        pw_launch_kwargs=None,
        html_read_kwargs=None,
        html_read_coroutine=None,
        file_cache_coroutine=None,
        browser_semaphore=None,
        use_scrapling_stealth=False,
        num_pw_html_retries=3,
        to_md_kwargs=None,
        pytesseract_exe_fp=None,
        **__,  # consume any extra kwargs
    ):
        """

        Parameters
        ----------
        header_template : dict, optional
            Optional GET header template. If not specified, uses
            :obj:`~elm.web.utilities.DEFAULT_HEADERS`.
            By default, ``None``.
        verify_ssl : bool, optional
            Option to use aiohttp's default SSL check. If ``False``,
            SSL certificate validation is skipped. By default, ``True``.
        aget_kwargs : dict, optional
            Other kwargs to pass to :meth:`aiohttp.ClientSession.get`.
            By default, ``None``.
        pw_launch_kwargs : dict, optional
            Keyword-value argument pairs to pass to
            ``async_playwright.chromium.launch`` (only used when
            reading HTML). By default, ``None``.
        html_read_kwargs : dict, optional
            Keyword-value argument pairs to pass to the
            `html_read_coroutine`. By default, ``None``.
        html_read_coroutine : callable, optional
            HTML file read coroutine. Must by an async function. Should
            accept HTML text as the first argument and kwargs as the
            rest. Must return a :obj:`elm.web.document.HTMLDocument`.
            If ``None``, a default function that runs in the main thread
            is used. By default, ``None``.
        file_cache_coroutine : callable, optional
            File caching coroutine. Can be used to cache files
            downloaded by this class. Must accept an
            :obj:`~elm.web.document.BaseDocument` instance as the first
            argument and the file content to be written as the second
            argument. If this method is not provided, no document
            caching is performed. By default, ``None``.
        browser_semaphore : asyncio.Semaphore, optional
            Semaphore instance that can be used to limit the number of
            playwright browsers open concurrently. If ``None``, no
            limits are applied. By default, ``None``.
        use_scrapling_stealth : bool, default=False
            Option to use scrapling stealth scripts instead of
            playwright-stealth. By default, ``False``.
        num_pw_html_retries : int, default=3
            Number of attempts to load HTML content. This is useful
            because the playwright parameters are stochastic, and
            sometimes a combination of them can fail to load HTML. The
            default value is likely a good balance between processing
            attempts and retrieval success. Note that the minimum number
            of attempts will always be 2, even if the user provides a
            value smaller than this. By default, ``3``.
        to_md_kwargs : dict, optional
            Keyword-value argument pairs to pass to to Docling's
            :func:`~docling_core.types.doc.DoclingDocument.export_to_markdown`
            method for converting the raw content to a markdown
            document. Can be useful to specify image placeholders (i.e.
            ``"image_placeholder"=""``) or page break placeholders (i.e.
            ``"page_break_placeholder"="<!-- page break -->").
            By default, ``None``.
        pytesseract_exe_fp : path-like, optional
            Path to the `pytesseract` executable. If specified, OCR will
            be used to extract text from scanned PDFs using Google's
            Tesseract.  By default ``None``.
        """
        super().__init__(file_cache_coroutine=file_cache_coroutine)
        self.content_fetcher = AsyncFetchWithRetry(
            header_template=header_template,
            verify_ssl=verify_ssl,
            aget_kwargs=aget_kwargs,
        )
        self.html_loader = _AsyncHTMLOnlyLoader(
            pw_launch_kwargs=pw_launch_kwargs,
            html_read_kwargs=html_read_kwargs,
            html_read_coroutine=html_read_coroutine,
            browser_semaphore=browser_semaphore,
            use_scrapling_stealth=use_scrapling_stealth,
            num_pw_html_retries=num_pw_html_retries,
        )
        self.to_md_kwargs = to_md_kwargs or {}
        self.pytesseract_exe_fp = pytesseract_exe_fp

    async def fetch_all(self, *sources):
        """Fetch documents for all requested sources.

        Parameters
        ----------
        *sources
            Iterable of sources (as strings) used to fetch the
            documents.

        Returns
        -------
        list
            List of parsed documents.
        """
        outer_task_name = asyncio.current_task().get_name()
        fetches = [
            asyncio.create_task(self.fetch(source), name=outer_task_name)
            for source in sources
        ]
        docs = await asyncio.gather(*fetches)
        if docs:
            logger.debug(
                "Got the following doc types from initial fetch:\n\t- %s",
                "\n\t- ".join(
                    [
                        f"{doc.attrs['source']} -> {doc.attrs['doc_type']!r}"
                        for doc in docs
                    ]
                ),
            )

        to_re_fetch = [
            doc.attrs["source"]
            for doc in docs
            if doc.attrs["doc_type"].casefold() == "html"
        ]
        if to_re_fetch:
            logger.debug(
                "Loading HTML with Playwright for %d source(s):\n%r",
                len(to_re_fetch),
                to_re_fetch,
            )
            docs += await self.html_loader.fetch_all(*to_re_fetch)
        return docs

    async def _fetch_doc(self, url):
        """Fetch a doc using Docling"""

        out = await self.content_fetcher.fetch(url)
        if out is None:
            return MDDocument(pages=[]), None

        logger.debug("Got content from %r", url)
        raw_content, __, __, headers = out
        doc = await read_docling_web_file(
            raw_content,
            url=resolve_remote_filename(
                http_url=AnyHttpUrl(url), response_headers=dict(headers)
            ),
            headers=dict(headers),
            pytesseract_exe_fp=self.pytesseract_exe_fp,
            **self.to_md_kwargs,
        )

        if doc.attrs["doc_type"].casefold() != "html":
            doc.WRITE_KWARGS = {"mode": "wb"}
            doc.FILE_EXTENSION = doc.attrs["doc_type"]
            return doc, raw_content

        return doc, doc.text


class AsyncLocalDoclingFileLoader(BaseAsyncFileLoader):
    """Async local file loader using Docling"""

    def __init__(
        self,
        file_cache_coroutine=None,
        doc_attrs=None,
        to_md_kwargs=None,
        pytesseract_exe_fp=None,
        **__,  # consume any extra kwargs
    ):
        """

        Parameters
        ----------
        file_cache_coroutine : callable, optional
            File caching coroutine. Can be used to cache files
            downloaded by this class. Must accept an
            :obj:`~elm.web.document.BaseDocument` instance as the first
            argument and the file content to be written as the second
            argument. If this method is not provided, no document
            caching is performed. By default, ``None``.
        doc_attrs : dict, optional
            Additional document attributes to add to each loaded
            document. By default, ``None``.
        to_md_kwargs : dict, optional
            Keyword-value argument pairs to pass to to Docling's
            :func:`~docling_core.types.doc.DoclingDocument.export_to_markdown`
            method for converting the raw content to a markdown
            document. Can be useful to specify image placeholders (i.e.
            ``"image_placeholder"=""``) or page break placeholders (i.e.
            ``"page_break_placeholder"="<!-- page break -->").
            By default, ``None``.
        pytesseract_exe_fp : path-like, optional
            Path to the `pytesseract` executable. If specified, OCR will
            be used to extract text from scanned PDFs using Google's
            Tesseract.  By default ``None``.
        """
        super().__init__(file_cache_coroutine=file_cache_coroutine)
        self.to_md_kwargs = to_md_kwargs or {}
        self.doc_attrs = doc_attrs or {}
        self.pytesseract_exe_fp = pytesseract_exe_fp

    async def _fetch_doc(self, source):
        """Load a doc by reading file based on extension"""
        doc, raw_content = await read_docling_local_file(
            source,
            pytesseract_exe_fp=self.pytesseract_exe_fp,
            **self.to_md_kwargs,
        )

        if doc.attrs["doc_type"].casefold() != "html":
            doc.WRITE_KWARGS = {"mode": "wb"}
            doc.FILE_EXTENSION = doc.attrs["doc_type"]
            return doc, raw_content

        return doc, doc.text

    async def _fetch_doc_with_url_in_metadata(self, source):
        """Fetch doc contents and add source to metadata"""
        doc, raw_content = await self._fetch_doc(source)
        for key, value in self.doc_attrs.items():
            doc.attrs[key] = value
        doc.attrs["source_fp"] = source
        return doc, raw_content


if os.environ.get("COMPASS_FILE_LOAD_BACKEND", "elm") == "docling":
    COMPASSWebFileLoader = AsyncDoclingWebFileLoader
    COMPASSLocalFileLoader = AsyncLocalFileLoader
else:
    COMPASSWebFileLoader = AsyncWebFileLoader
    COMPASSLocalFileLoader = AsyncLocalFileLoader
