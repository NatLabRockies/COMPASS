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

    # ruff:ignore[too-many-arguments, too-many-positional-arguments]
    def __init__(
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
        pdf_pipeline_options=None,
        re_fetch_failed_with_elm=False,
        **extra,
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
        pdf_pipeline_options : dict, optional
            Dictionary of keyword-value arguments to pass to
            :class:`docling.datamodel.pipeline_options.PdfPipelineOptions`
            initializer. Note that some options like
            ``do_table_structure``, ``table_structure_options``, and
            ``do_ocr`` are set automatically and cannot be overridden.
            If ``None``, the default options are used.
        re_fetch_failed_with_elm : bool, default=False
            Option to re-fetch failed sources using ELM's default
            fetcher. This can be useful if Docling fails to parse a
            document, but ELM's fetcher can still retrieve it. To make
            sure this functions properly, be sure to specify
            ``pdf_read_kwargs`` and ``pdf_read_coroutine``, in the
            ``extra`` kwargs as you would for the elm-based
            :class:`~elm.web.file_loader.AsyncWebFileLoader`.

            .. NOTE::

                This is meant to be a _fast_ fallback option for the
                longer Docling parse, so OCR PDF parsing is completely
                disabled for the ELM fallback.

            By default, ``False``.
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
        self.pdf_pipeline_options = pdf_pipeline_options

        self.failed_fetcher = None
        if re_fetch_failed_with_elm:
            self.failed_fetcher = AsyncWebFileLoader(
                header_template=header_template,
                verify_ssl=verify_ssl,
                aget_kwargs=aget_kwargs,
                pw_launch_kwargs=pw_launch_kwargs,
                pdf_read_kwargs=extra.get("pdf_read_kwargs"),
                html_read_kwargs=html_read_kwargs,
                pdf_read_coroutine=extra.get("pdf_read_coroutine"),
                html_read_coroutine=html_read_coroutine,
                pdf_ocr_read_coroutine=None,
                file_cache_coroutine=file_cache_coroutine,
                browser_semaphore=browser_semaphore,
                use_scrapling_stealth=use_scrapling_stealth,
                num_pw_html_retries=num_pw_html_retries,
            )

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
        docs = await self._fetch_docs_with_docling(sources)
        docs += await self._fetch_html_docs_again_using_playwright(docs)
        return await self._maybe_fetch_failed_docs_with_elm(docs, sources)

    async def _fetch_docs_with_docling(self, sources):
        """Fetch docs using Docling"""
        outer_task_name = asyncio.current_task().get_name()
        fetches = [
            asyncio.create_task(self.fetch(source), name=outer_task_name)
            for source in sources
        ]
        docs = await asyncio.gather(*fetches)
        docs = [doc for doc in docs if doc is not None and not doc.empty]
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
        return docs

    async def _fetch_html_docs_again_using_playwright(self, docs):
        """Fetch HTML docs using Playwright"""
        to_re_fetch = [
            doc.attrs["source"]
            for doc in docs
            if doc.attrs["doc_type"].casefold() == "html"
        ]
        if not to_re_fetch:
            return []

        logger.debug(
            "Loading HTML with Playwright for %d source(s):\n%r",
            len(to_re_fetch),
            to_re_fetch,
        )
        return await self.html_loader.fetch_all(*to_re_fetch)

    async def _maybe_fetch_failed_docs_with_elm(self, docs, sources):
        """Fetch docs that failed to load with ELM (if enabled)"""
        if self.failed_fetcher is None:
            return docs

        out_docs = []
        partial_fail_docs = {}
        failed_searches = []
        for source in sources:
            source_docs = [
                doc for doc in docs if doc.attrs["source"] == source
            ]
            if not source_docs:
                failed_searches.append(source)
                continue

            if len(source_docs) > 1:
                out_docs.extend(source_docs)
                continue

            doc = source_docs[0]
            if doc.attrs.get("conversion_status") != "success":
                failed_searches.append(source)
                partial_fail_docs[source] = doc
            else:
                out_docs.append(doc)

        if not failed_searches:
            return out_docs

        logger.debug(
            "Re-fetching %d failed source(s) with ELM:\n%r",
            len(failed_searches),
            failed_searches,
        )
        elm_docs = await self.failed_fetcher.fetch_all(*failed_searches)

        for elm_doc in elm_docs:
            docling_doc = partial_fail_docs.get(elm_doc.attrs["source"])
            elm_doc_failed = elm_doc.empty or "cache_fn" not in elm_doc.attrs
            if elm_doc_failed and docling_doc is not None:
                out_docs.append(docling_doc)
            else:
                out_docs.append(elm_doc)

        return out_docs

    async def _fetch_doc(self, url):
        """Fetch a doc using Docling"""

        out = await self.content_fetcher.fetch(url)
        if out is None:
            return MDDocument(pages=[]), None

        logger.debug("Got content from %r", url)
        raw_content, __, __, headers = out
        resolved_filename = resolve_remote_filename(
            http_url=AnyHttpUrl(url), response_headers=dict(headers)
        )
        logger.debug("Docling is starting content read from %r", url)
        try:
            doc = await read_docling_web_file(
                raw_content,
                url=resolved_filename,
                source_uri=url,
                headers=dict(headers),
                pytesseract_exe_fp=self.pytesseract_exe_fp,
                pdf_pipeline_options=self.pdf_pipeline_options,
                **self.to_md_kwargs,
            )
        except TimeoutError:
            logger.info("Docling parsing timed out for %r", url)
            return MDDocument(pages=[]), None

        if doc.empty:
            logger.info("Docling could not parse content from %s", url)
            return doc, None

        logger.debug(
            "Docling finished parsing %r:\n\t- Status: %r\n\t- "
            "Conversion time (s): %.2f\n\t- Num pages: %r\n\t- From OCR: %r",
            url,
            doc.attrs.get("conversion_status", "unknown"),
            doc.attrs.get("conversion_time_seconds", "unknown"),
            doc.attrs.get("num_pages", "unknown"),
            doc.attrs.get("from_ocr", "unknown"),
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
        pdf_pipeline_options=None,
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
        pdf_pipeline_options : dict, optional
            Dictionary of keyword-value arguments to pass to
            :class:`docling.datamodel.pipeline_options.PdfPipelineOptions`
            initializer. Note that some options like
            ``do_table_structure``, ``table_structure_options``, and
            ``do_ocr`` are set automatically and cannot be overridden.
            If ``None``, the default options are used.
        """
        super().__init__(file_cache_coroutine=file_cache_coroutine)
        self.to_md_kwargs = to_md_kwargs or {}
        self.doc_attrs = doc_attrs or {}
        self.pytesseract_exe_fp = pytesseract_exe_fp
        self.pdf_pipeline_options = pdf_pipeline_options

    async def _fetch_doc(self, source):
        """Load a doc by reading file based on extension"""
        logger.debug("Docling is starting content read from %s", source)
        try:
            doc, raw_content = await read_docling_local_file(
                source,
                pytesseract_exe_fp=self.pytesseract_exe_fp,
                pdf_pipeline_options=self.pdf_pipeline_options,
                **self.to_md_kwargs,
            )
        except TimeoutError:
            logger.info("Docling parsing timed out for %s", source)
            return MDDocument(pages=[]), None

        if doc.empty:
            logger.info("Docling could not parse content from %s", source)
            return doc, None

        logger.debug(
            "Docling finished parsing %s:\n\t- Status: %r\n\t- "
            "Conversion time (s): %.2f\n\t- Num pages: %r\n\t- From OCR: %r",
            source,
            doc.attrs.get("conversion_status", "unknown"),
            doc.attrs.get("conversion_time_seconds", "unknown"),
            doc.attrs.get("num_pages", "unknown"),
            doc.attrs.get("from_ocr", "unknown"),
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
    COMPASSLocalFileLoader = AsyncLocalDoclingFileLoader
else:
    COMPASSWebFileLoader = AsyncWebFileLoader
    COMPASSLocalFileLoader = AsyncLocalFileLoader
