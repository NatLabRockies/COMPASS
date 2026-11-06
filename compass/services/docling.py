"""COMPASS Docling-based services"""

import time
import math
import logging
import contextlib
import multiprocessing
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from elm.web.document import MDDocument
from pdf2image import convert_from_bytes, pdfinfo_from_bytes

from compass.services.cpu import (
    TIMEOUT_PARAMS,
    FileLoader,
    _force_shutdown_processes,
)
from compass.exceptions import COMPASSImportError, COMPASSValueError
from compass.utilities.logs import configure_docling_subprocess_logging


try:
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
except ModuleNotFoundError as exc:
    if exc.name not in {"docling_rs", "docling_core"}:
        raise

    msg = (
        "The Docling backend requires docling-rs and docling-core. "
        "These dependencies are installed by default except on macOS. "
        "Use COMPASS_FILE_LOAD_BACKEND=elm when they are unavailable, "
        "or install them explicitly with "
        'pip install "docling-rs>=1.104.3,<2" "docling-core>=2.101.1,<3".'
    )
    raise COMPASSImportError(msg) from exc


logger = logging.getLogger(__name__)


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


def _read_docling_catch_error(
    doc_bytes,
    file_source,
    headers=None,
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

    kwargs.pop("pytesseract_exe_fp", None)  # safeguard
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

    conv_result = _run_conversion_with_fallback(
        doc_converter,
        stream,
        input_format,
        pipeline_options,
        doc_bytes,
        file_source,
    )
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
        "from_ocr": configured_options["do_ocr"],
    }
    return MDDocument([doc_text], attrs=attrs, remove_comments=False)


def _run_conversion_with_fallback(
    doc_converter,
    stream,
    input_format,
    pipeline_options,
    doc_bytes,
    file_source,
):
    """Run conversion with fallback for pdfium error"""
    try:
        conv_result = doc_converter.convert(stream)
    except ConversionError as error:
        if (
            input_format != InputFormat.PDF
            or not pipeline_options.do_ocr
            or "pdfium support is not compiled in" not in str(error)
        ):
            raise
        logger.info(
            "Native PDF reader unavailable; retrying with Poppler renders"
        )
        conv_result = _read_docling_pdf_images(
            doc_converter, doc_bytes, file_source
        )
    return conv_result


def _read_docling_pdf_images(converter, doc_bytes, file_source):
    """Retry empty PDFs with Poppler renders and native OCR"""
    page_count = pdfinfo_from_bytes(doc_bytes)["Pages"]
    documents = []
    conversion_status = ConversionStatus.SUCCESS
    for page_number in range(1, page_count + 1):
        images = convert_from_bytes(
            doc_bytes, first_page=page_number, last_page=page_number
        )
        try:
            for image in images:
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
        msg = "`document_timeout` must be a positive number of seconds"
        raise COMPASSValueError(msg)


def _read_file_docling(fp, **kwargs):
    """Read a local file using Docling"""

    fp = Path(fp)
    doc_bytes = fp.read_bytes()
    doc = _read_docling_catch_error(
        doc_bytes, str(fp).replace(".txt", ".md"), headers=None, **kwargs
    )
    return doc, doc_bytes
