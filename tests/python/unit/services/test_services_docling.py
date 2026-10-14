"""Test COMPASS CPU Services"""

import logging
import os
import sys
import time
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from pdf2image import convert_from_bytes
from docling_core.types.doc import DocItemLabel, DoclingDocument
from docling_rs.datamodel.base_models import ConversionStatus, InputFormat
from docling_rs.exceptions import ConversionError

from compass.services.docling import (
    TIMEOUT_PARAMS,
    _read_docling,
    _read_docling_catch_error,
    _read_docling_pdf_images,
    _read_docling_without_timeout,
    _read_file_docling,
    _receive_docling_result,
    _run_docling_in_subprocess,
    _run_docling_subprocess,
)
from compass.exceptions import COMPASSValueError


logger = logging.getLogger("compass")


def _write_to_process_streams():
    """Write to stdout/stderr from a worker process"""
    print("PROCESS STDOUT", flush=True)
    print("PROCESS STDERR", file=sys.stderr, flush=True)
    return "STREAMED"


def _return_from_subprocess(value):
    """Return a serializable value from a child process"""
    return value


def _block_subprocess(seconds):
    """Block a child process long enough for a deadline to expire"""
    time.sleep(seconds)


def _fail_subprocess():
    """Raise a conversion-like failure in a child process"""
    raise ValueError("invalid document")


def _exit_subprocess():
    """Exit without sending a conversion result"""
    os._exit(1)


@pytest.fixture
def converter_state(monkeypatch):
    """Capture converter inputs without invoking external models"""
    state = {"status": "success"}

    def export_markdown(**kwargs):
        state["export_kwargs"] = kwargs
        return state.get("markdown", "<!-- page break -->\nmarkdown body")

    class FakeDocumentConverter:
        def __init__(self, format_options, document_timeout=None):
            state["format_options"] = format_options
            state["document_timeout"] = document_timeout

        def convert(self, stream):
            state["stream_name"] = stream.name
            state["stream_bytes"] = stream.stream.read()
            state["stream_format"] = stream.format
            if "error" in state:
                raise state["error"]
            return SimpleNamespace(
                input=SimpleNamespace(file=Path(stream.name)),
                document=SimpleNamespace(
                    export_to_markdown=export_markdown,
                    pages={1: "page 1", 2: "page 2"},
                ),
                status=SimpleNamespace(value=state["status"]),
            )

    monkeypatch.setattr(
        "compass.services.docling.DocumentConverter", FakeDocumentConverter
    )
    return state


@pytest.mark.parametrize(
    "configured_options,tesseract_path,expected_tables,expected_ocr",
    [
        ({}, None, True, True),
        ({}, "tesseract", True, True),
        (
            {"do_table_structure": False, "do_ocr": False},
            "tesseract",
            False,
            False,
        ),
        ({"do_table_structure": True, "do_ocr": True}, None, True, True),
        ({"document_timeout": 30}, None, True, True),
    ],
)
def test_read_docling_honors_native_pdf_options(
    converter_state,
    configured_options,
    tesseract_path,
    expected_tables,
    expected_ocr,
):
    """Honor native PDF options without requiring a Tesseract executable"""
    doc = _read_docling_without_timeout(
        b"<html></html>",
        "sample.html",
        pdf_pipeline_options=configured_options,
        pytesseract_exe_fp=tesseract_path,
    )

    assert "markdown body" in doc.text
    assert doc.attrs["from_ocr"] is configured_options.get("do_ocr", True)
    assert converter_state["document_timeout"] == configured_options.get(
        "document_timeout"
    )
    pdf_options = converter_state["format_options"][InputFormat.PDF]
    pdf_options = pdf_options.pipeline_options
    assert pdf_options.do_table_structure is expected_tables
    assert pdf_options.do_ocr is expected_ocr
    assert pdf_options.table_structure_options.do_cell_matching


@pytest.mark.parametrize("source_uri", [None, "https://example.com/code/"])
def test_docling_converter_preserves_inputs_and_metadata(
    converter_state, source_uri
):
    """Forward conversion inputs and preserve useful result metadata"""
    converter_state["status"] = "partial_success"
    configured_options = {
        "table_structure_options": {"do_cell_matching": False},
    }
    doc = _read_docling_without_timeout(
        b"<html>ordinance</html>",
        Path("sample.html"),
        headers=[("Content-Type", "text/html")],
        source_uri=source_uri,
        pdf_pipeline_options=configured_options,
        image_placeholder="",
        page_break_placeholder="<!-- page break -->",
    )

    assert converter_state["stream_name"] == "sample.html"
    assert converter_state["stream_bytes"] == b"<html>ordinance</html>"
    assert converter_state["stream_format"] == InputFormat.HTML
    assert converter_state["export_kwargs"] == {
        "image_placeholder": "",
        "page_break_placeholder": "<!-- page break -->",
    }
    options = converter_state["format_options"]
    assert set(options) == {InputFormat.PDF}
    assert not (
        options[
            InputFormat.PDF
        ].pipeline_options.table_structure_options.do_cell_matching
    )
    assert configured_options == {
        "table_structure_options": {"do_cell_matching": False},
    }
    assert doc.attrs["doc_filename"] == "sample"
    assert doc.attrs["doc_type"] == "html"
    assert doc.attrs["conversion_status"] == "partial_success"
    assert doc.attrs["num_pages"] == 2
    assert doc.attrs["from_ocr"] is True
    assert doc.attrs["conversion_time_seconds"] >= 0
    assert "<!-- page break -->" in doc.text


@pytest.mark.parametrize(
    "source,status,do_ocr",
    [
        ("empty.pdf", "success", False),
        ("empty.pdf", "partial_success", True),
        ("empty.html", "success", True),
    ],
)
def test_docling_empty_results_do_not_trigger_unwanted_ocr(
    converter_state, source, status, do_ocr
):
    """Respect OCR opt-outs and avoid retries for partial or non-PDF input"""
    converter_state.update(markdown="", status=status)

    doc = _read_docling_without_timeout(
        b"document", source, pdf_pipeline_options={"do_ocr": do_ocr}
    )

    assert doc.empty
    assert doc.attrs["conversion_status"] == status
    assert doc.attrs["doc_type"] == Path(source).suffix.lstrip(".")


@pytest.mark.parametrize(
    "source,do_ocr,message",
    [
        ("sample.pdf", False, "pdfium support is not compiled in"),
        ("sample.html", True, "pdfium support is not compiled in"),
        ("sample.pdf", True, "invalid PDF"),
    ],
)
def test_docling_pdfium_retry_does_not_mask_other_errors(
    converter_state, source, do_ocr, message
):
    """Propagate errors when the PDFium retry conditions are not met"""
    error = ConversionError(message)
    converter_state["error"] = error

    with pytest.raises(ConversionError, match=message) as raised:
        _read_docling_without_timeout(
            b"document", source, pdf_pipeline_options={"do_ocr": do_ocr}
        )

    assert raised.value is error


@pytest.mark.parametrize(
    "outcome,entry_point",
    [
        ("success", "helper"),
        ("partial_success", "helper"),
        ("success", "reader"),
        ("partial_success", "reader"),
        ("success", "pdfium_reader"),
        ("partial_success", "pdfium_reader"),
        ("conversion_error", "helper"),
        ("encoding_error", "helper"),
        ("render_error", "helper"),
    ],
)
# ruff: ignore[complex-structure]
def test_docling_pdf_retry_releases_each_page(
    monkeypatch, tmp_path, outcome, entry_point
):
    """Recover failed or empty PDFs and close rendered images"""
    source = tmp_path / "scanned.pdf"
    with Image.new("RGB", (20, 30), "white") as image:
        image.save(
            source, format="PDF", save_all=True, append_images=[image, image]
        )
    rendered_images = []
    converted_names = []
    native_pdf_inputs = []

    def fail_encoding(*args, **kwargs):
        raise RuntimeError("encoding failed")

    def render_page(doc_bytes, **kwargs):
        for previous_image in rendered_images:
            with pytest.raises(ValueError):
                previous_image.getpixel((0, 0))
        page_number = len(rendered_images) + 1
        assert kwargs.get("first_page") == page_number
        assert kwargs.get("last_page") == page_number
        if outcome == "render_error" and page_number == 2:
            raise RuntimeError("render failed")
        images = convert_from_bytes(doc_bytes, **kwargs)
        assert len(images) == 1
        rendered_images.extend(images)
        if outcome == "encoding_error" and page_number == 2:
            monkeypatch.setattr(images[0], "save", fail_encoding)
        return images

    def convert(stream):
        if stream.format == InputFormat.PDF:
            native_pdf_inputs.append(stream.name)
            assert stream.stream.read() == source.read_bytes()
            if entry_point == "pdfium_reader":
                raise ConversionError("pdfium support is not compiled in")
            return SimpleNamespace(
                document=DoclingDocument(name=stream.name),
                status=ConversionStatus.SUCCESS,
            )
        page_number = len(converted_names) + 1
        converted_names.append(stream.name)
        assert stream.format == InputFormat.IMAGE
        assert stream.stream.read().startswith(b"\x89PNG")
        if outcome == "conversion_error" and page_number == 2:
            raise RuntimeError("conversion failed")
        document = DoclingDocument.model_validate(
            {
                "name": stream.name,
                "pages": {
                    1: {
                        "page_no": 1,
                        "size": {"width": 20, "height": 30},
                    }
                },
            }
        )
        document.add_text(label=DocItemLabel.TEXT, text=f"Page {page_number}")
        document.add_text(
            label=DocItemLabel.SECTION_HEADER, text="Excluded heading"
        )
        status = (
            ConversionStatus.PARTIAL_SUCCESS
            if outcome == "partial_success" and page_number == 2
            else ConversionStatus.SUCCESS
        )
        return SimpleNamespace(document=document, status=status)

    monkeypatch.setattr(
        "compass.services.docling.convert_from_bytes", render_page
    )
    converter = SimpleNamespace(convert=convert)
    monkeypatch.setattr(
        "compass.services.docling.DocumentConverter",
        lambda **__: converter,
    )
    if outcome.endswith("error"):
        with pytest.raises(RuntimeError, match="failed"):
            _read_docling_pdf_images(
                converter, source.read_bytes(), str(source)
            )
    else:
        export_options = {"labels": {DocItemLabel.TEXT}}
        if entry_point in {"reader", "pdfium_reader"}:
            doc = _read_docling_without_timeout(
                source.read_bytes(),
                str(source),
                pdf_pipeline_options={"do_ocr": True},
                **export_options,
            )
            markdown = doc.text
            assert native_pdf_inputs == [str(source)]
            assert doc.attrs["doc_filename"] == source.stem
            assert doc.attrs["doc_type"] == "pdf"
            assert doc.attrs["conversion_status"] == outcome
            assert doc.attrs["num_pages"] == 3
            assert doc.attrs["from_ocr"] is True
        else:
            result = _read_docling_pdf_images(
                converter, source.read_bytes(), str(source)
            )
            assert result.status == ConversionStatus(outcome)
            assert result.input.file.name == source.name
            assert list(result.document.pages) == [1, 2, 3]
            markdown = result.document.export_to_markdown(**export_options)
        assert markdown.split() == [
            "Page",
            "1",
            "Page",
            "2",
            "Page",
            "3",
        ]
        assert converted_names == [
            "scanned-1.png",
            "scanned-2.png",
            "scanned-3.png",
        ]
    for image in rendered_images:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


@pytest.mark.parametrize(
    "timeout", [None, True, False, "120", 0, -1, np.nan, np.inf, -np.inf]
)
def test_docling_rejects_invalid_deadlines(timeout):
    """Reject invalid deadlines before starting conversion processes"""
    with pytest.raises(COMPASSValueError, match="positive number"):
        _read_docling(
            b"invalid",
            "sample.pdf",
            pdf_pipeline_options={"document_timeout": timeout},
        )


@pytest.mark.parametrize(
    "error", [TimeoutError("deadline"), ValueError("bad")]
)
def test_docling_conversion_errors_return_empty_documents(monkeypatch, error):
    """Return an empty unknown document for failed conversions"""

    def fail_conversion(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        "compass.services.docling._read_docling", fail_conversion
    )
    doc = _read_docling_catch_error(b"invalid", "sample.pdf")

    assert doc.empty
    assert doc.attrs["doc_type"] == "unknown"


@pytest.mark.parametrize(
    "configured_options,expected_timeout",
    [
        (None, 3600),
        ({"document_timeout": 120}, 132),
        ({"document_timeout": 0.5}, 0.55),
    ],
)
def test_read_docling_uses_process_deadline(
    monkeypatch, configured_options, expected_timeout
):
    """Docling deadlines should run outside the process-pool worker"""
    captured = {}
    expected = object()

    def _run_in_subprocess(fn, *, args, kwargs, timeout):
        captured["fn"] = fn
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["timeout"] = timeout
        return expected

    monkeypatch.setattr(
        "compass.services.docling._run_docling_in_subprocess",
        _run_in_subprocess,
    )

    result = _read_docling(
        b"%PDF",
        "sample.pdf",
        pdf_pipeline_options=configured_options,
        source_uri="https://example.com/code.pdf",
        headers={"Content-Type": "application/pdf"},
        pytesseract_exe_fp="tesseract",
        image_placeholder="",
    )

    assert result is expected
    assert captured["fn"] is _read_docling_without_timeout
    assert captured["args"] == (b"%PDF", "sample.pdf")
    assert captured["kwargs"] == {
        "pdf_pipeline_options": configured_options or {},
        "source_uri": "https://example.com/code.pdf",
        "headers": {"Content-Type": "application/pdf"},
        "pytesseract_exe_fp": "tesseract",
        "image_placeholder": "",
    }
    assert captured["timeout"] == pytest.approx(expected_timeout)


@pytest.mark.parametrize("extension", ["md", "txt"])
def test_docling_local_reader_returns_original_bytes(tmp_path, extension):
    """Infer Markdown for text files and return their original bytes"""
    source = tmp_path / f"ordinance.{extension}"
    content = (
        b"# Example County\n\nWind turbines require 125 feet of setback.\n"
    )
    source.write_bytes(content)

    doc, raw = _read_file_docling(
        source, pdf_pipeline_options={"document_timeout": 30}
    )

    assert raw == content
    assert not doc.empty
    assert "125 feet" in doc.text
    assert doc.attrs["doc_filename"] == "ordinance"
    assert doc.attrs["doc_type"] == "md"


def test_docling_local_reader_reports_missing_files(tmp_path):
    """Report missing local files before attempting a conversion"""
    with pytest.raises(FileNotFoundError):
        _read_file_docling(tmp_path / "missing.pdf")


@pytest.mark.parametrize(
    "source,headers",
    [
        ("sample.html", None),
        ("sample.htm", None),
        ("https://example.com/code.html?version=2", None),
        ("download", {"content-type": "text/html; charset=utf-8"}),
        ("sample.pdf", [("Content-Type", "text/html")]),
    ],
)
@pytest.mark.parametrize(
    "base,expected_url",
    [
        ("", "https://example.com/code/rules"),
        (
            '<base href="../ordinances/">',
            "https://example.com/ordinances/rules",
        ),
    ],
)
def test_docling_html_resolves_source_links(
    source, headers, base, expected_url
):
    """Convert real HTML with MIME hints and source-relative links"""
    content = (
        f"<html><head>{base}</head><body>"
        '<a href="rules">County ordinance</a></body></html>'
    ).encode()

    doc = _read_docling(
        content,
        source,
        headers=headers,
        source_uri="https://example.com/code/",
        pdf_pipeline_options={"document_timeout": 30},
        pytesseract_exe_fp="unused-tesseract",
    )

    assert not doc.empty
    assert f"[County ordinance]({expected_url})" in doc.text
    assert doc.attrs["doc_type"] == "html"
    assert doc.attrs["conversion_status"] == "success"
    assert doc.attrs["from_ocr"] is True


@pytest.mark.parametrize(
    "source,headers,expected_format",
    [
        ("CODE.PDF", None, InputFormat.PDF),
        ("code.pdf?download=1", None, InputFormat.PDF),
        ("code.bin", {"Content-Type": "application/pdf"}, InputFormat.PDF),
        ("code.docx", None, InputFormat.DOCX),
        ("code.png", None, InputFormat.IMAGE),
        ("code.bin", {"Content-Type": "image/png"}, InputFormat.IMAGE),
        ("code.txt", None, InputFormat.MD),
    ],
)
def test_docling_stream_preserves_format_hints(
    converter_state, source, headers, expected_format
):
    """Use native format hints without forwarding unsupported headers"""
    doc = _read_docling_without_timeout(b"document", source, headers=headers)

    assert converter_state["stream_format"] == expected_format
    assert converter_state["stream_bytes"] == b"document"
    assert doc.attrs["doc_type"] == expected_format.value


def test_docling_subprocess_returns_result():
    """The Docling child process should return completed conversions"""
    result = _run_docling_in_subprocess(
        _return_from_subprocess,
        args=("converted",),
        kwargs={},
        timeout=60,
    )

    assert result == "converted"


def test_docling_subprocess_enforces_deadline(monkeypatch):
    """The Docling child process should be stopped at its deadline"""
    monkeypatch.setitem(TIMEOUT_PARAMS, "shutdown_timeout", 0.1)
    monkeypatch.setitem(TIMEOUT_PARAMS, "force_shutdown_timeout", 0.1)

    start_time = time.monotonic()
    with pytest.raises(TimeoutError, match="Docling conversion exceeded"):
        _run_docling_in_subprocess(
            _block_subprocess,
            args=(10,),
            kwargs={},
            timeout=0.1,
        )

    assert time.monotonic() - start_time < 1


@pytest.mark.parametrize(
    "child,expected_message",
    [
        (_fail_subprocess, "ValueError: invalid document"),
        (_exit_subprocess, "exited without a result"),
    ],
)
def test_docling_subprocess_reports_failures(child, expected_message):
    """Report child exceptions and abrupt exits without hanging"""
    with pytest.raises(ConversionError, match=expected_message):
        _run_docling_in_subprocess(child, args=(), kwargs={}, timeout=60)


def test_docling_subprocess_forwards_streams(caplog):
    """Forward child stdout and stderr while waiting for the result"""
    with caplog.at_level(logging.INFO, logger="compass.services.cpu"):
        result = _run_docling_in_subprocess(
            _write_to_process_streams, args=(), kwargs={}, timeout=60
        )

    assert result == "STREAMED"
    assert "PROCESS STDOUT" in caplog.text
    assert "PROCESS STDERR" in caplog.text


@pytest.mark.parametrize("child", [_return_from_subprocess, _fail_subprocess])
@pytest.mark.parametrize("closed_receiver", [False, True])
def test_docling_worker_sends_outcome_and_closes_pipe(
    monkeypatch, child, closed_receiver
):
    """Serialize child outcomes and tolerate disconnected parent pipes"""
    monkeypatch.setattr(
        "compass.services.docling.configure_docling_subprocess_logging", Mock()
    )
    receiver, sender = multiprocessing.Pipe(duplex=False)
    with receiver, sender:
        if closed_receiver:
            receiver.close()
        args = ("converted",) if child is _return_from_subprocess else ()
        _run_docling_subprocess(sender, child, args, {})

        assert sender.closed
        if not closed_receiver:
            assert receiver.poll(1)
            expected = (
                ("success", "converted")
                if child is _return_from_subprocess
                else ("error", "ValueError: invalid document")
            )
            assert receiver.recv() == expected


@pytest.mark.parametrize("buffered_result", [False, True])
def test_docling_receive_handles_exited_process(buffered_result):
    """Read buffered results or report a child that exited silently"""
    receiver = Mock()
    receiver.poll.side_effect = (
        [False, True, True] if buffered_result else [False, False]
    )
    receiver.recv.return_value = ("success", "converted")
    process = Mock()
    process.is_alive.return_value = False

    if buffered_result:
        assert _receive_docling_result(receiver, process, 60) == (
            "success",
            "converted",
        )
    else:
        with pytest.raises(ConversionError, match="exited without a result"):
            _receive_docling_result(receiver, process, 60)
        receiver.recv.assert_not_called()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
