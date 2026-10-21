"""Test COMPASS CPU Services"""

import logging
import os
import sys
import time
import asyncio
import platform
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from docling.datamodel.base_models import InputFormat
from docling.exceptions import ConversionError

from compass.exceptions import COMPASSValueError
from compass.services.cpu import (
    TIMEOUT_PARAMS,
    ProcessPoolService,
    _read_docling,
    _read_docling_catch_error,
    _read_docling_without_timeout,
    _read_file_docling,
    _receive_docling_result,
    _run_docling_in_subprocess,
    _run_docling_subprocess,
)
from compass.services.provider import RunningAsyncServices
from compass.utilities.logs import LocationFileLog, LogListener


logger = logging.getLogger("compass")


def _log_from_process():
    """Call logger instance from a process"""
    msg = "A DEBUG LOG"
    logger.debug(msg)
    msg = "HELLO WORLD"
    logger.info(msg)
    return msg


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


def _mps_is_available():
    """Return whether the PyTorch MPS (Metal) backend is usable"""
    return torch.backends.mps.is_available()


def _sum_ones_on_mps():
    """Run a tiny tensor op on the MPS device from a child process"""
    return (torch.ones(3, device="mps") * 2).sum().item()


@pytest.fixture
def converter_state(monkeypatch):
    """Capture converter inputs without invoking external models"""
    state = {
        "mean_score": np.nan,
        "low_score": pd.NA,
        "ocr_score": np.nan,
        "status": "success",
    }

    def export_markdown(**kwargs):
        state["export_kwargs"] = kwargs
        return "<!-- page break -->\nmarkdown body"

    class FakeDocumentConverter:
        def __init__(self, format_options):
            state["format_options"] = format_options

        def convert(self, stream, headers=None):
            state["stream_name"] = stream.name
            state["stream_bytes"] = stream.stream.read()
            state["headers"] = headers
            return SimpleNamespace(
                confidence=SimpleNamespace(
                    mean_score=state["mean_score"],
                    low_score=state["low_score"],
                    pages={0: SimpleNamespace(ocr_score=state["ocr_score"])},
                ),
                input=SimpleNamespace(
                    file=Path(stream.name),
                    format=SimpleNamespace(value="html"),
                ),
                pages=["page 1", "page 2"],
                document=SimpleNamespace(export_to_markdown=export_markdown),
                status=SimpleNamespace(value=state["status"]),
            )

    monkeypatch.setattr(
        "compass.services.cpu.DocumentConverter", FakeDocumentConverter
    )
    return state


@pytest.mark.asyncio
async def test_logging_within_service(tmp_path):
    """Test that child-process logs are forwarded to the listener"""

    class ProcessLogging(ProcessPoolService):
        """Subclass for testing"""

        @property
        def can_process(self):
            return True

        async def process(self):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.pool, _log_from_process)

    log_listener = LogListener(["compass"], level="DEBUG")
    services = [ProcessLogging()]
    captured_records = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured_records.append(record)

    async with RunningAsyncServices(services), log_listener as ll:
        capture_handler = _CaptureHandler(level=logging.DEBUG)
        ll.addHandler(capture_handler)
        with LocationFileLog(ll, tmp_path, location="test_loc", level="DEBUG"):
            msg = await ProcessLogging.call()
            for _ in range(30):
                messages = {record.message for record in captured_records}
                if "[compass] HELLO WORLD" in messages:
                    break
                await asyncio.sleep(0.1)
        ll.removeHandler(capture_handler)

    assert msg == "HELLO WORLD"
    assert any(
        record.message == "[compass] HELLO WORLD"
        for record in captured_records
    ), {record.message for record in captured_records}
    assert not any(
        record.message == "[compass] A DEBUG LOG"
        for record in captured_records
    ), {record.message for record in captured_records}


@pytest.mark.asyncio
async def test_process_streams_are_forwarded_to_logs(capfd):
    """Test that worker stdout/stderr are logged instead of printed"""

    class ProcessStreamLogging(ProcessPoolService):
        """Subclass for testing"""

        @property
        def can_process(self):
            return True

        async def process(self):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self.pool, _write_to_process_streams
            )

    log_listener = LogListener(["compass"], level="DEBUG")
    services = [ProcessStreamLogging()]
    captured_records = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record):
            captured_records.append(record)

    async with RunningAsyncServices(services), log_listener as ll:
        capture_handler = _CaptureHandler(level=logging.INFO)
        ll.addHandler(capture_handler)
        msg = await ProcessStreamLogging.call()
        for _ in range(30):
            messages = {record.message for record in captured_records}
            if {
                "[compass.subprocess.stdout] PROCESS STDOUT",
                "[compass.subprocess.stderr] PROCESS STDERR",
            } <= messages:
                break
            await asyncio.sleep(0.1)
        ll.removeHandler(capture_handler)

    assert msg == "STREAMED"
    assert any(
        record.message == "[compass.subprocess.stdout] PROCESS STDOUT"
        for record in captured_records
    ), {record.message for record in captured_records}
    assert any(
        record.message == "[compass.subprocess.stderr] PROCESS STDERR"
        for record in captured_records
    ), {record.message for record in captured_records}


@pytest.mark.parametrize(
    "configured_options,tesseract_path,expected_tables,expected_ocr",
    [
        ({}, None, True, False),
        ({}, "tesseract", True, True),
        (
            {"do_table_structure": False, "do_ocr": False},
            "tesseract",
            False,
            False,
        ),
        ({"do_table_structure": True, "do_ocr": True}, None, True, True),
    ],
)
def test_read_docling_converts_missing_confidences_to_none(
    converter_state,
    configured_options,
    tesseract_path,
    expected_tables,
    expected_ocr,
):
    """Honor explicit PDF options while normalizing missing confidences"""
    doc = _read_docling_without_timeout(
        b"<html></html>",
        "sample.html",
        pdf_pipeline_options=configured_options,
        pytesseract_exe_fp=tesseract_path,
    )

    assert "markdown body" in doc.text
    assert doc.attrs["mean_confidence"] is None
    assert doc.attrs["low_score_confidence"] is None
    pdf_options = converter_state["format_options"][InputFormat.PDF]
    pdf_options = pdf_options.pipeline_options
    assert pdf_options.do_table_structure is expected_tables
    assert pdf_options.do_ocr is expected_ocr
    assert pdf_options.table_structure_options.do_cell_matching
    if tesseract_path:
        assert pdf_options.ocr_options.tesseract_cmd == tesseract_path


@pytest.mark.parametrize("source_uri", [None, "https://example.com/code/"])
def test_docling_converter_preserves_inputs_and_metadata(
    converter_state, source_uri
):
    """Forward conversion inputs and preserve useful result metadata"""
    converter_state.update(
        mean_score=0.9,
        low_score=0.5,
        ocr_score=0.8,
        status="partial_success",
    )
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
    assert converter_state["headers"] == {"Content-Type": "text/html"}
    assert converter_state["export_kwargs"] == {
        "image_placeholder": "",
        "page_break_placeholder": "<!-- page break -->",
    }
    options = converter_state["format_options"]
    assert str(options[InputFormat.HTML].backend_options.source_uri) == (
        source_uri or "sample.html"
    )
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
    assert doc.attrs["from_ocr"]
    assert doc.attrs["mean_confidence"] == pytest.approx(0.9)
    assert doc.attrs["low_score_confidence"] == pytest.approx(0.5)
    assert doc.attrs["conversion_time_seconds"] >= 0
    assert "<!-- page break -->" in doc.text


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

    monkeypatch.setattr("compass.services.cpu._read_docling", fail_conversion)
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
        "compass.services.cpu._run_docling_in_subprocess",
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


def test_docling_subprocess_returns_result():
    """The Docling child process should return completed conversions"""
    result = _run_docling_in_subprocess(
        _return_from_subprocess,
        args=("converted",),
        kwargs={},
        timeout=60,
    )

    assert result == "converted"


@pytest.mark.skipif(
    platform.system() != "Darwin" or not _mps_is_available(),
    reason="Fork-after-MPS crash requires the macOS MPS backend",
)
def test_docling_subprocess_survives_mps_initialized_parent():
    """Child conversions must run after the parent initializes MPS

    Docling pulls in PyTorch, which initializes the MPS (Metal)
    backend on macOS. Forking a child after that initialization
    crashes the child with SIGSEGV, so the conversion returns nothing
    and surfaces downstream as a misleading "no text found". Initialize
    MPS in the parent and confirm the subprocess still completes, which
    only holds when the spawn start method is used instead of fork.
    """
    torch.ones(1, device="mps").sum().item()

    result = _run_docling_in_subprocess(
        _sum_ones_on_mps, args=(), kwargs={}, timeout=120
    )

    assert result == pytest.approx(6.0)


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
        "compass.services.cpu.configure_docling_subprocess_logging", Mock()
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


def test_process_pool_release_resources_graceful_shutdown():  # ruff:ignore[complex-structure]
    """Graceful process-pool shutdown should not force worker exit"""

    class DummyService(ProcessPoolService):
        @property
        def can_process(self):
            return True

        async def process(self):
            return None

    class FakeThread:
        def __init__(self):
            self.join_calls = []
            self._alive = True

        def join(self, timeout=None):
            self.join_calls.append(timeout)
            self._alive = False

        def is_alive(self):
            return self._alive

    class FakeProcess:
        def __init__(self):
            self.terminate_calls = 0
            self.kill_calls = 0
            self.join_calls = []

        def is_alive(self):
            return False

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1

        def join(self, timeout=None):
            self.join_calls.append(timeout)

    class FakePool:
        def __init__(self):
            self.shutdown_calls = []
            self._executor_manager_thread = FakeThread()
            self._processes = {0: FakeProcess()}

        def shutdown(self, wait=True, cancel_futures=True):
            self.shutdown_calls.append((wait, cancel_futures))

    service = DummyService()
    pool = FakePool()
    service.pool = pool

    service.release_resources()

    assert service.pool is None
    assert pool.shutdown_calls == [(False, True)]
    assert pool._executor_manager_thread.join_calls == [
        TIMEOUT_PARAMS["shutdown_timeout"]
    ]
    process = pool._processes[0]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0


def test_process_pool_release_resources_forces_stuck_shutdown():  # ruff:ignore[complex-structure]
    """Stuck process-pool shutdown should terminate then kill workers"""

    class DummyService(ProcessPoolService):
        @property
        def can_process(self):
            return True

        async def process(self):
            return None

    class FakeThread:
        def __init__(self):
            self.join_calls = []
            self._alive = True

        def join(self, timeout=None):
            self.join_calls.append(timeout)
            if len(self.join_calls) > 1:
                self._alive = False

        def is_alive(self):
            return self._alive

    class FakeProcess:
        def __init__(self):
            self.terminate_calls = 0
            self.kill_calls = 0
            self.join_calls = []
            self._alive = True

        def is_alive(self):
            return self._alive

        def terminate(self):
            self.terminate_calls += 1

        def kill(self):
            self.kill_calls += 1
            self._alive = False

        def join(self, timeout=None):
            self.join_calls.append(timeout)

    class FakePool:
        def __init__(self):
            self.shutdown_calls = []
            self._executor_manager_thread = FakeThread()
            self._processes = {0: FakeProcess()}

        def shutdown(self, wait=True, cancel_futures=True):
            self.shutdown_calls.append((wait, cancel_futures))

    service = DummyService()
    pool = FakePool()
    service.pool = pool

    service.release_resources()

    assert service.pool is None
    assert pool.shutdown_calls == [(False, True)]
    assert pool._executor_manager_thread.join_calls == [
        TIMEOUT_PARAMS["shutdown_timeout"],
        TIMEOUT_PARAMS["force_shutdown_timeout"],
    ]
    process = pool._processes[0]
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.join_calls == [
        TIMEOUT_PARAMS["force_shutdown_timeout"],
        TIMEOUT_PARAMS["force_shutdown_timeout"],
    ]


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
