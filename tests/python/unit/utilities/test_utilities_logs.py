"""Test COMPASS Ordinance logging logic"""

import sys
import json
import logging
import asyncio
from pathlib import Path

import pytest

from compass.exceptions import COMPASSValueError
from compass.services.provider import RunningAsyncServices
from compass.utilities.logs import (
    AddLocationFilter,
    ExceptionOnlyFilter,
    LocationFileLog,
    LocationFilter,
    LogListener,
    log_versions,
    NoLocationFilter,
    _JsonExceptionFileHandler,
    _JsonFormatter,
    _LocalProcessQueueHandler,
    _get_version,
    LOGGING_QUEUE,
)


@pytest.fixture(autouse=True, scope="module")
def _speed_up_location_file_log_async_exit():
    """Speed up async LocationFileLog tests by shortening sleep"""

    original_sleep = LocationFileLog.ASYNC_EXIT_SLEEP_SECONDS
    LocationFileLog.ASYNC_EXIT_SLEEP_SECONDS = 0.01
    try:
        yield
    finally:
        LocationFileLog.ASYNC_EXIT_SLEEP_SECONDS = original_sleep


@pytest.fixture(scope="module")
def compass_logger():
    """Provide compass logger with DEBUG_TO_FILE level for tests"""
    logger = logging.getLogger("compass")
    prev_level = logger.level
    logger.setLevel("DEBUG_TO_FILE")
    try:
        yield logger
    finally:
        logger.setLevel(prev_level)


class _DummyListener:
    def __init__(self):
        self.added_handlers = []
        self.removed_handlers = []

    def addHandler(self, handler):  # noqa: N802
        self.added_handlers.append(handler)

    def removeHandler(self, handler):  # noqa: N802
        self.removed_handlers.append(handler)


def _sample_log_record(
    *,
    name="test",
    level=logging.INFO,
    pathname="",
    lineno=0,
    msg="test",
    args=(),
    exc_info=None,
    func=None,
):
    """Create a base log record configured with common defaults"""

    return logging.LogRecord(
        name=name,
        level=level,
        pathname=pathname,
        lineno=lineno,
        msg=msg,
        args=args,
        exc_info=exc_info,
        func=func,
    )


def _attach_value_error_exc_info(record, message):
    """Attach ValueError exc_info to the provided log record"""

    try:
        raise ValueError(message)
    except ValueError:
        record.exc_info = sys.exc_info()


@pytest.mark.asyncio
async def test_logs_sent_to_separate_files(tmp_path, service_base_class):
    """Test that logs are correctly sent to individual files"""

    logger = logging.getLogger("ords")
    test_locations = ["a", "bc", "def", "ghij"]
    __, TestService = service_base_class

    assert not logger.handlers

    class AlwaysThreeService(TestService):
        """Test service that returns ``3``"""

        NUMBER = 3
        LEN_SLEEP = 0.1

    async def process_single(val):
        """Call `AlwaysThreeService`"""
        logger.info(f"This location is {val!r}")
        return await AlwaysThreeService.call(len(val))

    async def process_location_with_logs(listener, log_dir, location):
        """Process location and record logs for tests"""
        with LocationFileLog(listener, log_dir, location=location):
            logger.info("A generic test log")
            return await process_single(location)

    log_dir = tmp_path / "ord_logs"
    services = [AlwaysThreeService()]
    loggers = ["ords"]

    async with RunningAsyncServices(services), LogListener(loggers) as ll:
        producers = [
            asyncio.create_task(
                process_location_with_logs(ll, log_dir, loc), name=loc
            )
            for loc in test_locations
        ]
        await asyncio.gather(*producers)

    assert not logger.handlers

    log_files = list(log_dir.glob("*.log"))
    json_log_files = list(log_dir.glob("*.json"))
    assert len(log_files) == len(test_locations)
    assert len(json_log_files) == len(test_locations)
    for loc in test_locations:
        expected_log_file = log_dir / f"{loc}.log"
        assert expected_log_file.exists()
        log_text = expected_log_file.read_text(encoding="utf-8")
        assert "A generic test log" in log_text
        assert f"This location is {loc!r}" in log_text


@pytest.mark.asyncio
async def test_location_file_log_async_context(tmp_path):
    """Test async LocationFileLog context captures text and exception logs"""

    logger_name = "async_location_logger"
    logger = logging.getLogger(logger_name)
    logger.handlers = []

    log_dir = tmp_path / "async_logs"

    async def _produce_logs(listener):
        async with LocationFileLog(listener, log_dir, location="async_loc"):
            logger.info("async info message")
            try:
                raise ValueError("async failure")
            except ValueError:
                logger.exception("async failure")

    async with LogListener([logger_name], level="INFO") as listener:
        task = asyncio.create_task(_produce_logs(listener), name="async_loc")
        await task

    text_log = log_dir / "async_loc.log"
    json_log = log_dir / "async_loc exceptions.json"
    assert text_log.exists()
    assert json_log.exists()

    log_text = text_log.read_text(encoding="utf-8")
    assert "async info message" in log_text
    assert "async failure" in log_text

    json_content = json.loads(json_log.read_text(encoding="utf-8"))
    assert "test_utilities_logs" in json_content
    assert "ValueError" in json_content["test_utilities_logs"]
    entries = json_content["test_utilities_logs"]["ValueError"]
    assert len(entries) == 1
    assert entries[0]["message"] == "async failure"
    assert entries[0]["exc_text"] == "async failure"
    assert entries[0]["taskName"] == "async_loc"


def test_location_file_log_breakdown_without_handler(tmp_path):
    """Ensure handler teardown skips when handler is missing"""

    listener = _DummyListener()
    log = LocationFileLog(listener, tmp_path, location="loc")

    log._break_down_handler()  # no AttributeError thrown
    log._remove_handler_from_listener()

    assert not listener.removed_handlers


def test_location_file_log_exception_breakdown_without_handler(tmp_path):
    """Ensure exception handler teardown skips when handler missing"""

    listener = _DummyListener()
    log = LocationFileLog(listener, tmp_path, location="loc")

    log._break_down_exception_handler()  # no AttributeError thrown
    log._remove_exception_handler_from_listener()

    assert not listener.removed_handlers


def test_location_file_log_add_handler_without_setup(tmp_path):
    """Ensure add handler raises when handler not set up"""

    listener = _DummyListener()
    log = LocationFileLog(listener, tmp_path, location="loc")

    with pytest.raises(
        COMPASSValueError, match="Must set up handler before listener!"
    ):
        log._add_handler_to_listener()


def test_location_file_log_add_exception_handler_without_setup(tmp_path):
    """Ensure add exception handler raises when handler not set up"""

    listener = _DummyListener()
    log = LocationFileLog(listener, tmp_path, location="loc")

    with pytest.raises(
        COMPASSValueError,
        match="Must set up exception handler before listener!",
    ):
        log._add_exception_handler_to_listener()


def test_no_location_filter():
    """Test NoLocationFilter correctly identifies records without location"""
    filter_obj = NoLocationFilter()

    record = _sample_log_record()
    assert filter_obj.filter(record)

    record.location = None
    assert filter_obj.filter(record)

    record.location = "Task-42"
    assert filter_obj.filter(record)

    record.location = "main"
    assert filter_obj.filter(record)

    record.location = "El Paso Colorado"
    assert not filter_obj.filter(record)


def test_location_filter():
    """Test LocationFilter correctly filters records by specific location"""
    location = "El Paso Colorado"
    filter_obj = LocationFilter(location)

    record = _sample_log_record()
    record.location = location
    assert filter_obj.filter(record)

    record.location = "Denver Colorado"
    assert not filter_obj.filter(record)

    record_no_loc = _sample_log_record()
    assert not filter_obj.filter(record_no_loc)

    record.location = None
    assert not filter_obj.filter(record)


@pytest.mark.asyncio
async def test_add_location_filter_with_async_task():
    """Test AddLocationFilter adds location from async task name"""
    filter_obj = AddLocationFilter()

    async def task_with_name():
        await asyncio.sleep(0)
        record = _sample_log_record()
        result = filter_obj.filter(record)
        assert result
        assert hasattr(record, "location")
        assert record.location == "test_location"
        return True

    task = asyncio.create_task(task_with_name(), name="test_location")
    await task


def test_add_location_filter_without_async_task():
    """Test AddLocationFilter defaults to 'main' when no async task"""
    filter_obj = AddLocationFilter()

    record = _sample_log_record()
    result = filter_obj.filter(record)
    assert result
    assert hasattr(record, "location")
    assert record.location == "main"


@pytest.mark.asyncio
async def test_add_location_filter_with_task_xx():
    """Test AddLocationFilter defaults to 'main' for Task-XX names"""
    filter_obj = AddLocationFilter()

    async def task_with_generic_name():
        await asyncio.sleep(0)
        record = _sample_log_record()
        result = filter_obj.filter(record)
        assert result
        assert hasattr(record, "location")
        assert record.location == "main"
        return True

    task = asyncio.create_task(task_with_generic_name(), name="Task-42")
    await task


def test_exception_only_filter():
    """Test ExceptionOnlyFilter only passes through exception records"""
    filter_obj = ExceptionOnlyFilter()

    record = _sample_log_record()
    assert not filter_obj.filter(record)

    _attach_value_error_exc_info(record, "test error")

    assert filter_obj.filter(record)
    non_exception_record = _sample_log_record(msg="plain")
    assert not filter_obj.filter(non_exception_record)


def test_json_formatter():
    """Test _JsonFormatter correctly formats log records to dictionaries"""
    formatter = _JsonFormatter()

    record = _sample_log_record(
        pathname="test.py",
        lineno=42,
        msg="test message",
        func="test_func",
    )
    record.taskName = "test_task"

    result = formatter.format(record)
    assert isinstance(result, dict)
    assert result["message"] == "test message"
    assert result["filename"] == "test.py"
    assert result["funcName"] == "test_func"
    assert result["taskName"] == "test_task"
    assert result["lineno"] == 42
    assert result["exc_info"] is None
    assert result["exc_text"] is None
    assert "timestamp" in result


def test_json_formatter_with_exception():
    """Test _JsonFormatter correctly formats exception information"""
    formatter = _JsonFormatter()

    record = _sample_log_record(
        level=logging.ERROR,
        pathname="test.py",
        lineno=42,
        msg="test error",
        func="test_func",
    )
    record.taskName = "test_task"

    _attach_value_error_exc_info(record, "custom error message")

    result = formatter.format(record)
    assert isinstance(result, dict)
    assert result["exc_info"] == "ValueError"
    assert result["exc_text"] == "custom error message"


def test_json_formatter_truncates_long_messages():
    """Test _JsonFormatter truncates messages longer than 103 chars"""
    formatter = _JsonFormatter()

    long_message = "a" * 200
    record = _sample_log_record(
        pathname="test.py",
        lineno=42,
        msg=long_message,
        func="test_func",
    )
    record.taskName = "test_task"

    result = formatter.format(record)
    assert len(result["message"]) == 103
    assert result["message"] == "a" * 103


def test_json_exception_file_handler_initialization(tmp_path):
    """Test _JsonExceptionFileHandler initializes correctly"""
    test_file = tmp_path / "test_exceptions.json"

    handler = _JsonExceptionFileHandler(test_file)

    assert handler.filename == test_file
    assert test_file.exists()

    content = json.loads(test_file.read_text(encoding="utf-8"))
    assert content == {}

    assert handler.level == logging.ERROR
    assert any(isinstance(f, ExceptionOnlyFilter) for f in handler.filters)

    handler.close()


def test_json_exception_file_handler_existing_file(tmp_path):
    """Test existing JSON exception files remain intact upon init"""
    test_file = tmp_path / "test_exceptions.json"
    existing_content = {
        "existing_module": {
            "ValueError": [
                {
                    "timestamp": "existing",
                    "message": "existing error",
                    "exc_text": "existing exception",
                    "filename": "existing.py",
                    "funcName": "existing_func",
                    "taskName": "existing_task",
                    "lineno": 12,
                }
            ]
        }
    }
    test_file.write_text(
        json.dumps(existing_content, indent=4), encoding="utf-8"
    )

    handler = _JsonExceptionFileHandler(test_file)

    content_after_init = json.loads(test_file.read_text(encoding="utf-8"))
    assert content_after_init == existing_content

    record = _sample_log_record(
        level=logging.ERROR,
        pathname="test.py",
        lineno=10,
        msg="new error",
        func="test_func",
    )
    record.taskName = "test_task"
    record.module = "new_module"

    _attach_value_error_exc_info(record, "new exception")

    handler.emit(record)
    handler.close()

    updated_content = json.loads(test_file.read_text(encoding="utf-8"))
    assert (
        updated_content["existing_module"]
        == existing_content["existing_module"]
    )
    assert "new_module" in updated_content
    assert "ValueError" in updated_content["new_module"]
    new_entries = updated_content["new_module"]["ValueError"]
    assert len(new_entries) == 1
    assert new_entries[0]["message"] == "new error"
    assert new_entries[0]["exc_text"] == "new exception"


def test_json_exception_file_handler_emit_type_error(tmp_path, monkeypatch):
    """Test emit returns early when json.dumps raises TypeError"""
    test_file = tmp_path / "test_exceptions.json"
    handler = _JsonExceptionFileHandler(test_file)
    original_content = test_file.read_text(encoding="utf-8")

    def _raise_type_error(*_, **__):
        raise TypeError("cannot serialize")

    monkeypatch.setattr("compass.utilities.logs.json.dumps", _raise_type_error)

    record = _sample_log_record(
        level=logging.ERROR,
        pathname="test.py",
        lineno=20,
        msg="bad error",
        func="test_func",
    )
    record.taskName = "test_task"
    record.module = "bad_module"

    _attach_value_error_exc_info(record, "bad exception")

    handler.emit(record)
    handler.close()

    assert test_file.read_text(encoding="utf-8") == original_content


def test_json_exception_file_handler_invalid_json(tmp_path):
    """Test _JsonExceptionFileHandler handles invalid JSON gracefully"""
    test_file = tmp_path / "test_exceptions.json"
    test_file.write_text("not a valid json!", encoding="utf-8")

    handler = _JsonExceptionFileHandler(test_file)

    record = _sample_log_record(
        level=logging.ERROR,
        pathname="test.py",
        lineno=30,
        msg="error after invalid json",
        func="test_func",
    )
    record.taskName = "test_task"
    record.module = "invalid_json_module"

    _attach_value_error_exc_info(record, "exception after invalid json")

    handler.emit(record)
    handler.close()

    content = json.loads(test_file.read_text(encoding="utf-8"))
    assert "invalid_json_module" in content
    assert "ValueError" in content["invalid_json_module"]
    entries = content["invalid_json_module"]["ValueError"]
    assert len(entries) == 1
    assert entries[0]["message"] == "error after invalid json"
    assert entries[0]["exc_text"] == "exception after invalid json"


def test_json_exception_file_handler_emit(tmp_path):
    """Test _JsonExceptionFileHandler correctly writes exceptions to JSON"""
    test_file = tmp_path / "test_exceptions.json"
    handler = _JsonExceptionFileHandler(test_file)

    record = _sample_log_record(
        level=logging.ERROR,
        pathname="test.py",
        lineno=42,
        msg="test error",
        func="test_func",
    )
    record.taskName = "test_task"
    record.module = "test_module"

    _attach_value_error_exc_info(record, "test exception")

    handler.emit(record)
    handler.close()

    content = json.loads(test_file.read_text(encoding="utf-8"))
    assert "test_module" in content
    assert "ValueError" in content["test_module"]
    assert len(content["test_module"]["ValueError"]) == 1

    entry = content["test_module"]["ValueError"][0]
    assert entry["message"] == "test error"
    assert entry["exc_text"] == "test exception"
    assert entry["filename"] == "test.py"
    assert entry["funcName"] == "test_func"
    assert entry["lineno"] == 42


def test_json_exception_file_handler_multiple_exceptions(tmp_path):
    """Test _JsonExceptionFileHandler handles multiple exceptions"""
    test_file = tmp_path / "test_exceptions.json"
    handler = _JsonExceptionFileHandler(test_file)

    for i in range(3):
        record = _sample_log_record(
            level=logging.ERROR,
            pathname="test.py",
            lineno=i,
            msg=f"error {i}",
            func="test_func",
        )
        record.taskName = "test_task"
        record.module = "test_module"

        msg = f"exception {i}"
        _attach_value_error_exc_info(record, msg)

        handler.emit(record)

    handler.close()

    content = json.loads(test_file.read_text(encoding="utf-8"))
    assert len(content["test_module"]["ValueError"]) == 3


def test_setup_logging_levels():
    """Test setup_logging_levels adds custom logging levels"""

    assert hasattr(logging, "TRACE")
    assert logging.TRACE == 5
    assert logging.getLevelName(logging.TRACE) == "TRACE"

    assert hasattr(logging, "DEBUG_TO_FILE")
    assert logging.DEBUG_TO_FILE == 9
    assert logging.getLevelName(logging.DEBUG_TO_FILE) == "DEBUG_TO_FILE"

    logger = logging.getLogger("test_custom_levels")
    assert hasattr(logger, "trace")
    assert hasattr(logger, "debug_to_file")
    assert callable(logger.trace)
    assert callable(logger.debug_to_file)


def test_local_process_queue_handler_emit():
    """Test _LocalProcessQueueHandler correctly enqueues records"""
    handler = _LocalProcessQueueHandler(LOGGING_QUEUE)

    while not LOGGING_QUEUE.empty():
        LOGGING_QUEUE.get()

    record = _sample_log_record(
        pathname="test.py",
        lineno=42,
        msg="test message",
        func="test_func",
    )

    handler.emit(record)

    assert not LOGGING_QUEUE.empty()
    queued_record = LOGGING_QUEUE.get()
    assert queued_record.msg == "test message"


def test_log_versions_logs_expected_packages(
    compass_logger, assert_message_was_logged
):
    """Test log_versions emits entries for each tracked package"""

    log_versions(compass_logger)

    expected_packages = [
        "NREL-ELM",
        "openai",
        "playwright",
        "tf-playwright-stealth",
        "rebrowser-playwright",
        "camoufox",
        "pdftotext",
        "pytesseract",
        "langchain-text-splitters",
        "crawl4ai",
        "nltk",
        "networkx",
        "pandas",
        "numpy",
    ]
    assert_message_was_logged("Running COMPASS version", log_level="INFO")
    for pkg in expected_packages:
        assert_message_was_logged(pkg, log_level="DEBUG_TO_FILE")


def test_log_listener_context_manager():
    """Test LogListener as a context manager"""
    logger_name = "test_listener_logger"
    logger = logging.getLogger(logger_name)

    logger.handlers = []

    with LogListener([logger_name], level="DEBUG") as listener:
        assert len(logger.handlers) == 1
        assert listener._listener is not None

    assert len(logger.handlers) == 0


@pytest.mark.asyncio
async def test_log_listener_async_context_manager():
    """Test LogListener as an async context manager"""
    logger_name = "test_async_listener_logger"
    logger = logging.getLogger(logger_name)

    logger.handlers = []

    async with LogListener([logger_name], level="INFO") as listener:
        assert len(logger.handlers) == 1
        assert listener._listener is not None

        captured_records = []

        class _CaptureHandler(logging.Handler):
            def emit(self, record):
                captured_records.append(record)

        capture_handler = _CaptureHandler(level=logging.INFO)
        listener.addHandler(capture_handler)

        logger.info("async listener message")

        for _ in range(30):
            if captured_records:
                break
            await asyncio.sleep(0.1)

        listener.removeHandler(capture_handler)

        assert captured_records
        assert captured_records[0].msg == "async listener message"
        assert getattr(captured_records[0], "location", None) == "main"

    assert len(logger.handlers) == 0


def test_get_dne_package():
    """Test _get_version for a non-existent package"""
    assert _get_version("DNE") == "not installed"


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
