"""COMPASS Ordinance logging utilities

This module implements queued logging, mostly following this blog:"
https://www.zopatista.com/python/2019/05/11/asyncio-logging/
"""

import os
import time
import json
import asyncio
import logging
import threading
from pathlib import Path
from queue import SimpleQueue
from functools import partial, partialmethod
from logging.handlers import QueueHandler, QueueListener

from compass.exceptions import COMPASSValueError


LOGGING_QUEUE = SimpleQueue()
COMPASS_DEBUG_LEVEL = int(os.environ.get("COMPASS_DEBUG_LEVEL", "0"))


class NoLocationFilter(logging.Filter):
    """Filter that catches all records without a location attribute."""

    def filter(self, record):  # noqa: PLR6301
        """Filter logging record.

        Parameters
        ----------
        record : logging.LogRecord
            Log record containing the log message + default attributes.
            If the ``location`` attribute is missing or is a string in
            the form "Task-XX", the filter returns ``True`` (i.e. record
            is emitted).

        Returns
        -------
        bool
            If the record's ``location`` attribute is "missing".
        """
        record_location = getattr(record, "location", None)
        return (
            record_location is None
            or "Task-" in record_location
            or record_location == "main"
        )


class LocationFilter(logging.Filter):
    """Filter down to logs for a specific location"""

    def __init__(self, location):
        """

        Parameters
        ----------
        location : str
            Location identifier. For example, ``"El Paso Colorado"``.
        """
        self.location = location

    def filter(self, record):
        """Filter logging record

        Parameters
        ----------
        record : logging.LogRecord
            Log record containing the log message + default attributes.
            Must have a ``location`` attribute that is a string
            identifier, or this function will return ``False`` every
            time. The ``location`` identifier will be checked against
            the filter's location attribute to determine the output
            result.

        Returns
        -------
        bool
            If the record's ``location`` attribute matches the filter's
            ``location`` attribute.
        """
        record_location = getattr(record, "location", None)
        return record_location is not None and record_location == self.location


class AddLocationFilter(logging.Filter):
    """Filter that injects location information into the log record"""

    def filter(self, record):  # noqa: PLR6301
        """Add location to record

        Parameters
        ----------
        record : logging.LogRecord
            Log record containing the log message + default attributes.
            This filter will add the location bing processed as a
            ``location`` attribute. If the there is no current async
            task (or if the task name is of the form "Task-XX"), the
            filter sets the location to "main".

        Returns
        -------
        bool
            Always true since we want the record to be passed along with
            the additional attribute.
        """
        try:
            location = asyncio.current_task().get_name()
        except RuntimeError:
            location = ""

        if not location or "Task" in location:
            location = "main"

        record.location = location
        return True


class _LocalProcessQueueHandler(QueueHandler):
    """QueueHandler that works within a single process (locally)"""

    def emit(self, record):
        """Emit record with a location attribute

        Parameters
        ----------
        record : logging.LogRecord
            Log record containing the log message + default attributes.
        """
        try:
            self.enqueue(record)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            self.handleError(record)


class LogListener:
    """Class to listen to logging queue and write logs to files"""

    def __init__(self, logger_names, level="INFO"):
        """

        Parameters
        ----------
        logger_names : iterable
            An iterable of string, where each string is a logger name.
            The logger corresponding to each of the names will be
            equipped with a logging queue handler.
        level : str, optional
            Log level to set for each logger. By default, ``"INFO"``.
        """
        self.logger_names = logger_names
        self.level = level
        self._listener = None
        self._queue_handler = _LocalProcessQueueHandler(LOGGING_QUEUE)
        self._queue_handler.addFilter(AddLocationFilter())

    def _setup_listener(self):
        """Set up the queue listener"""
        if self._listener is not None:
            return
        self._listener = QueueListener(
            LOGGING_QUEUE, logging.NullHandler(), respect_handler_level=True
        )
        self._listener.handlers = list(self._listener.handlers)

    def _add_queue_handler_to_loggers(self):
        """Add a queue handler to each logger"""
        for logger_name in self.logger_names:
            logger = logging.getLogger(logger_name)
            logger.addHandler(self._queue_handler)
            logger.setLevel(self.level)

    def _remove_queue_handler_from_loggers(self):
        """Remove the queue handler from each logger"""
        for logger_name in self.logger_names:
            logging.getLogger(logger_name).removeHandler(self._queue_handler)

    def _remove_all_handlers_from_listener(self):
        """Remove all handlers still attached to listener"""
        if self._listener is None:
            return

        for handler in self._listener.handlers:
            handler.close()
        self._listener.handlers = []

    def __enter__(self):
        self._setup_listener()
        self._add_queue_handler_to_loggers()
        self._listener.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._listener.stop()
        self._remove_queue_handler_from_loggers()
        self._remove_all_handlers_from_listener()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        self.__exit__(exc_type, exc, tb)

    def addHandler(self, handler):  # noqa: N802
        """Add a handler to the queue listener

        Logs that are sent to the queue will be emitted to the handler.

        Parameters
        ----------
        handler : logging.Handler
            Log handler to parse log records.
        """
        if handler not in self._listener.handlers:
            self._listener.handlers.append(handler)

    def removeHandler(self, handler):  # noqa: N802
        """Remove a handler from the queue listener

        Logs that are sent to the queue will no longer be emitted to the
        handler.

        Parameters
        ----------
        handler : logging.Handler
            Log handler to remove from queue listener.
        """
        if handler in self._listener.handlers:
            handler.close()
            self._listener.handlers.remove(handler)


class LocationFileLog:
    """Context manager to write logs for a location to a unique file"""

    TEXT_FORMATTER = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s"
    )
    """Formatter used for text logs"""

    def __init__(
        self, listener, log_dir, location, level="INFO", max_teardown_time=30
    ):
        """

        Parameters
        ----------
        listener : LogListener
            A listener instance. The file handler will be added to this
            listener.
        log_dir : path-like
            Path to output directory to contain log file.
        location : str
            Location identifier. For example, ``"El Paso Colorado"``.
            This string will become part of the file name, so it must
            contain only characters valid in a file name.
        level : str, optional
            Log level. By default, ``"INFO"``.
        max_teardown_time : int or float, default=30
            Max number of seconds to wait for logs to flush to file
            before unhooking the handler from the queue listener.
            By default, ``30``.
        """
        self.log_dir = Path(log_dir)
        self.location = location
        self.level = level
        self._handler = None
        self._exception_handler = None
        self._listener = listener
        self.max_teardown_time = max_teardown_time

    def _create_log_dir(self):
        """Create log output directory if it doesn't exist"""
        self.log_dir.mkdir(exist_ok=True, parents=True)

    def _setup_handler(self):
        """Setup the file handler for this location"""
        self._handler = logging.FileHandler(
            self.log_dir / f"{self.location}.log", encoding="utf-8"
        )
        self._handler.setLevel(self.level)
        self._handler.addFilter(LocationFilter(self.location))
        self._handler.setFormatter(self.TEXT_FORMATTER)

    def _setup_exception_handler(self):
        """Setup file handler for tracking errors for this location"""
        self._exception_handler = _JsonExceptionFileHandler(
            self.log_dir / f"{self.location} exceptions.json", encoding="utf-8"
        )
        self._exception_handler.addFilter(LocationFilter(self.location))

    def _break_down_handler(self):
        """Tear down the file handler for this location"""
        if self._handler is None:
            return

        self._handler.close()
        self._handler = None

    def _break_down_exception_handler(self):
        """Tear down the exception file handler for this location"""
        if self._exception_handler is None:
            return

        self._exception_handler.close()
        self._exception_handler = None

    def _add_handler_to_listener(self):
        """Add the file handler to the queue listener"""
        if self._handler is None:
            msg = "Must set up handler before listener!"
            raise COMPASSValueError(msg)

        self._listener.addHandler(self._handler)

    def _add_exception_handler_to_listener(self):
        """Add the exception file handler to the queue listener"""
        if self._exception_handler is None:
            msg = "Must set up exception handler before listener!"
            raise COMPASSValueError(msg)

        self._listener.addHandler(self._exception_handler)

    def _remove_handler_from_listener(self):
        """Remove the file handler from the listener"""
        if self._handler is None:
            return

        self._listener.removeHandler(self._handler)

    def _remove_exception_handler_from_listener(self):
        """Remove the exception file handler from the listener"""
        if self._exception_handler is None:
            return

        self._listener.removeHandler(self._exception_handler)

    def __enter__(self):
        self._create_log_dir()
        self._setup_handler()
        self._setup_exception_handler()
        self._add_handler_to_listener()
        self._add_exception_handler_to_listener()

    def __exit__(self, exc_type, exc, tb):
        self._remove_exception_handler_from_listener()
        self._remove_handler_from_listener()
        self._break_down_exception_handler()
        self._break_down_handler()

    async def __aenter__(self):
        self.__enter__()

    async def __aexit__(self, exc_type, exc, tb):
        start_time = time.monotonic()
        while (
            not LOGGING_QUEUE.empty()
            and (time.monotonic() - start_time) < self.max_teardown_time
        ):
            await asyncio.sleep(3)
        await asyncio.sleep(3)  # Final recording
        self.__exit__(exc_type, exc, tb)


class ExceptionOnlyFilter(logging.Filter):
    """Filter to only pass through Exception logging (errors)"""

    def filter(self, record):  # noqa: D102, PLR6301
        return bool(record.exc_info)


class _JsonFormatter(logging.Formatter):
    """Formatter that converts a record into a dictionary"""

    def format(self, record):
        exc_info = record.exc_info
        exc_text = None
        if exc_info:
            try:
                exc_text = exc_info[1].args[0]
            except Exception:  # noqa: BLE001
                exc_text = None

            try:
                exc_info = exc_info[0].__name__
            except Exception:  # noqa: BLE001
                exc_info = None

        message = record.getMessage()
        if message and len(message) > 103:  # noqa: PLR2004
            message = message[:103]
        return {
            "timestamp": self.formatTime(record),
            "message": message,
            "exc_text": exc_text,
            "filename": record.filename,
            "funcName": record.funcName,
            "taskName": record.taskName,
            "lineno": record.lineno,
            "exc_info": exc_info,
        }


class _JsonExceptionFileHandler(logging.Handler):
    """File handler that writes exception info to JSON file"""

    def __init__(self, filename, encoding="utf-8"):
        """

        Parameters
        ----------
        filename : path-like
            Path to file where JSON output on exceptions should be
            stored.
        encoding : str, default="utf-8"
            File read/write encoding. By default, ``"utf-8"``.
        """
        super().__init__()
        self.filename = Path(filename)
        self.encoding = encoding
        self._lock = threading.Lock()
        self._init_hooks()

    def _init_hooks(self):
        """Hooks to run in the initializer"""
        if not self.filename.exists():
            with (
                self._lock,
                self.filename.open("w", encoding=self.encoding) as f,
            ):
                json.dump({}, f, indent=4)

        self.addFilter(ExceptionOnlyFilter())
        self.setLevel(logging.ERROR)
        self.setFormatter(_JsonFormatter())

    def emit(self, record):
        """Log the specified logging record to a JSON file

        Parameters
        ----------
        record : logging.LogRecord
            A logging record containing log info.
        """
        module = record.module
        entry_dict = self.format(record)
        exc_info = entry_dict.pop("exc_info", None)

        with self._lock:
            records = self._get_existing_records()
            (
                records.setdefault(module, {})
                .setdefault(exc_info, [])
                .append(entry_dict)
            )
            try:
                serialized_exception_info = json.dumps(records, indent=4)
            except TypeError:
                return

            with self.filename.open("w", encoding=self.encoding) as f:
                f.write(serialized_exception_info)

    def _get_existing_records(self):
        """Get existing records from file"""
        with self.filename.open("r", encoding=self.encoding) as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = {}
        return records


def _setup_logging_levels():
    """Setup COMPASS logging levels"""
    logging.TRACE = 5
    logging.addLevelName(logging.TRACE, "TRACE")
    logging.Logger.trace = partialmethod(logging.Logger.log, logging.TRACE)
    logging.trace = partial(logging.log, logging.TRACE)

    logging.DEBUG_TO_FILE = 9
    logging.addLevelName(logging.DEBUG_TO_FILE, "DEBUG_TO_FILE")
    logging.Logger.debug_to_file = partialmethod(
        logging.Logger.log, logging.DEBUG_TO_FILE
    )
    logging.debug_to_file = partial(logging.log, logging.DEBUG_TO_FILE)
