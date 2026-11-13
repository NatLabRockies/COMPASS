"""Elapsed-time logging for asynchronous pipeline operations."""

import asyncio
import inspect
import time
from contextlib import asynccontextmanager, suppress


@asynccontextmanager
async def log_operation(logger, operation, target, *, interval=30):
    """Report elapsed time and the call chain during long waits."""
    owner = asyncio.current_task()
    started = time.monotonic()
    logger.info("Started %s: %s", operation, target)
    monitor = asyncio.create_task(
        _report_wait(logger, operation, target, owner, started, interval),
        name=owner.get_name(),
    )
    try:
        yield
    except BaseException as exc:
        logger.info(
            "Stopped %s after %.1fs (%s): %s",
            operation,
            time.monotonic() - started,
            type(exc).__name__,
            target,
        )
        raise
    else:
        logger.info(
            "Finished %s in %.1fs: %s",
            operation,
            time.monotonic() - started,
            target,
        )
    finally:
        monitor.cancel()
        with suppress(asyncio.CancelledError):
            await monitor


async def _report_wait(logger, operation, target, owner, started, interval):
    while True:
        await asyncio.sleep(interval)
        logger.info(
            "Still waiting for %s after %.1fs: %s | awaiting: %s",
            operation,
            time.monotonic() - started,
            target,
            _await_chain(owner),
        )


def _await_chain(task):
    frames = []
    pending = task.get_coro()
    while inspect.iscoroutine(pending) or inspect.isgenerator(pending):
        if inspect.iscoroutine(pending):
            frame, pending = pending.cr_frame, pending.cr_await
        else:
            frame, pending = pending.gi_frame, pending.gi_yieldfrom
        if frame is not None:
            frames.append(
                f"{frame.f_globals['__name__']}."
                f"{frame.f_code.co_name}:{frame.f_lineno}"
            )
    return " -> ".join(frames)
