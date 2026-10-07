"""Checks that wait diagnostics preserve asynchronous operation behavior."""

import asyncio
from unittest.mock import Mock

import pytest

from compass.utilities.timing import log_operation


async def test_wait_reports_call_chain_and_cleans_up_monitor():
    """A pending operation is identified without interrupting its work."""
    before = asyncio.all_tasks()
    reported, release = asyncio.Event(), asyncio.Event()
    messages, names = [], []

    def record(template, *args):
        messages.append(template % args)
        names.append(asyncio.current_task().get_name())
        if template.startswith("Still waiting"):
            reported.set()

    logger = Mock()
    logger.info.side_effect = record

    async def retrieve():
        async with log_operation(
            logger, "retrieval", "https://city.gov/code", interval=0.001
        ):
            await release.wait()
            return "document"

    task = asyncio.create_task(retrieve(), name="Example County")
    try:
        await asyncio.wait_for(reported.wait(), timeout=1)
        wait = next(m for m in messages if m.startswith("Still waiting"))
        assert "https://city.gov/code" in wait
        assert "retrieve:" in wait
        assert "asyncio.locks.wait:" in wait
        assert not task.done()
    finally:
        release.set()
        result = await task

    assert result == "document"
    assert messages[-1].startswith("Finished retrieval in")
    assert set(names) == {"Example County"}
    assert asyncio.all_tasks() == before


@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
async def test_failure_and_cancellation_propagate(error):
    """Diagnostics must not suppress errors or leave a monitor running."""
    before = asyncio.all_tasks()
    logger = Mock()
    with pytest.raises(error):
        async with log_operation(logger, "retrieval", "source"):
            raise error("test failure")

    messages = [args[0] % args[1:] for args, _ in logger.info.call_args_list]
    assert any(f"({error.__name__})" in message for message in messages)
    assert not any(message.startswith("Finished") for message in messages)
    assert asyncio.all_tasks() == before
