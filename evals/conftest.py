"""Pytest config for the evals suite."""

import pytest


def pytest_addoption(parser):
    """Add the ``--held-out`` flag to the test"""

    parser.addoption(
        "--held-out",
        action="store_true",
        default=False,
        help=("Run the eval against the held-out dataset"),
    )


def _is_controller(config):
    """True on the controller process (or when xdist is not in use)

    pytest-xdist sets ``config.workerinput`` only on worker processes, so
    its absence marks the controller -- the one process that outlives all
    workers and is the right place to aggregate per-jurisdiction results.
    """
    return not hasattr(config, "workerinput")


def pytest_sessionstart(session):
    """Clear stale per-jurisdiction results and logs before a run begins

    Only the controller clears, so workers (which start afterward) don't
    race to delete each other's freshly written files. Logs are cleared
    too: they're opened in append mode and the controller later reads
    them for explanations, so stale logs would cause confusion.
    """
    config = session.config
    if not _is_controller(config):
        return
    try:
        from test_run_date_extraction_evals import clear_logs, clear_results
    except ImportError:
        return  # the date eval module isn't part of this selection
    held_out = config.getoption("--held-out")
    clear_results(held_out)
    clear_logs(held_out)


def pytest_sessionfinish(session, exitstatus):
    """Aggregate per-jurisdiction results, write reports, and run the gate

    Runs on the controller only, after every worker has finished. Under
    pytest-xdist the per-case results are scattered across worker
    processes; reading the per-jurisdiction files here gives the full
    result set regardless of how many workers ran.
    """
    config = session.config
    if not _is_controller(config):
        return
    try:
        from test_run_date_extraction_evals import (
            load_results,
            report_and_gate,
        )
    except ImportError:
        return  # the date eval module isn't part of this selection

    held_out = config.getoption("--held-out")
    results = load_results(held_out)
    if not results:
        return

    failures = report_and_gate(session, results, held_out)
    if failures:
        # Surface the gate failure with a non-zero exit code so CI fails.
        # ``pytest.fail`` would be swallowed here since there is no test
        # item to attach the failure to.
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                "Eval regression gate FAILED:\n  " + "\n  ".join(failures),
                red=True,
            )
