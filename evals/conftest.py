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
    # xdist sets ``workerinput`` only on workers; its absence is the
    # controller, which outlives all workers and aggregates their results.
    return not hasattr(config, "workerinput")


def _date_eval():
    """Import the date eval module, or ``None`` if it isn't this run

    The session hooks need a few eval-specific entry points
    (``per_jurisdiction_results``, ``clear_logs``, ``report_and_gate``);
    the generic results I/O lives in ``utilities.PerJurisdictionResults``.
    """
    try:
        import test_run_date_extraction_evals as module  # noqa: PLC0415
    except ImportError:
        return None
    return module


def pytest_sessionstart(session):
    """Clear stale results and logs before a run (controller only)"""
    config = session.config
    if not _is_controller(config):
        return
    module = _date_eval()
    if module is None:
        return
    held_out = config.getoption("--held-out")
    module.per_jurisdiction_results(held_out).clear()
    module.clear_logs(held_out)


def pytest_sessionfinish(session, exitstatus):
    """Aggregate per-jurisdiction results, write reports, run the gate

    Controller only: under xdist the per-case results are scattered across
    workers, so reading the per-jurisdiction files here gives the full set.
    """
    config = session.config
    if not _is_controller(config):
        return
    module = _date_eval()
    if module is None:
        return

    held_out = config.getoption("--held-out")
    results = module.per_jurisdiction_results(held_out).load()
    if not results:
        return

    failures = module.report_and_gate(session, results, held_out)
    if failures:
        # ``pytest.fail`` is swallowed in a session hook (no test item to
        # attach to), so fail the run via the exit code instead.
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
        reporter = config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                "Eval regression gate FAILED:\n  " + "\n  ".join(failures),
                red=True,
            )
