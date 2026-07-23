"""Pytest config for the evals suite."""


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
    (``per_jurisdiction_results``, ``clear_logs``, ``report``); the
    generic results I/O lives in ``utilities.PerJurisdictionResults``.
    """
    try:
        import test_run_date_extraction_evals as module  # ruff:ignore[import-outside-top-level]
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
    """Aggregate per-jurisdiction results and write reports (controller only)

    Under xdist the per-case results are scattered across workers, so
    reading the per-jurisdiction files here gives the full set.
    """
    config = session.config
    if not _is_controller(config):
        return
    module = _date_eval()
    if module is None:
        return

    held_out = config.getoption("--held-out")
    results = module.per_jurisdiction_results(held_out).load()
    if results:
        module.report(session, results, held_out)
