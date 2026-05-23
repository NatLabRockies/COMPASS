"""Integration-test fixtures and hooks"""

import sys


def _date_accuracy_results():
    """Return the date-accuracy results list, or None if not collected

    The test module is imported by basename under pytest's default import
    mode, so look it up in ``sys.modules`` rather than via a package path.
    Returns ``None`` when the date-accuracy test never ran (deselected or
    skipped), so the breakdown is a no-op in normal runs.
    """
    module = sys.modules.get("test_extraction_date_accuracy")
    if module is None:
        return None
    return getattr(module, "DATE_ACCURACY_RESULTS", None)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a cost + accuracy breakdown for the date-accuracy test

    Renders a per-case table (sorted by cost), an accuracy tally, and the
    total LLM spend. Uses the terminal reporter so the summary always
    appears, regardless of ``-s`` / ``--log-cli-level``. No-ops when the
    date-accuracy test did not run (e.g. it was deselected or skipped).
    """
    results = _date_accuracy_results()
    if not results:
        return

    results = sorted(results, key=lambda r: r["cost"], reverse=True)
    total_cost = sum(r["cost"] for r in results)
    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])
    n_xfail = sum(
        1 for r in results if not r["correct"] and r["allow_failure"]
    )

    write = terminalreporter.write_line
    terminalreporter.section("Date extraction: cost & accuracy breakdown")
    write(f"{'result':>7}  {'exp':>5}  {'got':>5}  {'cost':>9}  jurisdiction")
    write("-" * 70)
    for r in results:
        if r["correct"]:
            status = "PASS"
        elif r["allow_failure"]:
            status = "xfail"
        else:
            status = "FAIL"
        write(
            f"{status:>7}  "
            f"{str(r['expected']):>5}  "
            f"{str(r['extracted']):>5}  "
            f"${r['cost']:>8.4f}  "
            f"{r['jurisdiction']}"
        )
    write("-" * 70)
    accuracy = (n_correct / n * 100) if n else 0.0
    write(
        f"accuracy: {n_correct}/{n} ({accuracy:.1f}%)"
        f"   xfail (known-hard): {n_xfail}"
    )
    write(f"total LLM cost: ${total_cost:.4f}  over {n} case(s)")
    if n:
        write(f"average cost/case: ${total_cost / n:.4f}")
