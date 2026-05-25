"""Eval fixtures and reporting hooks.

After an eval run, ``pytest_terminal_summary`` writes a detailed per-case
breakdown CSV and an overall metrics summary (accuracy / precision / recall /
F1) to ``tests/python/evals/results/``, and prints a short summary to the
terminal. No-ops in normal test runs (the evals are deselected by default).
"""

import csv
import sys
from operator import itemgetter
from datetime import datetime, UTC

import pytest


_CSV_FIELDS = [
    "cadence",
    "fips",
    "jurisdiction",
    "file",
    "source",
    "expected_year",
    "extracted_year",
    "extracted_month",
    "extracted_day",
    "correct",
    "category",
    "cost",
]

# Per-row regression tolerance: how many previously-correct rows may flip to
# wrong (e.g. from temperature sampling noise) before the gate fails.
_ROW_REGRESSION_TOLERANCE = 2


def _eval_module():
    """Return the date-accuracy test module if it ran, else None

    The test module is imported by basename under pytest's default import
    mode, so look it up in ``sys.modules`` rather than via a package path.
    """
    return sys.modules.get("test_extraction_date_accuracy")


def _compute_metrics(results):
    """Accuracy / precision / recall / F1 from per-case categories

    Positive class = "an enactment year exists". A WRONG year (extracted a
    different year than expected) counts as both a false positive and a
    false negative.

      accuracy  = (TP + TN) / N
      precision = TP / (TP + FP + WRONG)
      recall    = TP / (TP + FN + WRONG)
      f1        = 2PR / (P + R)
    """
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "WRONG": 0}
    for r in results:
        counts[r["category"]] += 1

    tp, tn, fp, fn, wrong = (
        counts["TP"],
        counts["TN"],
        counts["FP"],
        counts["FN"],
        counts["WRONG"],
    )
    n = len(results)

    def _safe_div(num, den):
        return num / den if den else 0.0

    accuracy = _safe_div(tp + tn, n)
    precision = _safe_div(tp, tp + fp + wrong)
    recall = _safe_div(tp, tp + fn + wrong)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "n": n,
        "counts": counts,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_cost": sum(r["cost"] for r in results),
    }


def _write_reports(results_dir, cadence, results, metrics):
    """Write the per-case breakdown CSV + metrics summary for a cadence"""
    results_dir.mkdir(parents=True, exist_ok=True)

    breakdown_fp = results_dir / f"{cadence}_eval_breakdown.csv"
    with breakdown_fp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in sorted(results, key=itemgetter("jurisdiction")):
            writer.writerow({k: row.get(k) for k in _CSV_FIELDS})

    summary_fp = results_dir / f"{cadence}_eval_metrics.csv"
    c = metrics["counts"]
    with summary_fp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["generated_utc", datetime.now(UTC).isoformat()])
        writer.writerow(["n_cases", metrics["n"]])
        writer.writerow(["accuracy", f"{metrics['accuracy']:.4f}"])
        writer.writerow(["precision", f"{metrics['precision']:.4f}"])
        writer.writerow(["recall", f"{metrics['recall']:.4f}"])
        writer.writerow(["f1", f"{metrics['f1']:.4f}"])
        writer.writerow(["true_positive", c["TP"]])
        writer.writerow(["true_negative", c["TN"]])
        writer.writerow(["false_positive", c["FP"]])
        writer.writerow(["false_negative", c["FN"]])
        writer.writerow(["wrong_year", c["WRONG"]])
        writer.writerow(["total_cost_usd", f"{metrics['total_cost']:.4f}"])

    return breakdown_fp, summary_fp


def _load_baseline_correct(breakdown_fp):
    """Map {fips: was_correct} from a committed baseline breakdown CSV

    Returns ``None`` if no baseline exists yet (first run establishes it).
    """
    if not breakdown_fp.exists():
        return None
    baseline = {}
    with breakdown_fp.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            baseline[str(row["fips"])] = row["correct"] == "True"
    return baseline


def _check_regression(rows, baseline):
    """Compare a cadence's rows to its baseline; return (failures, lines)

    Two gates (fails if EITHER trips):
      - aggregate (tight): more failing rows now than in the baseline.
      - per-row (loose): more than ``_ROW_REGRESSION_TOLERANCE`` rows that
        were correct in the baseline are now wrong.
    """
    if baseline is None:
        return [], ["  regression gate: no baseline yet (this run sets it)"]

    now_correct = {str(r["fips"]): r["correct"] for r in rows}
    fails_now = sum(1 for r in rows if not r["correct"])
    fails_base = sum(1 for ok in baseline.values() if not ok)

    regressed = sorted(
        fips
        for fips, was_ok in baseline.items()
        if was_ok and now_correct.get(fips) is False
    )

    failures, lines = [], []
    lines.append(
        f"  regression gate: failing rows now={fails_now} "
        f"baseline={fails_base}; "
        f"row regressions={len(regressed)} (tol={_ROW_REGRESSION_TOLERANCE})"
    )
    if fails_now > fails_base:
        failures.append(
            f"aggregate regression: {fails_now} failing rows > "
            f"baseline {fails_base}"
        )
    if len(regressed) > _ROW_REGRESSION_TOLERANCE:
        failures.append(
            f"{len(regressed)} previously-correct rows regressed "
            f"(tolerance {_ROW_REGRESSION_TOLERANCE}): {regressed}"
        )
    return failures, lines


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Write eval CSVs, print a summary, and enforce the regression gate

    No-ops when the eval did not run (deselected or skipped). If a cadence
    regresses against its committed baseline, fails the session (sets a
    non-zero exit) so the run goes red.
    """
    module = _eval_module()
    if module is None:
        return
    results = getattr(module, "DATE_ACCURACY_RESULTS", None)
    if not results:
        return

    results_dir = module.RESULTS_DIR
    write = terminalreporter.write_line

    # Group by cadence so dev and held-out get separate reports.
    by_cadence = {}
    for r in results:
        by_cadence.setdefault(r["cadence"], []).append(r)

    gate_failures = []
    for cadence, rows in sorted(by_cadence.items()):
        metrics = _compute_metrics(rows)
        # Read the committed baseline BEFORE overwriting the breakdown CSV.
        baseline = _load_baseline_correct(
            results_dir / f"{cadence}_eval_breakdown.csv"
        )
        breakdown_fp, summary_fp = _write_reports(
            results_dir, cadence, rows, metrics
        )
        regress_failures, regress_lines = _check_regression(rows, baseline)
        gate_failures.extend(f"[{cadence}] {m}" for m in regress_failures)

        terminalreporter.section(f"Eval summary: {cadence}")
        c = metrics["counts"]
        write(
            f"  cases={metrics['n']}  "
            f"TP={c['TP']} TN={c['TN']} FP={c['FP']} "
            f"FN={c['FN']} wrong={c['WRONG']}"
        )
        write(
            f"  accuracy={metrics['accuracy']:.3f}  "
            f"precision={metrics['precision']:.3f}  "
            f"recall={metrics['recall']:.3f}  "
            f"f1={metrics['f1']:.3f}"
        )
        write(f"  total LLM cost: ${metrics['total_cost']:.4f}")
        for line in regress_lines:
            write(line)
        write(f"  breakdown: {breakdown_fp}")
        write(f"  metrics:   {summary_fp}")

    if gate_failures:
        terminalreporter.section("Eval regression gate: FAILED")
        for f in gate_failures:
            write(f"  - {f}")
        # Force a non-zero session exit so the run goes red.
        terminalreporter._session.exitstatus = pytest.ExitCode.TESTS_FAILED
