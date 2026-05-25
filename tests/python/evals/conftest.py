"""Eval reporting + regression gate (``pytest_terminal_summary``).

Writes results to ``tests/python/evals/results/`` and gates against the
committed baseline; no-ops in normal runs (evals deselected by default).

- **dev**: per-case breakdown CSV + metrics CSV; gate = aggregate failing
  count AND per-row regression (tolerance for sampling noise).
- **held_out_eval**: metrics CSV only (no per-case detail, by design, to keep
  the held-out set hard to tune against); gate = aggregate failing count only.
"""

import csv
import sys
from operator import itemgetter
from datetime import datetime, UTC

import pytest
from statsmodels.stats.proportion import proportion_confint


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

# Prefix for result filenames, so this eval's CSVs are distinguishable from
# any other eval's output sharing the results/ directory.
_EVAL_NAME = "date_extraction"


def _eval_module():
    """Return the date-accuracy test module if it ran, else None

    The test module is imported by basename under pytest's default import
    mode, so look it up in ``sys.modules`` rather than via a package path.
    """
    return sys.modules.get("test_extraction_date_accuracy")


def _wilson_ci(k, n, alpha=0.05):
    """95% Wilson score interval for k/n, or (None, None) if n == 0

    IID (ignores clustering). Uses statsmodels' ``proportion_confint``.
    """
    if n == 0:
        return None, None
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def _compute_metrics(results):
    """Accuracy / precision / recall / F1 (+ 95% Wilson CIs) from categories

    Positive class = "an enactment year exists". A WRONG year (extracted a
    different year than expected) counts as both a false positive and a
    false negative. Each rate is a binomial proportion k/n with its own
    denominator, so it gets its own Wilson CI:

      accuracy  = (TP + TN) / N
      precision = TP / (TP + FP + WRONG)     # over cases that output a year
      recall    = TP / (TP + FN + WRONG)     # over cases where a year exists
      f1        = 2PR / (P + R)              # point estimate only
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
    pred_pos = tp + fp + wrong
    actual_pos = tp + fn + wrong

    def _safe_div(num, den):
        return num / den if den else 0.0

    precision = _safe_div(tp, pred_pos)
    recall = _safe_div(tp, actual_pos)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "n": n,
        "counts": counts,
        "accuracy": _safe_div(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_ci": _wilson_ci(tp + tn, n),
        "precision_ci": _wilson_ci(tp, pred_pos),
        "recall_ci": _wilson_ci(tp, actual_pos),
        "total_cost": sum(r["cost"] for r in results),
    }


def _fmt_ci(ci):
    """Format a (lo, hi) CI as 'lo-hi', or '' if undefined"""
    lo, hi = ci
    return "" if lo is None else f"{lo:.4f}-{hi:.4f}"


def _write_metrics_csv(fp, metrics):
    """Write the aggregate metrics summary CSV (with 95% Wilson CIs)"""
    c = metrics["counts"]
    fails = c["FP"] + c["FN"] + c["WRONG"]
    with fp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value", "ci95_wilson"])
        writer.writerow(["generated_utc", datetime.now(UTC).isoformat(), ""])
        writer.writerow(["n_cases", metrics["n"], ""])
        writer.writerow(
            ["accuracy", f"{metrics['accuracy']:.4f}",
             _fmt_ci(metrics["accuracy_ci"])]
        )
        writer.writerow(
            ["precision", f"{metrics['precision']:.4f}",
             _fmt_ci(metrics["precision_ci"])]
        )
        writer.writerow(
            ["recall", f"{metrics['recall']:.4f}",
             _fmt_ci(metrics["recall_ci"])]
        )
        writer.writerow(["f1", f"{metrics['f1']:.4f}", ""])
        writer.writerow(["true_positive", c["TP"], ""])
        writer.writerow(["true_negative", c["TN"], ""])
        writer.writerow(["false_positive", c["FP"], ""])
        writer.writerow(["false_negative", c["FN"], ""])
        writer.writerow(["wrong_year", c["WRONG"], ""])
        writer.writerow(["failing_cases", fails, ""])
        writer.writerow(
            ["total_cost_usd", f"{metrics['total_cost']:.4f}", ""]
        )


def _write_breakdown_csv(fp, results):
    """Write the detailed per-case breakdown CSV"""
    with fp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in sorted(results, key=itemgetter("jurisdiction")):
            writer.writerow({k: row.get(k) for k in _CSV_FIELDS})


def _load_baseline_correct(breakdown_fp):
    """Map {fips: was_correct} from a baseline breakdown CSV, or None"""
    if not breakdown_fp.exists():
        return None
    with breakdown_fp.open(newline="", encoding="utf-8") as fh:
        return {
            str(row["fips"]): row["correct"] == "True"
            for row in csv.DictReader(fh)
        }


def _load_baseline_failing(metrics_fp):
    """Read ``failing_cases`` from a baseline metrics CSV, or None"""
    if not metrics_fp.exists():
        return None
    with metrics_fp.open(newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if row and row[0] == "failing_cases":
                return int(row[1])
    return None


def _check_full_regression(rows, baseline):
    """dev gate: aggregate (failing count) + per-row regression checks"""
    if baseline is None:
        return [], ["  gate: no baseline yet (this run sets it)"]

    now_correct = {str(r["fips"]): r["correct"] for r in rows}
    fails_now = sum(1 for r in rows if not r["correct"])
    fails_base = sum(1 for ok in baseline.values() if not ok)
    regressed = sorted(
        fips
        for fips, was_ok in baseline.items()
        if was_ok and now_correct.get(fips) is False
    )

    failures = []
    lines = [
        (
            f"  gate: failing now={fails_now} baseline={fails_base}; "
            f"row regressions={len(regressed)} "
            f"(tol={_ROW_REGRESSION_TOLERANCE})"
        )
    ]
    if fails_now > fails_base:
        failures.append(
            f"aggregate regression: {fails_now} failing > {fails_base}"
        )
    if len(regressed) > _ROW_REGRESSION_TOLERANCE:
        failures.append(
            f"{len(regressed)} rows regressed "
            f"(tol {_ROW_REGRESSION_TOLERANCE}): {regressed}"
        )
    return failures, lines


def _check_aggregate_regression(metrics, baseline_failing):
    """held_out_eval gate: aggregate failing-count only (no per-row detail)"""
    c = metrics["counts"]
    fails_now = c["FP"] + c["FN"] + c["WRONG"]
    if baseline_failing is None:
        return [], ["  gate: no baseline yet (this run sets it)"]
    lines = [f"  gate: failing now={fails_now} baseline={baseline_failing}"]
    failures = []
    if fails_now > baseline_failing:
        failures.append(
            f"aggregate regression: {fails_now} failing > {baseline_failing}"
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
    results_dir.mkdir(parents=True, exist_ok=True)
    write = terminalreporter.write_line

    by_cadence = {}
    for r in results:
        by_cadence.setdefault(r["cadence"], []).append(r)

    gate_failures = []
    for cadence, rows in sorted(by_cadence.items()):
        metrics = _compute_metrics(rows)
        stem = f"{_EVAL_NAME}_{cadence}"
        metrics_fp = results_dir / f"{stem}_metrics.csv"
        breakdown_fp = results_dir / f"{stem}_breakdown.csv"

        # held_out_eval: only summary stats are surfaced/saved (no per-case
        # breakdown), and the gate is aggregate-only -- this keeps the
        # held-out set hard to inspect or tune against.
        if cadence == "held_out_eval":
            baseline_failing = _load_baseline_failing(metrics_fp)
            failures, lines = _check_aggregate_regression(
                metrics, baseline_failing
            )
            _write_metrics_csv(metrics_fp, metrics)
            extra = [f"  metrics: {metrics_fp}"]
        else:
            baseline = _load_baseline_correct(breakdown_fp)
            failures, lines = _check_full_regression(rows, baseline)
            _write_breakdown_csv(breakdown_fp, rows)
            _write_metrics_csv(metrics_fp, metrics)
            extra = [
                f"  breakdown: {breakdown_fp}",
                f"  metrics: {metrics_fp}",
            ]

        gate_failures.extend(f"[{cadence}] {m}" for m in failures)

        terminalreporter.section(f"Eval summary: {cadence}")
        c = metrics["counts"]
        write(
            f"  cases={metrics['n']}  "
            f"TP={c['TP']} TN={c['TN']} FP={c['FP']} "
            f"FN={c['FN']} wrong={c['WRONG']}"
        )
        write(
            f"  accuracy={metrics['accuracy']:.3f} "
            f"95%CI[{_fmt_ci(metrics['accuracy_ci'])}]"
        )
        write(
            f"  precision={metrics['precision']:.3f} "
            f"95%CI[{_fmt_ci(metrics['precision_ci'])}]  "
            f"recall={metrics['recall']:.3f} "
            f"95%CI[{_fmt_ci(metrics['recall_ci'])}]  "
            f"f1={metrics['f1']:.3f}"
        )
        write(f"  total LLM cost: ${metrics['total_cost']:.4f}")
        for line in lines:
            write(line)
        for line in extra:
            write(line)

    if gate_failures:
        terminalreporter.section("Eval regression gate: FAILED")
        for f in gate_failures:
            write(f"  - {f}")
        terminalreporter._session.exitstatus = pytest.ExitCode.TESTS_FAILED
