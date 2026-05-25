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
    "allow_failure",
    "cost",
]


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


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Write eval breakdown/metrics CSVs and print a terminal summary

    No-ops when the eval did not run (deselected or skipped).
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

    for cadence, rows in sorted(by_cadence.items()):
        metrics = _compute_metrics(rows)
        breakdown_fp, summary_fp = _write_reports(
            results_dir, cadence, rows, metrics
        )

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
        write(f"  breakdown: {breakdown_fp}")
        write(f"  metrics:   {summary_fp}")
