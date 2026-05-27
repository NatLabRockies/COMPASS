"""Eval reporting + regression gate.

Exposes ``report_evals(request, eval_name, results_by_type, results_dir)``
for each eval suite's teardown fixture to call. Writes results to the
given ``results_dir`` and gates against the committed baseline; no-ops
when no rows were recorded (evals deselected by default).

- **dev**: per-case breakdown CSV + metrics JSON; gate = aggregate failing
  count AND per-row regression (tolerance for sampling noise).
- **held_out**: metrics JSON only (no per-case detail, by design, to keep
  the held-out set hard to tune against); gate = aggregate failing count only.
"""

import csv
import json
from operator import itemgetter

import pytest


_CSV_FIELDS = [
    "fips",
    "jurisdiction",
    "file",
    "source",
    "feature",
    "expected",
    "extracted",
    "comparison_result",
    "input_tokens",
    "output_tokens",
    "time_taken_s",
    "cost",
]

# Comparison-result value that counts as "correct" (vs. failing).
_SUCCESS = "Success"

# Per-row regression tolerance: how many previously-correct rows may flip to
# wrong (e.g. from temperature sampling noise) before the gate fails.
_ROW_REGRESSION_TOLERANCE = 2


def _wilson_ci(k, n, alpha=0.05):
    """95% Wilson score interval for k/n, or (None, None) if n == 0

    IID (ignores clustering). Imported lazily so the base test session
    (which collects this conftest but deselects the evals) does not depend
    on statsmodels -- only an actual eval run needs it.
    """
    if n == 0:
        return None, None
    # Lazy import (see docstring): keep statsmodels out of the base session.
    from statsmodels.stats.proportion import (  # noqa: PLC0415
        proportion_confint,
    )

    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def _compute_metrics(results):
    """Accuracy / precision / recall / F1 (+ 95% Wilson CIs)

    Positive class = "a value exists" (``expected`` is not None).
    TP/TN/FP/FN are derived from ``(expected, extracted, comparison_result)``:
    a wrong-value case (both non-None but unequal) counts as **both** a
    false positive (predicted positive, wrong class) and a false negative
    (real positive went uncaught).

      accuracy  = (TP + TN) / N
      precision = TP / (TP + FP)             # over cases that output a value
      recall    = TP / (TP + FN)             # over cases where a value exists
      f1        = 2PR / (P + R)              # point estimate only
    """
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for r in results:
        exp, ext = r["expected"], r["extracted"]
        if r["comparison_result"] == "Success":
            counts["TP" if exp is not None else "TN"] += 1
        else:
            if ext is not None:
                counts["FP"] += 1
            if exp is not None:
                counts["FN"] += 1

    tp, tn, fp, fn = counts["TP"], counts["TN"], counts["FP"], counts["FN"]
    n = len(results)
    pred_pos = tp + fp
    actual_pos = tp + fn

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
        "total_input_tokens": sum(r["input_tokens"] for r in results),
        "total_output_tokens": sum(r["output_tokens"] for r in results),
        "total_time_taken_s": sum(r["time_taken_s"] for r in results),
    }


def _metrics_entry(feature, metrics):
    """Build the per-feature metrics dict written to the JSON list"""
    c = metrics["counts"]
    return {
        "feature": feature,
        "n_cases": metrics["n"],
        "accuracy": round(metrics["accuracy"], 4),
        "accuracy_95_percent_confidence_interval": _ci_str(
            metrics["accuracy_ci"]
        ),
        "precision": round(metrics["precision"], 4),
        "precision_95_percent_confidence_interval": _ci_str(
            metrics["precision_ci"]
        ),
        "recall": round(metrics["recall"], 4),
        "recall_95_percent_confidence_interval": _ci_str(
            metrics["recall_ci"]
        ),
        "f1": round(metrics["f1"], 4),
        "true_positive": c["TP"],
        "true_negative": c["TN"],
        "false_positive": c["FP"],
        "false_negative": c["FN"],
        "failing_cases": metrics["n"] - c["TP"] - c["TN"],
        "total_input_tokens": metrics["total_input_tokens"],
        "total_output_tokens": metrics["total_output_tokens"],
        "total_time_taken_s": round(metrics["total_time_taken_s"], 2),
        "total_cost_usd": round(metrics["total_cost"], 4),
    }


def _ci_str(ci):
    """Format (lo, hi) as 'lo - hi' string, or None if undefined"""
    lo, hi = ci
    return None if lo is None else f"{lo:.4f} - {hi:.4f}"


def _write_metrics_json(fp, entries):
    """Write the per-feature metrics list as JSON"""
    with fp.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)
        fh.write("\n")


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
            str(row["fips"]): row["comparison_result"] == _SUCCESS
            for row in csv.DictReader(fh)
        }


def _load_baseline_failing(metrics_fp):
    """Sum of ``failing_cases`` across features in a baseline metrics JSON"""
    if not metrics_fp.exists():
        return None
    with metrics_fp.open(encoding="utf-8") as fh:
        entries = json.load(fh)
    return sum(e["failing_cases"] for e in entries)


def _check_full_regression(rows, baseline):
    """dev gate: aggregate (failing count) + per-row regression checks"""
    if baseline is None:
        return [], ["  gate: no baseline yet (this run sets it)"]

    now_correct = {
        str(r["fips"]): r["comparison_result"] == _SUCCESS
        for r in rows
    }
    fails_now = sum(1 for ok in now_correct.values() if not ok)
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


def _check_aggregate_regression(fails_now, baseline_failing):
    """held_out gate: aggregate failing-count only (no per-row detail)"""
    if baseline_failing is None:
        return [], ["  gate: no baseline yet (this run sets it)"]
    lines = [f"  gate: failing now={fails_now} baseline={baseline_failing}"]
    failures = []
    if fails_now > baseline_failing:
        failures.append(
            f"aggregate regression: {fails_now} failing > {baseline_failing}"
        )
    return failures, lines


@pytest.fixture(scope="module")
def report_evals(request):
    """Return ``report(eval_name, results_by_type, results_dir)`` callable

    Used by each eval suite's teardown fixture to write CSVs/JSON, print
    a summary, and enforce the regression gate. No-ops when the eval did
    not run (deselected or skipped, so ``results_by_type`` has no rows).
    """
    def report(eval_name, results_by_type, results_dir):
        if not any(results_by_type.values()):
            return
        tr = request.config.pluginmanager.get_plugin("terminalreporter")
        write = tr.write_line
        results_dir.mkdir(parents=True, exist_ok=True)

        gate_failures = []
        for eval_type, rows in sorted(results_by_type.items()):
            if not rows:
                continue
            failures, summary_lines = _process_eval_type(
                eval_name, eval_type, rows, results_dir
            )
            gate_failures.extend(
                f"[{eval_name}/{eval_type}] {m}" for m in failures
            )
            tr.section(f"Eval summary: {eval_name} / {eval_type}")
            for line in summary_lines:
                write(line)

        if gate_failures:
            tr.section("Eval regression gate: FAILED")
            for f in gate_failures:
                write(f"  - {f}")
            tr._session.exitstatus = pytest.ExitCode.TESTS_FAILED

    return report


def _process_eval_type(eval_name, eval_type, rows, results_dir):
    """Compute metrics, write CSVs/JSON, run gate; return (failures, lines)"""
    eval_type_dir = results_dir / eval_type
    eval_type_dir.mkdir(parents=True, exist_ok=True)
    metrics_fp = eval_type_dir / f"{eval_name}_evals.json"
    breakdown_fp = eval_type_dir / f"{eval_name}_evals_breakdown.csv"

    by_feature = {}
    for r in rows:
        by_feature.setdefault(r["feature"], []).append(r)
    per_feature_metrics = {
        f: _compute_metrics(frows)
        for f, frows in sorted(by_feature.items())
    }
    entries = [
        _metrics_entry(f, m) for f, m in per_feature_metrics.items()
    ]

    # held_out: only summary stats are surfaced/saved (no per-case
    # breakdown), and the gate is aggregate-only -- this keeps the
    # held-out set hard to inspect or tune against.
    if eval_type == "held_out":
        baseline_failing = _load_baseline_failing(metrics_fp)
        fails_now = sum(e["failing_cases"] for e in entries)
        failures, gate_lines = _check_aggregate_regression(
            fails_now, baseline_failing
        )
        _write_metrics_json(metrics_fp, entries)
        extra = [f"  metrics: {metrics_fp}"]
    else:
        baseline = _load_baseline_correct(breakdown_fp)
        failures, gate_lines = _check_full_regression(rows, baseline)
        _write_breakdown_csv(breakdown_fp, rows)
        _write_metrics_json(metrics_fp, entries)
        extra = [
            f"  breakdown: {breakdown_fp}",
            f"  metrics: {metrics_fp}",
        ]

    summary_lines = []
    for feature, m in per_feature_metrics.items():
        c = m["counts"]
        summary_lines.extend([
            (
                f"  [{feature}] cases={m['n']}  "
                f"TP={c['TP']} TN={c['TN']} FP={c['FP']} FN={c['FN']}"
            ),
            (
                f"    accuracy={m['accuracy']:.3f} "
                f"95%CI[{_ci_str(m['accuracy_ci']) or ''}]  "
                f"precision={m['precision']:.3f} "
                f"95%CI[{_ci_str(m['precision_ci']) or ''}]  "
                f"recall={m['recall']:.3f} "
                f"95%CI[{_ci_str(m['recall_ci']) or ''}]  "
                f"f1={m['f1']:.3f}"
            ),
            f"    total LLM cost: ${m['total_cost']:.4f}",
        ])
    summary_lines.extend(gate_lines)
    summary_lines.extend(extra)
    return failures, summary_lines
