"""Eval reporting helpers (I/O + formatting, no policy).

Each eval suite (``test_run_<name>_evals.py``) calls :func:`report_evals`
from a module-scoped teardown fixture. ``report_evals`` writes the
per-case breakdown CSV (dev only) and the per-feature metrics JSON,
prints a summary, and returns the data each test needs to enforce its
own regression gate.

Regression-gate *policy* lives in the calling test module. This module
exposes two helpers each test can compose into a gate:

- :func:`load_baseline_failing` -- aggregate failing count from a
  committed metrics JSON
- :func:`regressed_rows` -- list of jurisdictions that were correct in
  the committed breakdown CSV but failing now

A typical gate compares ``failing_cases`` now vs baseline (and, for dev,
checks that no previously-correct row regressed).
"""

import csv
import json
from dataclasses import asdict

from compass.utilities.jurisdictions import Jurisdiction

from .base import RESULT_FIELDS, SUCCESS
from .metrics import compute_metrics


def _format_entry(feature, metrics):
    """Format a per-feature metrics dict for the JSON list"""
    round_4 = {"accuracy", "precision", "recall", "f1", "total_cost_usd"}
    round_2 = {"total_time_taken_s"}
    ci_keys = {
        "accuracy_95_percent_confidence_interval",
        "precision_95_percent_confidence_interval",
        "recall_95_percent_confidence_interval",
    }

    out = {"feature": feature}
    for key, val in metrics.items():
        if key in ci_keys:
            lo, hi = val
            out[key] = None if lo is None else f"{lo:.4f} - {hi:.4f}"
        elif key in round_4:
            out[key] = round(val, 4)
        elif key in round_2:
            out[key] = round(val, 2)
        else:
            out[key] = val
    return out


def _write_metrics_json(fp, entries):
    """Write the per-feature metrics list as JSON"""
    with fp.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)
        fh.write("\n")


def _write_breakdown_csv(fp, results):
    """Write the detailed per-case breakdown CSV"""
    with fp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in sorted(
            results,
            key=lambda r: (r.state, r.county, r.subdivision or ""),
        ):
            writer.writerow(asdict(row))


def load_baseline_failing(metrics_fp):
    """Sum of ``failing_cases`` across features in a baseline metrics JSON

    Returns ``None`` if the file does not exist (no baseline yet).
    """
    if not metrics_fp.exists():
        return None
    with metrics_fp.open(encoding="utf-8") as fh:
        entries = json.load(fh)
    return sum(e["failing_cases"] for e in entries)


def regressed_rows(rows, breakdown_fp):
    """Jurisdictions that were correct in the baseline CSV but failing now

    Returns a sorted list of :class:`Jurisdiction` instances, or ``None``
    if the baseline file does not exist (no baseline yet).
    """
    if not breakdown_fp.exists():
        return None
    with breakdown_fp.open(newline="", encoding="utf-8") as fh:
        baseline_correct = {
            Jurisdiction(
                subdivision_type=row["jurisdiction_type"],
                state=row["state"],
                county=row["county"],
                subdivision_name=row["subdivision"] or None,
            ): row["comparison_result"] == SUCCESS
            for row in csv.DictReader(fh)
        }
    now_correct = {
        r.jurisdiction: r.comparison_result == SUCCESS for r in rows
    }
    return sorted(
        (
            j for j, was_ok in baseline_correct.items()
            if was_ok and now_correct.get(j) is False
        ),
        key=str,
    )


def report_evals(
    request, eval_name, rows, results_dir, *, write_breakdown=True
):
    """Snapshot baselines, write CSVs/JSON, print a summary, return data

    Writes the per-feature metrics JSON to ``results_dir`` and (when
    ``write_breakdown`` is true) the per-case breakdown CSV. Baselines
    (``baseline_failing`` and ``regressed_rows``) are computed against
    the *committed* files **before** the new files are written, so the
    caller can compare them to the current run's metrics.

    Held-out runs typically pass ``write_breakdown=False`` -- the
    breakdown lists every document and how it scored, which makes the
    held-out set easy to tune against.

    Returns
    -------
    dict or None
        ``{rows, metrics, fails_now, baseline_failing, regressed_rows,
        breakdown_fp, metrics_fp}``. ``None`` when ``rows`` is empty
        (the eval did not run -- deselected or skipped).
        ``regressed_rows`` is ``None`` when the breakdown wasn't written
        or no baseline exists yet.
    """
    if not rows:
        return None

    tr = request.config.pluginmanager.get_plugin("terminalreporter")
    write = tr.write_line
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_fp = results_dir / f"{eval_name}_evals.json"
    breakdown_fp = results_dir / f"{eval_name}_evals_breakdown.csv"

    # Snapshot baselines BEFORE writing the new files.
    baseline_failing = load_baseline_failing(metrics_fp)
    regressed = regressed_rows(rows, breakdown_fp) if write_breakdown else None

    by_feature = {}
    for r in rows:
        by_feature.setdefault(r.feature, []).append(r)
    entries = [
        _format_entry(f, compute_metrics(frows))
        for f, frows in sorted(by_feature.items())
    ]
    fails_now = sum(e["failing_cases"] for e in entries)

    _write_metrics_json(metrics_fp, entries)
    extra = [f"  metrics: {metrics_fp}"]
    if write_breakdown:
        _write_breakdown_csv(breakdown_fp, rows)
        extra.insert(0, f"  breakdown: {breakdown_fp}")

    tr.section(f"Eval summary: {eval_name}")
    for entry in entries:
        for k, v in entry.items():
            write(f"  {k}={v}")
    for line in extra:
        write(line)

    return {
        "rows": rows,
        "metrics": entries,
        "fails_now": fails_now,
        "baseline_failing": baseline_failing,
        "regressed_rows": regressed,
        "breakdown_fp": breakdown_fp,
        "metrics_fp": metrics_fp,
    }
