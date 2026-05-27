"""Evals reporting helpers"""

import csv
import json
from dataclasses import asdict

from .base import RESULT_FIELDS, SUCCESS, Result
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


def _read_breakdown_csv(fp):
    """Read a committed breakdown CSV back into a list of Result rows"""
    if not fp.exists():
        return None
    with fp.open(newline="", encoding="utf-8") as fh:
        return [_result_from_csv_row(row) for row in csv.DictReader(fh)]


def _result_from_csv_row(row):
    """Reconstitute a Result from a breakdown-CSV row (cast numeric fields)"""
    return Result(
        state=row["state"],
        county=row["county"],
        subdivision=row["subdivision"] or None,
        jurisdiction_type=row["jurisdiction_type"],
        file=row["file"],
        source=row["source"],
        feature=row["feature"],
        expected=row["expected"],
        extracted=row["extracted"],
        comparison_result=row["comparison_result"],
        prompt_tokens=int(row["prompt_tokens"]),
        response_tokens=int(row["response_tokens"]),
        time_taken_s=float(row["time_taken_s"]),
        cost=float(row["cost"]),
    )


def load_baseline_failing(metrics_fp):
    """Sum of ``failing_cases`` across features in a baseline metrics JSON"""
    if not metrics_fp.exists():
        return None
    with metrics_fp.open(encoding="utf-8") as fh:
        entries = json.load(fh)
    return sum(e["failing_cases"] for e in entries)


def _get_regressed_jurisdictions(new_results, existing_results):
    """Jurisdictions correct in baseline but failing in current run"""
    if existing_results is None:
        return None
    now_correct = {
        r.jurisdiction for r in new_results if r.comparison_result == SUCCESS
    }
    return sorted(
        (
            r.jurisdiction
            for r in existing_results
            if r.comparison_result == SUCCESS
            and r.jurisdiction not in now_correct
        ),
        key=str,
    )


def report_evals(
    request, eval_name, results, results_dir, *, write_breakdown=True
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
        ``{metrics, fails_now, baseline_failing, regressed_rows,
        breakdown_fp, metrics_fp}``. ``None`` when ``results`` is empty
        (the eval did not run -- deselected or skipped).
        ``regressed_rows`` is ``None`` when the breakdown wasn't written
        or no baseline exists yet.
    """
    if not results:
        return None

    tr = request.config.pluginmanager.get_plugin("terminalreporter")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_fp = results_dir / f"{eval_name}_evals.json"
    breakdown_fp = results_dir / f"{eval_name}_evals_breakdown.csv"

    # Snapshot baselines BEFORE writing the new files.
    baseline_failing = load_baseline_failing(metrics_fp)
    existing_results = (
        _read_breakdown_csv(breakdown_fp) if write_breakdown else None
    )
    regressed = _get_regressed_jurisdictions(results, existing_results)

    results_by_feature = {}
    for result in results:
        results_by_feature.setdefault(result.feature, []).append(result)
    metrics_by_feature = [
        _format_entry(feature, compute_metrics(feature_results))
        for feature, feature_results in sorted(results_by_feature.items())
    ]
    fails_now = sum(e["failing_cases"] for e in metrics_by_feature)

    _write_metrics_json(metrics_fp, metrics_by_feature)
    extra = [f"  metrics: {metrics_fp}"]
    if write_breakdown:
        _write_breakdown_csv(breakdown_fp, results)
        extra.insert(0, f"  breakdown: {breakdown_fp}")

    tr.section(f"Eval summary: {eval_name}")
    for entry in metrics_by_feature:
        for k, v in entry.items():
            tr.write_line(f"  {k}={v}")
    for line in extra:
        tr.write_line(line)

    return {
        "metrics": metrics_by_feature,
        "fails_now": fails_now,
        "baseline_failing": baseline_failing,
        "regressed_rows": regressed,
        "breakdown_fp": breakdown_fp,
        "metrics_fp": metrics_fp,
    }
