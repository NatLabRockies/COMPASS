"""Evals reporting helpers"""

import json
from dataclasses import asdict
import pandas as pd

from .base import RESULT_FIELDS, SUCCESS, Result
from .metrics import compute_metrics


def report_evals(
    request, eval_name, results, results_dir, *, write_breakdown=True
):
    """Calculate the evals metrics and write the reports to results/"""
    if not results:
        return None

    tr = request.config.pluginmanager.get_plugin("terminalreporter")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_fp = results_dir / f"{eval_name}_evals.json"
    breakdown_fp = results_dir / f"{eval_name}_evals_breakdown.csv"

    # Snapshot baselines BEFORE writing the new files.
    existing_results = (
        _read_breakdown_csv(breakdown_fp) if write_breakdown else None
    )
    baseline_failing = _get_failing_count(existing_results)
    regressed = _get_regressed_jurisdictions(results, existing_results)
    fails_now = _get_failing_count(results)

    results_by_feature = {}
    for result in results:
        results_by_feature.setdefault(result.feature, []).append(result)
    metrics_by_feature = [
        _format_entry(feature, compute_metrics(feature_results))
        for feature, feature_results in sorted(results_by_feature.items())
    ]

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
        "current_nfail": fails_now,
        "prev_nfail": baseline_failing,
        "regressed_jurs": regressed,
    }


def _get_failing_count(results):
    if results is None:
        return None
    return sum(1 for r in results if r.comparison_result != SUCCESS)


def _read_breakdown_csv(fp):
    if not fp.exists():
        return None
    df = pd.read_csv(fp, dtype=str, keep_default_na=False, na_values=[])
    return [_result_from_csv_row(row) for row in df.to_dict("records")]


def _result_from_csv_row(row):
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


def _write_metrics_json(fp, entries):
    with fp.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)
        fh.write("\n")


def _write_breakdown_csv(fp, results):
    ordered = sorted(
        results, key=lambda r: (r.state, r.county, r.subdivision or "")
    )
    df = pd.DataFrame([asdict(r) for r in ordered], columns=RESULT_FIELDS)
    df.to_csv(fp, index=False)


def _format_entry(feature, metrics):
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
