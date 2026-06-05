"""Evals reporting helpers"""

import json
from pathlib import Path
from dataclasses import asdict

from .base import Result
from .metrics import compute_metrics


class _PrintReporter:
    """Stdout reporter used when there's no pytest terminal reporter"""

    def section(self, title):
        bar = "=" * 27
        print(f"\n{bar} {title} {bar}")

    def write_line(self, line, **_kwargs):
        print(line)


def _resolve_reporter(request):
    if request is None:
        return _PrintReporter()
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    return reporter if reporter is not None else _PrintReporter()


def report_evals(request, eval_name, results, results_dir):
    """Compute the eval metrics, write the metrics JSON, print the summary

    Pass ``request=None`` to report outside pytest (uses a stdout reporter).
    """
    if not results:
        return

    tr = _resolve_reporter(request)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_fp = results_dir / f"{eval_name}_evals.json"

    results_by_feature = {}
    for result in results:
        results_by_feature.setdefault(result.feature, []).append(result)
    metrics_by_feature = [
        _format_entry(feature, compute_metrics(feature_results))
        for feature, feature_results in sorted(results_by_feature.items())
    ]

    _write_metrics_json(metrics_fp, metrics_by_feature)

    tr.section(f"Eval summary: {eval_name}")
    for entry in metrics_by_feature:
        for k, v in entry.items():
            tr.write_line(f"  {k}={v}")
    tr.write_line(f"  metrics: {metrics_fp}")


class PerJurisdictionResults:
    """One ``Result`` JSON file per jurisdiction in a directory

    A simple sharded store that lets pytest-xdist workers each write their
    own cases (one file per jurisdiction, so concurrent writes never
    collide) while the controller reads them all back at session end. The
    directory is the only state -- nothing eval-specific lives here.
    """

    def __init__(self, results_dir):
        self.dir = Path(results_dir)

    def write(self, result, label):
        """Write one ``Result`` to ``<label>.json``"""
        self.dir.mkdir(parents=True, exist_ok=True)
        with (self.dir / f"{label}.json").open("w", encoding="utf-8") as fh:
            json.dump(asdict(result), fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    def load(self):
        """Read every ``<label>.json`` back into a list of ``Result``"""
        if not self.dir.exists():
            return []
        results = []
        for fp in sorted(self.dir.glob("*.json")):
            with fp.open(encoding="utf-8") as fh:
                results.append(Result(**json.load(fh)))
        return results

    def clear(self):
        """Delete all result files (stale shards from a previous run)"""
        if not self.dir.exists():
            return
        for fp in self.dir.glob("*.json"):
            fp.unlink()


def _write_metrics_json(fp, entries):
    with fp.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)
        fh.write("\n")


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
