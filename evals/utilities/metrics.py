"""Evals metrics computation"""

from .base import SUCCESS


def wilson_ci(k, n, alpha=0.05):
    """95% Wilson score interval for k/n, or ``(None, None)`` if n == 0

    Chosen over the normal approximation because it has better coverage
    at small sample sizes and near the 0/1 boundary. IID (ignores
    clustering). ``statsmodels`` is imported lazily so the base test
    session (which collects this module but deselects the evals) does
    not depend on it.
    """
    if n == 0:
        return None, None
    from statsmodels.stats.proportion import (  # noqa: PLC0415
        proportion_confint,
    )

    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return float(lo), float(hi)


def compute_metrics(results):
    """Accuracy / precision / recall / F1 (+ 95% Wilson CIs).

    A wrong value is counted toward both false positives and false
    negatives.

    accuracy  = (TP + TN) / N
    precision = TP / (TP + FP)             # over cases that output a value
    recall    = TP / (TP + FN)             # over cases where a value exists
    f1        = 2PR / (P + R)
    """
    counts = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    for r in results:
        success = r.comparison_result == SUCCESS
        if success and r.expected is not None:
            counts["true_positive"] += 1
        if success and r.expected is None:
            counts["true_negative"] += 1
        if not success and r.extracted is not None:
            counts["false_positive"] += 1
        if not success and r.expected is not None:
            counts["false_negative"] += 1

    n = len(results)
    pred_pos = counts["true_positive"] + counts["false_positive"]
    actual_pos = counts["true_positive"] + counts["false_negative"]
    correct = counts["true_positive"] + counts["true_negative"]

    def _safe_div(num, den):
        return num / den if den else 0.0

    precision = _safe_div(counts["true_positive"], pred_pos)
    recall = _safe_div(counts["true_positive"], actual_pos)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "n_cases": n,
        **counts,
        "failing_cases": n - correct,
        "accuracy": _safe_div(correct, n),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_95_percent_confidence_interval": wilson_ci(correct, n),
        "precision_95_percent_confidence_interval": wilson_ci(
            counts["true_positive"], pred_pos
        ),
        "recall_95_percent_confidence_interval": wilson_ci(
            counts["true_positive"], actual_pos
        ),
        "total_prompt_tokens": sum(r.prompt_tokens for r in results),
        "total_response_tokens": sum(r.response_tokens for r in results),
        "total_time_taken_s": sum(r.time_taken_s for r in results),
        "total_cost_usd": sum(r.cost for r in results),
    }
