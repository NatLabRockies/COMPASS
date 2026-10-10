"""Stage-independent minimum targets for persisted shard records"""

import json
import math

from compass.exceptions import COMPASSValueError


def normalize_targets(targets):
    """Validate and copy a mapping of metric names to minimum values

    Parameters
    ----------
    targets : dict or None
        Metric names and finite numeric lower bounds. ``None`` produces
        an empty mapping.

    Returns
    -------
    dict
        Validated targets, independent of the input mapping.
    """
    if targets is None:
        return {}

    if not isinstance(targets, dict):
        msg = "Targets must be a metric-to-minimum mapping"
        raise COMPASSValueError(msg)

    normalized = {}
    for metric, minimum in targets.items():
        if not isinstance(metric, str) or not metric.strip():
            msg = "Target metric names must be nonempty"
            raise COMPASSValueError(msg)

        if not _is_number(minimum):
            msg = f"Target '{metric}' requires a finite numeric minimum"
            raise COMPASSValueError(msg)

        normalized[metric.strip()] = minimum

    return normalized


def parse_target_options(options):
    """Parse repeated CLI targets, retaining the strictest minimum

    Parameters
    ----------
    options : iterable of str
        Expressions using ``METRIC=MINIMUM`` syntax.

    Returns
    -------
    dict
        Parsed targets. Repeated metrics use their largest minimum.
    """
    targets = {}
    for option in options:
        metric, separator, raw_minimum = option.partition("=")
        if not separator:
            msg = f"Invalid target '{option}'; use METRIC=MINIMUM"
            raise COMPASSValueError(msg)

        try:
            minimum = json.loads(raw_minimum)
        except (ValueError, TypeError) as exc:
            msg = f"Invalid minimum in target '{option}'"
            raise COMPASSValueError(msg) from exc

        for name, value in normalize_targets({metric: minimum}).items():
            targets[name] = max(targets.get(name, value), value)

    return targets


def evaluate_targets(record, targets, metrics):
    """Return unmet minimums using stage-provided metric accessors

    Parameters
    ----------
    record : dict
        One persisted shard record.
    targets : dict
        Validated metric names and numeric minimums.
    metrics : dict
        Metric names mapped to callables accepting a record. Accessors
        return a finite number or ``None`` for unavailable values.

    Returns
    -------
    list of dict
        Failures containing ``metric``, ``actual``, and ``minimum``.
        An empty list means every supplied target passed.
    """
    failures = []
    for metric, minimum in normalize_targets(targets).items():
        if metric not in metrics:
            msg = f"Unknown shard target '{metric}'"
            raise COMPASSValueError(msg)

        actual = metrics[metric](record)
        if actual is not None and not _is_number(actual):
            msg = f"Shard metric '{metric}' must be a finite number"
            raise COMPASSValueError(msg)

        if actual is None or actual < minimum:
            failures.append(
                {"metric": metric, "actual": actual, "minimum": minimum}
            )

    return failures


def _is_number(value):
    """Check for a finite numeric value without accepting booleans"""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    )
