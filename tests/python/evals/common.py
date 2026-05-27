"""Shared schema + helpers for eval suites

Eval-agnostic pieces that any ``test_run_<name>_evals.py`` can import.
Pytest-specific machinery (fixtures, gate, reporter) lives in
``conftest.py``.
"""

from dataclasses import dataclass, fields

SUCCESS = "Success"
FAILURE = "Failure"


@dataclass
class Result:
    """One row of an eval breakdown -- the per-case result schema

    Identity columns describe *which* case; result columns capture
    the ground-truth comparison; usage columns capture per-case cost.
    Field order is the canonical column order for the breakdown CSV.
    """

    # identity
    fips: int
    state: str
    county: str
    subdivision: str | None
    jurisdiction_type: str
    file: str
    source: str
    feature: str

    # ground-truth comparison
    expected: object
    extracted: object
    comparison_result: str

    # usage
    prompt_tokens: int
    response_tokens: int
    time_taken_s: float
    cost: float


RESULT_FIELDS = [f.name for f in fields(Result)]


def classify(expected, extracted):
    """Binary success: did the extractor match ground truth?"""
    return SUCCESS if extracted == expected else FAILURE


def display_name(case):
    """Human-readable jurisdiction label from a manifest case"""
    parts = [case["subdivision"], case["county"], case["state"]]
    return ", ".join(p for p in parts if p)
