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


def jurisdiction_from_case(case):
    """Build a :class:`compass.utilities.jurisdictions.Jurisdiction`

    Translates manifest field names (``subdivision``, ``jurisdiction_type``)
    to the constructor's parameter names (``subdivision_name``,
    ``subdivision_type``). Use ``.full_name`` on the returned object for
    a human-readable display label.
    """
    from compass.utilities.jurisdictions import Jurisdiction  # noqa: PLC0415

    return Jurisdiction(
        subdivision_type=case["jurisdiction_type"],
        state=case["state"],
        county=case["county"],
        subdivision_name=case["subdivision"],
    )
