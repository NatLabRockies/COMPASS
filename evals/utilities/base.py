"""Shared schema + doc loading for eval suites

Eval-agnostic pieces (the ``Result`` row schema, success/failure
constants, ``classify``, ``load_doc``) that any
``test_run_<name>_evals.py`` can import. Metric computation lives in
``utilities.metrics``; result formatting and I/O live in
``utilities.reports``.
"""

from dataclasses import dataclass, fields
from pathlib import Path

from elm.web.document import HTMLDocument, PDFDocument
from elm.utilities.parse import read_pdf
from compass.utilities.jurisdictions import Jurisdiction

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

    @property
    def jurisdiction(self):
        """Canonical Jurisdiction instance (hashable, for gate matching)"""
        return Jurisdiction(
            subdivision_type=self.jurisdiction_type,
            state=self.state,
            county=self.county,
            subdivision_name=self.subdivision or None,
        )


RESULT_FIELDS = [f.name for f in fields(Result)]


def classify(expected, extracted):
    """Binary success: did the extractor match ground truth?"""
    return SUCCESS if extracted == expected else FAILURE


def load_doc(fp, *, source=None):
    """Load a local file as an elm Document, dispatching on suffix"""
    fp = Path(fp)
    attrs = {"source": source} if source else {}
    if fp.suffix.casefold() == ".pdf":
        pages = read_pdf(fp.read_bytes(), verbose=False)
        return PDFDocument(pages, attrs=attrs)
    text = fp.read_text(encoding="utf-8", errors="ignore")
    return HTMLDocument([text], attrs=attrs)
