"""Common utilities for evals suites"""

from dataclasses import dataclass, fields
from pathlib import Path

from elm.web.document import HTMLDocument, PDFDocument
from elm.utilities.parse import read_pdf
from compass.utilities.jurisdictions import Jurisdiction

SUCCESS = "Success"
FAILURE = "Failure"


@dataclass
class Result:
    """One row of an evals breakdown -- the per-case result schema"""

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
        """Canonical Jurisdiction instance (hashable)"""
        return Jurisdiction(
            subdivision_type=self.jurisdiction_type,
            state=self.state,
            county=self.county,
            subdivision_name=self.subdivision or None,
        )


RESULT_FIELDS = [f.name for f in fields(Result)]


def classify(expected, extracted, match_type="exact"):
    """Binary success: did the extractor match ground truth?"""
    if match_type == "exact":
        return SUCCESS if extracted == expected else FAILURE
    # Add other match types here as needed
    return FAILURE


def load_doc(fp, *, source=None):
    """Load a local file as an elm Document, dispatching on suffix"""
    fp = Path(fp)
    attrs = {"source": source} if source else {}
    if fp.suffix.casefold() == ".pdf":
        pages = read_pdf(fp.read_bytes(), verbose=False)
        return PDFDocument(pages, attrs=attrs)
    text = fp.read_text(encoding="utf-8", errors="ignore")
    return HTMLDocument([text], attrs=attrs)
