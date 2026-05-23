"""Accuracy tests for :func:`compass.extraction.apply.extract_date`.

This test feeds real ordinance documents (loaded from local disk) to
``extract_date`` in the same type/format the production pipeline uses: a
:class:`~elm.web.document.BaseDocument` carrying the real ``source`` URL in
its ``attrs`` and the parsed document text in its ``raw_pages``. The known
"correct" enactment year for each document comes from the solar-validation
ground-truth dataset.

The dataset is described by the hand-maintained
``tests/data/solar_validation_files/manifest.json5`` (alongside the
documents it references). Each manifest entry has::

    {
        "fips": 13015,
        "jurisdiction": "Bartow Georgia",
        "file": "Bartow_County_Georgia.pdf",
        "source": "https://.../ordinance.pdf",
        "expected_year": 2020,
        // optional: mark a case as known-hard so a failure xfails
        // instead of failing the suite
        "allow_failure": false
    }

These cases make **live LLM calls**, so the whole module is skipped unless
Azure OpenAI credentials are available in the environment. Configure via:

    COMPASS_DATE_TEST_MODEL          (default: "compassop-gpt-5.4")
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_VERSION             (default: "2025-04-01-preview")

Notes
-----
The CSV ground truth contains only the enactment **year** (no month/day), so
each case strictly asserts the extracted year. Month and day are reported for
visibility but not asserted.

Cases where the ground truth says there is *no* enactment date have
``expected_year: null`` in the manifest; for those, the test asserts that
``extract_date`` returns ``None`` for the year (i.e. it does not invent a
date). This guards the false-positive direction.
"""

import os
import logging
from pathlib import Path

import pytest
from elm.web.document import HTMLDocument, PDFDocument
from elm.utilities.parse import read_pdf

from compass.llm.config import OpenAIConfig
from compass.extraction.apply import extract_date
from compass.utilities.io import load_config
from compass.utilities.costs import (
    LLM_COST_REGISTRY,
    compute_cost_from_totals,
)
from compass.services.openai import usage_from_response
from compass.services.usage import UsageTracker
from compass.services.provider import RunningAsyncServices


logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parents[2] / "data" / "solar_validation_files"
MANIFEST_FP = DATASET_DIR / "manifest.json5"

DEFAULT_MODEL = "compassop-gpt-5.4"

# $/million tokens, used only to report cost for the test run. Override via
# COMPASS_DATE_TEST_COST="<prompt>,<response>" if your rates differ.
DEFAULT_COST_PER_MTOK = (1.25, 7.5)

# Per-case cost/accuracy records, rendered as a breakdown at session end by
# the ``pytest_terminal_summary`` hook in this package's conftest.py.
DATE_ACCURACY_RESULTS = []
DEFAULT_API_VERSION = "2025-04-01-preview"


def _azure_credentials_available():
    """True if Azure OpenAI creds are present in the environment"""
    return bool(
        os.environ.get("AZURE_OPENAI_API_KEY")
        and os.environ.get("AZURE_OPENAI_ENDPOINT")
    )


def _load_manifest():
    """Load the date-accuracy ground-truth manifest"""
    if not MANIFEST_FP.exists():
        return []
    return load_config(MANIFEST_FP)


# This test makes live, billable LLM calls. It is opt-in only:
#   - `expensive` marker -> deselected by default; run with `-m expensive`
#   - credential skip    -> also requires Azure creds in the environment
#   - manifest skip      -> requires the ground-truth dataset
pytestmark = [
    pytest.mark.expensive,
    pytest.mark.skipif(
        not _azure_credentials_available(),
        reason="Azure OpenAI credentials not set "
        "(AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT)",
    ),
    pytest.mark.skipif(
        not MANIFEST_FP.exists(),
        reason=f"Date-accuracy manifest not found at {MANIFEST_FP}",
    ),
]

_MANIFEST = _load_manifest()


@pytest.fixture(scope="module")
def date_model_config():
    """Env-configurable LLM config for date extraction"""
    model_name = os.environ.get("COMPASS_DATE_TEST_MODEL", DEFAULT_MODEL)

    # Register a cost rate for this model so per-case $ can be reported.
    if model_name not in LLM_COST_REGISTRY:
        rate = os.environ.get("COMPASS_DATE_TEST_COST")
        prompt_rate, response_rate = (
            tuple(float(x) for x in rate.split(","))
            if rate
            else DEFAULT_COST_PER_MTOK
        )
        LLM_COST_REGISTRY[model_name] = {
            "prompt": prompt_rate,
            "response": response_rate,
        }

    return OpenAIConfig(
        name=model_name,
        llm_call_kwargs={"temperature": 1, "timeout": 300},
        client_type="azure",
        client_kwargs={
            "api_key": os.environ["AZURE_OPENAI_API_KEY"],
            "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
            "api_version": os.environ.get(
                "AZURE_OPENAI_VERSION", DEFAULT_API_VERSION
            ),
        },
    )


def _build_doc(case):
    """Build a production-shaped document from a manifest case

    Mirrors what the pipeline feeds ``extract_date``: a document carrying
    the real ``source`` URL in ``attrs`` and the parsed text in
    ``raw_pages``. No ``"date"`` attr is set, so extraction is not
    short-circuited.
    """
    fp = DATASET_DIR / case["file"]
    attrs = {"source": case["source"]}

    if fp.suffix.casefold() == ".pdf":
        pages = read_pdf(fp.read_bytes(), verbose=False)
        return PDFDocument(pages, attrs=attrs)

    text = fp.read_text(encoding="utf-8", errors="ignore")
    return HTMLDocument([text], attrs=attrs)


async def _extract_year(case, model_config):
    """Run ``extract_date`` for one case and return the extracted year"""
    doc = _build_doc(case)
    assert "date" not in doc.attrs, "doc should not be pre-dated"

    usage_tracker = UsageTracker(case["jurisdiction"], usage_from_response)
    async with RunningAsyncServices([model_config.llm_service]):
        doc = await extract_date(
            doc, model_config, usage_tracker=usage_tracker
        )

    cost = compute_cost_from_totals(usage_tracker.totals)
    year, month, day = doc.attrs["date"]

    if case["expected_year"] is None:
        correct = year is None
    else:
        correct = year == case["expected_year"]

    DATE_ACCURACY_RESULTS.append(
        {
            "jurisdiction": case["jurisdiction"],
            "file": case["file"],
            "expected": case["expected_year"],
            "extracted": year,
            "correct": correct,
            "allow_failure": bool(case.get("allow_failure")),
            "cost": cost,
        }
    )

    logger.info(
        "%s (FIPS %s): expected_year=%s -> extracted=(%s, %s, %s)  cost=$%.4f",
        case["jurisdiction"],
        case["fips"],
        case["expected_year"],
        year,
        month,
        day,
        cost,
    )
    return year


@pytest.mark.parametrize(
    "case",
    _MANIFEST,
    ids=[c["file"] for c in _MANIFEST],
)
async def test_extract_date_year_accuracy(case, date_model_config, request):
    """Extracted year matches the known ground truth for the document

    For documents with a known enactment year, the extracted year must
    match it. For documents the ground truth marks as having no date
    (``expected_year is None``), the extractor must not invent one.

    Cases flagged ``allow_failure: true`` in the manifest are treated as
    known-hard: a mismatch is reported as an xfail rather than failing
    the suite, while an unexpected pass shows up as an xpass.
    """
    extracted_year = await _extract_year(case, date_model_config)
    expected = case["expected_year"]

    if expected is None:
        correct = extracted_year is None
        detail = (
            f"expected NO year, but extracted {extracted_year}"
            if not correct
            else "correctly extracted no year"
        )
    else:
        correct = extracted_year == expected
        detail = f"expected year {expected}, got {extracted_year}"

    message = (
        f"{case['jurisdiction']} (FIPS {case['fips']}, {case['file']}): "
        f"{detail} [source: {case['source']}]"
    )

    if not correct and case.get("allow_failure"):
        pytest.xfail(f"known-hard case: {message}")

    assert correct, message


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
