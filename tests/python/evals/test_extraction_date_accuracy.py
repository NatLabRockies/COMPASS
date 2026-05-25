"""Accuracy tests for :func:`compass.extraction.apply.extract_date`.

This test feeds real ordinance documents (loaded from local disk) to
``extract_date`` in the same type/format the production pipeline uses: a
:class:`~elm.web.document.BaseDocument` carrying the real ``source`` URL in
its ``attrs`` and the parsed document text in its ``raw_pages``. The known
"correct" enactment year for each document comes from the solar-validation
ground-truth dataset.

There are two datasets, same eval logic, different cadence:
  - ``dev``: run frequently during development (`-m dev_eval`)
  - ``held-out``: run before a release (`-m held_out`)

Each lives under ``tests/python/evals/data/<cadence>/solar_validation_files/``
with a hand-maintained ``manifest.json5`` (alongside the documents). Each
manifest entry has::

    {
        "fips": 13015,
        "jurisdiction": "Bartow Georgia",
        "file": "Bartow_County_Georgia.pdf",
        "source": "https://.../ordinance.pdf",
        "expected_year": 2020
    }

(``expected_year: null`` means the ground truth is "no enactment date exists".)

These cases make **live LLM calls**, so the whole module is skipped unless
Azure OpenAI credentials are available in the environment. Configure via:

    COMPASS_DATE_TEST_MODEL          (default: "compassop-gpt-5.4")
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_VERSION             (default: "2025-04-01-preview")

What fails a case vs. what is measured
--------------------------------------
A per-case test fails **only** on a mechanical problem -- ``extract_date``
raising (LLM error, timeout, crash, unparseable document). A *wrong* (or
empty) prediction is **not** a failure: correctness is a measurement, recorded
in the breakdown CSV and rolled up into accuracy / precision / recall / F1.

After a run, the ``pytest_terminal_summary`` hook in this package's
``conftest.py`` writes a detailed per-case breakdown CSV (expected vs
predicted, correct flag, classification category, cost) plus overall metrics
to ``tests/python/evals/results/``, and enforces a regression gate against the
committed baseline breakdown (see ``conftest.py`` for the gate logic).

Notes
-----
The ground truth contains only the enactment **year** (no month/day); month
and day are recorded for visibility but not scored. A document the ground
truth marks as having no date should yield no extracted year (guards the
false-positive direction).
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

# Two eval datasets, same test logic, both committed in-repo. They differ
# only by cadence + which split of the data they hold (see data/README.md).
# Documents are resolved relative to each manifest's own directory.
_DATA_DIR = Path(__file__).parent / "data"
DEV_MANIFEST_FP = (
    _DATA_DIR / "dev" / "solar_validation_files" / "manifest.json5"
)
HELD_OUT_MANIFEST_FP = (
    _DATA_DIR / "held-out" / "solar_validation_files" / "manifest.json5"
)
RESULTS_DIR = Path(__file__).parent / "results"

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


def _load_manifest(manifest_fp):
    """Load a date-accuracy ground-truth manifest, or [] if absent"""
    if not manifest_fp.exists():
        return []
    return load_config(manifest_fp)


# These are evals (they make live, billable LLM calls). Opt-in only:
#   - deselected by default (the `dev_eval` / `held_out` markers below);
#     run one cadence at a time with `-m dev_eval` or `-m held_out`.
#   - credential skip -> also requires Azure creds in the environment.
# Each per-dataset test below carries exactly one cadence marker and skips
# if its dataset is absent. There is no umbrella marker that runs both.
pytestmark = [
    pytest.mark.skipif(
        not _azure_credentials_available(),
        reason="Azure OpenAI credentials not set "
        "(AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT)",
    ),
]

_DEV_CASES = _load_manifest(DEV_MANIFEST_FP)
_HELD_OUT_CASES = _load_manifest(HELD_OUT_MANIFEST_FP)


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


def _build_doc(case, dataset_dir):
    """Build a production-shaped document from a manifest case

    Mirrors what the pipeline feeds ``extract_date``: a document carrying
    the real ``source`` URL in ``attrs`` and the parsed text in
    ``raw_pages``. No ``"date"`` attr is set, so extraction is not
    short-circuited. Document files are resolved relative to the manifest's
    own directory (so dev and held-out sets stay independent).
    """
    fp = dataset_dir / case["file"]
    attrs = {"source": case["source"]}

    if fp.suffix.casefold() == ".pdf":
        pages = read_pdf(fp.read_bytes(), verbose=False)
        return PDFDocument(pages, attrs=attrs)

    text = fp.read_text(encoding="utf-8", errors="ignore")
    return HTMLDocument([text], attrs=attrs)


def _classify(expected, extracted):
    """Confusion-matrix category for an (expected, extracted) year pair

    Positive class = "an enactment year exists". A wrong year counts as
    BOTH a false positive and a false negative (matches the convention in
    solar_validation's ``margin_metrics``).

      TP  expected exists, extracted it correctly
      TN  no expected year, correctly extracted none
      FN  expected exists but nothing extracted (missed)
      FP  no expected year but a year was invented
      WRONG  expected exists, extracted a *different* year (FP and FN)
    """
    if expected is None:
        return "TN" if extracted is None else "FP"
    if extracted is None:
        return "FN"
    return "TP" if extracted == expected else "WRONG"


async def _extract_year(case, dataset_dir, cadence, model_config):
    """Run ``extract_date`` for one case and return the extracted year"""
    doc = _build_doc(case, dataset_dir)
    assert "date" not in doc.attrs, "doc should not be pre-dated"

    usage_tracker = UsageTracker(case["jurisdiction"], usage_from_response)
    async with RunningAsyncServices([model_config.llm_service]):
        doc = await extract_date(
            doc, model_config, usage_tracker=usage_tracker
        )

    cost = compute_cost_from_totals(usage_tracker.totals)
    year, month, day = doc.attrs["date"]
    expected = case["expected_year"]
    correct = (year is None) if expected is None else (year == expected)

    DATE_ACCURACY_RESULTS.append(
        {
            "cadence": cadence,
            "fips": case["fips"],
            "jurisdiction": case["jurisdiction"],
            "file": case["file"],
            "source": case["source"],
            "expected_year": expected,
            "extracted_year": year,
            "extracted_month": month,
            "extracted_day": day,
            "correct": correct,
            "category": _classify(expected, year),
            "cost": cost,
        }
    )

    logger.info(
        "%s (FIPS %s): expected_year=%s -> extracted=(%s, %s, %s)  cost=$%.4f",
        case["jurisdiction"],
        case["fips"],
        expected,
        year,
        month,
        day,
        cost,
    )
    return year


# A per-case test fails ONLY on a mechanical problem -- ``extract_date``
# raising (LLM error, timeout, crash, unparseable doc). A wrong (or empty)
# prediction is recorded, not failed: correctness is a *measurement* captured
# in the breakdown CSV + metrics, and regressions are caught by the aggregate
# checks in ``conftest.py`` (which compare against the committed baseline).


@pytest.mark.dev_eval
@pytest.mark.parametrize(
    "case", _DEV_CASES, ids=[c["file"] for c in _DEV_CASES]
)
async def test_date_year_accuracy_dev(case, date_model_config):
    """Run date extraction on a dev-dataset document and record the result"""
    await _extract_year(case, DEV_MANIFEST_FP.parent, "dev", date_model_config)


@pytest.mark.held_out
@pytest.mark.skipif(
    not HELD_OUT_MANIFEST_FP.exists(),
    reason=f"Held-out dataset not found at {HELD_OUT_MANIFEST_FP}",
)
@pytest.mark.parametrize(
    "case", _HELD_OUT_CASES, ids=[c["file"] for c in _HELD_OUT_CASES]
)
async def test_date_year_accuracy_held_out(case, date_model_config):
    """Run date extraction on a held-out document and record the result"""
    await _extract_year(
        case, HELD_OUT_MANIFEST_FP.parent, "held_out", date_model_config
    )


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
