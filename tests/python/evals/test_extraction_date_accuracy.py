"""Eval: ordinance enactment-year extraction (`extract_date`).

Feeds real documents to ``extract_date`` the way production does -- a doc
carrying the real ``source`` URL in ``attrs`` and parsed text in ``raw_pages``,
with no pre-set ``"date"`` (so extraction actually runs). Ground-truth years
come from per-cadence manifests under ``data/{dev,held-out}/`` (see
``data/README.md``).

Run one cadence at a time (deselected by default; needs Azure creds):

    pytest -m dev_eval     # frequent, during development
    pytest -m held_out     # before a release

A case fails ONLY on a mechanical error (``extract_date`` raising). A wrong
prediction is recorded, not failed -- correctness is scored in the breakdown
CSV + metrics and gated against the committed baseline (see ``conftest.py``).
Only the year is scored (ground truth has no month/day).
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
from compass.utilities.costs import LLM_COST_REGISTRY, compute_cost_from_totals
from compass.services.openai import usage_from_response
from compass.services.usage import UsageTracker
from compass.services.provider import RunningAsyncServices


logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
DEV_MANIFEST_FP = (
    _DATA_DIR / "dev" / "solar_validation_files" / "manifest.json5"
)
HELD_OUT_MANIFEST_FP = (
    _DATA_DIR / "held-out" / "solar_validation_files" / "manifest.json5"
)
RESULTS_DIR = Path(__file__).parent / "results"

MODEL = "compassop-gpt-5.4"
COST_PER_MTOK = {"prompt": 1.25, "response": 7.5}  # $/M tokens, for reporting

# Populated per case; consumed by the report/gate in conftest.py.
DATE_ACCURACY_RESULTS = []


def _azure_credentials_available():
    return bool(
        os.environ.get("AZURE_OPENAI_API_KEY")
        and os.environ.get("AZURE_OPENAI_ENDPOINT")
    )


def _load_manifest(manifest_fp):
    return load_config(manifest_fp) if manifest_fp.exists() else []


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
    """Azure config for the eval model (registers its cost rate)"""
    LLM_COST_REGISTRY.setdefault(MODEL, COST_PER_MTOK)
    return OpenAIConfig(
        name=MODEL,
        llm_call_kwargs={"temperature": 1, "timeout": 300},
        client_type="azure",
        client_kwargs={
            "api_key": os.environ["AZURE_OPENAI_API_KEY"],
            "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
            "api_version": os.environ.get(
                "AZURE_OPENAI_VERSION", "2025-04-01-preview"
            ),
        },
    )


def _build_doc(case, dataset_dir):
    """Production-shaped doc with the real source URL and no pre-set date"""
    fp = dataset_dir / case["file"]
    attrs = {"source": case["source"]}
    if fp.suffix.casefold() == ".pdf":
        pages = read_pdf(fp.read_bytes(), verbose=False)
        return PDFDocument(pages, attrs=attrs)
    text = fp.read_text(encoding="utf-8", errors="ignore")
    return HTMLDocument([text], attrs=attrs)


def _classify(expected, extracted):
    """Confusion category; a WRONG year counts as both FP and FN downstream"""
    if expected is None:
        return "TN" if extracted is None else "FP"
    if extracted is None:
        return "FN"
    return "TP" if extracted == expected else "WRONG"


async def _run_case(case, dataset_dir, cadence, model_config):
    """Extract the date for one case and record the result"""
    doc = _build_doc(case, dataset_dir)
    usage_tracker = UsageTracker(case["jurisdiction"], usage_from_response)
    async with RunningAsyncServices([model_config.llm_service]):
        doc = await extract_date(
            doc, model_config, usage_tracker=usage_tracker
        )

    year, month, day = doc.attrs["date"]
    expected = case["expected_year"]
    correct = (year is None) if expected is None else (year == expected)
    cost = compute_cost_from_totals(usage_tracker.totals)

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
    # Held-out per-case detail is intentionally not logged (only summary
    # stats are surfaced) so the held-out set stays hard to tune against.
    if cadence != "held_out":
        logger.info(
            "%s (FIPS %s): expected=%s extracted=(%s,%s,%s) cost=$%.4f",
            case["jurisdiction"],
            case["fips"],
            expected,
            year,
            month,
            day,
            cost,
        )


@pytest.mark.dev_eval
@pytest.mark.parametrize(
    "case", _DEV_CASES, ids=[c["file"] for c in _DEV_CASES]
)
async def test_date_year_accuracy_dev(case, date_model_config):
    """Run date extraction on each dev-dataset document"""
    await _run_case(case, DEV_MANIFEST_FP.parent, "dev", date_model_config)


@pytest.mark.held_out
@pytest.mark.skipif(
    not HELD_OUT_MANIFEST_FP.exists(),
    reason=f"Held-out dataset not found at {HELD_OUT_MANIFEST_FP}",
)
@pytest.mark.parametrize(
    "case", _HELD_OUT_CASES, ids=[c["file"] for c in _HELD_OUT_CASES]
)
async def test_date_year_accuracy_held_out(case, date_model_config):
    """Run date extraction on each held-out document"""
    await _run_case(
        case, HELD_OUT_MANIFEST_FP.parent, "held_out", date_model_config
    )
