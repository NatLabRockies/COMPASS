"""Date Extraction Evals"""

import os
import logging
import time
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
DEV_MANIFEST_FP = _DATA_DIR / "dev" / "solar" / "manifest.json5"
HELD_OUT_MANIFEST_FP = _DATA_DIR / "held-out" / "solar" / "manifest.json5"
RESULTS_DIR = Path(__file__).parent / "results"

RESULTS = {"dev": [], "held_out": []}


_DEV_CASES = load_config(DEV_MANIFEST_FP)
_HELD_OUT_CASES = load_config(HELD_OUT_MANIFEST_FP)


@pytest.fixture(scope="module")
def date_model_config():
    """Azure config for the eval model (registers its cost rate)"""
    model = "compassop-gpt-5.4"
    LLM_COST_REGISTRY.setdefault(
        model, {"prompt": 1.25, "response": 7.5}  # $/M tokens
    )
    return OpenAIConfig(
        name=model,
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
    fp = dataset_dir / case["file"]
    attrs = {"source": case["source"]}
    if fp.suffix.casefold() == ".pdf":
        pages = read_pdf(fp.read_bytes(), verbose=False)
        return PDFDocument(pages, attrs=attrs)
    text = fp.read_text(encoding="utf-8", errors="ignore")
    return HTMLDocument([text], attrs=attrs)


def _classify(expected, extracted):
    """Binary success: did the extractor match ground truth?"""
    return "Success" if extracted == expected else "Failure"


async def _run_case(case, dataset_dir, eval_type, model_config):
    """Extract the date for one case and record the result"""
    doc = _build_doc(case, dataset_dir)
    usage_tracker = UsageTracker(case["jurisdiction"], usage_from_response)
    start = time.perf_counter()
    async with RunningAsyncServices([model_config.llm_service]):
        doc = await extract_date(
            doc, model_config, usage_tracker=usage_tracker
        )
    elapsed = time.perf_counter() - start

    year, _month, _day = doc.attrs["date"]
    expected = case["expected"]["year"]
    totals = usage_tracker.totals
    cost = compute_cost_from_totals(totals)
    input_tokens = sum(u.get("prompt_tokens", 0) for u in totals.values())
    output_tokens = sum(u.get("response_tokens", 0) for u in totals.values())

    RESULTS[eval_type].append(
        {
            "fips": case["fips"],
            "jurisdiction": case["jurisdiction"],
            "file": case["file"],
            "source": case["source"],
            "feature": "year",
            "expected": expected,
            "extracted": year,
            "comparison_result": _classify(expected, year),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "time_taken_s": round(elapsed, 3),
            "cost": cost,
        }
    )
    # Held-out per-case detail is intentionally not logged (only summary
    # stats are surfaced) so the held-out set stays hard to tune against.
    if eval_type != "held_out":
        logger.info(
            "%s (FIPS %s): expected=%s extracted=%s cost=$%.4f",
            case["jurisdiction"],
            case["fips"],
            expected,
            year,
            cost,
        )


@pytest.mark.dev_evals
@pytest.mark.parametrize(
    "case", _DEV_CASES, ids=[c["file"] for c in _DEV_CASES]
)
async def test_date_year_accuracy_dev(case, date_model_config):
    """Run date extraction on each dev-dataset document"""
    await _run_case(
        case, DEV_MANIFEST_FP.parent, "dev", date_model_config
    )


@pytest.mark.held_out_evals
@pytest.mark.parametrize(
    "case", _HELD_OUT_CASES, ids=[c["file"] for c in _HELD_OUT_CASES]
)
async def test_date_year_accuracy_held_out(case, date_model_config):
    """Run date extraction on each held-out document"""
    await _run_case(
        case, HELD_OUT_MANIFEST_FP.parent, "held_out", date_model_config
    )
