"""Date Extraction Evals"""

import logging
import time
from pathlib import Path

import pytest

from compass.llm.config import OpenAIConfig
from compass.extraction.apply import extract_date
from compass.utilities.io import load_config
from compass.utilities.costs import (
    LLM_COST_REGISTRY,
    compute_total_cost_and_token_from_totals,
)
from compass.services.openai import usage_from_response
from compass.services.usage import UsageTracker
from compass.services.provider import RunningAsyncServices
from compass.utilities.jurisdictions import Jurisdiction

from utilities.base import Result, classify, load_doc
from utilities.reports import report_evals


logger = logging.getLogger(__name__)

EVAL_NAME = "date_extraction"

_DATA_DIR = Path(__file__).parent / "data"
_DEV_DATASET_DIR = _DATA_DIR / "dev" / "solar"
_HELD_OUT_DATASET_DIR = _DATA_DIR / "held-out" / "solar"
RESULTS_DIR = Path(__file__).parent / "results"

RESULTS = []

# Allow up to 2 rows to flip correct -> failing before failing the gate.
# The eval can otherwise pass while two flaky cases trade places (one
# correct becomes wrong, another wrong becomes correct), keeping
# aggregate accuracy flat -- the breakdown CSV always shows the swap,
# this gate just makes a large enough swap surface as a test failure.
_DEV_ROW_REGRESSION_TOLERANCE = 2


def pytest_generate_tests(metafunc):
    """Parametrize ``case`` with the dataset chosen by ``--held-out``

    Each case gets its resolved document path stamped on as ``case["fp"]``
    so the test body doesn't need to know which dataset it came from.
    """
    if "case" not in metafunc.fixturenames:
        return
    dataset_dir = (
        _HELD_OUT_DATASET_DIR
        if metafunc.config.getoption("--held-out")
        else _DEV_DATASET_DIR
    )
    cases = load_config(dataset_dir / "manifest.json5")
    for c in cases:
        c["fp"] = dataset_dir / c["file"]
    metafunc.parametrize("case", cases, ids=[c["file"] for c in cases])


@pytest.fixture(scope="module", autouse=True)
def _report(request):
    """Write CSVs/JSON, print summary, and (dev only) enforce the gate"""
    yield
    held_out = request.config.getoption("--held-out")
    eval_subdir = "held_out" if held_out else "dev"
    data = report_evals(
        request,
        EVAL_NAME,
        RESULTS,
        RESULTS_DIR / eval_subdir,
        write_breakdown=not held_out,
    )
    if not data or held_out:
        return  # held-out runs are unbiased reads, not gates

    failures = []
    base = data["baseline_failing"]
    if base is not None and data["fails_now"] > base:
        failures.append(
            f"aggregate regression: {data['fails_now']} failing "
            f"> baseline {base}"
        )
    regressed = data["regressed_rows"]
    if regressed and len(regressed) > _DEV_ROW_REGRESSION_TOLERANCE:
        failures.append(
            f"{len(regressed)} rows regressed "
            f"(tol {_DEV_ROW_REGRESSION_TOLERANCE}): {regressed}"
        )
    if failures:
        pytest.fail("Eval regression gate:\n  " + "\n  ".join(failures))


@pytest.fixture(scope="module")
def _model_config():
    model = "compassop-gpt-5.4"
    LLM_COST_REGISTRY.setdefault(
        model, {"prompt": 1.25, "response": 7.5}  # $/M tokens
    )
    return OpenAIConfig(name=model)


async def _run_case(case, model_config, *, log_detail):
    """Extract the date for one case and record the result"""
    label = Jurisdiction(
        subdivision_type=case["jurisdiction_type"],
        state=case["state"],
        county=case["county"],
        subdivision_name=case["subdivision"],
    ).full_name
    doc = load_doc(case["fp"], source=case["source"])
    usage_tracker = UsageTracker(label, usage_from_response)
    start = time.perf_counter()
    async with RunningAsyncServices([model_config.llm_service]):
        doc = await extract_date(
            doc, model_config, usage_tracker=usage_tracker
        )
    elapsed = time.perf_counter() - start

    year, _month, _day = doc.attrs["date"]
    expected = case["expected"]["year"]
    usage = compute_total_cost_and_token_from_totals(usage_tracker.totals)

    RESULTS.append(
        Result(
            state=case["state"],
            county=case["county"],
            subdivision=case["subdivision"],
            jurisdiction_type=case["jurisdiction_type"],
            file=case["file"],
            source=case["source"],
            feature="year",
            expected=expected,
            extracted=year,
            comparison_result=classify(expected, year),
            time_taken_s=round(elapsed, 3),
            **usage,
        )
    )
    if log_detail:
        logger.info(
            "%s: expected=%s extracted=%s cost=$%.4f",
            label,
            expected,
            year,
            usage["cost"],
        )


@pytest.mark.evals
async def test_date_year_accuracy(case, _model_config, request):
    """Run date extraction on each document in the active dataset"""
    held_out = request.config.getoption("--held-out")
    # held-out per-case detail hidden to prevent tuning against it
    await _run_case(case, _model_config, log_detail=not held_out)
