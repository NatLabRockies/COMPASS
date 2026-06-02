"""Date Extraction Evals"""

import asyncio
import logging
import time
from pathlib import Path

import pytest

from compass.extraction.apply import extract_date
from compass.utilities.io import load_config
from compass.utilities.costs import (
    LLM_COST_REGISTRY,
    compute_total_cost_and_token_from_totals,
)
from compass.utilities.enums import LLMTasks
from compass.services.openai import usage_from_response
from compass.services.usage import UsageTracker
from compass.services.provider import RunningAsyncServices
from compass.services.cpu import (
    FileLoader,
    OCRPDFLoader,
    read_pdf_file,
    read_pdf_file_ocr,
)
from compass.services.threaded import HTMLFileLoader, read_html_file
from compass.pipeline.data_classes import _build_models
from compass.utilities.jurisdictions import Jurisdiction
from compass.utilities.logs import (
    LocationFileLog,
    LogListener,
    setup_logging_levels,
)
from compass.web.file_loader import COMPASSLocalFileLoader

from utilities import Result, classify, report_evals


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
REGRESSION_TOL = 2


def _setup_pytesseract(exe_fp):
    """Set the pytesseract command"""
    import pytesseract  # noqa: PLC0415

    pytesseract.pytesseract.tesseract_cmd = exe_fp


def build_local_file_loader_kwargs(
    pytesseract_exe_fp=None, pdf_read_kwargs=None, html_read_kwargs=None
):
    """Build keyword arguments for ``COMPASSLocalFileLoader``

    Mirrors the file-loader-kwargs logic that lives in production at
    ``compass.pipeline.runtime.PipelineRuntime.local_file_loader_kwargs``
    (a cached_property on the runtime object). Inlined here so the eval
    stays decoupled from that runtime; the duplication is intentional
    until evals migrate to the production call path.
    """
    file_loader_kwargs = {
        "pdf_read_coroutine": read_pdf_file,
        "html_read_coroutine": read_html_file,
        "pdf_read_kwargs": pdf_read_kwargs,
        "html_read_kwargs": html_read_kwargs,
    }
    if pytesseract_exe_fp is not None:
        _setup_pytesseract(pytesseract_exe_fp)
        file_loader_kwargs.update(
            {
                "pdf_ocr_read_coroutine": read_pdf_file_ocr,
                "pytesseract_exe_fp": pytesseract_exe_fp,
            }
        )
    return file_loader_kwargs


def pytest_generate_tests(metafunc):
    """Generate evals cases with the dataset chosen by ``--held-out``"""
    if "case" not in metafunc.fixturenames:
        return
    dataset_dir = (
        _HELD_OUT_DATASET_DIR
        if metafunc.config.getoption("--held-out")
        else _DEV_DATASET_DIR
    )
    cases = load_config(dataset_dir / "manifest.json5")
    # Date extraction is only meaningful for enacted (Final) ordinances;
    # drafts/proposals have no adoption date by definition. Missing
    # document_type defaults to Final so legacy manifests aren't excluded.
    cases = [c for c in cases if c.get("document_type", "Final") == "Final"]
    metafunc.parametrize(
        "case",
        [(case, dataset_dir) for case in cases],
        ids=[case.get("file", f"case_{i}") for i, case in enumerate(cases)],
        indirect=True,  # send to the case fixture instead of test function
    )


@pytest.fixture
def case(request):
    """Receives the (raw_case, dataset_dir) from pytest_generate_tests."""
    case, dataset_dir = request.param
    case["fp"] = dataset_dir / case["file"]
    case["county"] = case["county"] or None
    case["subdivision"] = case["subdivision"] or None
    return case


@pytest.fixture(scope="module", autouse=True)
def _report(request):
    """Write CSVs/JSON, print summary, and (dev evals only) enforce the gate"""
    yield  # The code below runs after the test module finishes
    held_out = request.config.getoption("--held-out")
    eval_subdir = "held_out" if held_out else "dev"
    evals_data = report_evals(
        request,
        EVAL_NAME,
        RESULTS,
        RESULTS_DIR / eval_subdir,
        write_breakdown=not held_out,
    )
    if not evals_data or held_out:
        return  # held-out evals aren't compared against stored values

    failures = []
    if (
        evals_data["prev_nfail"] is not None
        and evals_data["current_nfail"] > evals_data["prev_nfail"]
    ):
        failures.append(
            f"aggregate regression: {evals_data['current_nfail']} failing"
            f" > previous {evals_data['prev_nfail']}"
        )
    if (
        evals_data["regressed_jurs"]
        and len(evals_data["regressed_jurs"]) > REGRESSION_TOL
    ):
        failures.append(
            f"{len(evals_data['regressed_jurs'])} rows regressed "
            f"(tol {REGRESSION_TOL}): {evals_data['regressed_jurs']}"
        )
    if failures:
        pytest.fail("Eval regression gate:\n  " + "\n  ".join(failures))


@pytest.fixture(scope="module")
def _model_config():
    config = load_config(Path(__file__).parent / "config.json5")
    LLM_COST_REGISTRY.update(config.get("llm_costs") or {})
    return _build_models(config["model"])[LLMTasks.DEFAULT]


@pytest.fixture(scope="session")
def _log_listener():
    """Run a COMPASS LogListener so per-case logs can be captured"""
    setup_logging_levels()
    with LogListener(["compass", "elm"], level="DEBUG") as listener:
        yield listener


async def _run_case(case, model_config, log_listener, log_dir, *, log_detail):
    """Extract the date for one case and record the result

    Each case's ``compass``/``elm`` logs (production detail) are written
    to ``<log_dir>/<jurisdiction>.log`` via :class:`LocationFileLog`.
    """
    label = Jurisdiction(
        subdivision_type=case["jurisdiction_type"],
        state=case["state"],
        county=case["county"],
        subdivision_name=case["subdivision"],
    ).full_name
    loader = COMPASSLocalFileLoader(
        **build_local_file_loader_kwargs(pytesseract_exe_fp="tesseract"),
        doc_attrs={"source": case["source"]},
    )
    usage_tracker = UsageTracker(label, usage_from_response)

    async def _load_and_extract():
        doc = await loader.fetch(case["fp"])
        return await extract_date(
            doc, model_config, usage_tracker=usage_tracker
        )

    start = time.perf_counter()
    async with (
        LocationFileLog(log_listener, log_dir, location=label, level="DEBUG"),
        RunningAsyncServices(
            [
                model_config.llm_service,
                FileLoader(),
                HTMLFileLoader(),
                OCRPDFLoader(max_workers=1),  # pytesseract locks up w/ >1 proc
            ]
        ),
    ):
        # Run in a task named after the jurisdiction so COMPASS's
        # location-aware logging routes records to this case's log file.
        doc = await asyncio.create_task(_load_and_extract(), name=label)
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
async def test_date_year_accuracy(case, _model_config, _log_listener, request):
    """Run date extraction on each document in the active dataset"""
    held_out = request.config.getoption("--held-out")
    eval_subdir = "held_out" if held_out else "dev"
    log_dir = RESULTS_DIR / eval_subdir / "logs"
    # held-out per-case detail hidden to prevent tuning against it
    await _run_case(
        case,
        _model_config,
        _log_listener,
        log_dir,
        log_detail=not held_out,
    )
