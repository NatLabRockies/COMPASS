"""Date Extraction Evals"""

import asyncio
import json
import logging
import time
from dataclasses import asdict
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

from utilities import Result, classify, report_evals, gate_failures


logger = logging.getLogger(__name__)

EVAL_NAME = "date_extraction"

_DATA_DIR = Path(__file__).parent / "data"
_DEV_DATASET_DIR = _DATA_DIR / "dev" / "solar"
_HELD_OUT_DATASET_DIR = _DATA_DIR / "held-out" / "solar"
RESULTS_DIR = Path(__file__).parent / "results"

# Each case writes its own result to ``per_jurisdiction/<label>.json``,
# a first-class run artifact that mirrors the per-jurisdiction ``logs/``
# directory (same ``<jurisdiction full name>`` stem). This makes the
# eval work under pytest-xdist: each xdist worker is a separate process
# with its own module globals, so a module-level results list would be
# invisible to the controller that runs the report hook. Writing one
# file per case sidesteps that -- every case owns a unique path (no
# cross-process write races, idempotent on rerun), and the controller's
# ``pytest_sessionfinish`` hook (in conftest.py) globs the directory to
# rebuild the full results list. See ``per_jurisdiction_dir`` /
# ``write_result`` / ``load_results`` below.
_PER_JURISDICTION_SUBDIR = "per_jurisdiction"


def _dataset_results_dir(held_out):
    """Top-level results directory for the active dataset"""
    return RESULTS_DIR / ("held_out" if held_out else "dev")


def per_jurisdiction_dir(held_out):
    """Directory holding one result file per jurisdiction for a dataset"""
    return _dataset_results_dir(held_out) / _PER_JURISDICTION_SUBDIR


def logs_dir(held_out):
    """Directory holding one ``<jurisdiction>.log`` per case"""
    return _dataset_results_dir(held_out) / "logs"


def write_result(result, label, held_out):
    """Write one case's ``Result`` to ``per_jurisdiction/<label>.json``

    ``label`` is the jurisdiction full name (same stem used for the
    per-jurisdiction log file). One file per case means concurrent
    writes from different xdist workers never collide, and a rerun of a
    single case overwrites just its own file.

    This file is a pure ``Result`` carrier across the xdist process
    boundary; the LLM's date ``explanation`` is not stored here but is
    recovered at assembly time from the per-jurisdiction log (see
    ``explanations_by_label``).
    """
    out_dir = per_jurisdiction_dir(held_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_fp = out_dir / f"{label}.json"
    with result_fp.open("w", encoding="utf-8") as fh:
        json.dump(asdict(result), fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def load_results(held_out):
    """Read every per-jurisdiction result file into a list of ``Result``

    Called from the controller-side ``pytest_sessionfinish`` hook after
    all workers finish. Returns ``[]`` if nothing was written (e.g. zero
    cases selected).
    """
    out_dir = per_jurisdiction_dir(held_out)
    if not out_dir.exists():
        return []
    results = []
    for result_fp in sorted(out_dir.glob("*.json")):
        with result_fp.open(encoding="utf-8") as fh:
            results.append(Result(**json.load(fh)))
    return results


def clear_results(held_out):
    """Delete stale per-jurisdiction files before a new run starts

    Ensures a run reflects exactly the current dataset -- files for
    jurisdictions no longer in the manifest don't leak into aggregation.
    """
    out_dir = per_jurisdiction_dir(held_out)
    if not out_dir.exists():
        return
    for result_fp in out_dir.glob("*.json"):
        result_fp.unlink()


def clear_logs(held_out):
    """Delete stale per-jurisdiction logs before a new run starts

    ``LocationFileLog`` opens log files in append mode, so without this a
    re-run would mix old and new records -- and logs for jurisdictions no
    longer in the run would linger and confuse the controller's
    explanation read. Clears every file (``<jur>.log`` and
    ``<jur> exceptions.json``) so the logs dir reflects only this run.
    """
    out_dir = logs_dir(held_out)
    if not out_dir.exists():
        return
    for log_fp in out_dir.iterdir():
        if log_fp.is_file():
            log_fp.unlink()


# DateExtractor logs the LLM's justification at DEBUG as
# "Date extraction explanation: <text>" but discards it from its return
# value. Rather than modify production date.py, the controller recovers
# it from each jurisdiction's DEBUG log at assembly time (dev only --
# held-out hides per-case detail). Brittle by design (depends on the log
# message format); a missing line yields ``None``.
_EXPLANATION_MARKER = "Date extraction explanation: "


def _read_explanation_from_log(log_dir, label):
    """Return the date explanation from a jurisdiction's log, or ``None``

    Reads ``<log_dir>/<label>.log`` (written by ``LocationFileLog``) and
    returns the text following the last ``_EXPLANATION_MARKER`` line.
    Called on the controller in ``pytest_sessionfinish``, well after all
    log handlers have flushed.
    """
    log_fp = Path(log_dir) / f"{label}.log"
    if not log_fp.exists():
        return None
    explanation = None
    with log_fp.open(encoding="utf-8") as fh:
        for line in fh:
            idx = line.find(_EXPLANATION_MARKER)
            if idx != -1:
                explanation = line[idx + len(_EXPLANATION_MARKER) :].strip()
    return explanation


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
    # Date extraction is only meaningful for enacted (Final) ordinances
    # since drafts/proposals have no "adoption date"
    cases = [c for c in cases if c["document_satus"].strip().lower() == "final"]
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


def _write_breakdown_json(results, held_out):
    """Write a JSON twin of the breakdown CSV, enriched with explanations

    Mirrors ``date_extraction_evals_breakdown.csv`` row-for-row (same
    state/county/subdivision sort order) but as JSON, and adds the LLM's
    free-text ``explanation`` per row -- recovered from each
    jurisdiction's log on the controller. Dev only: held-out hides
    per-case detail, so this is never written for held-out runs.
    """
    log_dir = logs_dir(held_out)
    ordered = sorted(
        results, key=lambda r: (r.state, r.county or "", r.subdivision or "")
    )
    rows = []
    for r in ordered:
        row = asdict(r)
        row["explanation"] = _read_explanation_from_log(
            log_dir, r.jurisdiction.full_name
        )
        rows.append(row)

    out_fp = (
        _dataset_results_dir(held_out)
        / f"{EVAL_NAME}_evals_breakdown.json"
    )
    with out_fp.open("w", encoding="utf-8") as fh:
        # ensure_ascii=False keeps explanation punctuation/accents readable
        json.dump(rows, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def report_and_gate(request, results, held_out):
    """Write CSVs/JSON, print the summary, and enforce the regression gate

    Pulled out of the old module-scoped ``_report`` fixture so it can run
    from the controller-side ``pytest_sessionfinish`` hook (which sees the
    aggregated results from every xdist worker) instead of inside a worker
    process. Returns the list of gate-failure messages (empty if it
    passed); the caller decides how to surface them.
    """
    eval_subdir = "held_out" if held_out else "dev"
    evals_data = report_evals(
        request,
        EVAL_NAME,
        results,
        RESULTS_DIR / eval_subdir,
        write_breakdown=not held_out,
    )
    if held_out:
        return []  # held-out evals aren't compared against stored values

    _write_breakdown_json(results, held_out)
    return gate_failures(evals_data, regression_tol=REGRESSION_TOL)


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


async def _run_case(
    case,
    model_config,
    log_listener,
    log_dir,
    *,
    held_out,
    log_detail,
):
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

    result = Result(
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
    write_result(result, label, held_out)
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
    # held-out per-case detail hidden to prevent tuning against it
    await _run_case(
        case,
        _model_config,
        _log_listener,
        logs_dir(held_out),
        held_out=held_out,
        log_detail=not held_out,
    )
