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
from compass.services.usage import LLMUsageTracker
from compass.services.provider import RunningAsyncServices
from compass.services.cpu import (
    FileLoader,
    OCRPDFLoader,
    read_pdf_file,
    read_pdf_file_ocr,
)
from compass.services.threaded import HTMLFileLoader, read_html_file
from compass.pipeline.data_classes import build_models
from compass.utilities.jurisdictions import Jurisdiction
from compass.utilities.logs import (
    LocationFileLog,
    LogListener,
    setup_logging_levels,
)
from compass.web.file_loader import COMPASSLocalFileLoader

from utilities import Result, classify, report_evals, PerJurisdictionResults


logger = logging.getLogger(__name__)

EVAL_NAME = "date_extraction"

_DATA_DIR = Path(__file__).parent / "data"
_DEV_DATASET_DIR = _DATA_DIR / "dev" / "solar"
_HELD_OUT_DATASET_DIR = _DATA_DIR / "held-out" / "solar"
RESULTS_DIR = Path(__file__).parent / "results"


def results_dir(held_out):
    """Top-level results directory for the active dataset"""
    return RESULTS_DIR / ("held_out" if held_out else "dev")


def per_jurisdiction_results(held_out):
    """Sharded per-jurisdiction result store for the active dataset"""
    return PerJurisdictionResults(results_dir(held_out) / "per_jurisdiction")


def _logs_dir(held_out):
    return results_dir(held_out) / "logs"


def clear_logs(held_out):
    """Delete stale logs before a run

    Logs are opened in append mode and re-read for explanations, so stale
    files would mix old/new records and confuse the explanation lookup.
    """
    out_dir = _logs_dir(held_out)
    if not out_dir.exists():
        return
    for log_fp in out_dir.iterdir():
        if log_fp.is_file():
            log_fp.unlink()


# DateExtractor logs the LLM's justification but drops it from its return
# value; we recover it from the DEBUG log rather than touch production
# date.py. A missing line yields ``None``.
_EXPLANATION_MARKER = "Date extraction explanation: "


def _read_explanation_from_log(log_dir, label):
    """Return the date explanation from a jurisdiction's log, or ``None``"""
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


def _setup_pytesseract(exe_fp):
    import pytesseract  # ruff:ignore[import-outside-top-level]

    pytesseract.pytesseract.tesseract_cmd = exe_fp


def build_local_file_loader_kwargs(
    pytesseract_exe_fp=None, pdf_read_kwargs=None, html_read_kwargs=None
):
    """Build kwargs for ``COMPASSLocalFileLoader``

    Intentionally duplicates
    ``PipelineRuntime.local_file_loader_kwargs`` to keep the eval
    decoupled from the production runtime.
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
    # Only enacted (Final) ordinances have an adoption date to extract.
    cases = [
        c for c in cases if c["document_satus"].strip().lower() == "final"
    ]
    metafunc.parametrize(
        "case",
        [(case, dataset_dir) for case in cases],
        ids=[case.get("file", f"case_{i}") for i, case in enumerate(cases)],
        indirect=True,
    )


@pytest.fixture
def case(request):
    """Resolve one parametrized (raw_case, dataset_dir) into a case dict"""
    case, dataset_dir = request.param
    case["fp"] = dataset_dir / case["file"]
    case["county"] = case["county"] or None
    case["subdivision"] = case["subdivision"] or None
    return case


def _write_breakdown_json(results, held_out):
    """Write a JSON twin of the breakdown CSV, enriched with explanations"""
    log_dir = _logs_dir(held_out)
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

    out_fp = results_dir(held_out) / f"{EVAL_NAME}_evals_breakdown.json"
    with out_fp.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def report(request, results, held_out):
    """Write the metrics JSON, print the summary, write the dev breakdown

    Runs on the controller in ``pytest_sessionfinish``. The explanation-rich
    breakdown JSON is dev only -- held-out hides per-case detail.
    """
    report_evals(request, EVAL_NAME, results, results_dir(held_out))
    if not held_out:
        _write_breakdown_json(results, held_out)


@pytest.fixture(scope="module")
def _model_config():
    config = load_config(Path(__file__).parent / "config.json5")
    LLM_COST_REGISTRY.update(config.get("llm_costs") or {})
    return build_models(config["model"])[LLMTasks.DEFAULT]


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
    """Extract the date for one case and write its ``Result``"""
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
    usage_tracker = LLMUsageTracker(label, usage_from_response)

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
        # Task name routes COMPASS's location-aware logs to this case's file.
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
    per_jurisdiction_results(held_out).write(result, label)
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
        _logs_dir(held_out),
        held_out=held_out,
        log_detail=not held_out,
    )
