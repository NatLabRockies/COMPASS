"""Update jurisdiction website URLs in a CSV file

This support script validates any existing jurisdiction website values
and searches for replacements when a value is missing or invalid. It
uses the same COMPASS website-search and validation primitives as the
pipeline, but wires the required model configs, semaphores, and usage
tracking directly instead of relying on a workflow object.

Example:
pixi run python update_jur_websites.py jurisdictions.csv config.json5
--start-index 100 --end-index 249
"""

import argparse
import asyncio
import contextlib
import logging
from pathlib import Path
from datetime import datetime, UTC

import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.theme import Theme
from elm.web.utilities import get_redirected_url

from compass.exceptions import COMPASSValueError
from compass.llm.config import OpenAIConfig
from compass.pb import COMPASS_PB
from compass.pipeline.data_classes import WebSearchParams
from compass.pipeline.runtime import MAX_CONCURRENT_SEARCH_ENGINE_QUERIES
from compass.scripts.download import find_jurisdiction_website
from compass.services.cpu import FileLoader
from compass.services.openai import usage_from_response
from compass.services.provider import RunningAsyncServices
from compass.services.usage import LLMUsageTracker
from compass.utilities.costs import (
    compute_cost_from_totals,
    compute_total_cost_from_usage,
)
from compass.utilities.finalize import _elapsed_time_as_str  # ruff:ignore[import-private-name]
from compass.utilities.enums import LLMTasks
from compass.utilities.io import load_config
from compass.utilities.jurisdictions import Jurisdiction
from compass.utilities.logs import AddLocationFilter, log_versions
from compass.utilities.url import base_website_url
from compass.web.file_loader import COMPASSWebFileLoader
from compass.validation.location import JurisdictionWebsiteValidator


logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = ("State",)
OPTIONAL_COLUMNS = (
    "County",
    "Subdivision",
    "Jurisdiction Type",
    "FIPS",
    "Website",
)


def _parse_args():
    """Parse command-line inputs"""
    parser = argparse.ArgumentParser(
        description=(
            "Validate and update jurisdiction websites for every row "
            "in a CSV file."
        )
    )
    parser.add_argument(
        "jurisdiction_csv",
        help="Path to the jurisdiction CSV to update.",
    )
    parser.add_argument(
        "config",
        help=("Path to a JSON/JSON5/YAML/TOML config."),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        help=(
            "Optional starting row index to process. The value is "
            "inclusive. By default, processing starts at row 0."
        ),
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help=(
            "Optional ending row index to process. The value is "
            "inclusive. By default, processing continues through the "
            "last row."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        help="Logging level for terminal output. By default, DEBUG.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase terminal logging verbosity.",
    )
    parser.add_argument(
        "-np",
        "--no-progress",
        action="store_true",
        help="Flag to hide progress bars during processing.",
    )
    return parser.parse_args()


def _resolve_log_path(config_fp):
    """Resolve the log path beside the runtime config file"""
    config_fp = Path(config_fp).expanduser().resolve()
    return config_fp.with_name(f"{config_fp.stem}_update_jur_websites.log")


def _setup_terminal_logging(
    console,
    verbosity_level,
    log_level,
    log_fp,
):
    """Set up terminal logging with COMPASS location tagging"""
    logger_names = [__name__, "compass"]
    if verbosity_level >= 2:  # ruff:ignore[magic-value-comparison]
        logger_names.extend(("elm", "docling"))
    if verbosity_level >= 3:  # ruff:ignore[magic-value-comparison]
        logger_names.append("openai")
    if verbosity_level >= 4:  # ruff:ignore[magic-value-comparison]
        logger_names.extend(
            ("networkx", "pytesseract", "pdf2image", "pdftotext")
        )

    log_fp = Path(log_fp).expanduser().resolve()
    log_fp.parent.mkdir(parents=True, exist_ok=True)

    for logger_name in dict.fromkeys(logger_names):
        target_logger = logging.getLogger(logger_name)
        terminal_handler = RichHandler(
            level=log_level,
            console=console,
            rich_tracebacks=True,
            omit_repeated_times=True,
            markup=True,
        )
        terminal_fmt = logging.Formatter(
            fmt="[[magenta]%(location)s[/magenta]]: %(message)s",
            defaults={"location": "main"},
        )
        terminal_handler.setFormatter(terminal_fmt)
        terminal_handler.addFilter(AddLocationFilter())
        target_logger.addHandler(terminal_handler)

        file_handler = logging.FileHandler(log_fp, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_fmt = logging.Formatter(
            fmt=("[%(asctime)s] [%(location)s] %(levelname)s: %(message)s"),
            defaults={"location": "main"},
        )
        file_handler.setFormatter(file_fmt)
        file_handler.addFilter(AddLocationFilter())
        target_logger.addHandler(file_handler)
        target_logger.setLevel(log_level)


def _resolve_output_path(input_fp):
    """Resolve the output path for the updated CSV"""
    input_fp = Path(input_fp).expanduser().resolve()
    return input_fp.with_name(f"{input_fp.stem}_updated.csv")


def _build_models(user_input):
    """Build configured model registry"""
    if isinstance(user_input, str):
        return {LLMTasks.DEFAULT: OpenAIConfig(name=user_input)}

    caller_instances = {}
    for raw_kwargs in user_input:
        kwargs = dict(raw_kwargs)
        tasks = kwargs.pop("tasks", LLMTasks.DEFAULT)
        if isinstance(tasks, str):
            tasks = [tasks]

        model_config = OpenAIConfig(**kwargs)
        for task_name in tasks:
            task = LLMTasks(task_name)
            if task in caller_instances:
                msg = (
                    f"Found duplicated task: {task!r}. Please ensure "
                    "each LLM caller definition has uniquely-assigned "
                    "tasks."
                )
                raise COMPASSValueError(msg)
            caller_instances[task] = model_config

    if LLMTasks.DEFAULT not in caller_instances:
        msg = (
            "No 'default' LLM caller defined in the model config. "
            "Please ensure exactly one model definition has 'tasks' "
            "set to 'default' or left unspecified."
        )
        raise COMPASSValueError(msg)

    return caller_instances


def _load_jurisdiction_csv(jurisdiction_csv):
    """Read the jurisdiction CSV while preserving text identifiers"""
    df = pd.read_csv(jurisdiction_csv, dtype=str).where(
        lambda df: df.notna(),
        None,
    )
    if df.empty:
        msg = "The jurisdiction CSV is empty"
        raise ValueError(msg)
    return df


def _slice_jurisdiction_df(df, start_index=None, end_index=None):
    """Slice the jurisdiction DataFrame by inclusive index bounds"""
    if start_index is None and end_index is None:
        return df

    max_index = len(df) - 1
    if max_index < 0:
        return df

    start_index = 0 if start_index is None else start_index
    end_index = max_index if end_index is None else end_index

    if start_index < 0 or end_index < 0:
        msg = "`start_index` and `end_index` must be non-negative"
        raise COMPASSValueError(msg)

    if start_index > end_index:
        msg = "`start_index` cannot be greater than `end_index`"
        raise COMPASSValueError(msg)

    if start_index > max_index:
        msg = (
            f"`start_index` ({start_index}) is outside the CSV row "
            f"range 0-{max_index}"
        )
        raise COMPASSValueError(msg)

    if end_index > max_index:
        msg = (
            f"`end_index` ({end_index}) is outside the CSV row range "
            f"0-{max_index}"
        )
        raise COMPASSValueError(msg)

    return df.iloc[start_index : end_index + 1].copy()


def _load_runtime_inputs(config_fp):
    """Load config-driven runtime inputs for the website update"""
    raw_config = load_config(config_fp, resolve_paths=False)

    model_configs = _build_models(raw_config["model"])

    search_params = WebSearchParams(
        search_engines=raw_config["search_engines"],
        url_ignore_substrings=[".k12.", ".edu/"],
    )

    file_loader_kwargs = raw_config.get("file_loader_kwargs") or {}
    file_loader_kwargs.pop("pdf_ocr_read_coroutine", None)

    max_num_concurrent_jurisdictions = raw_config.get(
        "max_num_concurrent_jurisdictions"
    )
    ppe_kwargs = raw_config.get("ppe_kwargs") or {}
    return (
        model_configs,
        search_params,
        file_loader_kwargs,
        max_num_concurrent_jurisdictions,
        ppe_kwargs,
    )


def _prepare_dataframe_inputs(jurisdiction_csv, start_index, end_index):
    """Load the full CSV and derive the processed row window"""
    df = _load_jurisdiction_csv(jurisdiction_csv)
    processing_df = _slice_jurisdiction_df(
        df,
        start_index=start_index,
        end_index=end_index,
    )
    col_map = {
        column: _find_column(df, column, required=column in REQUIRED_COLUMNS)
        for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS
    }

    website_col = col_map["Website"] or "Website"
    if col_map["Website"] is None:
        df[website_col] = None
        col_map["Website"] = website_col

    return df, processing_df, col_map, website_col


def _log_run_scope(df, processing_df, output_fp):
    """Log the selected processing window for the run"""
    selected_start = processing_df.index[0]
    selected_end = processing_df.index[-1]

    log_versions(logging.getLogger("compass"))
    logger.info("Loaded %d jurisdiction row(s)", len(df))
    logger.info(
        "Processing %d jurisdiction row(s) from index %d to %d",
        len(processing_df),
        selected_start,
        selected_end,
    )
    logger.info("Output CSV: %s", output_fp)
    return selected_start, selected_end


def _find_column(df, target, *, required=False):
    """Find a column in a DataFrame using case-insensitive matching"""
    target_cf = target.casefold()
    for col in df.columns:
        if str(col).casefold() == target_cf:
            return col

    if required:
        msg = f"Missing required column: {target!r}"
        raise ValueError(msg)
    return None


def _normalize_cell(value):
    """Normalize a CSV cell into a stripped string or None"""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return value


def _infer_jurisdiction_type(row, col_map):
    """Infer a jurisdiction type when the column is not present"""
    subdivision = _normalize_cell(row.get(col_map.get("Subdivision")))
    county = _normalize_cell(row.get(col_map.get("County")))
    if subdivision:
        msg = (
            "Rows with a subdivision must include a 'Jurisdiction Type' "
            "column."
        )
        raise ValueError(msg)
    if county:
        return "county"
    return "state"


def _build_jurisdiction(row, col_map):
    """Build a Jurisdiction instance from one CSV row"""
    jur_type_col = col_map.get("Jurisdiction Type")
    jur_type = _normalize_cell(row.get(jur_type_col))
    if jur_type is None:
        jur_type = _infer_jurisdiction_type(row, col_map)

    return Jurisdiction(
        subdivision_type=jur_type,
        state=_normalize_cell(row[col_map["State"]]),
        county=_normalize_cell(row.get(col_map.get("County"))),
        subdivision_name=_normalize_cell(row.get(col_map.get("Subdivision"))),
        code=_normalize_cell(row.get(col_map.get("FIPS"))),
        website_url=_normalize_cell(row.get(col_map.get("Website"))),
    )


def _task_label(jurisdiction, row_index):
    """Build a unique progress label for one CSV row"""
    return f"{jurisdiction.full_name} [{row_index + 1}]"


async def _validate_jurisdiction_website(
    jurisdiction,
    jurisdiction_website,
    model_configs,
    browser_semaphore,
    file_loader_kwargs,
    usage_tracker,
    location_label,
):
    """Validate a user-supplied jurisdiction website"""
    if jurisdiction_website is None:
        return jurisdiction_website

    try:
        jurisdiction_website = await get_redirected_url(
            jurisdiction_website,
            timeout=30,
        )
        jurisdiction_website = base_website_url(jurisdiction_website)
    except Exception:
        logger.exception(
            "Error redirect-checking website for %s",
            jurisdiction.full_name,
        )
        return None

    COMPASS_PB.update_jurisdiction_task(
        location_label,
        description=(f"Validating user input website: {jurisdiction_website}"),
    )
    model_config = model_configs.get(
        LLMTasks.DOCUMENT_JURISDICTION_VALIDATION,
        model_configs[LLMTasks.DEFAULT],
    )
    validator = JurisdictionWebsiteValidator(
        browser_semaphore=browser_semaphore,
        file_loader_kwargs=file_loader_kwargs,
        usage_tracker=usage_tracker,
        llm_service=model_config.llm_service,
        **model_config.llm_call_kwargs,
    )
    is_website_correct = await validator.check(
        jurisdiction_website,
        jurisdiction,
    )
    if not is_website_correct:
        return None

    return jurisdiction_website


async def _find_jurisdiction_website_for_jurisdiction(
    jurisdiction,
    model_configs,
    file_loader_kwargs,
    search_semaphore,
    browser_semaphore,
    usage_tracker,
    search_params,
    location_label,
):
    """Search for the main jurisdiction website"""
    COMPASS_PB.update_jurisdiction_task(
        location_label,
        description="Searching for jurisdiction website...",
    )
    return await find_jurisdiction_website(
        jurisdiction,
        model_configs,
        file_loader_kwargs=file_loader_kwargs,
        search_semaphore=search_semaphore,
        browser_semaphore=browser_semaphore,
        usage_tracker=usage_tracker,
        url_ignore_substrings=search_params.url_ignore_substrings,
        **search_params.se_kwargs,
    )


async def _process_one_jurisdiction(
    row_index,
    jurisdiction,
    model_configs,
    search_params,
    file_loader_kwargs,
    browser_semaphore,
    search_semaphore,
):
    """Validate or discover one jurisdiction website"""
    usage_tracker = LLMUsageTracker(
        f"row_{row_index + 1}_{jurisdiction.full_name}",
        usage_from_response,
    )
    location_label = _task_label(jurisdiction, row_index)
    original_website = jurisdiction.website_url
    website = original_website

    with COMPASS_PB.jurisdiction_prog_bar(location_label):
        if website:
            website = await _validate_jurisdiction_website(
                jurisdiction,
                website,
                model_configs,
                browser_semaphore,
                file_loader_kwargs,
                usage_tracker,
                location_label,
            )
            status = "validated_input" if website else "invalid_input"

        else:
            status = "missing_input"

        if not website:
            website = await _find_jurisdiction_website_for_jurisdiction(
                jurisdiction,
                model_configs,
                file_loader_kwargs,
                search_semaphore,
                browser_semaphore,
                usage_tracker,
                search_params,
                location_label,
            )
            if website:
                status = "found"
            elif original_website:
                status = "not_found_after_invalid_input"
            else:
                status = "not_found"

        row_cost = compute_cost_from_totals(usage_tracker.totals)
        logger.info(
            "Finished website processing for %s | status=%s | "
            "website=%s | row_cost=$%.4f",
            jurisdiction.full_name,
            status,
            website,
            row_cost,
        )

    return {
        "row_index": row_index,
        "website": website,
        "status": status,
        "cost": row_cost,
        "usage_totals": usage_tracker.totals,
    }


def _process_all_jurisdictions(
    df,
    col_map,
    model_configs,
    search_params,
    file_loader_kwargs,
    max_num_concurrent_jurisdictions,
):
    """Run website processing for the full CSV"""
    browser_semaphore = None
    if search_params.max_num_concurrent_browsers:
        browser_semaphore = asyncio.Semaphore(
            search_params.max_num_concurrent_browsers
        )

    search_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCH_ENGINE_QUERIES)
    jurisdiction_semaphore = None
    if max_num_concurrent_jurisdictions:
        jurisdiction_semaphore = asyncio.Semaphore(
            max_num_concurrent_jurisdictions
        )

    jobs = []
    for row_index, row in df.iterrows():
        jurisdiction = _build_jurisdiction(row, col_map)
        location_label = _task_label(jurisdiction, row_index)

        async def _run_row(
            row_index=row_index,
            jurisdiction=jurisdiction,
        ):
            sem_context = (
                jurisdiction_semaphore
                if jurisdiction_semaphore is not None
                else contextlib.AsyncExitStack()
            )
            async with sem_context:
                return await _process_one_jurisdiction(
                    row_index,
                    jurisdiction,
                    model_configs,
                    search_params,
                    file_loader_kwargs,
                    browser_semaphore,
                    search_semaphore,
                )

        jobs.append(asyncio.create_task(_run_row(), name=location_label))

    logger.info(
        "Submitted %d jurisdiction row tasks for processing", len(jobs)
    )
    return jobs


def _summarize_results(results, start_time):
    """Compute summary counts and total usage cost"""
    summary = {
        "validated_input": 0,
        "redirected_input": 0,
        "found": 0,
        "invalid_input": 0,
        "missing_input": 0,
        "not_found": 0,
        "not_found_after_invalid_input": 0,
    }
    tracked_usage = {}
    for result in results:
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        tracked_usage[str(result["row_index"])] = {
            "tracker_totals": result["usage_totals"]
        }

    summary["total_cost"] = compute_total_cost_from_usage(tracked_usage)
    summary["total_seconds"] = (datetime.now(UTC) - start_time).total_seconds()
    return summary


async def _cancel_pending_row_tasks(tasks):
    """Cancel any unfinished child tasks associated with row work"""
    current_task = asyncio.current_task()
    task_names = {task.get_name() for task in tasks}
    pending_tasks = [
        task
        for task in asyncio.all_tasks()
        if task is not current_task
        and not task.done()
        and task.get_name() in task_names
    ]
    if not pending_tasks:
        return

    logger.info(
        "Cancelling %d unfinished row task(s) before shutdown",
        len(pending_tasks),
    )
    for task in pending_tasks:
        task.cancel()

    await asyncio.gather(*pending_tasks, return_exceptions=True)


async def _run(args, console):  # ruff:ignore[too-many-locals]
    """Run the website update workflow"""

    start_time = datetime.now(UTC)
    (
        model_configs,
        search_params,
        file_loader_kwargs,
        max_num_concurrent_jurisdictions,
        ppe_kwargs,
    ) = _load_runtime_inputs(args.config)
    logger.debug("Loaded model configs: %r", model_configs)
    logger.debug("Loaded search params: %r", search_params)
    logger.debug("Loaded file loader kwargs: %r", file_loader_kwargs)
    logger.debug(
        "Loaded max num concurrent jurisdictions: %r",
        max_num_concurrent_jurisdictions,
    )
    logger.debug("Loaded ppe kwargs: %r", ppe_kwargs)

    df, processing_df, col_map, website_col = _prepare_dataframe_inputs(
        args.jurisdiction_csv,
        args.start_index,
        args.end_index,
    )
    output_fp = Path(args.jurisdiction_csv).expanduser().resolve()
    selected_start, selected_end = _log_run_scope(
        df,
        processing_df,
        output_fp,
    )

    COMPASS_PB.reset()
    COMPASS_PB.create_main_task(len(processing_df), action="Updating")

    llm_services = list(
        {model.llm_service for model in model_configs.values()}
    )
    if COMPASSWebFileLoader.__name__ == "AsyncDoclingWebFileLoader":
        llm_services.append(FileLoader(**ppe_kwargs))
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    results = []
    async with RunningAsyncServices(llm_services):
        tasks = _process_all_jurisdictions(
            processing_df,
            col_map,
            model_configs,
            search_params,
            file_loader_kwargs,
            max_num_concurrent_jurisdictions,
        )
        try:
            for completed, task in enumerate(asyncio.as_completed(tasks)):
                result = await task
                results.append(result)
                df.loc[result["row_index"], website_col] = result["website"]
                if completed % 100 == 0:
                    summary = _summarize_results(results, start_time)
                    logger.info(
                        "Processed %d/%d rows so far | found=%d | "
                        "total_cost=$%.4f",
                        completed,
                        len(processing_df),
                        summary["found"],
                        summary["total_cost"],
                    )
                    df.to_csv(output_fp, index=False)
        finally:
            await _cancel_pending_row_tasks(tasks)

    df.to_csv(output_fp, index=False)
    summary = _summarize_results(results, start_time)
    summary_message = "\n".join(
        [
            "Website update complete.",
            f"Rows processed: {len(processing_df):,}",
            f"Processed index range: {selected_start:,}-{selected_end:,}",
            (
                "Existing websites kept: "
                f"{
                    sum(
                        (
                            summary['validated_input'],
                            summary['redirected_input'],
                        )
                    ):,}"
            ),
            f"Websites found by search: {summary['found']:,}",
            (
                "Rows still missing websites: "
                f"{
                    sum(
                        (
                            summary['not_found'],
                            summary['not_found_after_invalid_input'],
                        )
                    ):,}"
            ),
            f"Total cost: ${summary['total_cost']:.4f}",
            f"Time elapsed: {_elapsed_time_as_str(summary['total_seconds'])}",
            f"Wrote updated CSV to: {output_fp}",
        ]
    )
    logger.info(summary_message)
    console.print(summary_message)

    COMPASS_PB.reset()


def main():
    """Run the update-jurisdiction-websites CLI"""
    args = _parse_args()
    log_fp = _resolve_log_path(args.config)
    custom_theme = Theme({"logging.level.trace": "rgb(94,79,162)"})
    console = Console(theme=custom_theme)
    _setup_terminal_logging(
        console,
        args.verbose,
        args.log_level,
        log_fp,
    )

    if args.no_progress:
        asyncio.run(_run(args, console))
        return

    COMPASS_PB.console = console
    live_display = Live(
        COMPASS_PB.group,
        console=console,
        refresh_per_second=20,
        transient=True,
    )
    with live_display:
        asyncio.run(_run(args, console))
    COMPASS_PB.console = None


if __name__ == "__main__":
    main()
