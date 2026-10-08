"""Search-only orchestration for COMPASS

Runs the web-search portion of the COMPASS pipeline (no download,
filtering, validation, or extraction) and emits a JSON report of the
ranked URLs returned by each configured search engine for each
jurisdiction. The output is intended to help diagnose retrieval
quality before invoking the full pipeline.
"""

import asyncio
import json
import logging
from statistics import median
from datetime import datetime, UTC
from pathlib import Path

from elm.version import __version__ as elm_version

from compass import __version__ as compass_version
from compass.web.search import search_single_jurisdiction
from compass.pipeline.runtime import PipelineRuntime
from compass.utilities.jurisdictions import (
    jurisdictions_from_df,
    load_jurisdictions_from_fp,
)


logger = logging.getLogger(__name__)


async def run_search_from_request(request, config_path=None):
    """Run search-engine queries for every jurisdiction in a config

    The function loads jurisdictions, fetches query templates from the
    plugin registered for ``tech``, formats them, and submits each
    query to the configured search engines (with fallback). All ranked
    URLs are returned in a JSON-serializable structure annotated with
    filtering reasons (blacklist, duplicate, or beyond requested
    top-N).

    Parameters
    ----------
    request : compass.pipeline.data_classes.BaseRequest
        The request object containing all user-specified settings and
        configurations for the pipeline run. This should be an instance
        of one of the specific request types (e.g., ProcessRequest,
        CollectionRequest, ExtractionRequest) that inherit from
        BaseRequest, and should include all necessary information such
        as the mode to run in, output directories, jurisdiction
        information, model configurations, and any other relevant
        settings.
    config_path : path-like, optional
        Absolute path of the originating config file, embedded in the
        returned report for traceability. By default, ``None``.

    Returns
    -------
    dict
        JSON-serializable report containing per-jurisdiction ranked
        URLs and filtering reasons.
    """

    runtime = PipelineRuntime(request)
    jurisdictions_df = load_jurisdictions_from_fp(request.jurisdiction_fp)
    return await run_search(runtime, jurisdictions_df, config_path)


async def run_search(
    runtime, jurisdictions_df, config_path=None, *, persist=False
):
    """Search jurisdictions and aggregate their results

    Parameters
    ----------
    runtime : compass.pipeline.runtime.PipelineRuntime
        The runtime object containing all necessary context and
        configurations for executing the search.
    jurisdictions_df : pandas.DataFrame
        DataFrame containing jurisdiction information loaded from the
        config.
    config_path : path-like, optional
        Absolute path of the originating config file, embedded in the
        returned report for traceability. By default, ``None``.
    persist : bool, optional
        Whether to persist intermediate search results to disk.
        By default, ``False``.

    Returns
    -------
    dict
        JSON-serializable report containing per-jurisdiction ranked
        URLs and filtering reasons.
    """
    time_start_utc = datetime.now(UTC)

    qt = await runtime.extractor_class(None, None).get_query_templates()
    se_kwargs = runtime.search_params.se_kwargs
    num_urls = runtime.search_params.num_urls_to_check_per_jurisdiction

    tasks = [
        search_single_jurisdiction(
            qt,
            jur,
            num_urls,
            runtime.search_engine_semaphore,
            runtime.search_params.url_ignore_substrings,
            runtime.search_params.url_keep_substrings,
            simple=False,
            **se_kwargs,
        )
        for jur in jurisdictions_from_df(jurisdictions_df)
    ]
    num_jurisdictions_searched = len(tasks)
    jur_results = await asyncio.gather(*tasks)

    time_end_utc = datetime.now(UTC)
    time_elapsed = time_end_utc - time_start_utc
    config_path = str(Path(config_path).resolve()) if config_path else None

    result_counts = []
    filtered_counts = []
    se_counts = {}
    for results in jur_results:
        result_counts.append(results.get("num_results", 0))
        filtered_counts.append(
            sum(
                row.get("filtered_reason") is None
                for row in results["results"]
            )
        )
        for se, count in results["search_engine_counts"].items():
            se_counts[se] = se_counts.get(se, 0) + count

    return {
        "tech": runtime.tech,
        "versions": {"compass": compass_version, "elm": elm_version},
        "config_path": config_path,
        "num_urls_requested": num_urls,
        "search_engines": list(se_kwargs.get("search_engines", [])),
        "query_templates": list(qt),
        "time_start_utc": time_start_utc.isoformat(),
        "time_end_utc": time_end_utc.isoformat(),
        "total_time": time_elapsed.total_seconds(),
        "total_time_string": str(time_elapsed),
        "num_jurisdictions_searched": num_jurisdictions_searched,
        "num_jurisdictions_found": sum(
            results.get("num_results", 0) > 0 for results in jur_results
        ),
        "search_engine_totals": dict(se_counts),
        "result_stats": {
            "min": min(result_counts, default=0),
            "max": max(result_counts, default=0),
            "median": median(result_counts) if result_counts else 0,
            "total": sum(result_counts),
        },
        "filtered_result_stats": {
            "min": min(filtered_counts, default=0),
            "max": max(filtered_counts, default=0),
            "median": median(filtered_counts) if filtered_counts else 0,
            "total": sum(filtered_counts),
        },
        "jurisdictions": jur_results,
    }


def write_search_report(report, out_path):
    """Write a search-only report as JSON

    Parameters
    ----------
    report : dict
        Report returned by :func:`run_search`.
    out_path : path-like
        Destination file path.
    """
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")


def summary(report):
    """Format search-only output as readable plain text

    Parameters
    ----------
    report : dict
        Dictionary produced by :func:`run_search`.

    Returns
    -------
    str
        Multi-line summary containing only records that were not
        filtered, sorted by ``overall_rank`` within each jurisdiction.
    """
    lines = []
    lines.extend(
        (
            "COMPASS search-only summary",
            f"tech: {report.get('tech')}",
            f"timestamp: {report.get('time_end_utc')}",
            f"requested top urls: {report.get('num_urls_requested')}",
            "",
        )
    )

    jurisdictions = report.get("jurisdictions", [])
    for jur in jurisdictions:
        lines.append(f"jurisdiction: {jur.get('full_name')}")

        if jur.get("error"):
            lines.extend((f"  error: {jur.get('error')}", ""))
            continue

        kept = [
            entry
            for entry in jur.get("results", [])
            if entry.get("filtered_reason") is None
        ]
        kept.sort(
            key=lambda entry: (
                entry.get("overall_rank")
                if entry.get("overall_rank") is not None
                else float("inf"),
                entry.get("query_rank")
                if entry.get("query_rank") is not None
                else float("inf"),
            )
        )

        if not kept:
            lines.extend(("  no unfiltered results", ""))
            continue

        for entry in kept:
            lines.extend(
                (
                    (
                        "  "
                        f"[{entry.get('overall_rank')}] "
                        f"{entry.get('search_engine')} "
                        f"(query_rank={entry.get('query_rank')})"
                    ),
                    f"    query: {entry.get('query')}",
                    f"    url: {entry.get('url')}",
                )
            )

        lines.append("")

    return "\n".join(lines).rstrip()
