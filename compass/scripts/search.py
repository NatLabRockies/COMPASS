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

from datetime import datetime, UTC
from pathlib import Path

from elm.web.search.run import search_all_se

from compass.exceptions import COMPASSValueError
from compass.plugin.registry import resolve_plugin
from compass.pipeline.data_classes import WebSearchParams
from compass.utilities.jurisdictions import (
    jurisdictions_from_df,
    load_jurisdictions_from_fp,
)


logger = logging.getLogger(__name__)


async def run_search(
    tech,
    jurisdiction_fp,
    num_urls_to_check_per_jurisdiction=5,
    max_num_concurrent_browsers=10,
    max_num_concurrent_website_searches=None,
    url_ignore_substrings=None,
    search_engines=None,
    config_path=None,
    **__,
):
    """Run search-engine queries for every jurisdiction in a config

    The function loads jurisdictions, fetches query templates from the
    plugin registered for ``tech``, formats them, and submits each
    query to the configured search engines (with fallback). All ranked
    URLs are returned in a JSON-serializable structure annotated with
    filtering reasons (blacklist, duplicate, or beyond requested
    top-N).

    Parameters
    ----------
    tech : str
        Technology identifier used to look up the registered plugin in
        :data:`compass.plugin.registry.PLUGIN_REGISTRY`.
    jurisdiction_fp : path-like
        Path to a CSV describing the jurisdictions to search.
    num_urls_to_check_per_jurisdiction : int, optional
        Number of top URLs to retain (per jurisdiction) before marking
        the remainder as filtered. By default, ``5``.
    max_num_concurrent_browsers : int, optional
        Maximum number of Playwright browser instances allowed to run
        concurrently across all jurisdictions. By default, ``10``.
    max_num_concurrent_website_searches : int, optional
        Unused; accepted for parity with the full pipeline config.
        By default, ``None``.
    url_ignore_substrings : list of str, optional
        Substrings used to mark matching URLs as filtered.
        By default, ``None``.
    search_engines : list of dict, optional
        Ordered search engine configurations (see
        :class:`~compass.pipeline.data_classes.WebSearchParams`). If
        omitted, the elm default fallback chain is used.
        By default, ``None``.
    config_path : path-like, optional
        Absolute path of the originating config file, embedded in the
        returned report for traceability. By default, ``None``.

    Returns
    -------
    dict
        JSON-serializable report containing per-jurisdiction ranked
        URLs and filtering reasons.
    """
    wsp = WebSearchParams(
        num_urls_to_check_per_jurisdiction=(
            num_urls_to_check_per_jurisdiction
        ),
        max_num_concurrent_browsers=max_num_concurrent_browsers,
        max_num_concurrent_website_searches=(
            max_num_concurrent_website_searches
        ),
        url_ignore_substrings=url_ignore_substrings,
        search_engines=search_engines,
    )

    plugin_cls = resolve_plugin(tech)
    query_templates = await _get_query_templates(plugin_cls)

    jurisdictions = load_jurisdictions_from_fp(jurisdiction_fp)

    browser_semaphore = asyncio.Semaphore(max_num_concurrent_browsers)
    blacklist = list(url_ignore_substrings or [])

    se_kwargs = wsp.se_kwargs

    tasks = [
        _search_one_jurisdiction(
            jur,
            query_templates,
            browser_semaphore,
            blacklist,
            wsp.num_urls_to_check_per_jurisdiction,
            se_kwargs,
        )
        for jur in jurisdictions_from_df(jurisdictions)
    ]
    jur_results = await asyncio.gather(*tasks)

    return {
        "timestamp": datetime.now(UTC)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "config_path": str(Path(config_path).resolve())
        if config_path
        else None,
        "tech": tech,
        "num_urls_requested": wsp.num_urls_to_check_per_jurisdiction,
        "search_engines": list(se_kwargs["search_engines"]),
        "query_templates": list(query_templates),
        "jurisdictions": jur_results,
    }


async def _get_query_templates(plugin_cls):
    """Pull query templates from a plugin without LLM model configs"""
    plugin = plugin_cls(None, None)
    templates = await plugin.get_query_templates()
    if not templates:
        msg = (
            f"Plugin {plugin_cls.__name__} returned no query templates. "
            "Pre-generate templates or provide them in the config before "
            "running search-only."
        )
        raise COMPASSValueError(msg)
    return list(templates)


async def _search_one_jurisdiction(
    jurisdiction,
    query_templates,
    browser_semaphore,
    blacklist,
    num_urls,
    se_kwargs,
):
    """Search every query/engine combo for a single jurisdiction"""
    queries = [
        template.format(jurisdiction=jurisdiction.full_name)
        for template in query_templates
    ]

    base = {
        "jurisdiction": jurisdiction.full_name,
        "state": jurisdiction.state,
        "county": jurisdiction.county,
        "subdivision": jurisdiction.subdivision_name,
        "queries": queries,
        "results": [],
        "error": None,
    }

    try:
        out = await search_all_se(
            queries,
            num_urls=num_urls,
            ignore_url_parts=None,  # custom filters applied later
            browser_semaphore=browser_semaphore,
            task_name=jurisdiction.full_name,
            **se_kwargs,
        )
    except Exception as exc:
        logger.exception("Search failed for %s", jurisdiction.full_name)
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base

    base["results"] = _apply_filters(out, blacklist, num_urls)
    return base


def _apply_filters(results, blacklist, num_urls):
    """Mark blacklisted URLs, duplicates, and beyond top-N entries"""

    results = _flatten_results(results)
    _apply_blacklist_filters(results, blacklist)
    _apply_duplicate_filters(results)
    _apply_top_n_filters(results, num_urls)

    for entry in results:
        entry.pop("_order", None)
        entry.pop("query_index", None)
        entry.pop("se_order", None)

    return results


def _flatten_results(results):
    """Flatten results from nested structure to a single list"""
    flat = []
    result_order = 1
    for se_ind, se_results in enumerate(results, start=1):
        for query_ind, single_query_results in enumerate(se_results, start=1):
            for link_info in single_query_results:
                link_info["filtered_reason"] = None
                link_info["overall_rank"] = None
                link_info["query_index"] = query_ind
                link_info["se_order"] = se_ind
                link_info["_order"] = result_order
                flat.append(link_info)
                result_order += 1
    return flat


def _apply_blacklist_filters(results, blacklist):
    """Mark rows that match any blacklist substring"""
    blacklist_terms = [sub for sub in blacklist if sub]
    blacklist_terms_cf = [sub.casefold() for sub in blacklist_terms]
    for entry in results:
        url_cf = entry["url"].casefold()
        match_index = next(
            (
                i
                for i, sub_cf in enumerate(blacklist_terms_cf)
                if sub_cf in url_cf
            ),
            None,
        )
        if match_index is None:
            continue
        entry["filtered_reason"] = f"blacklist:{blacklist_terms[match_index]}"


def _apply_duplicate_filters(results):
    """Mark duplicate rows per search engine and URL"""
    winners = {}
    for entry in _active_results_sorted(results):
        key = (entry["search_engine"], entry["url"])
        winner = winners.get(key)
        if winner is None:
            winners[key] = entry
            continue

        winner.setdefault("duplicates", []).append(
            {
                "url": entry["url"],
                "query": entry["query"],
                "search_engine": entry["search_engine"],
                "query_rank": entry["query_rank"],
            }
        )

        entry["filtered_reason"] = "duplicate"


def _apply_top_n_filters(results, num_urls):
    """Mark entries past top-N after filtering"""
    for overall_rank, entry in enumerate(
        _active_results_sorted(results), start=1
    ):
        entry["overall_rank"] = overall_rank
        if overall_rank <= num_urls:
            continue
        entry["filtered_reason"] = "beyond_top_n"


def _active_results_sorted(results):
    """Return filtered-in rows sorted by ranking priority"""
    active_results = [
        entry for entry in results if entry["filtered_reason"] is None
    ]

    active_results.sort(key=_link_sort_key)
    return active_results


def _link_sort_key(entry):
    """Get a sort key for a search result entry

    Lower values indicate more confidence in result
    """
    duplicate_count = len(entry.get("duplicates", []))
    return (  # lower is better
        entry["query_rank"],
        -duplicate_count,
        entry["se_order"],
        entry["query_index"],
        entry["_order"],
    )


def write_search_report(report, out_path):
    """Write a search-only report as JSON

    Parameters
    ----------
    report : dict
        Report returned by :func:`run_search`.
    out_path : path-like
        Destination file path. If ``None``, the report is written to
        stdout. By default, ``None``.
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
            f"timestamp: {report.get('timestamp')}",
            f"requested top urls: {report.get('num_urls_requested')}",
            "",
        )
    )

    jurisdictions = report.get("jurisdictions", [])
    for jur in jurisdictions:
        lines.append(f"jurisdiction: {jur.get('jurisdiction')}")

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
