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
import random
from datetime import datetime, UTC
from operator import itemgetter
from pathlib import Path

from elm.web.search.run import SEARCH_ENGINE_OPTIONS

from compass.plugin import PLUGIN_REGISTRY
from compass.utilities.base import WebSearchParams
from compass.utilities.jurisdictions import (
    jurisdictions_from_df,
    load_jurisdictions_from_fp,
)


logger = logging.getLogger(__name__)


_DEFAULT_SEARCH_ENGINES = (
    "PlaywrightGoogleLinkSearch",
    "PlaywrightDuckDuckGoLinkSearch",
    "DuxDistributedGlobalSearch",
)


async def run_search_only(
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
        :obj:`~compass.plugin.PLUGIN_REGISTRY`.
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
        :class:`~compass.utilities.base.WebSearchParams`). If omitted,
        the elm default fallback chain is used. By default, ``None``.
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

    se_names, init_kwargs_by_se = _resolve_search_engines(wsp)

    plugin_cls = _resolve_plugin(tech)
    query_templates = await _get_query_templates(plugin_cls)

    jurisdictions = list(
        jurisdictions_from_df(load_jurisdictions_from_fp(jurisdiction_fp))
    )

    browser_semaphore = asyncio.Semaphore(max_num_concurrent_browsers)
    blacklist = list(url_ignore_substrings or [])

    tasks = [
        _search_one_jurisdiction(
            jur,
            query_templates,
            se_names,
            init_kwargs_by_se,
            browser_semaphore,
            blacklist,
            wsp.num_urls_to_check_per_jurisdiction,
        )
        for jur in jurisdictions
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
        "search_engines": list(se_names),
        "query_templates": list(query_templates),
        "jurisdictions": jur_results,
    }


def _resolve_search_engines(wsp):
    """Return ordered engine names and per-engine init kwargs"""
    se_kwargs = dict(wsp.se_kwargs)
    se_names = se_kwargs.pop("search_engines", None) or list(
        _DEFAULT_SEARCH_ENGINES
    )
    pw_launch_kwargs = se_kwargs.get("pw_launch_kwargs", {})

    init_kwargs_by_se = {}
    for se_name in se_names:
        opt = SEARCH_ENGINE_OPTIONS[se_name]
        init_kwargs = dict(pw_launch_kwargs) if opt.uses_browser else {}
        init_kwargs.update(se_kwargs.get(opt.kwg_key_name, {}))
        init_kwargs_by_se[se_name] = init_kwargs

    return se_names, init_kwargs_by_se


def _resolve_plugin(tech):
    """Look up the registered plugin class for a technology"""
    plugin_cls = PLUGIN_REGISTRY.get(tech.casefold())
    if plugin_cls is None:
        msg = (
            f"No plugin registered for tech={tech!r}. Available: "
            f"{sorted(PLUGIN_REGISTRY)}"
        )
        raise KeyError(msg)
    return plugin_cls


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
        raise ValueError(msg)
    return list(templates)


async def _search_one_jurisdiction(
    jurisdiction,
    query_templates,
    se_names,
    init_kwargs_by_se,
    browser_semaphore,
    blacklist,
    num_urls,
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
        per_query = await asyncio.gather(
            *[
                _search_one_query(
                    query,
                    query_index,
                    se_names,
                    init_kwargs_by_se,
                    browser_semaphore,
                    jurisdiction.full_name,
                )
                for query_index, query in enumerate(queries)
            ]
        )
    except Exception as exc:
        logger.exception(
            "Search failed for %s", jurisdiction.full_name
        )
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base

    flat = [entry for entries in per_query for entry in entries]
    base["results"] = _apply_filters(flat, blacklist, num_urls)
    return base


async def _search_one_query(
    query,
    query_index,
    se_names,
    init_kwargs_by_se,
    browser_semaphore,
    location,
):
    """Run a single query through the engine fallback chain"""
    for se_name in se_names:
        opt = SEARCH_ENGINE_OPTIONS[se_name]
        try:
            engine = opt.se_class(**init_kwargs_by_se[se_name])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] could not instantiate %s: %s",
                location,
                se_name,
                exc,
            )
            continue

        try:
            raw = await _run_query_with_engine(
                engine,
                query,
                uses_browser=opt.uses_browser,
                browser_semaphore=browser_semaphore,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[%s] %s search failed for %r: %s",
                location,
                se_name,
                query,
                exc,
            )
            continue

        urls = raw[0] if raw else []
        if not urls:
            continue

        return [
            {
                "url": url,
                "query": query,
                "query_index": query_index,
                "search_engine": se_name,
                "query_rank": rank,
                "overall_rank": None,
                "filtered_reason": None,
            }
            for rank, url in enumerate(urls, start=1)
        ]

    return []


async def _run_query_with_engine(
    engine, query, uses_browser, browser_semaphore
):
    """Execute one query for a pre-initialized engine"""
    if uses_browser:
        await asyncio.sleep(random.uniform(1, 10))  # noqa: S311
        async with browser_semaphore:
            return await engine.results(query, num_results=10)

    return await engine.results(query, num_results=10)


def _apply_filters(results, blacklist, num_urls):
    """Mark blacklisted URLs, duplicates, and beyond top-N entries"""
    for order, entry in enumerate(results):
        entry["_order"] = order
        entry["overall_rank"] = None

    _apply_blacklist_filters(results, blacklist)
    _apply_duplicate_filters(results)
    _apply_top_n_filters(results, num_urls)

    for entry in results:
        entry.pop("_order", None)
        entry.pop("query_index", None)

    return results


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
                "overall_rank": entry["overall_rank"],
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
    """Return filtered-in rows sorted by query rank and query order"""
    active_results = [
        entry for entry in results if entry["filtered_reason"] is None
    ]
    active_results.sort(
        key=itemgetter("query_rank", "query_index", "_order")
    )
    return active_results


def write_search_only_report(report, out_path=None):
    """Write or print a search-only report as JSON

    Parameters
    ----------
    report : dict
        Report returned by :func:`run_search_only`.
    out_path : path-like, optional
        Destination file path. If ``None``, the report is written to
        stdout. By default, ``None``.
    """
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if out_path is None:
        print(payload)
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")


def format_search_only_report_human(report):
    """Format search-only output as readable plain text

    Parameters
    ----------
    report : dict
        Dictionary produced by :func:`run_search_only`.

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
            (
                "requested top urls: "
                f"{report.get('num_urls_requested')}"
            ),
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
