"""COMPASS ordinance document web search functionality"""

import logging
from urllib.parse import urlsplit, urlunsplit

from elm.web.search.run import search_all_se, search_with_fallback_with_attrs

from compass.utilities.url import URLPartFilter


logger = logging.getLogger(__name__)
_JURISDICTION_WEBSITE_PLACEHOLDER = "{jurisdiction_website}"


async def search_single_jurisdiction(
    query_templates,
    jurisdiction,
    num_urls=5,
    browser_semaphore=None,
    url_ignore_substrings=None,
    url_keep_substrings=None,
    simple=True,
    **se_kwargs,
):
    """Search the web for relevant links and return a sorted output

    Parameters
    ----------
    query_templates : iterable of str
        Query templates to format with the jurisdiction name and search.
        Each template should include a ``{jurisdiction}`` placeholder
        for the jurisdiction name. Templates may include a
        ``{jurisdiction_website}`` placeholder for the jurisdiction's
        known website URL. Website templates are skipped when the
        jurisdiction does not have a known website.
    jurisdiction : Jurisdiction
        Jurisdiction instance representing the jurisdiction to search
        documents for.
    num_urls : int, optional
        Number of unique search result URL's to check for each
        jurisdiction. By default, ``5``.
    browser_semaphore : asyncio.Semaphore
        Semaphore instance that can be used to limit the number of
        playwright browsers used to submit search engine queries open
        concurrently. By default, ``None``.
    url_ignore_substrings : list of str, optional
        URL substrings that should be excluded from search results.
        Substrings are applied case-insensitively. By default, ``None``.
    url_keep_substrings : list of str, optional
        URL substrings that should be included in search results even if
        they match an ignore substring. Substrings are applied
        case-insensitively. By default, ``None``.
    simple : bool, optional
        Flag indicating whether to use a simple top-n sort from the
        first search engine that gives results (``True``) or to apply a
        holistic link sorting based on all results from all search
        engines (``False``). By default, ``True``.
    **se_kwargs
        Additional keyword arguments forwarded to
        :func:`elm.web.search.run.web_search_links_as_docs`. Common
        entries include ``usage_tracker`` for logging LLM usage and
        extra Playwright configuration.

    Returns
    -------
    dict
        Dictionary containing the following keys:

            - ``jurisdiction``: Full jurisdiction name
            - ``state``: Jurisdiction state
            - ``county``: Jurisdiction county
            - ``subdivision``: Jurisdiction subdivision name
            - ``queries``: List of formatted query strings that were
              searched
            - ``results``: List of search results dictionaries, with at
              least one key: ``"url"``, which contains the URL of the
              search result.

    """

    queries = _format_queries(jurisdiction, query_templates)
    base = {
        "jurisdiction": jurisdiction.full_name,
        "state": jurisdiction.state,
        "county": jurisdiction.county,
        "subdivision": jurisdiction.subdivision_name,
        "queries": queries,
        "results": [],
        "error": None,
    }
    if not queries:
        return base

    run_meth = _run_simple_sort_search if simple else _run_holistic_sort_search

    try:
        out = await run_meth(
            queries,
            num_urls,
            url_ignore_substrings,
            url_keep_substrings,
            browser_semaphore,
            jurisdiction.full_name,
            **se_kwargs,
        )

    except Exception as exc:
        logger.exception("Search failed for %s", jurisdiction.full_name)
        base["error"] = f"{type(exc).__name__}: {exc}"
        return base

    base["results"] = out
    return base


def _format_queries(jurisdiction, query_templates):
    """Format query templates for a given jurisdiction"""
    jurisdiction_website = _format_jurisdiction_website(jurisdiction)
    return [
        query.format(
            jurisdiction=jurisdiction.full_name,
            jurisdiction_website=jurisdiction_website,
        )
        for query in query_templates
        if _JURISDICTION_WEBSITE_PLACEHOLDER not in query
        or jurisdiction.website_url
    ]


def _format_jurisdiction_website(jurisdiction):
    """Strip paths and ensure a valid URL scheme for a jur website"""
    jurisdiction_website = jurisdiction.website_url
    if jurisdiction_website:
        website_parts = urlsplit(jurisdiction_website)
        if not website_parts.netloc:
            website_parts = urlsplit(f"//{jurisdiction_website}")
        jurisdiction_website = urlunsplit(
            (website_parts.scheme, website_parts.netloc, "", "", "")
        ).removeprefix("//")
    return jurisdiction_website


async def _run_simple_sort_search(
    queries,
    num_urls,
    url_ignore_substrings,
    url_keep_substrings,
    search_semaphore,
    jurisdiction_full_name,
    **se_kwargs,
):
    """Run search with fallback search engines, applying simple sort"""
    return await search_with_fallback_with_attrs(
        queries,
        num_urls=num_urls,
        url_ignore_substrings=url_ignore_substrings,
        url_keep_substrings=url_keep_substrings,
        browser_semaphore=search_semaphore,
        task_name=jurisdiction_full_name,
        **se_kwargs,
    )


async def _run_holistic_sort_search(
    queries,
    num_urls,
    url_blacklist,
    url_whitelist,
    browser_semaphore,
    jurisdiction_full_name,
    **se_kwargs,
):
    """Run search with all search engines and apply holistic sorting"""
    out = await search_all_se(
        queries,
        num_urls=10,  # Need as many results as possible for holistic sort
        url_ignore_substrings=None,  # custom filters applied later
        url_keep_substrings=None,  # custom filters applied later
        browser_semaphore=browser_semaphore,
        task_name=jurisdiction_full_name,
        **se_kwargs,
    )
    return _apply_filters(out, url_blacklist, url_whitelist, num_urls)


def _apply_filters(results, url_blacklist, url_whitelist, num_urls):
    """Mark blacklisted URLs, duplicates, and beyond top-N entries"""

    results = _flatten_results(results)
    _apply_blacklist_filters(results, url_blacklist, url_whitelist)
    _apply_duplicate_filters(results)
    _apply_top_n_filters(results, num_urls)

    for entry in results:
        entry.pop("_order", None)
        entry.pop("query_index", None)
        entry.pop("se_order", None)

    return sorted(results, key=_overall_sort_key)


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


def _apply_blacklist_filters(results, url_blacklist, url_whitelist):
    """Mark rows that match any blacklist substring"""
    url_filter = URLPartFilter(url_blacklist, url_whitelist)

    for entry in results:
        blacklist_match = url_filter.blacklist_match(entry["url"])
        if blacklist_match is None:
            continue
        entry["filtered_reason"] = f"blacklist:{blacklist_match}"


def _apply_duplicate_filters(results):
    """Mark duplicate rows by URL, across all search engines

    Winner selection follows the priority encoded in
    :func:`_link_sort_key`, so the engine listed first in the user
    config only wins among entries that are otherwise tied (same
    ``query_rank`` and ``query_index``).
    """
    for entry in results:
        entry["search_engines"] = [entry["search_engine"]]

    winners = {}
    for entry in _active_results_sorted(results):
        key = entry["url"]
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
        if entry["search_engine"] not in winner["search_engines"]:
            winner["search_engines"].append(entry["search_engine"])

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
        entry["query_index"],
        entry["se_order"],
        entry["_order"],
    )


def _overall_sort_key(result):
    """Get overall sort key for a search result item"""
    return (
        result.get("overall_rank") or float("inf"),
        result.get("filtered_reason") or "",
    )
