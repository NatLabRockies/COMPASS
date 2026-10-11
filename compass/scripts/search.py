"""Search-only orchestration for COMPASS

Runs the web-search portion of the COMPASS pipeline (no download,
filtering, validation, or extraction) and emits a JSON report of the
ranked URLs returned by each configured search engine for each
jurisdiction. The output is intended to help diagnose retrieval
quality before invoking the full pipeline.
"""

import json
from datetime import UTC, datetime
from functools import partial
from os import PathLike
from pathlib import Path
from statistics import median
from warnings import warn

from elm.version import __version__ as elm_version
from elm.web.search.run import SEARCH_ENGINE_OPTIONS

from compass import __version__ as compass_version
from compass.exceptions import COMPASSFileNotFoundError, COMPASSValueError
from compass.pipeline.targets import evaluate_targets, normalize_targets
from compass.utilities.io import write_text_atomic, load_config
from compass.warn import COMPASSWarning
from compass.web.search import _search_result_shard_path

SEARCH_RESULT_MANIFEST_FILENAME = "search_result_manifest.json"


async def build_search_report(
    runtime, jur_results, time_start_utc, failed_targets=None
):
    """Aggregate jurisdiction search results into a report

    Parameters
    ----------
    runtime : compass.pipeline.runtime.PipelineRuntime
        Runtime containing the search settings and technology.
    jur_results : list of dict or None
        Results for each requested jurisdiction. Failed workflows may
        return ``None``.
    time_start_utc : datetime.datetime
        UTC timestamp when the search run started.
    failed_targets : list of str, optional
        List of failed target descriptions/specifications, if any.
        By default, ``None``.

    Returns
    -------
    dict
        JSON-serializable report containing per-jurisdiction ranked
        URLs and filtering reasons.
    """
    qt = await runtime.extractor_class(None, None).get_query_templates()
    se_kwargs = runtime.search_params.se_kwargs
    num_urls = runtime.search_params.num_urls_to_check_per_jurisdiction
    num_jurisdictions_searched = len(jur_results)

    time_end_utc = datetime.now(UTC)
    time_elapsed = time_end_utc - time_start_utc

    result_counts = []
    filtered_counts = []
    se_counts = {}
    out_results = []
    for results in jur_results:
        if results is None:
            result_counts.append(0)
            filtered_counts.append(0)
            continue

        out_results.append(results)
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
        "num_urls_requested": num_urls,
        "search_engines": list(se_kwargs.get("search_engines", [])),
        "query_templates": list(qt),
        "time_start_utc": time_start_utc.isoformat(),
        "time_end_utc": time_end_utc.isoformat(),
        "total_time": time_elapsed.total_seconds(),
        "total_time_string": str(time_elapsed),
        "num_jurisdictions_searched": num_jurisdictions_searched,
        "num_jurisdictions_found": sum(
            results.get("num_results", 0) > 0 for results in out_results
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
        "failed_targets": failed_targets,
        "jurisdictions": out_results,
    }


def load_search_result_jurisdictions(manifest_fp, expected_tech):
    """Load saved search results indexed by jurisdiction code

    Parameters
    ----------
    manifest_fp : path-like or list of path-like
        Manifest, shard, run directory, shard directory, or glob paths.
    expected_tech : str
        Technology required when the input declares its technology.

    Returns
    -------
    dict
        Saved jurisdiction results indexed by their FIPS codes.
    """
    if isinstance(manifest_fp, (str, PathLike)):
        manifest_fp = [manifest_fp]

    jurisdictions = {}
    for pattern in manifest_fp:
        jurisdictions.update(
            _records_from_files(pattern, expected_tech, jurisdictions)
        )

    return jurisdictions


def _records_from_files(pattern, expected_tech, jurisdictions):
    """Yield search records from files matching the given pattern"""
    for shard_fp in _search_result_paths(Path(pattern).expanduser()):
        yield from _records_from_single_file(
            shard_fp, expected_tech, jurisdictions
        )


def _search_result_paths(path):
    """Select an aggregate manifest or the existing result shards"""
    if not path.exists():
        return _try_find_from_relative(path)

    if not path.is_dir():
        return [path]

    manifest = path / SEARCH_RESULT_MANIFEST_FILENAME
    if manifest.is_file():
        return [manifest]

    shard_dir = path / "se_results"
    if not shard_dir.is_dir():
        shard_dir = path

    paths = sorted(shard_dir.glob("*.json"))
    if not paths:
        msg = f"No search result shards found in {shard_dir}"
        raise COMPASSFileNotFoundError(msg)

    return paths


def _try_find_from_relative(path):
    """Try to find search result files from a relative path"""
    base = Path(path.anchor or ".")
    pattern = path.relative_to(base) if path.is_absolute() else path
    matches = sorted(base.glob(str(pattern)))
    if not matches:
        msg = f"Search result input not found: {path}"
        raise COMPASSFileNotFoundError(msg)

    return [
        shard_fp
        for match in matches
        for shard_fp in _search_result_paths(match)
    ]


def _records_from_single_file(shard_fp, expected_tech, jurisdictions):
    """Extract search result records from a single file"""
    payload = load_config(shard_fp, resolve_paths=False)
    _validate_search_result_tech(payload, expected_tech, shard_fp)
    if payload.get("tech") is None:
        msg = (
            f"Search result input {shard_fp} has no technology "
            "metadata; its technology cannot be verified"
        )
        warn(msg, COMPASSWarning)

    records = payload.get("jurisdictions", [payload])
    if not isinstance(records, list):
        msg = f"Invalid search result jurisdictions: {shard_fp}"
        raise COMPASSValueError(msg)

    for record in records:
        _validate_search_result_record(record, shard_fp)
        _validate_search_result_tech(record, expected_tech, shard_fp)
        code = str(record["FIPS"])
        if code in jurisdictions:
            msg = f"Duplicate search result entry for FIPS '{code}'"
            raise COMPASSValueError(msg)

        yield code, record


def _validate_search_result_tech(payload, expected_tech, path):
    """Validate declared search result technology"""
    if not isinstance(payload, dict):
        msg = f"Invalid search result payload: {path}"
        raise COMPASSValueError(msg)

    if (tech := payload.get("tech")) is not None and tech != expected_tech:
        msg = (
            f"Search result technology '{tech}' does not match "
            f"'{expected_tech}': {path}"
        )
        raise COMPASSValueError(msg)


def _validate_search_result_record(record, path):
    """Validate replay identity and result structure"""
    if _invalid_record(record):
        msg = f"Invalid search result FIPS: {path}"
        raise COMPASSValueError(msg)

    if _invalid_results_list(record.get("results")):
        msg = f"Invalid search results: {path}"
        raise COMPASSValueError(msg)


def _invalid_record(record):
    """Validate a SE record

    Checks:
        - record is a dict
        - record contains a non-empty "FIPS" string
    """
    if not isinstance(record, dict):
        return True
    if not isinstance(record.get("FIPS"), str):
        return True
    return not record["FIPS"].strip()


def _invalid_results_list(results):
    """Validate a SE results list

    Checks:
        - results is a list
        - all result dicts within are valid
    """
    if not isinstance(results, list):
        return True
    return any(_invalid_result(result) for result in results)


def _invalid_result(result):
    """Validate a SE result dictionary

    Checks:
        - result is a dict
        - URL exists and is str
        - URL is non-empty
        - overall_rank is either None or a positive integer
    """
    if not isinstance(result, dict):
        return True
    if not isinstance(result.get("url"), str):
        return True
    if not result["url"].strip():
        return True

    return result.get("overall_rank") is not None and (
        type(result["overall_rank"]) is not int or result["overall_rank"] < 1
    )


def validate_search_targets(targets, num_urls=5, search_engines=None):
    """Validate shard count targets and return search metric accessors

    Parameters
    ----------
    targets : dict
        Shard metric names and nonnegative integer minimums.
    num_urls : int, default=5
        Maximum retained URLs per jurisdiction. By default, ``5``.
    search_engines : list of str or dict, optional
        Configured engines. ``None`` allows registered engine labels.
        By default, ``None``.

    Returns
    -------
    dict
        Supported metric names mapped to shard value accessors.
    """

    if search_engines:
        names = [
            params if isinstance(params, str) else params["se_name"]
            for params in search_engines
        ]
        unknown = set(names).difference(SEARCH_ENGINE_OPTIONS)
        if unknown:
            msg = f"Unknown search engines: {sorted(unknown)}"
            raise COMPASSValueError(msg)

        options = {name: SEARCH_ENGINE_OPTIONS[name] for name in names}

    metrics = {
        "num_results": lambda record: record.get("num_results", 0),
        "num_kept_results": _kept_result_count,
    }
    for option in options.values():
        label = option.se_class._SE_NAME  # ruff:ignore[private-member-access]
        metrics[f"search_engine_counts.{label}"] = partial(
            _engine_result_count, engine=label
        )

    for metric, minimum in normalize_targets(targets).items():
        if metric not in metrics:
            msg = (
                f"Unknown search shard target '{metric}'. "
                f"Available targets: {', '.join(sorted(metrics))}"
            )
            raise COMPASSValueError(msg)

        if not isinstance(minimum, int) or minimum < 0:
            msg = f"Search target '{metric}' requires a nonnegative integer"
            raise COMPASSValueError(msg)

        if metric == "num_kept_results" and minimum > num_urls:
            msg = (
                f"Target num_kept_results={minimum} exceeds the configured "
                f"retained URL limit ({num_urls})"
            )
            raise COMPASSValueError(msg)

    return metrics


def _kept_result_count(record):
    """Read retained counts or derive them for older search shards"""
    return record.get(
        "num_kept_results",
        sum(row.get("filtered_reason") is None for row in record["results"]),
    )


def _engine_result_count(record, engine):
    """Read one engine count, treating an absent engine as zero"""
    return record.get("search_engine_counts", {}).get(engine, 0)


def search_shard_failures(record, targets, metrics):
    """Evaluate default nonempty results and explicit shard minimums

    Parameters
    ----------
    record : dict or None
        One search shard, or ``None`` when its file is missing.
    targets : dict
        Validated search minimum targets.
    metrics : dict
        Search metric accessors returned by target validation.

    Returns
    -------
    list of dict
        Unmet default and explicit requirements for this shard.
    """
    record = record or {"results": []}
    failures = []
    if not record["results"]:
        failures.append({"metric": "results", "actual": 0, "minimum": 1})
    return failures + evaluate_targets(record, targets, metrics)


def load_search_shards(out_dir, jurisdictions, expected_tech):
    """Inventory actual requested shards without trusting the manifest

    Parameters
    ----------
    out_dir : path-like
        Existing search output directory.
    jurisdictions : iterable of Jurisdiction
        Jurisdictions requested in the current configuration.
    expected_tech : str
        Required technology when saved outputs declare one.

    Returns
    -------
    dict
        FIPS codes mapped to dictionaries with ``record`` and ``path``.
        Missing shards have no entry.
    """
    out_dir = Path(out_dir)
    manifest = out_dir / SEARCH_RESULT_MANIFEST_FILENAME
    if manifest.is_file():
        _validate_search_result_tech(
            load_config(manifest, resolve_paths=False), expected_tech, manifest
        )

    requested = {
        jurisdiction.code: jurisdiction for jurisdiction in jurisdictions
    }
    shard_dir = out_dir / "se_results"
    expected_paths = {
        _search_result_shard_path(shard_dir, jurisdiction): code
        for code, jurisdiction in requested.items()
    }
    if len(expected_paths) != len(requested):
        msg = "Requested jurisdiction shard filenames collide"
        raise COMPASSValueError(msg)

    shards = {}
    for path in sorted(shard_dir.glob("*.json")):
        record = load_config(path, resolve_paths=False)
        _validate_search_result_record(record, path)
        code = record["FIPS"]
        if path in expected_paths and code != expected_paths[path]:
            msg = f"Search shard FIPS does not match {path}"
            raise COMPASSValueError(msg)

        if code not in requested:
            continue

        _validate_search_result_tech(record, expected_tech, path)
        if code in shards:
            msg = f"Duplicate search result entry for FIPS '{code}'"
            raise COMPASSValueError(msg)

        _validate_search_shard_counts(record, path)
        shards[code] = {"record": record, "path": path}

    return shards


def _validate_search_shard_counts(record, path):
    """Reject malformed shard counts before any continuation purge"""
    counts = record.get("search_engine_counts")
    if not isinstance(counts, dict):
        msg = f"Invalid search engine counts: {path}"
        raise COMPASSValueError(msg)

    values = [
        record.get("num_results"),
        _kept_result_count(record),
        *counts.values(),
    ]
    if any(type(value) is not int or value < 0 for value in values):
        msg = f"Invalid search shard counts: {path}"
        raise COMPASSValueError(msg)


def write_search_report(report, out_path):
    """Write a search-only report as JSON

    Parameters
    ----------
    report : dict
        Report returned by :func:`build_search_report`.
    out_path : path-like
        Destination file path.
    """
    payload = json.dumps(report, indent=4, ensure_ascii=False)
    write_text_atomic(out_path, payload)


def summary(report):
    """Format search-only output as readable plain text

    Parameters
    ----------
    report : dict
        Dictionary produced by :func:`build_search_report`.

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
            "---------------------------",
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
