"""COMPASS utilities for finalizing a run directory"""

import json
import getpass
import logging
from pathlib import Path

import pandas as pd
from elm.version import __version__ as elm_version

from compass import __version__ as compass_version
from compass.utilities.parsing import (
    num_ordinances_dataframe,
    ordinances_bool_index,
)


logger = logging.getLogger(__name__)

def save_run_meta(
    dirs,
    tech,
    start_date,
    end_date,
    num_jurisdictions_searched,
    num_jurisdictions_found,
    total_cost,
    models,
):
    """Persist metadata describing an ordinance collection run

    Parameters
    ----------
    dirs : compass.utilities.base.Directories
        Directory container describing where outputs, logs, and working
        files should be written during the run.
    tech : {"wind", "solar", "small wind"}
        Technology targeted by the collection run. The value is stored
        verbatim in the metadata file for downstream reporting.
    start_date : datetime.datetime
        Timestamp marking when the run began.
    end_date : datetime.datetime
        Timestamp marking when the run finished.
    num_jurisdictions_searched : int
        Number of jurisdictions evaluated during the run.
    num_jurisdictions_found : int
        Number of jurisdictions that produced at least one ordinance.
    total_cost : float
        Aggregate cost incurred by LLM usage for the run. ``None`` or
        zero values are recorded as ``null`` in the metadata.
    models : dict
        Mapping from LLM task identifiers (as str) to configuration
        objects (:class:`~compass.llm.config.OpenAIConfig`) used
        throughout the run. The function records a condensed summary of
        each configuration.

    Returns
    -------
    float
        Total runtime of the collection, expressed in seconds.

    Notes
    -----
    The function writes ``meta.json`` into ``dirs.out`` alongside
    references to other artifacts generated during the run. The return
    value mirrors the ``total_time`` entry stored in the metadata.
    """

    try:
        username = getpass.getuser()
    except OSError:
        username = "Unknown"

    time_elapsed = end_date - start_date
    meta_data = {
        "username": username,
        "versions": {"compass": compass_version, "elm": elm_version},
        "technology": tech,
        "models": _extract_model_info_from_all_models(models),
        "time_start_utc": start_date.isoformat(),
        "time_end_utc": end_date.isoformat(),
        "total_time": time_elapsed.total_seconds(),
        "total_time_string": str(time_elapsed),
        "num_jurisdictions_searched": num_jurisdictions_searched,
        "num_jurisdictions_found": num_jurisdictions_found,
        "cost": total_cost or None,
        "manifest": {},
    }
    manifest = {
        "LOG_DIR": dirs.logs,
        "CLEAN_FILE_DIR": dirs.clean_files,
        "JURISDICTION_DBS_DIR": dirs.jurisdiction_dbs,
        "ORDINANCE_FILES_DIR": dirs.ordinance_files,
        "USAGE_FILE": dirs.out / "usage.json",
        "JURISDICTION_FILE": dirs.out / "jurisdictions.json",
        "QUANT_DATA_FILE": dirs.out / "quantitative_ordinances.csv",
        "QUAL_DATA_FILE": dirs.out / "quantitative_ordinances.csv",
    }
    for name, file_path in manifest.items():
        if file_path.exists():
            meta_data["manifest"][name] = str(
                file_path.relative_to(dirs.out, walk_up=True)
            )
        else:
            meta_data["manifest"][name] = None

    meta_data["manifest"]["META_FILE"] = "meta.json"
    with (dirs.out / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta_data, fh, indent=4)

    return time_elapsed.total_seconds()


def doc_infos_to_db(doc_infos, output_columns):
    """Aggregate parsed ordinance CSV files into a normalized database

    Parameters
    ----------
    doc_infos : Iterable
        Iterable of dictionaries describing ordinance extraction
        results. Each dictionary must contain ``"ord_db_fp"`` (path to a
        parsed CSV), ``"source"`` (document URL), ``"date"`` (tuple of
        year, month, day, with ``None`` allowed), and ``"jurisdiction"``
        (a :class:`~compass.utilities.jurisdictions.Jurisdiction`
        instance).
    output_columns : list
        List of expected output columns (as
        :class:`compass.plugin.interface.OutputColumn` instances)
        for the consolidated database. The function enforces the
        presence of these columns in the output
        :class:`pandas.DataFrame`, filling in missing columns with null
        values as needed.

    Returns
    -------
    pandas.DataFrame
        Consolidated ordinance dataset.
    int
        Number of jurisdictions contributing at least one ordinance to
        the consolidated dataset.

    Notes
    -----
    Empty or ``None`` entries in ``doc_infos`` are skipped. Ordinance
    CSVs that lack parsed values (``num_ordinances_dataframe`` equals
    zero) are ignored. The returned DataFrame enforces an ordered column
    layout and casts the ``quantitative`` flag to nullable boolean.
    """
    db = []
    parsed_cols = [col.name for col in output_columns]
    for doc_info in doc_infos:
        if doc_info is None:
            continue

        ord_db_fp = doc_info.get("ord_db_fp")
        if ord_db_fp is None:
            continue

        ord_db = pd.read_csv(ord_db_fp, encoding="utf-8-sig")

        if num_ordinances_dataframe(ord_db) == 0:
            continue

        results = _db_results(ord_db, doc_info["jurisdiction"])
        results = _formatted_db(results, parsed_cols=parsed_cols)
        db.append(results)

    if not db:
        return pd.DataFrame(columns=parsed_cols), 0

    logger.info("Compiling final database for %d jurisdiction(s)", len(db))
    num_jurisdictions_found = len(db)
    db = pd.concat([df.dropna(axis=1, how="all") for df in db], axis=0)
    db = _empirical_adjustments(db)
    return _formatted_db(db, parsed_cols=parsed_cols), num_jurisdictions_found


def save_db(db, out_dir, output_columns):
    """Write qualitative and quantitative ordinance outputs to disk

    Parameters
    ----------
    db : pandas.DataFrame
        Ordinance dataset containing the full set of output columns,
        plus the ``quantitative`` boolean flag that dictates output
        routing.
    out_dir : path-like
        Directory where ``qualitative_ordinances.csv`` and
        ``quantitative_ordinances.csv`` should be written. The directory
        is created by :class:`pathlib.Path` if necessary.
    output_columns : list
        List of expected output columns (as
        :class:`compass.plugin.interface.OutputColumn` instances)
        for the consolidated database. The function enforces the
        presence of these columns in the output
        :class:`pandas.DataFrame`, filling in missing columns with null
        values as needed.

    Notes
    -----
    Empty DataFrames short-circuit without creating output files. The
    function respects the boolean ``quantitative`` column and assumes it
    has already been sanitized by :func:`doc_infos_to_db`.
    """
    if db.empty:
        return

    qual_out_cols = [
        col.name for col in output_columns if col.include_in_qual_output
    ]
    quant_out_cols = [
        col.name for col in output_columns if col.include_in_quant_output
    ]

    out_dir = Path(out_dir)
    qual_db = db[~db["quantitative"]][qual_out_cols]
    quant_db = db[db["quantitative"]][quant_out_cols]
    qual_db.to_csv(
        out_dir / "qualitative_ordinances.csv",
        index=False,
        encoding="utf-8-sig",
    )
    quant_db.to_csv(
        out_dir / "quantitative_ordinances.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _db_results(results, jurisdiction):
    """Extract results from doc attrs to DataFrame"""

    results["FIPS"] = jurisdiction.code
    results["county"] = jurisdiction.county
    results["state"] = jurisdiction.state
    results["subdivision"] = jurisdiction.subdivision_name
    results["jurisdiction_type"] = jurisdiction.type
    return results


def _empirical_adjustments(db):
    """Post-processing adjustments based on empirical observations

    Current adjustments include:

        - Limit adder to max of 250 ft.
            - Chat GPT likes to report large values here, but in
            practice all values manually observed in ordinance documents
            are below 250 ft. If large value is detected, assume it's an
            error on Chat GPT's part and remove it.

    """
    if "adder" in db.columns:
        db.loc[db["adder"] > 250, "adder"] = None  # ruff:ignore[magic-value-comparison]
    return db


def _formatted_db(db, parsed_cols):
    """Format DataFrame for output"""
    for col in parsed_cols:
        if col not in db.columns:
            db[col] = None

    db["quantitative"] = db["quantitative"].astype("boolean").fillna(True)
    ord_rows = ordinances_bool_index(db)
    return db[ord_rows][parsed_cols].reset_index(drop=True)


def _extract_model_info_from_all_models(models):
    """Group model info together"""
    models_to_tasks = {}
    for task, caller_args in models.items():
        models_to_tasks.setdefault(caller_args, []).append(task)

    return [
        {
            "name": caller_args.name,
            "llm_call_kwargs": caller_args.llm_call_kwargs or None,
            "llm_service_rate_limit": caller_args.llm_service_rate_limit,
            "text_splitter_chunk_size": caller_args.text_splitter_chunk_size,
            "text_splitter_chunk_overlap": (
                caller_args.text_splitter_chunk_overlap
            ),
            "client_type": caller_args.client_type,
            "tasks": tasks,
        }
        for caller_args, tasks in models_to_tasks.items()
    ]


def compile_run_summary_message(
    total_seconds, total_cost, out_dir, document_count
):
    """Create a human-readable summary of a completed run

    Parameters
    ----------
    total_seconds : float or int
        Duration of the run in seconds.
    total_cost : float or int or None
        Monetary cost incurred by the run. ``None`` or zero suppresses
        the cost line in the summary.
    out_dir : path-like
        Location of the run output directory. The value is embedded in
        the summary text.
    document_count : int
        Number of documents discovered across all jurisdictions.

    Returns
    -------
    str
        Summary string formatted for CLI presentation with ``rich``
        markup.

    Notes
    -----
    The function does not perform I/O; callers may log or display the
    returned string as needed.
    """
    runtime = _elapsed_time_as_str(total_seconds)
    total_cost = (
        f"\nTotal cost: [#71906e]${total_cost:,.2f}[/#71906e]"
        if total_cost
        else ""
    )

    return (
        f"✅ Scraping complete!\nOutput Directory: {out_dir}\n"
        f"Total runtime: {runtime} {total_cost}\n"
        f"Number of jurisdictions with extracted data: {document_count}"
    )


def compile_collection_summary_message(manifest_fp, collection_manifest):
    """Compile a short collection summary message

    Parameters
    ----------
    manifest_fp : path-like
        File path where the collection manifest was written. The value
        is embedded in the summary text.
    collection_manifest : dict
        Dictionary describing the collection results, including a list
        of jurisdictions and their associated documents. The function
        extracts jurisdiction and document counts from the manifest for
        inclusion in the summary.
    total_seconds : float or int
        Duration of the collection phase in seconds, used to report
        total runtime in the summary.

    Returns
    -------
    str
        Summary string formatted for CLI presentation with ``rich``
        markup.
    """
    num_jurisdictions = len(collection_manifest.get("jurisdictions", []))
    num_documents = sum(
        len((info or {}).get("documents") or [])
        for info in collection_manifest.get("jurisdictions", [])
    )
    runtime = _elapsed_time_as_str(collection_manifest["total_time"])
    locs = "jurisdiction" if num_jurisdictions == 1 else "jurisdictions"
    docs = "document" if num_documents == 1 else "documents"
    return (
        f"✅ Collection complete!\nCollection manifest: {manifest_fp}\n"
        f"Total runtime: {runtime}\n"
        f"{num_documents:,d} {docs} collected for "
        f"{num_jurisdictions:,d} {locs}"
    )


def _elapsed_time_as_str(seconds_elapsed):
    """Format elapsed time into human readable string"""
    days, seconds = divmod(int(seconds_elapsed), 24 * 3600)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    time_str = f"{hours:d}:{minutes:02d}:{seconds:02d}"
    if days:
        time_str = f"{days:,d} day{'s' if abs(days) != 1 else ''}, {time_str}"
    return time_str
