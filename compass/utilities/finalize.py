"""COMPASS utilities for finalizing a run directory"""

import json
import getpass
import logging
from pathlib import Path

import pandas as pd
from elm.version import __version__ as elm_version

from compass import __version__ as compass_version
from compass.utilities.parsing import (
    extract_ord_year_from_doc_attrs,
    num_ordinances_dataframe,
    ordinances_bool_index,
)


logger = logging.getLogger(__name__)
_PARSED_COLS = [
    # TODO: Put these in an enum
    "county",
    "state",
    "subdivision",
    "jurisdiction_type",
    "FIPS",
    "feature",
    "value",
    "units",
    "adder",
    "min_dist",
    "max_dist",
    "summary",
    "ord_year",
    "section",
    "source",
    "quantitative",
]
QUANT_OUT_COLS = _PARSED_COLS[:-1]
"""Output columns in quantitative ordinance file"""
QUAL_OUT_COLS = _PARSED_COLS[:6] + _PARSED_COLS[-5:-1]
"""Output columns in qualitative ordinance file"""


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
    """Write out meta information about ordinance collection run

    Parameters
    ----------
    dirs : :class:`~compass.utilities.base.Directories`
        Directories instance containing information about the output
        directories used for the run.
    tech : {"wind", "solar", "small wind"}
        Technology that was the target of the run.
    start_date, end_date : datetime.datetime
        Instances representing the start and end dates, respectively.
    num_jurisdictions_searched, num_jurisdictions_found : int
        Total number of jurisdictions that were searched and actually
        found, respectively.
    total_cost : float
        Total cost of the processing, in $.
    models : dict
        Dictionary mapping task names (from
        :class:`~compass.utilities.enums.LLMTasks`) to
        :class:`~compass.llm.config.OpenAIConfig` instances used for the
        run.

    Returns
    -------
    run_time : float
        Total processing run-time, in seconds.
    """

    try:
        username = getpass.getuser()
    except OSError:
        username = "Unknown"

    time_elapsed = end_date - start_date
    meta_data = {
        "username": username,
        "versions": {"elm": elm_version, "compass": compass_version},
        "technology": tech,
        "models": _extract_model_info_from_all_models(models),
        "time_start_utc": start_date.isoformat(),
        "time_end_utc": end_date.isoformat(),
        "total_time": time_elapsed.seconds,
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
            meta_data["manifest"][name] = str(file_path.relative_to(dirs.out))
        else:
            meta_data["manifest"][name] = None

    meta_data["manifest"]["META_FILE"] = "meta.json"
    with (dirs.out / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta_data, fh, indent=4)

    return time_elapsed.seconds


def doc_infos_to_db(doc_infos):
    """Convert list of docs to output database

    Parameters
    ----------
    doc_infos : iterable of dict
        Iterable of dictionaries, where each dictionary has at least the
        following keys:

            - "ord_db_fp": Path to parsed ordinance CSV file
            - "source": URL of the file from which ordinances were
              extracted
            - "date": Tuple of (year, month, day). Any of the values can
              be ``None``.
            - "jurisdiction": Instance of Jurisdiction representing the
              jurisdiction associated with these ordinance values.

        If this iterable is empty, and empty DataFrame (with the correct
        columns) is returned.

    Returns
    -------
    ordinances : pandas.DataFrame
        DataFrame containing ordinances collected from all individual
        CSV's.
    count : int
        Total number jurisdictions for which ordinances were found.
    """
    db = []
    for doc_info in doc_infos:
        if doc_info is None:
            continue

        ord_db_fp = doc_info.get("ord_db_fp")
        if ord_db_fp is None:
            continue

        ord_db = pd.read_csv(ord_db_fp)

        if num_ordinances_dataframe(ord_db) == 0:
            continue

        results = _db_results(ord_db, doc_info)
        results = _formatted_db(results)
        db.append(results)

    if not db:
        return pd.DataFrame(columns=_PARSED_COLS), 0

    logger.info("Compiling final database for %d jurisdiction(s)", len(db))
    num_jurisdictions_found = len(db)
    db = pd.concat([df.dropna(axis=1, how="all") for df in db], axis=0)
    db = _empirical_adjustments(db)
    return _formatted_db(db), num_jurisdictions_found


def save_db(db, out_dir):
    """Split DB into qualitative vs quantitative and save to disk

    Parameters
    ----------
    db : pandas.DataFrame
        Pandas DataFrame containing ordinance data to save. Must have
        all columns in :obj:`QUANT_OUT_COLS` and :obj:`QUAL_OUT_COLS`
        as well as a ``"quantitative"`` column that contains a boolean
        determining whether the rwo belongs in the quantitative output
        file (``True``) or the qualitative output file (``False``).
    out_dir : path-like
        Path to output directory where ordinance database csv files
        should be written.
    """
    if db.empty:
        return

    out_dir = Path(out_dir)
    qual_db = db[~db["quantitative"]][QUAL_OUT_COLS]
    quant_db = db[db["quantitative"]][QUANT_OUT_COLS]
    qual_db.to_csv(out_dir / "qualitative_ordinances.csv", index=False)
    quant_db.to_csv(out_dir / "quantitative_ordinances.csv", index=False)


def _db_results(results, doc_info):
    """Extract results from doc attrs to DataFrame"""

    results["source"] = doc_info.get("source")
    results["ord_year"] = extract_ord_year_from_doc_attrs(doc_info)

    jurisdiction = doc_info["jurisdiction"]
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
        db.loc[db["adder"] > 250, "adder"] = None  # noqa: PLR2004
    return db


def _formatted_db(db):
    """Format DataFrame for output"""
    for col in _PARSED_COLS:
        if col not in db.columns:
            db[col] = None

    db["quantitative"] = db["quantitative"].astype("boolean").fillna(True)
    ord_rows = ordinances_bool_index(db)
    return db[ord_rows][_PARSED_COLS].reset_index(drop=True)


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
    """Summarize the run results into a formatted string

    Parameters
    ----------
    total_seconds : int or float
        Total number of seconds the run took to complete.
    total_cost : int or float
        Total cost of the run, in $.
    out_dir : path-like
        Path to output directory where the run results are saved.
    document_count : int
        Number of documents found during the run.

    Returns
    -------
    str
        Formatted string summarizing the run results.
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
        f"Number of documents found: {document_count}"
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
