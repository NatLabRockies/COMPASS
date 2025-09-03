"""COMPASS CLI finalize subcommand"""

import json
from datetime import datetime

import click
from rich.theme import Theme
from rich.console import Console

from compass.utilities import Directories
from compass.utilities.location import Jurisdiction
from compass.utilities.parsing import load_config
from compass.utilities.finalize import save_run_meta, doc_infos_to_db, save_db
from compass.scripts.process import _initialize_model_params


@click.command
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to ordinance configuration JSON or JSON5 file. This file "
    "should contain any/all the arguments to pass to "
    ":func:`compass.scripts.process.process_jurisdictions_with_openai`."
    "The directory that this config points to will be finalized.",
)
def finalize(config):
    """Finalize a partially-completed COMPASS run"""
    config = load_config(config)
    tech = config["tech"]  # fail early

    dirs = Directories(
        config["out_dir"],
        config.get("log_dir"),
        config.get("clean_dir"),
        config.get("ordinance_file_dir"),
        config.get("jurisdiction_dbs_dir"),
    )
    meta_fp = dirs.out / "meta.json"
    jurisdictions_fp = dirs.out / "jurisdictions.json"

    if meta_fp.exists():
        msg = (
            f"Found meta file in output directory: '{meta_fp!s}'. "
            f"Will not finalize COMPASS run in this directory. "
        )
        raise click.ClickException(msg)

    if not jurisdictions_fp.exists():
        msg = (
            f"Could not find jurisdictions JSON file: '{jurisdictions_fp!s}'. "
            f"Cannot finalize COMPASS run in this directory. "
        )
        raise click.ClickException(msg)

    custom_theme = Theme({"logging.level.trace": "rgb(94,79,162)"})
    console = Console(theme=custom_theme)
    console.print(f"Finalizing COMPASS run in {dirs.out!s}...")

    models = _initialize_model_params(config.get("model", "gpt-4o-mini"))
    start_datetime = datetime.fromtimestamp(dirs.out.stat().st_ctime)
    end_datetime = datetime.fromtimestamp(jurisdictions_fp.stat().st_mtime)

    with jurisdictions_fp.open("r", encoding="utf-8") as fh:
        jurisdictions = json.load(fh)

    console.print("Compiling databases...")
    jurisdictions = jurisdictions.get("jurisdictions", [])

    _compile_db(jurisdictions, dirs)

    console.print("Saving meta info...")
    num_jurisdictions_searched = len(jurisdictions)
    num_jurisdictions_found = sum(
        jur.get("found", False) for jur in jurisdictions
    )
    total_cost = sum(jur.get("cost") or 0 for jur in jurisdictions)

    save_run_meta(
        dirs,
        tech,
        start_date=start_datetime,
        end_date=end_datetime,
        num_jurisdictions_searched=num_jurisdictions_searched,
        num_jurisdictions_found=num_jurisdictions_found,
        total_cost=total_cost,
        models=models,
    )
    console.print(f"✅ Finalized COMPASS run in {dirs.out!s}!")


def _compile_db(jurisdictions, dirs):
    """Merge all jurisdiction dbs into one"""
    all_doc_infos = []
    for jur_info in jurisdictions:
        if not jur_info.get("found", False):
            continue

        jurisdiction = Jurisdiction(
            subdivision_type=jur_info.get("jurisdiction_type"),
            state=jur_info.get("state"),
            county=jur_info.get("county"),
            subdivision_name=jur_info.get("subdivision"),
            code=jur_info.get("FIPS"),
        )
        ord_db_fp = (
            dirs.jurisdiction_dbs / f"{jurisdiction.full_name} Ordinances.csv"
        )
        if not ord_db_fp.exists():
            continue

        doc_info = jur_info.get("documents", [])
        if not doc_info:
            continue

        doc_info = doc_info[0]
        all_doc_infos.append(
            {
                "ord_db_fp": ord_db_fp,
                "source": doc_info.get("source"),
                "date": (
                    doc_info.get("effective_year"),
                    doc_info.get("effective_month"),
                    doc_info.get("effective_day"),
                ),
                "jurisdiction": jurisdiction,
            }
        )

    db, __ = doc_infos_to_db(all_doc_infos)
    save_db(db, dirs.out)
