"""COMPASS CLI process subcommand"""

import asyncio
import logging
import shutil
import sys
import warnings
import multiprocessing

from pathlib import Path

import click

from compass._cli.common import run_async_command
from compass.plugin import create_schema_based_one_shot_extraction_plugin
from compass.scripts.process import process_jurisdictions_with_openai
from compass.utilities.io import load_config
from compass.utilities.logs import AddLocationFilter


OUT_DIR_POLICY_CHOICES = ["fail", "increment", "overwrite", "prompt"]


@click.command
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to ordinance configuration JSON or JSON5 file. This file "
    "should contain any/all the arguments to pass to "
    ":func:`compass.scripts.process.process_jurisdictions_with_openai`.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Show logs on the terminal. Add extra libraries to get logs from by "
    "increasing the input (-v, -vv, -vvv). Does not affect log level, which "
    "is controlled via the config input.",
)
@click.option(
    "-np",
    "--no_progress",
    is_flag=True,
    help="Flag to hide progress bars during processing.",
)
@click.option(
    "--plugin",
    "-p",
    required=False,
    default=None,
    help="One-shot plugin configuration to add to COMPASS before processing",
)
@click.option(
    "--out_dir_exists",
    required=False,
    default=None,
    type=click.Choice(OUT_DIR_POLICY_CHOICES, case_sensitive=False),
    help="How to handle an existing output directory."
    " Choices: fail, increment, overwrite, prompt."
    " If omitted, prompts interactively when running in a terminal,"
    " or fails when running non-interactively (e.g. CI).",
)
def process(config, verbose, no_progress, plugin, out_dir_exists):
    """Download and extract ordinances for a list of jurisdictions"""
    config = load_config(config)

    config["out_dir"] = _resolve_out_dir_conflict(
        config["out_dir"], out_dir_exists
    )

    if plugin is not None:
        create_schema_based_one_shot_extraction_plugin(
            config=plugin, tech=config["tech"]
        )

    run_async_command(
        process_jurisdictions_with_openai,
        config,
        verbose,
        no_progress,
    )
