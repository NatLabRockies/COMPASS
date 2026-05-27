"""COMPASS CLI extract subcommand"""

import click

from compass._cli.common import run_async_command, OUT_DIR_POLICY_CHOICES
from compass.plugin import create_schema_based_one_shot_extraction_plugin
from compass.pipeline import ExtractionRequest
from compass.utilities.io import load_config


@click.command
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to an extraction configuration JSON or JSON5 file. This file "
    "should contain any/all the arguments to pass to "
    ":class:`~compass.pipeline.data_classes.ExtractionRequest`.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Show logs on the terminal.",
)
@click.option(
    "-np",
    "--no-progress",
    is_flag=True,
    help="Flag to hide progress bars during extraction.",
)
@click.option(
    "--plugin",
    "-p",
    required=False,
    default=None,
    help="One-shot plugin configuration to add to COMPASS before extraction",
)
@click.option(
    "--out-dir-exists",
    "-o",
    required=False,
    default=None,
    type=click.Choice(OUT_DIR_POLICY_CHOICES, case_sensitive=False),
    help="How to handle an existing output directory."
    " Choices: fail, increment, overwrite, prompt."
    " If omitted, prompts interactively when running in a terminal,"
    " or fails when running non-interactively (e.g. CI).",
)
def extract(config, verbose, no_progress, plugin, out_dir_exists):
    """Extract structured data from a saved collection manifest"""
    config = load_config(config)

    if plugin is not None:
        create_schema_based_one_shot_extraction_plugin(
            config=plugin, tech=config["tech"]
        )

    run_async_command(
        config, ExtractionRequest, verbose, no_progress, out_dir_exists
    )
