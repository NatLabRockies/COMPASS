"""COMPASS CLI extract subcommand"""

import click

from compass._cli.common import run_async_command
from compass.plugin import create_schema_based_one_shot_extraction_plugin
from compass.scripts.process import extract_collected_jurisdiction_documents
from compass.utilities.io import load_config


@click.command
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to an extraction configuration JSON or JSON5 file. This file "
    "should contain any/all the arguments to pass to :func:"
    "`compass.scripts.process.extract_collected_jurisdiction_documents`.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Show logs on the terminal.",
)
@click.option(
    "-np",
    "--no_progress",
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
def extract(config, verbose, no_progress, plugin):
    """Extract structured data from a saved collection manifest"""
    config = load_config(config)

    if plugin is not None:
        create_schema_based_one_shot_extraction_plugin(
            config=plugin, tech=config["tech"]
        )

    run_async_command(
        extract_collected_jurisdiction_documents,
        config,
        verbose,
        no_progress,
    )
