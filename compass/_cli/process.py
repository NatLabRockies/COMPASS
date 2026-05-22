"""COMPASS CLI process subcommand"""

import click

from compass._cli.common import run_async_command
from compass.plugin import create_schema_based_one_shot_extraction_plugin
from compass.scripts.process import process_jurisdictions_with_openai
from compass.utilities.io import load_config


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
def process(config, verbose, no_progress, plugin):
    """Download and extract ordinances for a list of jurisdictions"""
    config = load_config(config)

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
