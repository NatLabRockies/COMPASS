"""COMPASS CLI collect subcommand"""

import click

from compass._cli.common import run_async_command
from compass.plugin import create_schema_based_one_shot_extraction_plugin
from compass.pipeline import CollectionRequest
from compass.utilities.io import load_config


@click.command
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to a collection configuration JSON or JSON5 file. This file "
    "should contain any/all the arguments to pass to "
    ":class:`~compass.pipeline.data_classes.CollectionRequest`.",
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
    help="Flag to hide progress bars during collection.",
)
@click.option(
    "--plugin",
    "-p",
    required=False,
    default=None,
    help="One-shot plugin configuration to add to COMPASS before collection",
)
def collect(config, verbose, no_progress, plugin):
    """Collect ordinance documents for a list of jurisdictions

    This command runs the "first half" (i.e. the collection portion) of
    the COMPASS pipeline on for a set of jurisdictions. It finds and
    parses the documents but does not do any filtering or validation. As
    such, it does not require an LLM endpoint and thus can be
    parallelized and scaled without worrying about rate limits. The
    output is a manifest of parsed documents that can be passed to the
    extraction command.
    """
    config = load_config(config)

    if plugin is not None:
        create_schema_based_one_shot_extraction_plugin(
            config=plugin, tech=config["tech"]
        )

    run_async_command(
        config,
        CollectionRequest,
        verbose,
        no_progress,
        out_dir_exists="continue",
    )
