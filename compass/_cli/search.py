"""COMPASS CLI search subcommand"""

from pathlib import Path

import click

from compass._cli.common import (
    CONFIG_OVERRIDE_CONTEXT_SETTINGS,
    run_async_command,
    OUT_DIR_POLICY_CHOICES,
)
from compass.pipeline import SearchRequest
from compass.plugin import create_schema_based_one_shot_extraction_plugin
from compass.scripts.search import summary, SEARCH_RESULT_MANIFEST_FILENAME
from compass.utilities.io import load_config


@click.command(context_settings=CONFIG_OVERRIDE_CONTEXT_SETTINGS)
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to a search configuration JSON or JSON5 file. This file "
    "should contain any/all the arguments to pass to "
    ":class:`~compass.pipeline.data_classes.SearchRequest`. Any top-level "
    "config may also be passed as an extra CLI option (using the syntax "
    "`--my_param=new`) to override the config value.",
)
@click.option("-v", "--verbose", count=True, help="Show logs on the terminal.")
@click.option(
    "-np",
    "--no-progress",
    is_flag=True,
    help="Hide progress bars during search.",
)
@click.option(
    "--plugin",
    "-p",
    default=None,
    help="One-shot plugin configuration to add to COMPASS before search",
)
@click.option(
    "--out-dir-exists",
    "-o",
    default=None,
    type=click.Choice(
        [*OUT_DIR_POLICY_CHOICES, "continue"], case_sensitive=False
    ),
    help="Policy for an existing output directory, as in collect/process.",
)
@click.option(
    "--summarize", "-s", is_flag=True, help="Summarize search results"
)
@click.pass_context
def search(
    ctx, config, verbose, no_progress, plugin, out_dir_exists, summarize
):
    """Save ranked search results and logs for later collection"""
    config = load_config(config)
    if plugin is not None:
        create_schema_based_one_shot_extraction_plugin(
            config=plugin, tech=config["tech"]
        )

    run_async_command(
        config,
        request_class=SearchRequest,
        verbose=verbose,
        no_progress=no_progress,
        out_dir_exists=out_dir_exists,
        override_args=ctx.args,
    )

    if summarize:
        report_fp = Path(config["out_dir"]) / SEARCH_RESULT_MANIFEST_FILENAME
        report = load_config(report_fp)
        text_report = summary(report)
        print()
        print(text_report)
