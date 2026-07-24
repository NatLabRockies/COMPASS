"""COMPASS CLI process subcommand"""

import click

from compass._cli.common import (
    CONFIG_OVERRIDE_CONTEXT_SETTINGS,
    OUT_DIR_POLICY_CHOICES,
    run_async_command,
)
from compass.plugin import create_schema_based_one_shot_extraction_plugin
from compass.pipeline import ProcessRequest
from compass.utilities.io import load_config


@click.command(context_settings=CONFIG_OVERRIDE_CONTEXT_SETTINGS)
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to ordinance configuration JSON or JSON5 file. This file "
    "should contain any/all the arguments to pass to "
    ":class:`~compass.pipeline.data_classes.ProcessRequest`. Any top-level "
    "config may also be passed as an extra CLI option (using the syntax "
    "`--my_param=new`) to override the config value.",
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
    "--no-progress",
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
@click.pass_context
def process(ctx, config, verbose, no_progress, plugin, out_dir_exists):
    """Download and extract ordinances for a list of jurisdictions"""
    config = load_config(config)

    if plugin is not None:
        create_schema_based_one_shot_extraction_plugin(
            config=plugin, tech=config["tech"]
        )

    run_async_command(
        config,
        request_class=ProcessRequest,
        verbose=verbose,
        no_progress=no_progress,
        out_dir_exists=out_dir_exists,
        override_args=ctx.args,
    )
