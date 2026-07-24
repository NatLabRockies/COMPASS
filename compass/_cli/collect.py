"""COMPASS CLI collect subcommand"""

import click

from compass._cli.common import (
    CONFIG_OVERRIDE_CONTEXT_SETTINGS,
    OUT_DIR_POLICY_CHOICES,
    run_async_command,
)
from compass.plugin import create_schema_based_one_shot_extraction_plugin
from compass.pipeline import CollectionRequest
from compass.utilities.io import load_config


@click.command(context_settings=CONFIG_OVERRIDE_CONTEXT_SETTINGS)
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to a collection configuration JSON or JSON5 file. This file "
    "should contain any/all the arguments to pass to "
    ":class:`~compass.pipeline.data_classes.CollectionRequest`. Any top-level "
    "config may also be passed as an extra CLI option (using the syntax "
    "`--my_param=new`) to override the config value.",
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
@click.option(
    "--out-dir-exists",
    "-o",
    required=False,
    default=None,
    type=click.Choice(
        [*OUT_DIR_POLICY_CHOICES, "continue"], case_sensitive=False
    ),
    help="How to handle an existing output directory."
    " Choices: continue, fail, increment, overwrite, prompt."
    " If omitted, prompts interactively when running in a terminal,"
    " or fails when running non-interactively (e.g. CI).",
)
@click.pass_context
def collect(ctx, config, verbose, no_progress, plugin, out_dir_exists):
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
        request_class=CollectionRequest,
        verbose=verbose,
        no_progress=no_progress,
        out_dir_exists=out_dir_exists,
        override_args=ctx.args,
    )
