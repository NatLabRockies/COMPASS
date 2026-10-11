"""COMPASS CLI collect subcommand"""

from pathlib import Path

import click

from compass._cli.common import (
    CONFIG_OVERRIDE_CONTEXT_SETTINGS,
    OUT_DIR_POLICY_CHOICES,
    run_async_command,
)
from compass.plugin import create_schema_based_one_shot_extraction_plugin
from compass.pipeline import CollectionRequest
from compass.pipeline.targets import parse_target_options
from compass.pipeline.collection.persistence import (
    COLLECTION_MANIFEST_FILENAME,
)
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
@click.option("-v", "--verbose", count=True, help="Show logs on the terminal.")
@click.option(
    "-np",
    "--no-progress",
    is_flag=True,
    help="Hide progress bars during collection.",
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
@click.option(
    "--target",
    "targets",
    multiple=True,
    help="Per-shard minimum METRIC=VALUE. Repeat for multiple requirements. "
    "Supports num_docs and completed_step_document_counts.STEP. "
    "Use --out-dir-exists continue for existing output directories.",
)
@click.pass_context
def collect(
    ctx, config, verbose, no_progress, plugin, out_dir_exists, targets
):
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

    policy = out_dir_exists.lower() if out_dir_exists else None
    targets = parse_target_options(targets)
    if targets and policy not in {None, "continue"}:
        msg = "--targets conflicts with --out-dir-exists"
        raise click.ClickException(msg)

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
        re_run_targets=targets,
    )

    if targets:
        report_fp = Path(config["out_dir"]) / COLLECTION_MANIFEST_FILENAME
        report = load_config(report_fp)
        if report["failed_targets"]:
            msg = f"Failed {len(report['failed_targets']):,d} targets"
            raise click.ClickException(msg)
