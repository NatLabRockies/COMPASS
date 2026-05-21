"""COMPASS CLI process subcommand"""

import asyncio
import logging
import shutil
import sys
import warnings
import multiprocessing

from pathlib import Path

import click
from rich.live import Live
from rich.theme import Theme
from rich.logging import RichHandler
from rich.console import Console

from compass.pb import COMPASS_PB
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

    custom_theme = Theme({"logging.level.trace": "rgb(94,79,162)"})
    console = Console(theme=custom_theme)

    _setup_cli_logging(
        console, verbose, log_level=config.get("log_level", "INFO")
    )

    # Need to set start method to "spawn" instead of "fork" for unix
    # systems. If this call is not present, software hangs when process
    # pool executor is launched.
    # More info here: https://stackoverflow.com/a/63897175/20650649
    multiprocessing.set_start_method("spawn")

    # asyncio.run(...) doesn't throw exceptions correctly for some
    # reason...
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if no_progress:
        loop.run_until_complete(process_jurisdictions_with_openai(**config))
        return

    # warnings will be logged to file (and terminal if verbose >= 1)
    warnings.filterwarnings("ignore")

    COMPASS_PB.console = console
    live_display = Live(
        COMPASS_PB.group,
        console=console,
        refresh_per_second=20,
        transient=True,
    )
    with live_display:
        run_msg = loop.run_until_complete(
            process_jurisdictions_with_openai(**config)
        )

    console.print(run_msg)
    COMPASS_PB.console = None


def _setup_cli_logging(console, verbosity_level, log_level="INFO"):
    """Setup logging for CLI"""
    libs = []
    if verbosity_level >= 1:
        libs.append("compass")
    if verbosity_level >= 2:  # noqa: PLR2004
        libs.extend(("elm", "docling"))
    if verbosity_level >= 3:  # noqa: PLR2004
        libs.append("openai")
    if verbosity_level >= 4:  # noqa: PLR2004
        libs.extend(("networkx", "pytesseract", "pdf2image", "pdftotext"))

    for lib in libs:
        logger = logging.getLogger(lib)
        handler = RichHandler(
            level=log_level,
            console=console,
            rich_tracebacks=True,
            omit_repeated_times=True,
            markup=True,
        )
        fmt = logging.Formatter(
            fmt="[[magenta]%(location)s[/magenta]]: %(message)s",
            defaults={"location": "main"},
        )
        handler.setFormatter(fmt)
        handler.addFilter(AddLocationFilter())
        logger.addHandler(handler)
        logger.setLevel(log_level)


def _resolve_out_dir_conflict(out_dir, policy):
    """Handle existing output directory using the selected policy"""
    out_dir = Path(out_dir)
    policy = _resolve_out_dir_policy(policy)

    if not out_dir.exists() or policy == "fail":
        return out_dir

    if policy == "increment":
        new_out_dir = _next_versioned_directory(out_dir)
        click.echo(
            "Output directory exists. "
            f"Using incremented directory: {new_out_dir!s}"
        )
        return new_out_dir

    if policy == "overwrite":
        click.echo(f"Overwriting existing output directory: {out_dir!s}")
        shutil.rmtree(out_dir)
        return out_dir

    if policy == "prompt":
        return _resolve_prompt_out_dir_conflict(out_dir)

    msg = (
        f"Unknown out_dir_exists policy '{policy}'. "
        f"Supported values: {OUT_DIR_POLICY_CHOICES}."
    )
    raise click.ClickException(msg)


def _next_versioned_directory(out_dir):
    """
    Create the next available output directory suffix with
    versioning
    """
    idx = 2
    max_idx = 1_000_000
    while idx <= max_idx:
        candidate = out_dir.parent / f"{out_dir.name}_v{idx}"
        if not candidate.exists():
            return candidate
        idx += 1

    msg = (
        f"Unable to find an available versioned directory for '{out_dir!s}' "
        f"up to suffix _v{max_idx}."
    )
    raise click.ClickException(msg)


def _resolve_out_dir_policy(policy):
    """Resolve output directory policy from explicit input

    Falls back to terminal mode defaults when no policy is set.
    """
    if policy is not None:
        return policy.lower()
    if sys.stdin.isatty():
        return "prompt"
    return "fail"


def _resolve_prompt_out_dir_conflict(out_dir):
    """Handle interactive prompt flow for existing output directory"""
    if not sys.stdin.isatty():
        msg = (
            "Cannot use out_dir_exists='prompt' in non-interactive mode. "
            "Use one of: fail, increment, overwrite."
        )
        raise click.ClickException(msg)

    create_incremented = click.confirm(
        f"Output directory '{out_dir!s}' already exists. "
        "Create a new incremented directory automatically?",
        default=True,
    )
    if create_incremented:
        new_out_dir = _next_versioned_directory(out_dir)
        click.echo(f"Using incremented directory: {new_out_dir!s}")
        return new_out_dir

    overwrite = click.confirm(
        f"Overwrite '{out_dir!s}' by deleting it and continuing?",
        default=False,
    )
    if overwrite:
        click.echo(f"Overwriting existing output directory: {out_dir!s}")
        shutil.rmtree(out_dir)
        return out_dir

    msg = (
        "Run cancelled. Please update out_dir in config, or rerun with "
        "--out_dir_exists increment/overwrite."
    )
    raise click.ClickException(msg)
