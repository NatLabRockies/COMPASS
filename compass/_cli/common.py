"""Shared helpers for COMPASS CLI subcommands"""

import sys
import shutil
import asyncio
import logging
import warnings
import contextlib
import multiprocessing
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.theme import Theme

from compass.pb import COMPASS_PB
from compass.utilities.logs import AddLocationFilter
from compass.pipeline.coordinator import run_compass


OUT_DIR_POLICY_CHOICES = ["fail", "increment", "overwrite", "prompt"]


def run_async_command(
    config, request_class, verbose, no_progress, out_dir_exists=None
):
    """Run a COMPASS async command with shared CLI behavior

    Parameters
    ----------
    config : dict
        Configuration dictionary passed as keyword arguments to
        `command`. This mapping must include an ``"out_dir"`` entry,
        which is resolved according to `out_dir_exists` before command
        execution.
    request_class : callable
        The COMPASS request class to instantiate and pass to the command
        function, e.g.
        :class:`~compass.pipeline.data_classes.CollectionRequest`.
    verbose : int
        CLI verbosity level controlling which library loggers are shown
        in the console. Higher values enable logs from more underlying
        libraries.
    no_progress : bool
        Option to disable the Rich live progress display. If ``True``,
        the command is executed directly without attaching COMPASS
        progress bars.
    out_dir_exists : str, optional
        Policy controlling how an existing output directory should be
        handled. Supported values are ``"fail"``, ``"increment"``,
        ``"overwrite"``, and ``"prompt"``. If ``None``, the policy is
        chosen automatically based on whether the session is
        interactive. By default, ``None``.
    """
    custom_theme = Theme({"logging.level.trace": "rgb(94,79,162)"})
    console = Console(theme=custom_theme)

    setup_cli_logging(
        console, verbose, log_level=config.get("log_level", "INFO")
    )

    config["out_dir"] = _resolve_out_dir_conflict(
        config["out_dir"], out_dir_exists
    )

    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    request = request_class(**config)
    if no_progress:
        loop.run_until_complete(run_compass(request))
        return

    warnings.filterwarnings("ignore")

    COMPASS_PB.console = console
    live_display = Live(
        COMPASS_PB.group,
        console=console,
        refresh_per_second=20,
        transient=True,
    )
    with live_display:
        run_msg = loop.run_until_complete(run_compass(request))

    console.print(run_msg)
    COMPASS_PB.console = None


def setup_cli_logging(console, verbosity_level, log_level="INFO"):
    """[NOT PUBLIC API] Setup logging for CLI"""
    libs = []
    if verbosity_level >= 1:
        libs.append("compass")
    if verbosity_level >= 2:  # ruff:ignore[magic-value-comparison]
        libs.extend(("elm", "docling"))
    if verbosity_level >= 3:  # ruff:ignore[magic-value-comparison]
        libs.append("openai")
    if verbosity_level >= 4:  # ruff:ignore[magic-value-comparison]
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
    """Create the next available output directory with versioning"""
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
