"""Ordinances CLI"""

import click
import asyncio
import logging
import multiprocessing
from pathlib import Path

import pyjson5
from rich.live import Live
from rich.theme import Theme
from rich.logging import RichHandler
from rich.console import Console

from compass import __version__
from compass.pb import COMPASS_PB
from compass.scripts.process import process_counties_with_openai
from compass.utilities.logs import AddLocationFilter


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def main(ctx):
    """Ordinance command line interface"""
    ctx.ensure_object(dict)


@main.command()
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to ordinance configuration JSON file. This file "
    "should contain any/all the arguments to pass to "
    ":func:`compass.scripts.process.process_counties_with_openai`.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Flag to show logging on the terminal. Default is not "
    "to show any logs on the terminal.",
)
@click.option(
    "-np",
    "--no_progress",
    is_flag=True,
    help="Flag to hide progress bars during processing.",
)
def process(config, verbose, no_progress):
    """Download and extract ordinances for a list of counties"""
    with Path(config).open(encoding="utf-8") as fh:
        config = pyjson5.decode_io(fh)

    custom_theme = Theme({"logging.level.trace": "rgb(94,79,162)"})
    console = Console(theme=custom_theme)

    if verbose:
        _setup_cli_logging(console, log_level=config.get("log_level", "INFO"))

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
        loop.run_until_complete(process_counties_with_openai(**config))
        return

    COMPASS_PB.console = console
    with Live(
        COMPASS_PB.group,
        console=console,
        refresh_per_second=20,
        transient=True,
    ):
        out = loop.run_until_complete(process_counties_with_openai(**config))

    print(f"✅ Scraping complete! Final database contains {len(out):,} rows.")
    COMPASS_PB.console = None


def _setup_cli_logging(console, log_level="INFO"):
    """Setup logging for CLI"""
    for lib in ["compass", "elm"]:
        logger = logging.getLogger(lib)
        handler = RichHandler(
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


if __name__ == "__main__":
    main(obj={})
