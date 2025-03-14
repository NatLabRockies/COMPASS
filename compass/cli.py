"""Ordinances CLI"""

import sys
import click
import asyncio
import logging
import multiprocessing
from pathlib import Path

import pyjson5

from compass import __version__
from compass.scripts.process import process_counties_with_openai


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
    ":func:`compass.process.process_counties_with_openai`.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Flag to show logging on the terminal. Default is not "
    "to show any logs on the terminal.",
)
def process(config, verbose):
    """Download and extract ordinances for a list of counties"""
    with Path(config).open(encoding="utf-8") as fh:
        config = pyjson5.decode_io(fh)

    if verbose:
        for lib in ["compass", "elm"]:
            logger = logging.getLogger(lib)
            logger.addHandler(logging.StreamHandler(stream=sys.stdout))
            logger.setLevel(config.get("log_level", "INFO"))

    # Need to set start method to "spawn" instead of "fork" for unix
    # systems. If this call is not present, software hangs when process
    # pool executor is launched.
    # More info here: https://stackoverflow.com/a/63897175/20650649
    multiprocessing.set_start_method("spawn")

    # asyncio.run(...) doesn't throw exceptions correctly for some
    # reason...
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(process_counties_with_openai(**config))


if __name__ == "__main__":
    main(obj={})
