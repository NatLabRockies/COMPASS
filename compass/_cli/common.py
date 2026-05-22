"""Shared helpers for COMPASS CLI subcommands"""

import asyncio
import logging
import warnings
import contextlib
import multiprocessing

from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.theme import Theme

from compass.pb import COMPASS_PB
from compass.utilities.logs import AddLocationFilter


def run_async_command(command, config, verbose, no_progress):
    """Run a COMPASS async command with shared CLI behavior"""
    custom_theme = Theme({"logging.level.trace": "rgb(94,79,162)"})
    console = Console(theme=custom_theme)

    setup_cli_logging(
        console, verbose, log_level=config.get("log_level", "INFO")
    )

    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if no_progress:
        loop.run_until_complete(command(**config))
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
        run_msg = loop.run_until_complete(command(**config))

    console.print(run_msg)
    COMPASS_PB.console = None


def setup_cli_logging(console, verbosity_level, log_level="INFO"):
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
