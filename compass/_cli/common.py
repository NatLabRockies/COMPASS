"""Shared helpers for COMPASS CLI subcommands"""

import logging

from rich.logging import RichHandler

from compass.utilities.logs import AddLocationFilter


def setup_cli_logging(console, verbosity_level, log_level="INFO"):
    """Attach a Rich log handler to selected libraries

    Parameters
    ----------
    console : rich.console.Console
        Console instance used by the Rich log handler.
    verbosity_level : int
        Number of ``-v`` flags supplied on the command line. Each
        increment opts an additional set of libraries into terminal
        logging.
    log_level : str, optional
        Log level applied to each attached library logger and handler.
        By default, ``"INFO"``.
    """
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
