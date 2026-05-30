"""COMPASS CLI search subcommand"""

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.theme import Theme

from compass._cli.common import setup_cli_logging
from compass.plugin import create_schema_based_one_shot_extraction_plugin
from compass.scripts.search import run_search, summary, write_search_report
from compass.utilities.io import load_config


@click.command(name="search")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to ordinance configuration JSON or JSON5 file. Only the "
    "search-related keys (``tech``, ``jurisdiction_fp``, "
    "``search_engines``, ``url_ignore_substrings``, "
    "``num_urls_to_check_per_jurisdiction``, "
    "``max_num_concurrent_browsers``) are read.",
)
@click.option(
    "-n",
    "--n-top-urls",
    "n_top_urls",
    type=int,
    default=None,
    help="Override the number of top URLs to retain per jurisdiction "
    "(``num_urls_to_check_per_jurisdiction``).",
)
@click.option(
    "-o",
    "--output",
    "output",
    type=click.Path(),
    default=None,
    help="Optional path to write the report. If omitted, the report is "
    "written to stdout.",
)
@click.option(
    "--output-format",
    "output_format",
    type=click.Choice(["json", "summary"], case_sensitive=False),
    default="json",
    show_default=True,
    help="Output representation for search results.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Show logs on stderr. Add extra libraries to get logs from by "
    "increasing the input (-v, -vv, -vvv).",
)
@click.option(
    "--plugin",
    "-p",
    required=False,
    default=None,
    help="One-shot plugin configuration to register before searching",
)
def search(config, n_top_urls, output, output_format, verbose, plugin):
    """Run only the search step and emit ranked URL results"""
    config_path = config
    config = load_config(config)

    if plugin is not None:
        create_schema_based_one_shot_extraction_plugin(
            config=plugin, tech=config["tech"]
        )

    if n_top_urls is not None:
        config["num_urls_to_check_per_jurisdiction"] = n_top_urls

    custom_theme = Theme({"logging.level.trace": "rgb(94,79,162)"})
    console = Console(theme=custom_theme, stderr=True)
    setup_cli_logging(
        console, verbose, log_level=config.get("log_level", "INFO")
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    report = loop.run_until_complete(
        run_search(config_path=config_path, **config)
    )
    if output_format == "json":
        if output is None:
            console.print_json(data=report)
        else:
            write_search_report(report, output)
        return

    text_report = summary(report)
    if output is None:
        console.print(text_report)
        return

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"{text_report}\n", encoding="utf-8")
