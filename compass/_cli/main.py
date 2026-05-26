"""COMPASS main CLI entrypoint"""

import click

from compass import __version__
from compass._cli.process import process
from compass._cli.finalize import finalize
from compass._cli.search_only import search_only


@click.group()
@click.version_option(version=__version__)
@click.pass_context
def main(ctx):
    """COMPASS command line interface"""
    ctx.ensure_object(dict)


main.add_command(process)
main.add_command(finalize)
main.add_command(search_only)


if __name__ == "__main__":
    main(obj={})
