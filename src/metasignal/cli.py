"""Main CLI for metasignal."""

from importlib import metadata

import click


@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True}
)
@click.argument("input_", metavar="INPUT")
@click.option(
    "-r",
    "--reverse",
    is_flag=True,
    help="Reverse the input.",
)
@click.version_option(
    metadata.version("metasignal"), "-v", "--version"
)
def cli(input_: str, *, reverse: bool = False) -> None:
    """Repeat the input.

    SDT using python.
    """
    click.echo(input_ if not reverse else input_[::-1])
