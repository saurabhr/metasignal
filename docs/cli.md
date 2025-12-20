# CLI Reference

This page lists the `--help` for `metasignal`.

## metasignal

Running `metasignal --help` or `python -m metasignal --help` shows a list of all of the available options and arguments:

<!-- [[[cog
import cog
from metasignal import cli
from click.testing import CliRunner
result = CliRunner().invoke(cli.cli, ["--help"], terminal_width=88)
help = result.output.replace("Usage: cli", "Usage: metasignal")
cog.outl(f"\n```sh\nmetasignal --help\n{help.rstrip()}\n```\n")
]]] -->
<!-- [[[end]]] -->
