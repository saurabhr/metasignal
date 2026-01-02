"""Main CLI for metasignal."""

from importlib import metadata

import click
import numpy as np

from metasignal import MetaSignal


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True}
)
@click.version_option(metadata.version("metasignal"), "-v", "--version")
def cli() -> None:
    """SDT using python."""


@cli.command()
def test_matlab() -> None:
    """Test the MATLAB engine connection."""
    click.echo("Starting MATLAB engine...")
    try:
        ms = MetaSignal()
        click.echo("MATLAB engine started successfully.")

        # Test a simple call with dummy data
        stim = np.array([0, 1, 0, 1])
        resp = np.array([0, 1, 1, 0])
        conf = np.array([1, 2, 2, 1])
        n_ratings = 2

        click.echo(
            f"Dummy data: stim={stim}, resp={resp}, conf={conf}, n_ratings={n_ratings}"
        )

        ms.stop()
        click.echo("MATLAB engine stopped.")
    except Exception as err:
        click.echo(f"Error: {err}")
        raise click.Abort from err


@cli.command()
@click.option(
    "--stim", type=str, required=True, help="Comma-separated stimulus values (0/1)."
)
@click.option(
    "--resp", type=str, required=True, help="Comma-separated response values (0/1)."
)
@click.option(
    "--conf", type=str, required=True, help="Comma-separated confidence values."
)
@click.option(
    "--n-ratings", type=int, required=True, help="Number of rating categories."
)
def compute(stim: str, resp: str, conf: str, n_ratings: int) -> None:
    """Compute meta-signal measures from comma-separated input."""
    try:
        stim_arr = np.fromstring(stim, sep=",")
        resp_arr = np.fromstring(resp, sep=",")
        conf_arr = np.fromstring(conf, sep=",")

        ms = MetaSignal()
        result = ms.compute_all_measures(stim_arr, resp_arr, conf_arr, n_ratings)
        click.echo(f"Computed measures: {result}")
        ms.stop()
    except Exception as err:
        click.echo(f"Error: {err}")
        raise click.Abort from err
