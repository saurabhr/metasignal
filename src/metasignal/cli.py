"""Main CLI for metasignal."""

from importlib import metadata

import click
import numpy as np

from metasignal.analysis.group import MEASURE_LABELS
from metasignal.stdpy.compute_all import compute_all_measures


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True}
)
@click.version_option(metadata.version("metasignal"), "-v", "--version")
def cli() -> None:
    """Signal Detection Theory and metacognitive measures (pure Python)."""


@cli.command()
@click.option(
    "--stim", type=str, required=True, help="Comma-separated stimulus values (0/1)."
)
@click.option(
    "--resp", type=str, required=True, help="Comma-separated response values (0/1)."
)
@click.option(
    "--conf", type=str, required=True, help="Comma-separated confidence ratings (1 to n-ratings)."
)
@click.option(
    "--n-ratings", type=int, required=True, help="Number of confidence rating categories."
)
def compute(stim: str, resp: str, conf: str, n_ratings: int) -> None:
    """Compute all 20 SDT and metacognitive measures from trial-level data."""
    stim_arr = np.fromstring(stim, sep=",")
    resp_arr = np.fromstring(resp, sep=",")
    conf_arr = np.fromstring(conf, sep=",")

    result = compute_all_measures(stim_arr, resp_arr, conf_arr, n_ratings)

    click.echo(f"{'Measure':<20} {'Value':>10}")
    click.echo("-" * 32)
    for label, value in zip(MEASURE_LABELS, result):
        val_str = f"{value:.4f}" if not np.isnan(value) else "NaN"
        click.echo(f"{label:<20} {val_str:>10}")
