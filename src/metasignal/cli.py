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


# ── compute ───────────────────────────────────────────────────────────────────

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

    if len(stim_arr) == 0 or len(resp_arr) == 0 or len(conf_arr) == 0:
        raise click.BadParameter(
            "Could not parse input — expected comma-separated numbers (e.g. 0,1,0,1).",
            param_hint="'--stim' / '--resp' / '--conf'",
        )
    if not (len(stim_arr) == len(resp_arr) == len(conf_arr)):
        raise click.UsageError(
            f"--stim ({len(stim_arr)}), --resp ({len(resp_arr)}), and --conf ({len(conf_arr)}) "
            "must all have the same number of values."
        )

    result = compute_all_measures(stim_arr, resp_arr, conf_arr, n_ratings)

    click.echo(f"{'Measure':<20} {'Value':>10}")
    click.echo("-" * 32)
    for label, value in zip(MEASURE_LABELS, result):
        val_str = f"{value:.4f}" if not np.isnan(value) else "NaN"
        click.echo(f"{label:<20} {val_str:>10}")


# ── bayes ─────────────────────────────────────────────────────────────────────

def _import_sdtbayes():
    """Lazy-import sdtbayes or surface a clear install hint."""
    try:
        import metasignal.sdtbayes as _mod
        return _mod
    except ImportError:
        raise click.ClickException(
            "The sdtbayes extra is not installed. Run:\n"
            "    pip install metasignal[sdtbayes]"
        )


def _read_csv(csv_path: str, *required_cols: str) -> "pd.DataFrame":
    """Read a CSV and validate that every required column is present."""
    try:
        import pandas as pd
    except ImportError:
        raise click.ClickException(
            "pandas is required. Run:\n    pip install metasignal[sdtbayes]"
        )
    df = pd.read_csv(csv_path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise click.UsageError(
            f"Column(s) {missing} not found in CSV. "
            f"Available columns: {df.columns.tolist()}"
        )
    return df


def _df_to_participants(
    df: "pd.DataFrame",
    participant_col: str,
    stim_col: str,
    resp_col: str,
    conf_col: str,
) -> list:
    return [
        (
            g[stim_col].to_numpy(dtype=int),
            g[resp_col].to_numpy(dtype=int),
            g[conf_col].to_numpy(dtype=int),
        )
        for _, g in df.groupby(participant_col, sort=False)
    ]


def _show_summary(fit, sdt, var_names_str: str | None) -> None:
    """Print convergence warning (if any) then the posterior summary table."""
    diag = sdt.convergence_diagnostics(fit)
    n_bad = int((~diag["converged"]).sum())
    if n_bad:
        click.secho(
            f"Warning: {n_bad} parameter(s) have R-hat > 1.01 — "
            "consider increasing --iter.",
            fg="yellow",
            err=True,
        )
    vn = [v.strip() for v in var_names_str.split(",")] if var_names_str else None
    summary = sdt.posterior_summary(fit, var_names=vn)
    click.echo("\nPosterior summary:")
    click.echo(summary.to_string())


@cli.group()
def bayes() -> None:
    """Bayesian hierarchical meta-d' models (requires metasignal[sdtbayes])."""


@bayes.command(name="two-stage")
@click.option(
    "--csv", "csv_path",
    type=click.Path(exists=True, dir_okay=False), required=True,
    help="Long-format CSV with one trial per row.",
)
@click.option("--n-ratings", type=int, required=True,
              help="Number of confidence rating categories.")
@click.option("--participant-col", default="participant", show_default=True,
              help="Column name for participant IDs.")
@click.option("--stim-col", default="stim", show_default=True,
              help="Column name for stimulus values (0/1).")
@click.option("--resp-col", default="resp", show_default=True,
              help="Column name for response values (0/1).")
@click.option("--conf-col", default="conf", show_default=True,
              help="Column name for confidence ratings.")
@click.option("--chains", default=4, show_default=True,
              help="Number of MCMC chains.")
@click.option("--iter", default=2000, show_default=True,
              help="Total iterations per chain (including warmup).")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--var-names", default=None,
              help="Comma-separated parameter names to show (default: all).")
def two_stage(
    csv_path: str, n_ratings: int,
    participant_col: str, stim_col: str, resp_col: str, conf_col: str,
    chains: int, iter: int, seed: int, var_names: str | None,
) -> None:
    """Two-stage Bayesian group-level M-ratio from a CSV of trial data.

    Stage 1 fits per-participant MLE meta-d'. Stage 2 fits a Bayesian
    hierarchical model over log M-ratio. Key parameters in the output:

    \b
      b_Intercept  — group mean log M-ratio  (exp → M-ratio)
      sigma        — between-subject SD on the log scale

    CSV must have one trial per row with columns for participant ID,
    stimulus (0/1), response (0/1), and confidence rating.
    """
    sdt = _import_sdtbayes()
    df = _read_csv(csv_path, participant_col, stim_col, resp_col, conf_col)
    participants = _df_to_participants(df, participant_col, stim_col, resp_col, conf_col)

    click.echo(f"Loaded {len(participants)} participants from '{csv_path}'.")
    click.echo("Fitting two-stage Bayesian model — this may take a minute...")

    fit, mle_df = sdt.fit_two_stage_group(
        participants, n_ratings=n_ratings,
        chains=chains, iter=iter, seed=seed,
    )

    click.echo("\nStage 1 — per-participant MLE estimates:")
    click.echo(mle_df[["participant", "dprime", "meta_da", "m_ratio"]].to_string(index=False))

    _show_summary(fit, sdt, var_names)


@bayes.command(name="compare")
@click.option(
    "--csv", "csv_path",
    type=click.Path(exists=True, dir_okay=False), required=True,
    help="Long-format CSV with one trial per row and a group column.",
)
@click.option("--n-ratings", type=int, required=True,
              help="Number of confidence rating categories.")
@click.option("--group-col", default="group", show_default=True,
              help="Column name for group labels (must have exactly 2 unique values).")
@click.option("--participant-col", default="participant", show_default=True,
              help="Column name for participant IDs.")
@click.option("--stim-col", default="stim", show_default=True,
              help="Column name for stimulus values (0/1).")
@click.option("--resp-col", default="resp", show_default=True,
              help="Column name for response values (0/1).")
@click.option("--conf-col", default="conf", show_default=True,
              help="Column name for confidence ratings.")
@click.option("--chains", default=4, show_default=True,
              help="Number of MCMC chains.")
@click.option("--iter", default=2000, show_default=True,
              help="Total iterations per chain (including warmup).")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--var-names", default=None,
              help="Comma-separated parameter names to show (default: all).")
def compare(
    csv_path: str, n_ratings: int, group_col: str,
    participant_col: str, stim_col: str, resp_col: str, conf_col: str,
    chains: int, iter: int, seed: int, var_names: str | None,
) -> None:
    """Two-stage Bayesian comparison of M-ratio between two groups.

    The CSV must have a group column with exactly two unique values. Groups
    are sorted alphabetically — the first is group A, the second is group B.
    Key parameter in the output:

    \b
      b_group1  — posterior difference in log M-ratio (group B − group A)
                  exp(b_group1) > 1 means group B has higher M-ratio

    CSV must have one trial per row with columns for group, participant ID,
    stimulus (0/1), response (0/1), and confidence rating.
    """
    sdt = _import_sdtbayes()
    df = _read_csv(csv_path, group_col, participant_col, stim_col, resp_col, conf_col)

    groups = sorted(df[group_col].unique())
    if len(groups) != 2:
        raise click.UsageError(
            f"--group-col '{group_col}' must have exactly 2 unique values, "
            f"found {len(groups)}: {groups}"
        )
    label_a, label_b = groups

    group_a = _df_to_participants(
        df[df[group_col] == label_a], participant_col, stim_col, resp_col, conf_col
    )
    group_b = _df_to_participants(
        df[df[group_col] == label_b], participant_col, stim_col, resp_col, conf_col
    )

    click.echo(
        f"Group '{label_a}': {len(group_a)} participants  |  "
        f"Group '{label_b}': {len(group_b)} participants"
    )
    click.echo("Fitting two-stage Bayesian comparison — this may take a minute...")

    fit, mle_df = sdt.fit_two_stage_comparison(
        group_a, group_b, n_ratings=n_ratings,
        chains=chains, iter=iter, seed=seed,
    )

    click.echo("\nStage 1 — per-participant MLE estimates:")
    mle_df["group_label"] = mle_df["group"].map({0: label_a, 1: label_b})
    click.echo(
        mle_df[["group_label", "participant", "dprime", "meta_da", "m_ratio"]]
        .to_string(index=False)
    )

    _show_summary(fit, sdt, var_names)
