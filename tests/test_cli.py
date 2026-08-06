from importlib import import_module, metadata

import numpy as np
import pytest
from click.testing import CliRunner

from metasignal.cli import cli

from .utils import run_command_in_shell

INLINE_ARGS = [
    "compute",
    "--stim", "0,1,0,1,0,1,0,1",
    "--resp", "0,1,0,0,0,1,1,1",
    "--conf", "4,4,3,1,3,4,2,3",
    "--n-ratings", "4",
]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_main_module() -> None:
    """Exercise (most of) the code in the `__main__` module."""
    import_module("metasignal.__main__")


def test_run_as_module() -> None:
    """Is the script runnable as a Python module?"""
    result = run_command_in_shell("python -m metasignal --help")
    assert result.exit_code == 0


def test_run_as_executable() -> None:
    """Is the script installed (as a `console_script`) and runnable as an executable?"""
    import shutil
    import os
    # The console_script may be installed outside the default PATH (e.g. ~/.local/bin).
    # Resolve the full path so the test works regardless of shell PATH configuration.
    exe = shutil.which("metasignal") or os.path.expanduser("~/.local/bin/metasignal")
    result = run_command_in_shell(f"{exe} --help")
    assert result.exit_code == 0


def test_version_runner(runner: CliRunner) -> None:
    """Does `--version` display the correct version?"""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert (
        result.output
        == f"cli, version {metadata.version('metasignal')}\n"
    )


def test_compute_inline(runner: CliRunner) -> None:
    """`compute` works with inline --stim/--resp/--conf (the original mode)."""
    result = runner.invoke(cli, INLINE_ARGS)
    assert result.exit_code == 0
    assert "meta_d" in result.output
    assert "dprime" in result.output


def test_compute_csv_matches_inline(tmp_path, runner: CliRunner) -> None:
    """`compute --csv` on the same trials gives the same output as inline."""
    csv_path = tmp_path / "trials.csv"
    csv_path.write_text(
        "stim,resp,conf\n"
        "0,0,4\n1,1,4\n0,0,3\n1,0,1\n0,0,3\n1,1,4\n0,1,2\n1,1,3\n"
    )
    inline = runner.invoke(cli, INLINE_ARGS)
    from_csv = runner.invoke(
        cli, ["compute", "--csv", str(csv_path), "--n-ratings", "4"]
    )
    assert from_csv.exit_code == 0
    assert from_csv.output == inline.output


def test_compute_csv_custom_column_names(tmp_path, runner: CliRunner) -> None:
    """`compute --csv` respects --stim-col/--resp-col/--conf-col."""
    csv_path = tmp_path / "trials.csv"
    csv_path.write_text(
        "stimulus,response,confidence\n0,0,4\n1,1,4\n0,0,3\n1,0,1\n"
    )
    result = runner.invoke(
        cli,
        [
            "compute", "--csv", str(csv_path),
            "--stim-col", "stimulus", "--resp-col", "response", "--conf-col", "confidence",
            "--n-ratings", "4",
        ],
    )
    assert result.exit_code == 0
    assert "meta_d" in result.output


def test_compute_csv_and_inline_conflict(tmp_path, runner: CliRunner) -> None:
    """Passing both --csv and --stim/--resp/--conf is a usage error."""
    csv_path = tmp_path / "unused.csv"
    csv_path.write_text("stim,resp,conf\n0,0,1\n")
    result = runner.invoke(
        cli,
        ["compute", "--csv", str(csv_path), "--stim", "0,1", "--resp", "0,1",
         "--conf", "1,2", "--n-ratings", "2"],
    )
    assert result.exit_code != 0
    assert "not both" in result.output


def test_compute_neither_csv_nor_inline(runner: CliRunner) -> None:
    """Passing neither --csv nor --stim/--resp/--conf is a usage error."""
    result = runner.invoke(cli, ["compute", "--n-ratings", "4"])
    assert result.exit_code != 0
    assert "Provide --csv" in result.output


def _itmc_csv(tmp_path, n_participants: int = 2):
    """Write a multi-participant trials CSV via trialSimulation, return its path."""
    from metasignal.stdpy.simulate import trialSimulation

    rows = []
    for i in range(n_participants):
        df = trialSimulation(d=1.5, metad=1.2, nTrials=300, rng=np.random.default_rng(i))
        rows.append(
            {
                "participant": f"s{i}",
                "stim": df["Stimuli"].astype(int).tolist(),
                "resp": df["Responses"].astype(int).tolist(),
                "conf": df["Confidence"].astype(int).tolist(),
            }
        )
    csv_path = tmp_path / "itmc_trials.csv"
    lines = ["participant,stim,resp,conf"]
    for r in rows:
        for stim, resp, conf in zip(r["stim"], r["resp"], r["conf"]):
            lines.append(f"{r['participant']},{stim},{resp},{conf}")
    csv_path.write_text("\n".join(lines) + "\n")
    return csv_path


def test_itmc_csv(tmp_path, runner: CliRunner) -> None:
    """`itmc` reports one row per participant with informative meta_I."""
    csv_path = _itmc_csv(tmp_path)
    result = runner.invoke(cli, ["itmc", "--csv", str(csv_path)])
    assert result.exit_code == 0
    assert "meta_I" in result.output
    assert "s0" in result.output
    assert "s1" in result.output


def test_itmc_matches_python_api(tmp_path, runner: CliRunner) -> None:
    """`itmc` output matches calling `estimate_meta_I` directly."""
    from metasignal.itmc import estimate_meta_I
    import pandas as pd

    csv_path = _itmc_csv(tmp_path)
    expected = estimate_meta_I(
        pd.read_csv(csv_path),
        stimulus_col="stim", response_col="resp", rating_col="conf",
        participant_col="participant",
    )

    result = runner.invoke(cli, ["itmc", "--csv", str(csv_path)])
    assert result.exit_code == 0
    assert result.output == expected.to_string(index=False) + "\n"


def test_itmc_custom_column_names(tmp_path, runner: CliRunner) -> None:
    """`itmc` respects --participant-col/--stim-col/--resp-col/--conf-col."""
    csv_path = tmp_path / "itmc_custom.csv"
    csv_path.write_text(_itmc_csv(tmp_path).read_text().replace(
        "participant,stim,resp,conf", "subj,stimulus,response,confidence"
    ))
    result = runner.invoke(
        cli,
        [
            "itmc", "--csv", str(csv_path),
            "--participant-col", "subj", "--stim-col", "stimulus",
            "--resp-col", "response", "--conf-col", "confidence",
        ],
    )
    assert result.exit_code == 0
    assert "meta_I" in result.output


def test_itmc_backend_statconfr(tmp_path, runner: CliRunner) -> None:
    """`itmc --backend statconfr` runs and produces informative output."""
    csv_path = _itmc_csv(tmp_path, n_participants=1)
    result = runner.invoke(cli, ["itmc", "--csv", str(csv_path), "--backend", "statconfr"])
    assert result.exit_code == 0
    assert "RMI" in result.output


def test_itmc_missing_csv_column(tmp_path, runner: CliRunner) -> None:
    """`itmc` gives a clear usage error when a required column is absent."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("participant,stim,resp\ns1,0,0\n")
    result = runner.invoke(cli, ["itmc", "--csv", str(csv_path)])
    assert result.exit_code != 0
    assert "not found in CSV" in result.output


def _sdtr_csv(tmp_path):
    """Write the Macho (2020) Ch. 5.1.1 Yes/No golden-example data as trial-level rows."""
    import numpy as np

    counts = np.array([[1780, 763], [883, 1025]])
    lines = ["participant,signal,response"]
    for signal_idx in (0, 1):
        for cat_idx, n in enumerate(counts[signal_idx], start=1):
            lines.extend(f"p1,{signal_idx},{cat_idx}" for _ in range(int(n)))
    csv_path = tmp_path / "sdtr_trials.csv"
    csv_path.write_text("\n".join(lines) + "\n")
    return csv_path


def test_sdtr_csv_matches_golden_values(tmp_path, runner: CliRunner) -> None:
    """`sdtr` on the manual's own Yes/No example recovers its published values."""
    csv_path = _sdtr_csv(tmp_path)
    result = runner.invoke(cli, ["sdtr", "--csv", str(csv_path), "--restriction", "equalvar"])
    assert result.exit_code == 0
    assert "mean_1" in result.output
    assert "0.617699" in result.output
    assert "0.524287" in result.output


def test_sdtr_custom_column_names(tmp_path, runner: CliRunner) -> None:
    """`sdtr` respects --participant-col/--signal-col/--response-col."""
    csv_path = tmp_path / "sdtr_custom.csv"
    csv_path.write_text(_sdtr_csv(tmp_path).read_text().replace(
        "participant,signal,response", "subj,stim_class,resp_cat"
    ))
    result = runner.invoke(
        cli,
        [
            "sdtr", "--csv", str(csv_path),
            "--participant-col", "subj", "--signal-col", "stim_class",
            "--response-col", "resp_cat", "--restriction", "equalvar",
        ],
    )
    assert result.exit_code == 0
    assert "mean_1" in result.output


def test_sdtr_missing_csv_column(tmp_path, runner: CliRunner) -> None:
    """`sdtr` gives a clear usage error when a required column is absent."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("participant,signal\np1,0\n")
    result = runner.invoke(cli, ["sdtr", "--csv", str(csv_path)])
    assert result.exit_code != 0
    assert "not found in CSV" in result.output
