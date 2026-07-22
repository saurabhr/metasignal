from importlib import import_module, metadata

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
