from importlib import import_module, metadata

import pytest
from click.testing import CliRunner

from metasignal.cli import cli

from .utils import run_command_in_shell


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
