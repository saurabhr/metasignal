# Contributing to metasignal

<!-- start docs-include-contributing -->

Thanks for your interest in improving metasignal. This document covers how to set up a
development environment, run the checks that CI runs, and the conventions the project follows.
Participation in this project is governed by the
[Code of Conduct](https://github.com/saurabhr/metasignal/blob/main/CODE_OF_CONDUCT.md).

## Development setup

metasignal uses [uv](https://docs.astral.sh/uv/) for environment and dependency management, and
[nox](https://nox.thea.codes/) to run tasks in isolated sessions.

```bash
git clone https://github.com/saurabhr/metasignal.git
cd metasignal
uvx nox -s dev
source .venv/bin/activate
```

This creates a `.venv` with metasignal installed in editable mode plus the `dev` dependency group
(tests + docs).

## Running checks

| Task                                     | Command                 |
| ---------------------------------------- | ----------------------- |
| Run the test suite                       | `uvx nox -s tests`      |
| Run pre-commit hooks (lint/format)       | `uvx nox -s pre-commit` |
| Build the docs (strict mode, matches CI) | `uvx nox -s docs`       |
| Serve the docs with live reload          | `uvx nox -s docs-live`  |

Run `uvx nox` with no arguments to run the default sessions (`pre-commit`, `tests`, `docs`) — this
is what CI checks on every PR.

Install the pre-commit hooks locally so lint issues are caught before you push:

```bash
uv run pre-commit install
```

## Code style

- Linting/formatting is enforced by [ruff](https://docs.astral.sh/ruff/); see `[tool.ruff]` in
  [pyproject.toml](https://github.com/saurabhr/metasignal/blob/main/pyproject.toml) for the
  active rule set and per-file exceptions.
- Type hints are expected on public functions.
- Docstrings follow the Google style (see the `mkdocstrings` config in
  [mkdocs.yml](https://github.com/saurabhr/metasignal/blob/main/mkdocs.yml)).

## Tests

- Tests live in [tests/](https://github.com/saurabhr/metasignal/tree/main/tests) and run via
  `pytest`.
- Add or update tests for any behavior change, including bug fixes.
- New measures added to `stdpy.compute_all_measures` should include a comparison against the
  reference MATLAB implementation where one exists (see
  [matlab/](https://github.com/saurabhr/metasignal/tree/main/matlab) and
  [analysis/rahnev_comparison/](https://github.com/saurabhr/metasignal/tree/main/analysis/rahnev_comparison)).

## Documentation

Docs are built with MkDocs + Material and published to
[Read the Docs](https://metasignal.readthedocs.io/). If you change public API behavior, update:

- The relevant page under `docs/` (installation, usage, API reference, tutorials).
- [CHANGELOG.md](https://github.com/saurabhr/metasignal/blob/main/CHANGELOG.md) under an
  `## [Unreleased]` heading (see existing entries for format — this project follows
  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
  [CalVer](https://calver.org/)).

## Submitting a pull request

1. Fork the repo and create a branch from `main`.
2. Make your change, with tests and docs updated as above.
3. Run `uvx nox` locally and make sure it passes.
4. Open a PR describing the change and, for anything non-trivial, the motivation behind it.
5. CI (tests, lint, docs build) must pass before merge.

For larger changes (new measures, new estimation backends, breaking API changes), please open an
issue first to discuss the approach — see
[Future development](https://github.com/saurabhr/metasignal/blob/main/docs/roadmap.md) for areas
that are already planned or under consideration.

## Reporting bugs / requesting features

Use the [issue tracker](https://github.com/saurabhr/metasignal/issues). For bug reports, include
a minimal reproducible example (stim/resp/conf arrays are usually enough) and the metasignal
version (`python -c "from importlib.metadata import version; print(version('metasignal'))"`).

<!-- end docs-include-contributing -->
