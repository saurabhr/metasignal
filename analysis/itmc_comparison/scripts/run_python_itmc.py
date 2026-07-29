#!/usr/bin/env python3
"""Compute itmc measures (Python, backend='statconfr') on the shared dataset.

Mirrors what analysis/rahnev_comparison does for stdpy vs. MATLAB, but for
itmc vs. the R statConfR package it was ported from.

Usage:
    python analysis/itmc_comparison/scripts/run_python_itmc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from metasignal.itmc import fit_group  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data" / "shared_trials.csv"
OUT_DIR = HERE / "results"


def _per_participant(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run fit_group once per participant so each call sees only that
    participant's own (consistent) rating scale -- avoids cross-participant
    scale-inference warnings/bias when n_ratings varies across participants."""
    rows = []
    for pid, sub in df.groupby("participant", sort=False):
        res = fit_group(
            sub,
            stimuli="stimulus",
            responses="response",
            confidence="rating",
            **kwargs,
        )
        res.insert(0, "participant", pid)
        rows.append(res)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    df = pd.read_csv(DATA)

    no_bc = _per_participant(df, backend="statconfr", bias_correction=False)
    no_bc.to_csv(OUT_DIR / "python_no_bias_correction.csv", index=False)

    bc = _per_participant(df, backend="statconfr", bias_correction=True, seed=42)
    bc.to_csv(OUT_DIR / "python_bias_corrected.csv", index=False)

    print(f"Wrote {len(no_bc)} rows -> python_no_bias_correction.csv")
    print(f"Wrote {len(bc)} rows -> python_bias_corrected.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
