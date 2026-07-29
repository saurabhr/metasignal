#!/usr/bin/env python3
"""Reproduce Dayan (2023)'s own hand-worked meta-I example (Table 1, p.397).

Dayan (2023) works through the full calculation of meta-I by hand for
subject 5, condition 3 of the Shekhar & Rahnev (2021) dataset, binarizing
confidence at a single threshold. He reports the exact trial counts (Table
1) and the resulting value, meta-I = 0.094 (also shown in Figure 1's table
as mI[2] for that subject/condition).

This script reconstructs those exact trial counts and checks that
metasignal.itmc.meta_I (backend='simple') reproduces the same value -- a
direct check against Dayan's own published number, not just against the R
statConfR port (see run_python_itmc.py / run_r_statconfr.R for that).

Usage:
    python analysis/itmc_comparison/scripts/check_dayan_table1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from metasignal.itmc import meta_I  # noqa: E402

# Table 1, Dayan (2023), p.397: subject 5, condition 3.
# rectitude (r) x confidence (c, binarized at threshold 0.72) contingency table.
N_INCORRECT_LOW, N_INCORRECT_HIGH = 138, 5
N_CORRECT_LOW, N_CORRECT_HIGH = 440, 415
PUBLISHED_META_I = 0.094  # Dayan's reported value (also Figure 1 table, mI[2] for sub 5, con 3)


def build_trials() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct stim/response/rating arrays with exactly Dayan's Table 1 counts.

    Only accuracy and rating matter for meta_I (backend='simple'); stimulus
    and response are constructed so accuracy == correct/incorrect as given.
    """
    stim = np.concatenate(
        [
            np.zeros(N_CORRECT_LOW + N_CORRECT_HIGH, dtype=int),
            np.zeros(N_INCORRECT_LOW + N_INCORRECT_HIGH, dtype=int),
        ]
    )
    resp = np.concatenate(
        [
            np.zeros(N_CORRECT_LOW + N_CORRECT_HIGH, dtype=int),
            np.ones(N_INCORRECT_LOW + N_INCORRECT_HIGH, dtype=int),
        ]
    )
    rating = np.concatenate(
        [
            np.ones(N_CORRECT_LOW, dtype=int),
            np.full(N_CORRECT_HIGH, 2, dtype=int),
            np.ones(N_INCORRECT_LOW, dtype=int),
            np.full(N_INCORRECT_HIGH, 2, dtype=int),
        ]
    )
    return stim, resp, rating


def main() -> int:
    stim, resp, rating = build_trials()
    n_total = N_INCORRECT_LOW + N_INCORRECT_HIGH + N_CORRECT_LOW + N_CORRECT_HIGH
    assert len(stim) == n_total == 998

    computed = meta_I(stim, resp, rating, backend="simple")
    matches = round(computed, 3) == PUBLISHED_META_I

    print(f"Trial counts (Dayan 2023, Table 1): "
          f"r=0,c=l:{N_INCORRECT_LOW}  r=0,c=h:{N_INCORRECT_HIGH}  "
          f"r=1,c=l:{N_CORRECT_LOW}  r=1,c=h:{N_CORRECT_HIGH}  (n={n_total})")
    print(f"metasignal.itmc.meta_I (backend='simple'): {computed:.6f}")
    print(f"Dayan (2023) published value:               {PUBLISHED_META_I}")
    print(f"Match (rounded to 3 d.p.): {matches}")

    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
