#!/usr/bin/env python3
"""Generate shared synthetic trial data for itmc vs. statConfR cross-validation.

Produces one long-format CSV with a "participant" column, where each
participant is a simulated SDT observer at a different (d', meta-d') and
n_ratings combination. Both the Python (metasignal.itmc) and R (statConfR)
sides read this exact same CSV, so any numeric difference in downstream
measures reflects a real implementation discrepancy, not a data difference.

Usage:
    python analysis/itmc_comparison/scripts/generate_shared_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from metasignal.stdpy.simulate import trialSimulation  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "shared_trials.csv"

# (d', meta-d', n_ratings, n_trials) per simulated participant.
# Spans: well-calibrated / over-/under-confident, low/high sensitivity,
# coarse/fine confidence scales, small/large samples.
SPECS = [
    (0.5, 0.5, 4, 400),
    (1.0, 1.0, 4, 400),
    (1.5, 1.5, 4, 400),
    (2.0, 2.0, 4, 400),
    (1.0, 0.5, 4, 400),   # under-confident (meta-d' < d')
    (1.0, 1.8, 4, 400),   # over-confident (meta-d' > d')
    (1.5, 1.5, 2, 400),   # coarse (binary) confidence
    (1.5, 1.5, 6, 400),   # fine confidence scale
    (1.5, 1.5, 4, 150),   # small sample
    (1.5, 1.5, 4, 1200),  # large sample
    (0.2, 0.2, 4, 400),   # near-chance sensitivity
    (2.5, 2.5, 4, 400),   # high sensitivity
]


def main() -> int:
    frames = []
    for i, (d, metad, n_ratings, n_trials) in enumerate(SPECS):
        rng = np.random.default_rng(1000 + i)
        df = trialSimulation(d=d, metad=metad, nTrials=n_trials, nRatings=n_ratings, rng=rng)
        out = pd.DataFrame(
            {
                "participant": f"sim{i:02d}_d{d}_meta{metad}_r{n_ratings}_n{n_trials}",
                "stimulus": df["Stimuli"].to_numpy(dtype=int),
                "response": df["Responses"].to_numpy(dtype=int),
                "rating": df["Confidence"].to_numpy(dtype=int),
            }
        )
        frames.append(out)

    combined = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT, index=False)
    print(f"Wrote {len(combined)} trials across {len(SPECS)} participants -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
