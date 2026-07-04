"""Group-level summary statistics over SDT measures.

Computes per-participant measures and aggregates them across a group,
following the within-subjects analysis approach common in metacognition
research (e.g. Rahnev, 2025).

Usage example::

    import numpy as np
    from metasignal.analysis import group_summary

    # Simulate 10 participants, each with 80 trials
    rng = np.random.default_rng(0)
    participants = [
        (
            rng.integers(0, 2, 80).astype(float),  # stim
            rng.integers(0, 2, 80).astype(float),  # resp
            rng.integers(1, 3, 80).astype(float),  # conf
        )
        for _ in range(10)
    ]

    summary = group_summary(participants, n_ratings=2)
    print(summary["mean"])    # shape (26,)
    print(summary["sem"])     # shape (26,)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from metasignal.stdpy.compute_all import compute_all_measures

# Friendly names for the 26-element output array (index → label)
MEASURE_LABELS: list[str] = [
    "meta_d",          # 0
    "AUC2",            # 1
    "gamma",           # 2
    "phi",             # 3
    "deltaConf",       # 4
    "M_ratio",         # 5
    "AUC2_ratio",      # 6
    "gamma_ratio",     # 7
    "phi_ratio",       # 8
    "deltaConf_ratio", # 9
    "M_diff",          # 10
    "AUC2_diff",       # 11
    "gamma_diff",      # 12
    "phi_diff",        # 13
    "deltaConf_diff",  # 14
    "metaNoise",       # 15
    "metaUncertainty", # 16
    "dprime",          # 17
    "c",               # 18
    "mean_conf",       # 19
    "logL",            # 20
    "AIC",             # 21
    "BIC",             # 22
    "AICc",            # 23
    "k",               # 24
    "n",               # 25
]
N_MEASURES = len(MEASURE_LABELS)


def group_summary(
    participants: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_ratings: int,
) -> dict[str, Any]:
    """Compute group-level descriptive statistics over the 26-measure array.

    Runs ``compute_all_measures`` for each participant and aggregates across
    the group, ignoring NaN values (participants where a measure could not
    be estimated are excluded from that measure's statistics).

    Args:
        participants: List of ``(stim, resp, conf)`` tuples, one per participant.
        n_ratings: Number of confidence rating categories.

    Returns:
        Dictionary with keys:

        - ``"individual"`` — ``np.ndarray`` of shape ``(n_participants, 26)``
        - ``"mean"`` — ``np.ndarray`` of shape ``(26,)``, nanmean across participants
        - ``"median"`` — ``np.ndarray`` of shape ``(26,)``
        - ``"sem"`` — ``np.ndarray`` of shape ``(26,)``, standard error of the mean
        - ``"n_valid"`` — ``np.ndarray`` of shape ``(26,)`` int, participants with
            non-NaN values per measure
        - ``"labels"`` — list of measure name strings (length 26)
    """
    n_participants = len(participants)
    individual = np.full((n_participants, N_MEASURES), np.nan)

    for i, (stim, resp, conf) in enumerate(participants):
        individual[i] = compute_all_measures(
            np.asarray(stim), np.asarray(resp), np.asarray(conf), n_ratings
        )

    n_valid = np.sum(~np.isnan(individual), axis=0).astype(int)
    mean = np.nanmean(individual, axis=0)
    median = np.nanmedian(individual, axis=0)
    # Require at least 2 valid values for a meaningful SEM; return NaN otherwise
    std = np.nanstd(individual, axis=0, ddof=1)
    sem = np.where(n_valid >= 2, std / np.sqrt(n_valid), np.nan)

    return {
        "individual": individual,
        "mean": mean,
        "median": median,
        "sem": sem,
        "n_valid": n_valid,
        "labels": MEASURE_LABELS,
    }
