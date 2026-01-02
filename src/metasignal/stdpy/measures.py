"""Standard meta-measures (AUC, Gamma, Phi)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm


def sdt_expect_conf(nr_s1: np.ndarray, nr_s2: np.ndarray) -> dict[str, Any]:
    """Compute expected counts based on SDT assumptions.

    Replicates MATLAB SDTexpectConf behavior.
    """
    n_ratings = len(nr_s1) // 2

    # Correct for empty cells
    if np.any(nr_s1 == 0) or np.any(nr_s2 == 0):
        nr_s1_c = nr_s1 + 1 / (2 * n_ratings)
        nr_s2_c = nr_s2 + 1 / (2 * n_ratings)
    else:
        nr_s1_c = nr_s1
        nr_s2_c = nr_s2

    sum1 = np.sum(nr_s1_c)
    sum2 = np.sum(nr_s2_c)

    # Cumulative sums for HR and FAR across criteria
    hr = np.cumsum(nr_s2_c[::-1]) / sum2
    far = np.cumsum(nr_s1_c[::-1]) / sum1

    # We want HR/FAR for all 2*nRatings-1 criteria
    hr_orig = hr[:-1][::-1]
    far_orig = far[:-1][::-1]

    # d' and c at the primary (middle) criterion
    t1_idx = n_ratings - 1
    dprime = norm.ppf(hr_orig[t1_idx]) - norm.ppf(far_orig[t1_idx])

    # c for all criteria
    c = -0.5 * (norm.ppf(hr_orig) + norm.ppf(far_orig))

    # Expected HR and FAR based on d' and c
    exp_far = 1 - norm.cdf(c, -dprime / 2, 1)
    exp_hr = 1 - norm.cdf(c, dprime / 2, 1)

    # Expected proportions
    exp_nr_s1 = np.diff(np.concatenate([[0], exp_far[::-1], [1]]))[::-1]
    exp_nr_s2 = np.diff(np.concatenate([[0], exp_hr[::-1], [1]]))[::-1]

    return {
        "nR_S1_exp": exp_nr_s1 * sum1,
        "nR_S2_exp": exp_nr_s2 * sum2,
        "nR_S1_act": nr_s1,
        "nR_S2_act": nr_s2,
        "dprime": dprime,
    }


def compute_type2_auc(nr_s1: np.ndarray, nr_s2: np.ndarray) -> float:
    """Compute Type-2 AUC from counts."""
    n_ratings = len(nr_s1) // 2

    # correct/incorrect counts
    counts_c = nr_s2[n_ratings:] + nr_s1[:n_ratings][::-1]
    counts_i = nr_s1[n_ratings:] + nr_s2[:n_ratings][::-1]

    # Type-2 HR and FAR
    hr2 = np.cumsum(counts_c[::-1]) / np.sum(counts_c)
    far2 = np.cumsum(counts_i[::-1]) / np.sum(counts_i)

    # Add (0,0) and sort
    hr2 = np.concatenate([[0], hr2])
    far2 = np.concatenate([[0], far2])

    # Trapezoidal integration
    auc = 0.0
    for i in range(len(hr2) - 1):
        auc += (far2[i + 1] - far2[i]) * (hr2[i + 1] + hr2[i]) / 2
    return auc


def compute_gamma(nr_s1: np.ndarray, nr_s2: np.ndarray) -> float:
    """Compute Goodman-Kruskal's gamma."""
    n_ratings = len(nr_s1) // 2
    counts_i = nr_s1[n_ratings:] + nr_s2[:n_ratings][::-1]
    counts_c = nr_s2[n_ratings:] + nr_s1[:n_ratings][::-1]

    table = np.vstack([counts_i, counts_c])

    concordant = 0.0
    discordant = 0.0
    for i in range(2):
        for j in range(n_ratings):
            if i == 0:
                concordant += table[i, j] * np.sum(table[1, j + 1 :])
                discordant += table[i, j] * np.sum(table[1, :j])
            else:
                concordant += table[i, j] * np.sum(table[0, :j])
                discordant += table[i, j] * np.sum(table[0, j + 1 :])

    if (concordant + discordant) == 0:
        return np.nan
    return (concordant - discordant) / (concordant + discordant)


def compute_phi(nr_s1: np.ndarray, nr_s2: np.ndarray) -> float:
    """Compute Phi coefficient (correlation between correctness and confidence)."""
    n_ratings = len(nr_s1) // 2
    counts_i = nr_s1[n_ratings:] + nr_s2[:n_ratings][::-1]
    counts_c = nr_s2[n_ratings:] + nr_s1[:n_ratings][::-1]

    correct = np.concatenate(
        [np.zeros(int(np.sum(counts_i))), np.ones(int(np.sum(counts_c)))]
    )
    conf = np.concatenate(
        [
            np.repeat(np.arange(1, n_ratings + 1), counts_i.astype(int)),
            np.repeat(np.arange(1, n_ratings + 1), counts_c.astype(int)),
        ]
    )

    if len(np.unique(correct)) < 2 or len(np.unique(conf)) < 2:
        return np.nan

    return float(np.corrcoef(correct, conf)[0, 1])
