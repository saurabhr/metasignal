"""Type-2 (metacognitive) SDT measures: AUC2, Gamma, Phi, DeltaConf."""

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
    multiplier = np.concatenate([np.arange(n_ratings, 0, -1), np.arange(1, n_ratings + 1)])
    correct_trials_s1 = np.concatenate([np.ones(n_ratings), np.zeros(n_ratings)])
    correct_trials_s2 = np.concatenate([np.zeros(n_ratings), np.ones(n_ratings)])

    correct_cells = np.concatenate([nr_s1[:n_ratings], nr_s2[n_ratings:]])
    incorrect_cells = np.concatenate([nr_s2[:n_ratings], nr_s1[n_ratings:]])

    total = np.sum(correct_cells) + np.sum(incorrect_cells)
    if total == 0:
        return np.nan

    av_acc = np.sum(correct_cells) / total
    av_conf = np.sum((correct_cells + incorrect_cells) * multiplier) / total

    numerator = np.sum(
        (multiplier - av_conf) * (correct_trials_s1 - av_acc) * nr_s1 +
        (multiplier - av_conf) * (correct_trials_s2 - av_acc) * nr_s2
    )

    den1 = np.sum((multiplier - av_conf)**2 * (nr_s1 + nr_s2))
    den2 = np.sum((correct_trials_s1 - av_acc)**2 * (nr_s1 + nr_s2[::-1]))
    denominator = np.sqrt(den1 * den2)

    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def compute_delta_conf(nr_s1: np.ndarray, nr_s2: np.ndarray) -> dict[str, float]:
    """Compute Delta Confidence and related expected measures.

    Args:
        nr_s1: Counts array for S1 stimulus responses
        nr_s2: Counts array for S2 stimulus responses

    Returns:
        Dictionary containing delta_conf, delta_conf_ratio, and delta_conf_diff.
    """
    n_ratings = len(nr_s1) // 2

    def _compute_delta(n_s1: np.ndarray, n_s2: np.ndarray) -> float:
        multiplier = np.concatenate(
            [np.arange(n_ratings, 0, -1), np.arange(1, n_ratings + 1)]
        )
        correct_cells = np.concatenate([n_s1[:n_ratings], n_s2[n_ratings:]])
        incorrect_cells = np.concatenate([n_s2[:n_ratings], n_s1[n_ratings:]])

        sum_c = np.sum(correct_cells)
        sum_i = np.sum(incorrect_cells)

        mean_conf_correct = np.sum(correct_cells * multiplier) / sum_c if sum_c > 0 else 0.0
        mean_conf_incorrect = np.sum(incorrect_cells * multiplier) / sum_i if sum_i > 0 else 0.0

        return float(mean_conf_correct - mean_conf_incorrect)

    # Actual delta conf
    delta_conf = _compute_delta(nr_s1, nr_s2)

    # Expected delta conf from SDT expectations
    sdt_expect = sdt_expect_conf(nr_s1, nr_s2)
    conf_diff_expected = _compute_delta(
        np.array(sdt_expect["nR_S1_exp"]),
        np.array(sdt_expect["nR_S2_exp"])
    )

    return {
        "delta_conf": delta_conf,
        "delta_conf_ratio": delta_conf / conf_diff_expected if conf_diff_expected != 0 else np.nan,
        "delta_conf_diff": delta_conf - conf_diff_expected,
    }
