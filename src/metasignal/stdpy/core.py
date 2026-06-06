"""Core Signal Detection Theory (SDT) functions."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def compute_sdt_resp(
    stimulus: np.ndarray, response: np.ndarray
) -> tuple[float, float, float]:
    """Compute SDT parameters (d', c, ln_beta).

    Args:
        stimulus: Vector of 2 values (lower is noise, higher is stimulus).
        response: Vector with the same values as stimulus.

    Returns:
        tuple: (dprime, c, ln_beta)
    """
    s_min = np.min(stimulus)
    s_max = np.max(stimulus)

    n_s1 = np.sum(stimulus == s_min)
    n_s2 = np.sum(stimulus == s_max)

    if n_s1 == 0 or n_s2 == 0:
        raise ValueError(
            "stimulus must contain both classes; found only one class in input."
        )

    # Determine hit and FA rate
    hit_rate = np.sum((stimulus == s_max) & (response == s_max)) / n_s2
    fa_rate = np.sum((stimulus == s_min) & (response == s_max)) / n_s1

    # Correct for values of 0 or 1
    if hit_rate == 0:
        hit_rate = 0.5 / n_s2
    elif hit_rate == 1:
        hit_rate = 1 - 0.5 / n_s2

    if fa_rate == 0:
        fa_rate = 0.5 / n_s1
    elif fa_rate == 1:
        fa_rate = 1 - 0.5 / n_s1

    # Compute d' and criterion c
    zh = norm.ppf(hit_rate)
    zfa = norm.ppf(fa_rate)

    dprime = zh - zfa
    c = -0.5 * (zh + zfa)
    ln_beta = dprime * c

    return dprime, c, ln_beta


def trials_to_counts(
    stim_id: np.ndarray, response: np.ndarray, rating: np.ndarray, n_ratings: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert trial data to response counts for each stimulus category.

    Args:
        stim_id: 0=S1, 1=S2.
        response: 0=S1, 1=S2.
        rating: 1 to n_ratings.
        n_ratings: Total number of ratings.

    Returns:
        tuple: (nr_s1, nr_s2)
    """
    # Filter bad trials
    f = (
        (np.isin(stim_id, [0, 1]))
        & (np.isin(response, [0, 1]))
        & (rating >= 1)
        & (rating <= n_ratings)
    )
    stim_id_f = stim_id[f]
    response_f = response[f]
    rating_f = rating[f]

    nr_s1 = np.zeros(2 * n_ratings)
    nr_s2 = np.zeros(2 * n_ratings)

    # get tallies of "S1" rating responses for S1 and S2 stim
    for i in range(1, n_ratings + 1):
        nr_s1[i - 1] = np.sum(
            (stim_id_f == 0) & (response_f == 0) & (rating_f == n_ratings + 1 - i)
        )
        nr_s2[i - 1] = np.sum(
            (stim_id_f == 1) & (response_f == 0) & (rating_f == n_ratings + 1 - i)
        )

    # get tallies of "S2" rating responses for S1 and S2 stim
    for i in range(1, n_ratings + 1):
        nr_s1[i + n_ratings - 1] = np.sum(
            (stim_id_f == 0) & (response_f == 1) & (rating_f == i)
        )
        nr_s2[i + n_ratings - 1] = np.sum(
            (stim_id_f == 1) & (response_f == 1) & (rating_f == i)
        )

    return nr_s1, nr_s2
