"""Maximum Likelihood Estimation of meta-d'."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def fit_meta_d_mle(
    nr_s1: np.ndarray,
    nr_s2: np.ndarray,
    s: float = 1.0,
) -> dict[str, Any]:
    """Fit meta-d' using Maximum Likelihood Estimation.

    Args:
        nr_s1: Counts for S1 stimulus.
        nr_s2: Counts for S2 stimulus.
        s: Ratio of standard deviations (sd(S1)/sd(S2)). Default is 1.0.

    Returns:
        dict: Results of the fit.
    """
    if len(nr_s1) % 2 != 0:
        val_err = "input arrays must have an even number of elements"
        raise ValueError(val_err)
    if len(nr_s1) != len(nr_s2):
        val_err = "input arrays must have the same number of elements"
        raise ValueError(val_err)

    n_ratings = len(nr_s1) // 2
    n_criteria = 2 * n_ratings - 1

    # Calculate initial d1 and t1c
    rating_hr = np.cumsum(nr_s2[::-1]) / np.sum(nr_s2)
    rating_far = np.cumsum(nr_s1[::-1]) / np.sum(nr_s1)
    rating_hr = rating_hr[:-1][::-1]
    rating_far = rating_far[:-1][::-1]

    t1_index = n_ratings - 1

    eps = 1e-5
    rating_hr = np.clip(rating_hr, eps, 1 - eps)
    rating_far = np.clip(rating_far, eps, 1 - eps)

    d1 = (1 / s) * norm.ppf(rating_hr[t1_index]) - norm.ppf(rating_far[t1_index])
    t1c1 = (-1 / (1 + s)) * (
        norm.ppf(rating_hr[t1_index]) + norm.ppf(rating_far[t1_index])
    )

    constant_criterion_val = d1 * (t1c1 / d1) if d1 != 0 else 0

    c1 = (-1 / (1 + s)) * (norm.ppf(rating_hr) + norm.ppf(rating_far))
    t2c1_guess = np.delete(c1, t1_index)

    guess = np.concatenate([[d1], t2c1_guess - constant_criterion_val])

    lb = np.concatenate(
        [[-10], -20 * np.ones((n_criteria - 1) // 2), np.zeros((n_criteria - 1) // 2)]
    )
    ub = np.concatenate(
        [[10], np.zeros((n_criteria - 1) // 2), 20 * np.ones((n_criteria - 1) // 2)]
    )
    bounds = list(zip(lb, ub))

    def monotonicity_constraint(params: np.ndarray) -> np.ndarray:
        t2c_rel = params[1:]
        half = (n_criteria - 1) // 2
        full_c = np.concatenate([t2c_rel[:half], [0], t2c_rel[half:]])
        return np.diff(full_c) - 1e-5

    cons = {"type": "ineq", "fun": monotonicity_constraint}

    def neg_log_likelihood(params: np.ndarray) -> float:
        meta_d1_fit = params[0]
        t2c_rel = params[1:]

        half = (n_criteria - 1) // 2
        curr_constant_criterion = meta_d1_fit * (t1c1 / d1) if d1 != 0 else 0

        s1mu = -meta_d1_fit / 2 - curr_constant_criterion
        s1sd = 1.0
        s2mu = meta_d1_fit / 2 - curr_constant_criterion
        s2sd = s1sd / s

        t2c_full = np.concatenate([t2c_rel[:half], [0], t2c_rel[half:]])
        t2c_full_ext = np.concatenate([[-np.inf], t2c_full, [np.inf]])

        nc_rs1 = nr_s1[:n_ratings]
        ni_rs1 = nr_s2[:n_ratings]
        nc_rs2 = nr_s2[n_ratings:]
        ni_rs2 = nr_s1[n_ratings:]

        c_area_rs1 = norm.cdf(0, s1mu, s1sd)
        i_area_rs1 = norm.cdf(0, s2mu, s2sd)
        c_area_rs2 = 1 - norm.cdf(0, s2mu, s2sd)
        i_area_rs2 = 1 - norm.cdf(0, s1mu, s1sd)

        c_area_rs1 = max(c_area_rs1, 1e-10)
        i_area_rs1 = max(i_area_rs1, 1e-10)
        c_area_rs2 = max(c_area_rs2, 1e-10)
        i_area_rs2 = max(i_area_rs2, 1e-10)

        log_l = 0.0
        for i in range(n_ratings):
            pc = (
                norm.cdf(t2c_full_ext[i + 1], s1mu, s1sd)
                - norm.cdf(t2c_full_ext[i], s1mu, s1sd)
            ) / c_area_rs1
            pi = (
                norm.cdf(t2c_full_ext[i + 1], s2mu, s2sd)
                - norm.cdf(t2c_full_ext[i], s2mu, s2sd)
            ) / i_area_rs1

            pc = max(pc, 1e-10)
            pi = max(pi, 1e-10)
            log_l += nc_rs1[i] * np.log(pc) + ni_rs1[i] * np.log(pi)

            pc2 = (
                (1 - norm.cdf(t2c_full_ext[n_ratings + i], s2mu, s2sd))
                - (1 - norm.cdf(t2c_full_ext[n_ratings + i + 1], s2mu, s2sd))
            ) / c_area_rs2
            pi2 = (
                (1 - norm.cdf(t2c_full_ext[n_ratings + i], s1mu, s1sd))
                - (1 - norm.cdf(t2c_full_ext[n_ratings + i + 1], s1mu, s1sd))
            ) / i_area_rs2

            pc2 = max(pc2, 1e-10)
            pi2 = max(pi2, 1e-10)
            log_l += nc_rs2[i] * np.log(pc2) + ni_rs2[i] * np.log(pi2)

        return -log_l

    res = minimize(
        neg_log_likelihood,
        guess,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 1000},
    )
    meta_d1_final = res.x[0]
    scale = np.sqrt(2 / (1 + s**2)) * s
    da = scale * d1
    meta_da = scale * meta_d1_final

    return {
        "da": da,
        "meta_da": meta_da,
        "M_ratio": meta_da / da if da != 0 else np.nan,
        "M_diff": meta_da - da,
        "s": s,
        "logL": -res.fun,
        "success": res.success,
    }
