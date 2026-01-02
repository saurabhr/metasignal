"""Meta-uncertainty model implementation."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, lognorm


def get_llh_choice(stim_value: np.ndarray, model_params: np.ndarray) -> np.ndarray:
    """Compute likelihood of each response alternative.

    Replicates MATLAB getLlhChoice.
    """
    stim_sens = model_params[0]
    stim_crit = model_params[1]
    noise_meta = model_params[2]
    conf_crit = np.cumsum(model_params[3:])

    noise_sens = 1.0
    guess_rate = 1e-5
    sample_rate = 100

    sens_mean = stim_value * stim_sens
    sens_crit = stim_crit * stim_sens

    n_stim = len(stim_value)

    x_bounds = np.sort(np.concatenate([[-c for c in conf_crit[::-1]], [0], conf_crit]))
    n_resp_alt = len(x_bounds) + 1

    choice_llh = np.zeros((n_resp_alt, n_stim))

    for i in range(n_stim):
        mu_log_n = np.log((noise_sens**2) / np.sqrt(noise_meta**2 + noise_sens**2))
        sigma_log_n = np.sqrt(np.log((noise_meta**2) / (noise_sens**2) + 1))

        q = np.linspace(0.5 / sample_rate, 1 - 0.5 / sample_rate, sample_rate)
        dv_den_x = lognorm.ppf(q, sigma_log_n, scale=np.exp(mu_log_n))

        mu = (1.0 / dv_den_x) * (sens_mean[i] - sens_crit)
        sigma = (1.0 / dv_den_x) * noise_sens

        ratio_dist_p = np.zeros(len(x_bounds))
        for j, xb in enumerate(x_bounds):
            ratio_dist_p[j] = np.mean(norm.cdf(xb, mu, sigma))

        for ix in range(n_resp_alt):
            if ix == 0:
                val = ratio_dist_p[0]
            elif ix < n_resp_alt - 1:
                val = ratio_dist_p[ix] - ratio_dist_p[ix - 1]
            else:
                val = 1 - ratio_dist_p[-1]

            choice_llh[ix, i] = (guess_rate / n_resp_alt) + (1 - guess_rate) * val

    return choice_llh


def compute_meta_uncertainty(
    stim: np.ndarray, resp: np.ndarray, conf: np.ndarray, n_ratings: int
) -> float:
    """Estimate meta-uncertainty.

    Replicates MATLAB compute_metaUncertainty.
    """
    stim_values = np.unique(stim)
    n_stim = len(stim_values)

    n_choice = np.zeros((n_ratings * 2, n_stim))

    resp_min = np.min(resp)
    resp_max = np.max(resp)

    for i, sv in enumerate(stim_values):
        mask = stim == sv
        for r in range(1, n_ratings + 1):
            n_choice[n_ratings - r, i] = np.sum(
                (resp[mask] == resp_min) & (conf[mask] == r)
            )
        for r in range(1, n_ratings + 1):
            n_choice[n_ratings + r - 1, i] = np.sum(
                (resp[mask] == resp_max) & (conf[mask] == r)
            )

    def objective(params: np.ndarray) -> float:
        llh = get_llh_choice(stim_values, params)
        llh = np.clip(llh, 1e-10, 1.0)
        return float(-np.sum(n_choice * np.log(llh)))

    n_conf_crit = n_ratings - 1
    guess = np.concatenate([[1.0, 0.0, 0.2], np.sort(2 * np.random.rand(n_conf_crit))])

    bounds = [
        (0.0, 10.0),  # stim_sens
        (-3.0, 3.0),  # stim_crit
        (0.01, 5.0),  # meta_uncertainty
    ] + [(0.0, 5.0)] * n_conf_crit

    res = minimize(objective, guess, method="L-BFGS-B", bounds=bounds)

    return float(res.x[2])
