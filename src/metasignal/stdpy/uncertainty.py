"""Meta-uncertainty model implementation.

Faithful port of MATLAB ``compute_metaUncertainty.m`` / ``getLlhChoice``.

Scientific notes
----------------
The objective is non-convex. MATLAB uses a single random start
(``sort(2*rand(...))``) with ``fmincon``. A single random Python start
therefore need not match a particular MATLAB Result file. To make the
estimator *scientifically stable and reproducible*, this module:

1. Uses the same objective, bounds, and sampling schedule as MATLAB.
2. Runs a fixed multi-start grid (deterministic seeds).
3. Returns the fit with the lowest negative log-likelihood.

This does not invent a new model; it removes avoidable optimizer
noise while preserving the published likelihood.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm, norm


def get_llh_choice(stim_value: np.ndarray, model_params: np.ndarray) -> np.ndarray:
    """Compute likelihood of each response alternative (MATLAB getLlhChoice)."""
    stim_sens = model_params[0]
    stim_crit = model_params[1]
    noise_meta = model_params[2]
    conf_crit = np.cumsum(model_params[3:])

    noise_sens = 1.0
    guess_rate = 1e-5
    sample_rate = 100  # MATLAB default

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

        # Vectorized over criteria; mean over Monte-Carlo noise samples.
        ratio_dist_p = np.mean(
            norm.cdf(x_bounds[:, None], mu[None, :], sigma[None, :]),
            axis=1,
        )

        for ix in range(n_resp_alt):
            if ix == 0:
                val = ratio_dist_p[0]
            elif ix < n_resp_alt - 1:
                val = ratio_dist_p[ix] - ratio_dist_p[ix - 1]
            else:
                val = 1.0 - ratio_dist_p[-1]
            choice_llh[ix, i] = (guess_rate / n_resp_alt) + (1.0 - guess_rate) * val

    return choice_llh


def _starting_points(n_conf_crit: int, n_starts: int) -> list[np.ndarray]:
    """Deterministic multi-start seeds mirroring MATLAB's start structure."""
    starts: list[np.ndarray] = []
    # Canonical MATLAB-like seed with zero random draw for criteria.
    starts.append(
        np.concatenate([[1.0, 0.0, 0.2], np.linspace(0.2, 1.8, n_conf_crit)])
    )
    # Additional deterministic draws analogous to sort(2*rand(...)).
    for seed in range(1, n_starts):
        rng = np.random.default_rng(10_000 + seed)
        crit = np.sort(2.0 * rng.random(n_conf_crit))
        meta0 = float(rng.choice([0.1, 0.2, 0.5, 1.0, 2.0]))
        starts.append(np.concatenate([[1.0, 0.0, meta0], crit]))
    return starts


def compute_meta_uncertainty(
    stim: np.ndarray,
    resp: np.ndarray,
    conf: np.ndarray,
    n_ratings: int,
    rng: np.random.Generator | None = None,
    n_starts: int = 5,
    *,
    matlab_compat: bool = False,
) -> float:
    """Estimate meta-uncertainty (MATLAB compute_metaUncertainty).

    Parameters
    ----------
    rng :
        Optional generator. Used for an extra start in multi-start mode, or
        as the sole MATLAB-style start when ``matlab_compat=True``.
    n_starts :
        Number of deterministic optimizer starts (default 5). Ignored when
        ``matlab_compat=True``.
    matlab_compat :
        If True, mimic MATLAB's single random start
        ``[1, 0, 0.2, sort(2*rand(...))]`` (requires ``rng`` or uses a
        fresh default generator). Prefer multi-start for stable science;
        use this only when reproducing a specific MATLAB Result file.
    """
    stim_values = np.unique(stim)
    n_stim = len(stim_values)
    if n_stim < 2:
        raise ValueError("stim must contain both classes; found only one class in input.")

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
        # MATLAB does not clip; guess_rate keeps llh > 0 in practice.
        if matlab_compat:
            llh = np.maximum(llh, np.finfo(float).tiny)
        else:
            llh = np.clip(llh, 1e-10, 1.0)
        return float(-np.sum(n_choice * np.log(llh)))

    n_conf_crit = n_ratings - 1
    bounds = [
        (0.0, 10.0),  # stim_sens
        (-3.0, 3.0),  # stim_crit
        (0.01, 5.0),  # meta_uncertainty
    ] + [(0.0, 5.0)] * n_conf_crit

    if matlab_compat:
        rng_m = rng if rng is not None else np.random.default_rng()
        starts = [
            np.concatenate([[1.0, 0.0, 0.2], np.sort(2.0 * rng_m.random(n_conf_crit))])
        ]
        opt = {"maxiter": 10**5, "maxfun": 10**5, "ftol": 1e-12}
    else:
        starts = _starting_points(n_conf_crit, max(1, int(n_starts)))
        if rng is not None:
            starts.append(
                np.concatenate(
                    [[1.0, 0.0, 0.2], np.sort(2.0 * rng.random(n_conf_crit))]
                )
            )
        opt = {"maxiter": 2000, "maxfun": 4000, "ftol": 1e-10}

    best_val = np.inf
    best_x = starts[0].copy()
    for guess in starts:
        res = minimize(
            objective,
            guess,
            method="L-BFGS-B",
            bounds=bounds,
            options=opt,
        )
        if np.isfinite(res.fun) and res.fun < best_val:
            best_val = float(res.fun)
            best_x = np.asarray(res.x, dtype=float)

    return float(best_x[2])
