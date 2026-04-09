"""Lognormal Meta Noise Model implementation."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize_scalar
from scipy.stats import norm

# Setup global lookup table cache
_LOOKUP_TABLE: dict[str, Any] | None = None

def _load_lookup_interpolator() -> RegularGridInterpolator:
    """Load the lookup table array and create an interpolator."""
    global _LOOKUP_TABLE
    if _LOOKUP_TABLE is not None:
        return _LOOKUP_TABLE["interpolator"]

    npz_path = pathlib.Path(__file__).parent / "lookupTable.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing lookup table file at {npz_path}")

    with np.load(npz_path) as data:
        # Load the raw grids
        mus = np.array(data["mus"]).flatten()
        crit = np.array(data["crit"]).flatten()
        meta_noise = np.array(data["metaNoise"]).flatten()

        # Original MATLAB table was [mus, crit, metaNoise]
        table = data["lookupTable"]

        # Create interpolator
        # We use 'linear' to closely approximate MATLAB's weighted distance
        interp = RegularGridInterpolator(
            (mus, crit, meta_noise),
            table,
            method="linear",
            bounds_error=False,
            fill_value=None  # extrapolate using nearest edge behavior implicitly by not failing
        )
        _LOOKUP_TABLE = {"interpolator": interp}

    return _LOOKUP_TABLE["interpolator"]


def _evaluate_integral(mu_gauss: float, mu_lognormal: float, meta_noise: float) -> float:
    """Evaluate lognormal/gauss integral using precomputed grid."""
    interp = _load_lookup_interpolator()
    # Ensure inputs are bound reasonably to avoid wild extrapolation
    # The grid has its own limits but we let RegularGridInterpolator handle it
    val = interp((mu_gauss, mu_lognormal, meta_noise))
    return float(np.clip(val, 0.0, 1.0))


def _compute_sdt_criteria(stim: np.ndarray, resp: np.ndarray, conf: np.ndarray, n_ratings: int) -> tuple[float, np.ndarray]:
    """Calculate d' and initial criteria similar to the MATLAB compute_SDTcriteria."""
    hr = np.zeros(2 * n_ratings - 1)
    far = np.zeros(2 * n_ratings - 1)

    min_stim = int(np.min(stim))
    max_stim = int(np.max(stim))

    # Criteria on the left of decision criterion
    for roc_point in range(1, n_ratings):
        hr[roc_point - 1] = np.sum((stim == max_stim) & ((resp == max_stim) | (conf <= n_ratings - roc_point))) / np.sum(stim == max_stim)
        far[roc_point - 1] = np.sum((stim == min_stim) & ((resp == max_stim) | (conf <= n_ratings - roc_point))) / np.sum(stim == min_stim)

    # Decision criterion and right confidence criteria
    for roc_point in range(n_ratings, 2 * n_ratings):
        idx = roc_point - 1
        hr[idx] = np.sum((stim == max_stim) & (resp == max_stim) & (conf > roc_point - n_ratings)) / np.sum(stim == max_stim)
        far[idx] = np.sum((stim == min_stim) & (resp == max_stim) & (conf > roc_point - n_ratings)) / np.sum(stim == min_stim)

    eps = 1e-10
    hr = np.clip(hr, eps, 1 - eps)
    far = np.clip(far, eps, 1 - eps)

    t1_idx = n_ratings - 1
    dprime = float(norm.ppf(hr[t1_idx]) - norm.ppf(far[t1_idx]))
    c = -0.5 * (norm.ppf(hr) + norm.ppf(far))

    return dprime, c


def _logl_func_criteria(mu_conf: float, mus: tuple[float, float], meta_noise: float, data_counts_binary: np.ndarray) -> dict[str, Any]:
    """Calculate log likelihood for a given confidence criterion."""
    p_hc_s1 = _evaluate_integral(mus[0], mu_conf, meta_noise)
    p_hc_s2 = _evaluate_integral(mus[1], mu_conf, meta_noise)

    prob_r_model = np.array([
        [1 - p_hc_s1, p_hc_s1],
        [1 - p_hc_s2, p_hc_s2],
    ])
    prob_r_model = np.clip(prob_r_model, 1e-5, 1.0)

    logl = -np.sum(np.log(prob_r_model) * data_counts_binary)
    return {"logL": logl, "p_HC": np.array([p_hc_s1, p_hc_s2]), "x": mu_conf}


def compute_meta_noise(stim: np.ndarray, resp: np.ndarray, conf: np.ndarray, n_ratings: int) -> dict[str, Any]:
    """Fit Lognormal Meta Noise model to data.

    Args:
        stim: Stimulus array (e.g. 0/1)
        resp: Response array (e.g. 0/1)
        conf: Confidence array (1 to n_ratings)
        n_ratings: Number of rating categories

    Returns:
        Dictionary containing meta_noise, dprime, criteria, and logl.
    """
    stim = np.asarray(stim)
    resp = np.asarray(resp)
    conf = np.asarray(conf)

    min_stim = int(np.min(stim))
    max_stim = int(np.max(stim))

    nr_stim_s1 = np.zeros(2 * n_ratings)
    nr_stim_s2 = np.zeros(2 * n_ratings)

    for rating in range(1, n_ratings + 1):
        # S1 responses
        sr_s1 = (stim == min_stim) & (resp == min_stim) & (conf == rating)
        sr_s2 = (stim == max_stim) & (resp == min_stim) & (conf == rating)
        nr_stim_s1[n_ratings - rating] = np.sum(sr_s1)
        nr_stim_s2[n_ratings - rating] = np.sum(sr_s2)

        # S2 responses
        sr_s1_2 = (stim == min_stim) & (resp == max_stim) & (conf == rating)
        sr_s2_2 = (stim == max_stim) & (resp == max_stim) & (conf == rating)
        nr_stim_s1[n_ratings + rating - 1] = np.sum(sr_s1_2)
        nr_stim_s2[n_ratings + rating - 1] = np.sum(sr_s2_2)

    data_counts = np.vstack([nr_stim_s1, nr_stim_s2])

    dprime, c_init = _compute_sdt_criteria(stim, resp, conf, n_ratings)
    c = np.copy(c_init)

    # 1. Optimize criteria for metaNoise = 0
    for crit in range(2 * n_ratings - 1):
        if np.isinf(c_init[crit]) or np.isnan(c_init[crit]):
            r_range = np.arange(-5, 5.01, 0.01)
        else:
            r_range = np.arange(c_init[crit] - 0.5, c_init[crit] + 0.51, 0.01)

        pr_s1_m = norm.cdf(r_range, -dprime/2, 1)
        pr_s2_m = norm.cdf(r_range, dprime/2, 1)
        pr_model = np.vstack([pr_s1_m, pr_s2_m, 1 - pr_s1_m, 1 - pr_s2_m])
        pr_model = np.clip(pr_model, 1e-5, 1.0)

        data = np.concatenate([
            np.sum(data_counts[:, :crit+1], axis=1),
            np.sum(data_counts[:, crit+1:], axis=1)
        ])

        logl_grid = -np.sum(np.log(pr_model) * data[:, None], axis=0)
        c[crit] = r_range[np.argmin(logl_grid)]

    # Compute baseline logL for metaNoise=0
    c_pad = np.concatenate([[-np.inf], c, [np.inf]])
    pr_s1 = np.diff(norm.cdf(c_pad, -dprime/2, 1))
    pr_s2 = np.diff(norm.cdf(c_pad, dprime/2, 1))
    pr_model = np.vstack([pr_s1, pr_s2])
    prob_dec_resp = np.array([
        np.sum(pr_model[:, :n_ratings], axis=1),
        np.sum(pr_model[:, n_ratings:], axis=1)
    ]).T
    pr_model = np.clip(pr_model, 1e-5, 1.0)

    # 2. Setup the target function for metaNoise
    mu_s1 = -dprime/2 - c[n_ratings - 1]
    mu_s2 = dprime/2 - c[n_ratings - 1]

    def _meta_noise_objective(test_mn: float) -> float:
        if test_mn <= 0:
            return 1e9  # Penalty

        crit_values = np.zeros((2, n_ratings - 1))
        p_hc_pos = np.zeros((2, n_ratings - 1))
        p_hc_neg = np.zeros((2, n_ratings - 1))

        for crit_side in (1, 2):
            mus = (mu_s1, mu_s2) if crit_side == 1 else (-mu_s1, -mu_s2)

            for crt in range(n_ratings - 1):
                if crit_side == 1:
                    dc_bin = np.array([
                        np.sum(data_counts[:, :n_ratings + crt + 1], axis=1),
                        np.sum(data_counts[:, n_ratings + crt + 1:], axis=1)
                    ]).T
                else:
                    dc_bin = np.array([
                        np.sum(data_counts[:, n_ratings - crt - 1:], axis=1),
                        np.sum(data_counts[:, :n_ratings - crt - 1], axis=1)
                    ]).T

                # Internal optimization for criterion mu
                def _crit_obj(mc: float) -> float:
                    return _logl_func_criteria(mc, mus, test_mn, dc_bin)["logL"]

                # Constrain internal search roughly [-6, 6]
                res_crit = minimize_scalar(_crit_obj, bounds=(-6.0, 6.0), method='bounded')
                best_mc = res_crit.x

                info = _logl_func_criteria(best_mc, mus, test_mn, dc_bin)
                crit_values[crit_side - 1, crt] = info["x"]
                if crit_side == 1:
                    p_hc_pos[:, crt] = info["p_HC"]
                else:
                    p_hc_neg[:, crt] = info["p_HC"]

        # Reconstruct prob_model across all bins
        hc_pos = np.hstack([prob_dec_resp[:, 1:2], p_hc_pos, np.zeros((2, 1))])
        pr_pos = hc_pos[:, :-1] - hc_pos[:, 1:]

        hc_neg = np.hstack([prob_dec_resp[:, 0:1], p_hc_neg, np.zeros((2, 1))])
        pr_neg = hc_neg[:, :-1] - hc_neg[:, 1:]

        pr_full = np.hstack([np.fliplr(pr_neg), pr_pos])
        pr_full = np.clip(pr_full, 1e-5, 1.0)

        return float(-np.sum(np.log(pr_full) * data_counts))

    # 3. Global optimization for meta_noise
    res = minimize_scalar(_meta_noise_objective, bounds=(0.0, 5.0), method='bounded')
    best_meta_noise = res.x
    best_logl = res.fun

    return {
        "meta_noise": float(best_meta_noise),
        "dprime": float(dprime),
        "c": c.tolist(),
        "logL": float(best_logl)
    }
