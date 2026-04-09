"""Unified script to run all meta-measures and output the comprehensive 20-variable array."""

import numpy as np

from metasignal.stdpy.core import compute_sdt_resp, trials_to_counts
from metasignal.stdpy.measures import (
    sdt_expect_conf,
    compute_type2_auc,
    compute_gamma,
    compute_phi,
    compute_delta_conf,
)
from metasignal.stdpy.metad import fit_meta_d_mle
from metasignal.stdpy.metanoise import compute_meta_noise
from metasignal.stdpy.uncertainty import compute_meta_uncertainty


def compute_all_measures(stim: np.ndarray, resp: np.ndarray, conf: np.ndarray, n_ratings: int) -> np.ndarray:
    """Compute all 20 meta-signal measures mimicking the MATLAB output.

    Returns an array of 20 elements corresponding to:
    [
      1: meta_d, 2: AUC2, 3: gamma, 4: phi, 5: deltaConf,
      6: M_ratio, 7: AUC2_ratio, 8: gamma_ratio, 9: phi_ratio, 10: deltaConf_ratio,
      11: M_diff, 12: AUC2_diff, 13: gamma_diff, 14: phi_diff, 15: deltaConf_diff,
      16: metaNoise, 17: metaUncertainty, 18: dprime, 19: c, 20: mean_conf
    ]
    """
    stim = np.asarray(stim, dtype=float)
    resp = np.asarray(resp, dtype=float)
    conf = np.asarray(conf, dtype=float)

    # Remove NaNs
    valid = ~np.isnan(stim) & ~np.isnan(resp) & ~np.isnan(conf)
    stim = stim[valid]
    resp = resp[valid]
    conf = conf[valid]

    if len(stim) == 0:
        return np.full(20, np.nan)

    # Make input 0/1
    stim_min, stim_max = np.min(stim), np.max(stim)
    resp_min, resp_max = np.min(resp), np.max(resp)

    stim_bin = np.zeros_like(stim, dtype=int)
    stim_bin[stim == stim_max] = 1

    resp_bin = np.zeros_like(resp, dtype=int)
    resp_bin[resp == resp_max] = 1

    # Basic quantities
    dprime, c, _ = compute_sdt_resp(stim_bin, resp_bin)
    mean_conf = np.mean(conf)

    if np.array_equal(stim_bin, resp_bin) or dprime == 0 or len(np.unique(conf)) == 1:
        return np.full(20, np.nan)

    # Convert to counts
    nr_s1, nr_s2 = trials_to_counts(stim_bin, resp_bin, conf.astype(int), n_ratings)
    nr_s1 = np.array(nr_s1)
    nr_s2 = np.array(nr_s2)

    # meta-d', M-Ratio, M-Diff
    try:
        meta_d_res = fit_meta_d_mle(nr_s1, nr_s2)
        meta_d = meta_d_res["meta_da"]
        m_ratio = meta_d_res["M_ratio"]
        m_diff = meta_d_res["M_diff"]
    except Exception:
        meta_d, m_ratio, m_diff = np.nan, np.nan, np.nan

    if dprime < 0.2:
        m_ratio = np.nan

    # SDT Expectations for ratios/diffs
    sdt_exp = sdt_expect_conf(nr_s1, nr_s2)
    nr_s1_exp = np.array(sdt_exp["nR_S1_exp"])
    nr_s2_exp = np.array(sdt_exp["nR_S2_exp"])

    # AUC2
    auc2 = compute_type2_auc(nr_s1, nr_s2)
    auc2_exp = compute_type2_auc(nr_s1_exp, nr_s2_exp)
    auc2_ratio = auc2 / auc2_exp if auc2_exp != 0 else np.nan
    auc2_diff = auc2 - auc2_exp

    # Gamma
    gamma = compute_gamma(nr_s1, nr_s2)
    gamma_exp = compute_gamma(nr_s1_exp, nr_s2_exp)
    gamma_ratio = gamma / gamma_exp if gamma_exp != 0 else np.nan
    gamma_diff = gamma - gamma_exp

    # Phi
    phi = compute_phi(nr_s1, nr_s2)
    phi_exp = compute_phi(nr_s1_exp, nr_s2_exp)
    phi_ratio = phi / phi_exp if phi_exp != 0 else np.nan
    phi_diff = phi - phi_exp

    # DeltaConf
    dc_res = compute_delta_conf(nr_s1, nr_s2)
    delta_conf = dc_res["delta_conf"]
    delta_conf_ratio = dc_res["delta_conf_ratio"]
    delta_conf_diff = dc_res["delta_conf_diff"]

    # metaNoise
    try:
        mn_res = compute_meta_noise(stim_bin, resp_bin, conf.astype(int), n_ratings)
        meta_noise = mn_res["meta_noise"]
    except Exception:
        meta_noise = np.nan

    # metaUncertainty
    try:
        uncert_res = compute_meta_uncertainty(stim_bin, resp_bin, conf.astype(int), n_ratings)
        meta_uncert = uncert_res["meta_uncertainty"]
    except Exception:
        meta_uncert = np.nan

    # Combine all
    return np.array([
        meta_d, auc2, gamma, phi, delta_conf,
        m_ratio, auc2_ratio, gamma_ratio, phi_ratio, delta_conf_ratio,
        m_diff, auc2_diff, gamma_diff, phi_diff, delta_conf_diff,
        meta_noise, meta_uncert, float(dprime), float(c), float(mean_conf)
    ])
