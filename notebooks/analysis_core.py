"""
analysis_core.py
================
Pure-Python replication of the MATLAB benchmark analyses from:
  "A comprehensive assessment of current methods for measuring metacognition"
  (Rahnev, Nature Communications 2025)

This module reproduces:
  - Supplementary Tables 3-9
  - Preprocessing logic matching step1_importDataToMatlab.m / step2_preprocessData.m
  - Statistical tests matching perform_ttest.m and the ana_*.m scripts

Run standalone: python analysis_core.py
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "matlab", "metasignal_mat", "Preprocess", "orig_csv_files")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from metasignal.stdpy.compute_all import compute_all_measures

# ── constants ────────────────────────────────────────────────────────────────
MEASURE_NAMES = [
    "meta-d'", "AUC2", "Gamma", "Phi", "DeltaConf",
    "M-Ratio", "AUC2-Ratio", "Gamma-Ratio", "Phi-Ratio", "DeltaConf-Ratio",
    "M-Diff", "AUC2-Diff", "Gamma-Diff", "Phi-Diff", "DeltaConf-Diff",
    "meta-noise", "meta-uncertainty", "d'", "Criterion", "Confidence",
    "logL", "AIC", "BIC", "AICc", "k", "n",
]
N_MEASURES = 26

ACC_LO, ACC_HI = 0.60, 0.95
MAX_PROP_SAME  = 0.85


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_filter(df, conf_col="Confidence", extra_cols=None):
    """
    Apply standard exclusion criteria to a per-trial DataFrame.

    Returns a list of dicts, one per included subject, with keys:
      stim, resp, conf  (plus any extras named in extra_cols)
    """
    subjects = []
    for sid, grp in df.groupby("Subj_idx"):
        grp = grp.dropna(subset=["Stimulus", "Response", conf_col])
        stim = grp["Stimulus"].to_numpy(float)
        resp = grp["Response"].to_numpy(float)
        conf = grp[conf_col].to_numpy(float)

        correct = stim == resp
        acc = np.mean(correct)
        if acc < ACC_LO or acc > ACC_HI:
            continue
        if np.max(np.bincount(resp.astype(int))) / len(resp) > MAX_PROP_SAME:
            continue
        if np.max(np.bincount(conf.astype(int))) / len(conf) > MAX_PROP_SAME:
            continue

        rec = {"sid": sid, "stim": stim, "resp": resp, "conf": conf}
        if extra_cols:
            for col in extra_cols:
                rec[col] = grp[col].to_numpy(float)
        subjects.append(rec)
    return subjects


def preprocess_haddara():
    df = pd.read_csv(os.path.join(DATA, "data_Haddara_2022_Expt2.csv"))
    # Day column (1-7); keep days 2-7
    subjects = []
    n_ratings = 4
    for sid, grp in df.groupby("Subj_idx"):
        grp = grp.dropna(subset=["Stimulus", "Response", "Confidence"])
        stim = grp["Stimulus"].to_numpy(float)
        resp = grp["Response"].to_numpy(float)
        conf = grp["Confidence"].to_numpy(float)
        day  = grp["Day"].to_numpy(float)

        correct = stim == resp
        acc = np.mean(correct)
        if acc < ACC_LO or acc > ACC_HI:
            continue
        if np.max(np.unique(resp, return_counts=True)[1]) / len(resp) > MAX_PROP_SAME:
            continue
        if np.max(np.unique(conf, return_counts=True)[1]) / len(conf) > MAX_PROP_SAME:
            continue
        subjects.append({"sid": sid, "stim": stim, "resp": resp,
                          "conf": conf, "day": day, "n_ratings": n_ratings})
    return subjects


def preprocess_maniscalco():
    df = pd.read_csv(os.path.join(DATA, "data_Maniscalco_2017_expt1.csv"))
    n_ratings = 4
    subjects = []
    for sid, grp in df.groupby("Subj_idx"):
        # MATLAB: NaN responses count as incorrect (stim==NaN → 0)
        # Do NOT dropna before computing accuracy/mode filters
        stim_all = grp["Stimulus"].to_numpy(float)
        resp_all = grp["Response"].to_numpy(float)
        conf_all = grp["Confidence"].to_numpy(float)

        # NaN responses → incorrect (matches MATLAB correct = (stim==resp)+0)
        correct = np.where(np.isnan(resp_all), 0.0, (stim_all == resp_all).astype(float))
        acc = np.mean(correct)
        if acc < ACC_LO or acc > ACC_HI:
            continue
        if np.max(np.bincount(resp_all[~np.isnan(resp_all)].astype(int))) / len(resp_all) > MAX_PROP_SAME:
            continue
        conf_valid = conf_all[~np.isnan(conf_all)]
        if np.max(np.bincount(conf_valid.astype(int))) / len(conf_all) > MAX_PROP_SAME:
            continue

        # After filtering, drop NaN rows for actual computation
        valid = ~(np.isnan(stim_all) | np.isnan(resp_all) | np.isnan(conf_all))
        stim = stim_all[valid]
        resp = resp_all[valid]
        conf = conf_all[valid]
        subjects.append({"sid": sid, "stim": stim, "resp": resp,
                          "conf": conf, "n_ratings": n_ratings})
    return subjects


def preprocess_shekhar():
    df = pd.read_csv(os.path.join(DATA, "data_Shekhar_2021.csv"))
    n_ratings = 6
    # Confidence is continuous 50-100; discretize into n_ratings bins
    edges = np.linspace(50, 100, n_ratings + 1)
    subjects = []
    for sid, grp in df.groupby("Subj_idx"):
        stim_raw = grp["Stimulus"].to_numpy(float)
        resp_raw = grp["Response"].to_numpy(float)
        conf_raw = grp["Confidence"].to_numpy(float)
        contrast  = grp["Contrast"].to_numpy(float)

        conf = np.digitize(conf_raw, edges, right=True)
        conf = np.clip(conf, 1, n_ratings)

        correct = stim_raw == resp_raw
        acc = np.mean(correct)
        if acc < ACC_LO or acc > ACC_HI:
            continue
        if np.max(np.unique(resp_raw, return_counts=True)[1]) / len(resp_raw) > MAX_PROP_SAME:
            continue
        if np.max(np.unique(conf, return_counts=True)[1]) / len(conf) > MAX_PROP_SAME:
            continue
        subjects.append({"sid": sid, "stim": stim_raw, "resp": resp_raw,
                          "conf": conf, "contrast": contrast, "n_ratings": n_ratings})
    return subjects


def preprocess_rouault(expt=1):
    fname = f"data_Rouault_2018_Expt{expt}.csv"
    df = pd.read_csv(os.path.join(DATA, fname))
    n_ratings = 6
    subjects = []
    for sid, grp in df.groupby("Subj_idx"):
        grp = grp.dropna(subset=["Stimulus", "Response", "Confidence"])
        stim     = grp["Stimulus"].to_numpy(float)
        resp     = grp["Response"].to_numpy(float)
        conf_raw = grp["Confidence"].to_numpy(float)
        dotdiff  = grp["DotDiff"].to_numpy(float)

        correct = stim == resp
        acc = np.mean(correct)
        if acc < ACC_LO or acc > ACC_HI:
            continue
        if np.max(np.bincount(resp.astype(int))) / len(resp) > MAX_PROP_SAME:
            continue

        # MATLAB applies conf stereotypy filter on RAW conf (before any transformation)
        if np.max(np.bincount(conf_raw.astype(int))) / len(conf_raw) > MAX_PROP_SAME:
            continue

        # Apply Rouault1 conf transformation AFTER filtering
        if expt == 1:
            conf = conf_raw - 5
            conf[conf < 1] = 1
        else:
            conf = conf_raw.copy()   # Rouault2 conf is already 1-6

        subjects.append({"sid": sid, "stim": stim, "resp": resp,
                          "conf": conf, "contrast": dotdiff, "n_ratings": n_ratings})
    return subjects


def preprocess_locke():
    df = pd.read_csv(os.path.join(DATA, "data_Locke_2020.csv"))
    df = df[df["Training"] == 0].copy()
    df["Confidence"] = df["Confidence"] + 1   # 0/1 → 1/2
    n_ratings = 2
    subjects = []
    for sid, grp in df.groupby("Subj_idx"):
        grp = grp.dropna(subset=["Stimulus", "Response", "Confidence"])
        stim      = grp["Stimulus"].to_numpy(float)
        resp      = grp["Response"].to_numpy(float)
        conf      = grp["Confidence"].to_numpy(float)
        condition = grp["Condition"].to_numpy(float)

        correct = stim == resp
        acc = np.mean(correct)
        if acc < ACC_LO or acc > ACC_HI:
            continue
        if np.max(np.unique(resp, return_counts=True)[1]) / len(resp) > MAX_PROP_SAME:
            continue
        subjects.append({"sid": sid, "stim": stim, "resp": resp,
                          "conf": conf, "condition": condition,
                          "n_ratings": n_ratings})
    return subjects


# ══════════════════════════════════════════════════════════════════════════════
# XUE RECODING  (replicates xue_recode.m)
# ══════════════════════════════════════════════════════════════════════════════

def xue_recode(conf, low_high_recoding):
    """
    Recode confidence ratings per Xue et al. (2021):
      low_high_recoding=1 → removes the lowest rating (shifts toward HIGH confidence)
      low_high_recoding=2 → removes the highest rating (shifts toward LOW confidence)
    Returns new conf array with (n_ratings-1) categories.
    """
    valid = conf[~np.isnan(conf)]
    if len(np.unique(valid)) < 3 or not np.allclose(valid, np.round(valid)):
        return np.full_like(conf, np.nan)

    conf_new = conf.copy().astype(float)
    if low_high_recoding == 1:
        conf_new -= 1
        cmin = np.nanmin(conf_new)
        conf_new[conf_new == cmin] = cmin + 1
    elif low_high_recoding == 2:
        cmax = np.nanmax(conf_new)
        conf_new[conf_new == cmax] = cmax - 1
    return conf_new


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL HELPERS  (replicates perform_ttest.m)
# ══════════════════════════════════════════════════════════════════════════════

def ttest_1samp(data):
    """
    One-sample t-test against 0, returns (t, df, p, cohen_d, ci_lo, ci_hi).
    Matches MATLAB perform_ttest.m.
    """
    data = np.asarray(data, dtype=float)
    valid = data[~np.isnan(data)]
    n = len(valid)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    df = n - 1
    t, p = stats.ttest_1samp(valid, 0)
    d = t / np.sqrt(n)           # Cohen's d = t / sqrt(n)
    sem = np.std(valid, ddof=1) / np.sqrt(n)
    ci = np.mean(valid) + stats.t.ppf([0.025, 0.975], df) * sem
    return t, df, p, d, ci[0], ci[1]


def rm_anova_1way(data_2d):
    """
    One-way repeated-measures ANOVA.
    data_2d shape: (n_subjects, n_conditions)
    Returns (F, df_between, df_error, p, eta2_p).
    """
    n, k = data_2d.shape
    grand = np.nanmean(data_2d)
    row_means = np.nanmean(data_2d, axis=1, keepdims=True)
    col_means = np.nanmean(data_2d, axis=0, keepdims=True)

    ss_between = n * np.sum((col_means - grand) ** 2)
    ss_subjects = k * np.sum((row_means - grand) ** 2)
    ss_total    = np.sum((data_2d - grand) ** 2)
    ss_error    = ss_total - ss_between - ss_subjects

    df_between = k - 1
    df_error   = (n - 1) * (k - 1)

    ms_between = ss_between / df_between
    ms_error   = ss_error   / df_error

    F = ms_between / ms_error
    p = stats.f.sf(F, df_between, df_error)
    eta2_p = ss_between / (ss_between + ss_error)
    return F, df_between, df_error, p, eta2_p


# ══════════════════════════════════════════════════════════════════════════════
# DIFFICULTY ANALYSES  (Supp Tables 3-5)
# ══════════════════════════════════════════════════════════════════════════════

def compute_difficulty_shekhar(subjects):
    """
    Per subject: measures at contrast 1 (hardest) and contrast 3 (easiest).
    Returns array (n_subjects, 2, 20) where dim1 = [hard, easy].
    """
    out = []
    for s in subjects:
        row = np.full((2, N_MEASURES), np.nan)
        for ci, c in enumerate([1, 3]):
            mask = s["contrast"] == c
            if mask.sum() < 10:
                continue
            row[ci] = compute_all_measures(
                s["stim"][mask], s["resp"][mask],
                s["conf"][mask], s["n_ratings"]
            )
        out.append(row)
    return np.array(out)   # (n_sub, 2, 20)


def compute_difficulty_rouault(subjects):
    """
    Per subject: measures at low (<=median) and high (>median) contrast.
    Returns array (n_subjects, 2, 20).
    """
    out = []
    for s in subjects:
        med = np.median(s["contrast"])
        lo  = s["contrast"] <= med
        hi  = s["contrast"] >  med
        row = np.full((2, N_MEASURES), np.nan)
        if lo.sum() >= 10:
            row[0] = compute_all_measures(
                s["stim"][lo], s["resp"][lo], s["conf"][lo], s["n_ratings"])
        if hi.sum() >= 10:
            row[1] = compute_all_measures(
                s["stim"][hi], s["resp"][hi], s["conf"][hi], s["n_ratings"])
        out.append(row)
    return np.array(out)


def apply_3sd_outlier_removal(arr):
    """
    Replicate MATLAB ana_taskPerformance.m outlier removal:
    For each measure and each difficulty level, set values outside mean±3SD to NaN.
    Then: if any difficulty level is NaN for a measure, set all difficulty levels to NaN.

    arr: (n_sub, n_levels, n_measures)
    Returns cleaned copy.
    """
    out = arr.copy()
    n_sub, n_levels, n_meas = out.shape
    for m in range(n_meas):
        for dl in range(n_levels):
            col = out[:, dl, m]
            mu = np.nanmean(col)
            sd = np.nanstd(col, ddof=1)
            if not np.isnan(mu) and sd > 0:
                outlier = (col < mu - 3*sd) | (col > mu + 3*sd)
                out[outlier, dl, m] = np.nan
        # Propagate NaN across difficulty levels
        has_nan = np.isnan(out[:, :, m]).any(axis=1)
        out[has_nan, :, m] = np.nan
    return out


def difficulty_table(diff_arr, label):
    """
    diff_arr: (n_sub, 2, 20) — dim1 = [hard, easy]
    Returns a DataFrame matching Supp Tables 3-5.
    Applies ±3SD outlier removal (matching MATLAB ana_taskPerformance.m) before t-test.
    """
    # Apply MATLAB outlier removal
    diff_clean = apply_3sd_outlier_removal(diff_arr)

    # easy (index 1) - hard (index 0)
    delta = diff_clean[:, 1, :] - diff_clean[:, 0, :]   # (n_sub, 20)
    rows = []
    for m, name in enumerate(MEASURE_NAMES):
        t, df, p, d, ci_lo, ci_hi = ttest_1samp(delta[:, m])
        rows.append({
            "Measure": name, "t": t, "df": df, "p": p,
            "Cohen's d": d, "CI lower": ci_lo, "CI upper": ci_hi
        })
    df_out = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print(f"  Difficulty dependence — {label}")
    print(f"{'='*70}")
    print(df_out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# METACOGNITIVE BIAS ANALYSES  (Supp Tables 6-8)
# ══════════════════════════════════════════════════════════════════════════════

def compute_bias(subjects):
    """
    For each subject compute all measures under Xue recode 1 and 2.
    Returns (n_sub, 2, 20) where dim1 = [recode1 (high-conf bias), recode2 (low-conf bias)].
    """
    out = []
    for s in subjects:
        nr = s["n_ratings"]
        row = np.full((2, N_MEASURES), np.nan)
        for ri, rtype in enumerate([1, 2]):
            conf_r = xue_recode(s["conf"], rtype)
            if not np.all(np.isnan(conf_r)):
                row[ri] = compute_all_measures(
                    s["stim"], s["resp"], conf_r, nr - 1)
        out.append(row)
    return np.array(out)


def compute_bias_shekhar(subjects):
    """Shekhar: average recode over 3 contrast levels."""
    out = []
    for s in subjects:
        nr = s["n_ratings"]
        per_contrast = np.full((3, 2, N_MEASURES), np.nan)
        for ci, c in enumerate([1, 2, 3]):
            mask = s["contrast"] == c
            if mask.sum() < 10:
                continue
            for ri, rtype in enumerate([1, 2]):
                conf_r = xue_recode(s["conf"][mask], rtype)
                if not np.all(np.isnan(conf_r)):
                    per_contrast[ci, ri] = compute_all_measures(
                        s["stim"][mask], s["resp"][mask], conf_r, nr - 1)
        row = np.nanmean(per_contrast, axis=0)   # (2, 20)
        out.append(row)
    return np.array(out)


def bias_table(bias_arr, label):
    """
    bias_arr: (n_sub, 2, 20) — dim1 = [recode1 (high-conf), recode2 (low-conf)]
    Test: recode1 - recode2 (do confidence differences bias measures?)
    Supp Tables 6-8 exclude d' and Criterion (Xue recode doesn't affect responses).
    """
    delta = bias_arr[:, 1, :] - bias_arr[:, 0, :]   # recode2 − recode1 (low-conf minus high-conf bias)
    rows = []
    for m, name in enumerate(MEASURE_NAMES):
        if name in ("d'", "Criterion"):
            continue   # excluded from Tables 6-8
        t, df, p, d, ci_lo, ci_hi = ttest_1samp(delta[:, m])
        rows.append({
            "Measure": name, "t": t, "df": df, "p": p,
            "Cohen's d": d, "CI lower": ci_lo, "CI upper": ci_hi
        })
    df_out = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print(f"  Metacognitive bias — {label}")
    print(f"{'='*70}")
    print(df_out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE BIAS ANALYSIS  (Supp Table 9)
# ══════════════════════════════════════════════════════════════════════════════

def compute_response_bias_locke(subjects):
    """
    Per subject compute measures for each of the 7 conditions (1-7).
    Returns (n_sub, 7, 20).
    """
    conditions = list(range(1, 8))
    out = []
    for s in subjects:
        row = np.full((7, N_MEASURES), np.nan)
        for ci, cond in enumerate(conditions):
            mask = s["condition"] == cond
            if mask.sum() < 5:
                continue
            row[ci] = compute_all_measures(
                s["stim"][mask], s["resp"][mask],
                s["conf"][mask], s["n_ratings"])
        out.append(row)
    return np.array(out)   # (n_sub, 7, 20)


def response_bias_table(rb_arr, label):
    """
    rb_arr: (n_sub, 7, 20)
    Repeated-measures ANOVA across 7 conditions for each measure.
    """
    rows = []
    for m, name in enumerate(MEASURE_NAMES):
        data = rb_arr[:, :, m]   # (n_sub, 7)
        # Use only subjects with complete data
        complete = ~np.any(np.isnan(data), axis=1)
        data_c = data[complete]
        if data_c.shape[0] < 2:
            rows.append({"Measure": name, "F(6,54)": np.nan, "p": np.nan, "eta2_p": np.nan})
            continue
        F, df_b, df_e, p, eta2p = rm_anova_1way(data_c)
        rows.append({"Measure": name,
                     f"F({df_b},{df_e})": round(F, 3),
                     "p": p, "η²p": round(eta2p, 3)})
    df_out = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print(f"  Response bias — {label}")
    print(f"{'='*70}")
    print(df_out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# PRECISION ANALYSIS  (Supp Fig 1)
# ══════════════════════════════════════════════════════════════════════════════

def metas_altered_conf(stim, resp, conf, n_ratings, prop_altered):
    """
    Artificially corrupt confidence ratings:
      correct trials → decrease conf by 1 (toward miscalibration)
      incorrect trials → increase conf by 1
    prop_altered fraction of trials are altered.
    """
    n = len(conf)
    n_alter = int(round(n * prop_altered))
    conf_new = conf.copy().astype(float)
    altered = 0
    for i in range(n):
        if stim[i] == resp[i] and conf_new[i] > 1:
            conf_new[i] -= 1
            altered += 1
        elif stim[i] != resp[i] and conf_new[i] < n_ratings:
            conf_new[i] += 1
            altered += 1
        if altered >= n_alter:
            break
    return compute_all_measures(stim, resp, conf_new, n_ratings)


# ══════════════════════════════════════════════════════════════════════════════
# SPLIT-HALF & TEST-RETEST RELIABILITY  (Supp Fig not shown here but stored)
# ══════════════════════════════════════════════════════════════════════════════

def compute_splithalf(stim, resp, conf, n_ratings):
    """Returns (measures_odd, measures_even)."""
    odd  = compute_all_measures(stim[0::2], resp[0::2], conf[0::2], n_ratings)
    even = compute_all_measures(stim[1::2], resp[1::2], conf[1::2], n_ratings)
    return odd, even


# ══════════════════════════════════════════════════════════════════════════════
# MAIN  —  reproduce all supplementary tables
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading and filtering datasets...")

    shekhar    = preprocess_shekhar()
    rouault1   = preprocess_rouault(1)
    rouault2   = preprocess_rouault(2)
    haddara    = preprocess_haddara()
    maniscalco = preprocess_maniscalco()
    locke      = preprocess_locke()

    print(f"  Shekhar:    {len(shekhar)} subjects")
    print(f"  Rouault1:   {len(rouault1)} subjects")
    print(f"  Rouault2:   {len(rouault2)} subjects")
    print(f"  Haddara:    {len(haddara)} subjects")
    print(f"  Maniscalco: {len(maniscalco)} subjects")
    print(f"  Locke:      {len(locke)} subjects")

    # ── Tables 3-5: Difficulty ──
    print("\n[Computing difficulty analyses — may take a few minutes...]")
    shekhar_diff   = compute_difficulty_shekhar(shekhar)
    rouault1_diff  = compute_difficulty_rouault(rouault1)
    rouault2_diff  = compute_difficulty_rouault(rouault2)

    t3 = difficulty_table(shekhar_diff,  "Shekhar (n=20) — Supp Table 3")
    t4 = difficulty_table(rouault1_diff, "Rouault1 — Supp Table 4")
    t5 = difficulty_table(rouault2_diff, "Rouault2 — Supp Table 5")

    # ── Tables 6-8: Metacognitive bias ──
    print("\n[Computing metacognitive bias analyses...]")
    haddara_bias    = compute_bias(haddara)
    maniscalco_bias = compute_bias(maniscalco)
    shekhar_bias    = compute_bias_shekhar(shekhar)

    t6 = bias_table(haddara_bias,    "Haddara (n=70) — Supp Table 6")
    t7 = bias_table(maniscalco_bias, "Maniscalco (n=22) — Supp Table 7")
    t8 = bias_table(shekhar_bias,    "Shekhar (n=20) — Supp Table 8")

    # ── Table 9: Response bias ──
    print("\n[Computing response bias analysis (Locke)...]")
    locke_rb = compute_response_bias_locke(locke)
    t9 = response_bias_table(locke_rb, "Locke (n=10) — Supp Table 9")

    print("\n\nDone. All supplementary tables reproduced.")
