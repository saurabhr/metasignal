#!/usr/bin/env python3
"""Bin-stratified split-half and precision, matching Rahnev (2025) Methods exactly.

The scripts this replaces (notebooks/precompute_ha_ma_split.py,
notebooks/precompute_precision.py) use a single whole-dataset odd/even split
and a single whole-dataset confidence-corruption run. The published protocol
instead requires non-overlapping bins of 50/100/200/400 trials, analyzed
separately per bin and then averaged (Methods: "Assessing split-half
reliability", "Assessing validity and precision").

Split-half (bin=100, matches Fig. 7's reported quantity exactly):
  - Haddara: days 2-7 (500 trials/day), 5 non-overlapping 100-trial bins/day
    -> 30 bin-instances.
  - Maniscalco: floor(n_trials/100) non-overlapping bins per subject.
  - Shekhar: per contrast (difficulty) level, floor(n_trials_at_level/100)
    bins; averaged across the 3 levels per Methods.
  For each bin instance: split into odd/even trial positions (local to the
  bin), compute all 26 measures per half per subject, Pearson-correlate
  odd-vs-even across subjects -> one r per bin instance. Fisher-z average
  across bin instances -> one r per dataset. Simple mean across the three
  datasets -> final value, comparable to PAPER["split_half"].

Precision (Eq. 3, averaged over bin sizes 50/100/200/400 and corruption
2/4/6%, Haddara + Maniscalco only, matching Fig. 1b/7):
  ponytail: the fully exhaustive bin-instance grid is ~30k
  compute_all_measures calls (~6h). Capped at MAX_BINS_PER_SIZE
  non-overlapping bins per size (randomly subsampled when more are
  available) to keep this tractable in one background run. Raise the cap
  (or remove it) if a future run has hours to spare and full exhaustiveness
  matters more than turnaround time.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "notebooks"))

import numpy as np
from analysis_core import preprocess_haddara, preprocess_maniscalco, preprocess_shekhar
from metasignal.stdpy.compute_all import compute_all_measures

OUT_JSON = os.path.join(REPO, "analysis", "rahnev_comparison", "figures", "validation", "reliability_proper.json")
N_MEAS = 26
MAX_BINS_PER_SIZE = 15
RNG = np.random.default_rng(0)


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_measures(stim, resp, conf, n_ratings):
    try:
        return compute_all_measures(stim, resp, conf, int(n_ratings), matlab_compat=True)
    except Exception:
        return np.full(N_MEAS, np.nan)


def fisher_mean(rs):
    rs = np.asarray(rs, float)
    rs = rs[np.isfinite(rs)]
    if not len(rs):
        return np.nan
    rs = np.clip(rs, -0.999999, 0.999999)
    return float(np.tanh(np.mean(np.arctanh(rs))))


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def bin_starts(n_trials: int, bin_size: int) -> list[int]:
    n_bins = n_trials // bin_size
    return [i * bin_size for i in range(n_bins)]


# ---------------------------------------------------------------------------
# Split-half (bin = 100 only, matches Fig. 7)
#
# Rahnev (2025) Methods: "a bin size of k here means that 2k trials were
# examined with both the odd and even trials having a sample size of k."
# bin_size below is therefore the PER-HALF trial count k; the window read
# from the trial stream is 2*bin_size, split into odd/even halves of
# bin_size each. (Previously this function read a window of only
# `bin_size` trials and split THAT in half, giving bin_size/2 trials per
# half -- half the intended sample size, which is why the split-half
# correlations undershot the published Fig. 7 values.)
# ---------------------------------------------------------------------------


def split_half_for_bin(stim, resp, conf, n_ratings, start, bin_size):
    window = 2 * bin_size
    seg_stim = stim[start : start + window]
    seg_resp = resp[start : start + window]
    seg_conf = conf[start : start + window]
    odd = slice(0, None, 2)
    even = slice(1, None, 2)
    m_odd = safe_measures(seg_stim[odd], seg_resp[odd], seg_conf[odd], n_ratings)
    m_even = safe_measures(seg_stim[even], seg_resp[even], seg_conf[even], n_ratings)
    return m_odd, m_even


def dataset_split_half_haddara(bin_size=100):
    subjects = preprocess_haddara()
    per_bin_r = []
    n_bins_total = 0
    for day in range(2, 8):
        for si, s in enumerate(subjects):
            pass
        # Collect, per (day, local-bin-index), the odd/even vectors across subjects.
        day_mask_trials = None
    # Re-organize: iterate over days, then over local bin index within the day,
    # collecting one (n_subj, 26) odd matrix and even matrix per bin instance.
    for day in range(2, 8):
        day_subjects = []
        for s in subjects:
            mask = s["day"] == day
            day_subjects.append((s["stim"][mask], s["resp"][mask], s["conf"][mask], s["n_ratings"]))
        n_trials_day = min(len(ds[0]) for ds in day_subjects) if day_subjects else 0
        starts = bin_starts(n_trials_day, 2 * bin_size)
        for start in starts:
            odd_mat, even_mat = [], []
            for stim, resp, conf, nr in day_subjects:
                m_odd, m_even = split_half_for_bin(stim, resp, conf, nr, start, bin_size)
                odd_mat.append(m_odd)
                even_mat.append(m_even)
            odd_mat = np.array(odd_mat)
            even_mat = np.array(even_mat)
            rs = [correlation(odd_mat[:, m], even_mat[:, m]) for m in range(17)]
            per_bin_r.append(rs)
            n_bins_total += 1
            log(f"    Haddara day{day} bin@{start}: done ({n_bins_total} bin-instances so far)")
    per_bin_r = np.array(per_bin_r)  # (n_bins, 17)
    return np.array([fisher_mean(per_bin_r[:, m]) for m in range(17)])


def dataset_split_half_maniscalco(bin_size=100):
    subjects = preprocess_maniscalco()
    n_trials = min(len(s["stim"]) for s in subjects)
    starts = bin_starts(n_trials, 2 * bin_size)
    per_bin_r = []
    for bi, start in enumerate(starts):
        odd_mat, even_mat = [], []
        for s in subjects:
            m_odd, m_even = split_half_for_bin(s["stim"], s["resp"], s["conf"], s["n_ratings"], start, bin_size)
            odd_mat.append(m_odd)
            even_mat.append(m_even)
        odd_mat, even_mat = np.array(odd_mat), np.array(even_mat)
        rs = [correlation(odd_mat[:, m], even_mat[:, m]) for m in range(17)]
        per_bin_r.append(rs)
        log(f"    Maniscalco bin@{start}: done ({bi + 1}/{len(starts)})")
    per_bin_r = np.array(per_bin_r)
    return np.array([fisher_mean(per_bin_r[:, m]) for m in range(17)])


def dataset_split_half_shekhar(bin_size=100):
    subjects = preprocess_shekhar()
    levels = np.unique(subjects[0]["contrast"])
    level_results = []
    for level in levels:
        level_subjects = []
        for s in subjects:
            mask = s["contrast"] == level
            level_subjects.append((s["stim"][mask], s["resp"][mask], s["conf"][mask], s["n_ratings"]))
        n_trials = min(len(ls[0]) for ls in level_subjects)
        starts = bin_starts(n_trials, 2 * bin_size)
        per_bin_r = []
        for bi, start in enumerate(starts):
            odd_mat, even_mat = [], []
            for stim, resp, conf, nr in level_subjects:
                m_odd, m_even = split_half_for_bin(stim, resp, conf, nr, start, bin_size)
                odd_mat.append(m_odd)
                even_mat.append(m_even)
            odd_mat, even_mat = np.array(odd_mat), np.array(even_mat)
            rs = [correlation(odd_mat[:, m], even_mat[:, m]) for m in range(17)]
            per_bin_r.append(rs)
            log(f"    Shekhar level {level} bin@{start}: done ({bi + 1}/{len(starts)})")
        per_bin_r = np.array(per_bin_r)
        level_results.append([fisher_mean(per_bin_r[:, m]) for m in range(17)])
    return np.nanmean(np.array(level_results), axis=0)


def run_split_half():
    log("=== Split-half (bin=100) ===")
    t0 = time.time()
    haddara = dataset_split_half_haddara()
    log(f"Haddara done in {time.time() - t0:.0f}s")
    t1 = time.time()
    maniscalco = dataset_split_half_maniscalco()
    log(f"Maniscalco done in {time.time() - t1:.0f}s")
    t2 = time.time()
    shekhar = dataset_split_half_shekhar()
    log(f"Shekhar done in {time.time() - t2:.0f}s")
    final = np.nanmean(np.array([haddara, maniscalco, shekhar]), axis=0)
    return {
        "haddara": haddara.tolist(),
        "maniscalco": maniscalco.tolist(),
        "shekhar": shekhar.tolist(),
        "combined": final.tolist(),
    }


# ---------------------------------------------------------------------------
# Precision (Eq. 3, bin-averaged, capped bin-instance count for tractability)
# ---------------------------------------------------------------------------

PROPS = [0.02, 0.04, 0.06]
BIN_SIZES = [50, 100, 200, 400]


def corrupt(stim, resp, conf, n_ratings, prop, rng):
    conf_c = conf.copy().astype(float)
    correct = stim == resp
    n = len(stim)
    n_corrupt = max(1, int(round(prop * n)))
    idx = rng.choice(n, size=n_corrupt, replace=False)
    for i in idx:
        if correct[i]:
            conf_c[i] = max(1, conf[i] - 1)
        else:
            conf_c[i] = min(n_ratings, conf[i] + 1)
    return conf_c


def precision_for_dataset(name, subjects, day_key=None):
    """Bin-stratified precision (Eq. 3) for one dataset, capped bin count per size."""
    results_by_size = []
    for bin_size in BIN_SIZES:
        if day_key is None:
            # Maniscalco: flat trial pool per subject.
            all_starts_per_subj = [bin_starts(len(s["stim"]), bin_size) for s in subjects]
            n_bins_avail = min(len(x) for x in all_starts_per_subj)
        else:
            # Haddara: per-day bins, days 2-7.
            n_trials_day = min(np.sum(s["day"] == 2) for s in subjects)
            n_bins_avail = (n_trials_day // bin_size) * 6  # 6 days
        n_use = min(n_bins_avail, MAX_BINS_PER_SIZE)
        log(f"  {name} bin={bin_size}: {n_bins_avail} bin-instances available, using {n_use}")

        orig_vals, drop_vals = [], {p: [] for p in PROPS}
        bins_done = 0
        if day_key is None:
            starts_pool = list(range(min(len(s["stim"]) for s in subjects) // bin_size))
            chosen = RNG.choice(starts_pool, size=n_use, replace=False) if len(starts_pool) > n_use else np.array(starts_pool)
            for local_bin in chosen:
                start = local_bin * bin_size
                subj_orig, subj_drop = [], {p: [] for p in PROPS}
                for s in subjects:
                    seg_stim = s["stim"][start : start + bin_size]
                    seg_resp = s["resp"][start : start + bin_size]
                    seg_conf = s["conf"][start : start + bin_size]
                    m_orig = safe_measures(seg_stim, seg_resp, seg_conf, s["n_ratings"])
                    subj_orig.append(m_orig)
                    for p in PROPS:
                        conf_c = corrupt(seg_stim, seg_resp, seg_conf, s["n_ratings"], p, RNG)
                        m_c = safe_measures(seg_stim, seg_resp, conf_c, s["n_ratings"])
                        subj_drop[p].append(m_orig - m_c)
                orig_vals.append(np.array(subj_orig))
                for p in PROPS:
                    drop_vals[p].append(np.array(subj_drop[p]))
                bins_done += 1
                log(f"    {name} bin={bin_size} instance {bins_done}/{n_use} done")
        else:
            days = list(range(2, 8))
            day_bin_pairs = []
            bins_per_day = (min(np.sum(s["day"] == 2) for s in subjects) // bin_size)
            for day in days:
                for lb in range(bins_per_day):
                    day_bin_pairs.append((day, lb))
            chosen_idx = RNG.choice(len(day_bin_pairs), size=n_use, replace=False) if len(day_bin_pairs) > n_use else np.arange(len(day_bin_pairs))
            for ci in chosen_idx:
                day, local_bin = day_bin_pairs[ci]
                start = local_bin * bin_size
                subj_orig, subj_drop = [], {p: [] for p in PROPS}
                for s in subjects:
                    mask = s["day"] == day
                    seg_stim = s["stim"][mask][start : start + bin_size]
                    seg_resp = s["resp"][mask][start : start + bin_size]
                    seg_conf = s["conf"][mask][start : start + bin_size]
                    m_orig = safe_measures(seg_stim, seg_resp, seg_conf, s["n_ratings"])
                    subj_orig.append(m_orig)
                    for p in PROPS:
                        conf_c = corrupt(seg_stim, seg_resp, seg_conf, s["n_ratings"], p, RNG)
                        m_c = safe_measures(seg_stim, seg_resp, conf_c, s["n_ratings"])
                        subj_drop[p].append(m_orig - m_c)
                orig_vals.append(np.array(subj_orig))
                for p in PROPS:
                    drop_vals[p].append(np.array(subj_drop[p]))
                bins_done += 1
                log(f"    {name} bin={bin_size} instance {bins_done}/{n_use} done (day {day})")

        orig_stack = np.concatenate(orig_vals, axis=0)  # (n_use*n_subj, 26)
        sd = np.nanstd(orig_stack, axis=0, ddof=1)
        sd[sd == 0] = np.nan
        prop_precisions = []
        for p in PROPS:
            drop_stack = np.concatenate(drop_vals[p], axis=0)
            prop_precisions.append(np.nanmean(drop_stack, axis=0) / sd)
        results_by_size.append(np.nanmean(np.array(prop_precisions), axis=0))
    return np.nanmean(np.array(results_by_size), axis=0)[:17]


def run_precision():
    log("=== Precision (bin-averaged, capped) ===")
    t0 = time.time()
    haddara_subjects = preprocess_haddara()
    haddara = precision_for_dataset("Haddara", haddara_subjects, day_key="day")
    log(f"Haddara precision done in {time.time() - t0:.0f}s")
    t1 = time.time()
    maniscalco_subjects = preprocess_maniscalco()
    maniscalco = precision_for_dataset("Maniscalco", maniscalco_subjects, day_key=None)
    log(f"Maniscalco precision done in {time.time() - t1:.0f}s")
    combined_raw = np.nanmean(np.array([haddara, maniscalco]), axis=0)
    combined_renorm = combined_raw / np.nanmean(combined_raw[:16])
    return {
        "haddara": haddara.tolist(),
        "maniscalco": maniscalco.tolist(),
        "combined_raw": combined_raw.tolist(),
        "combined_renormalized_mean16": combined_renorm.tolist(),
    }


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    out = {}
    out["split_half"] = run_split_half()
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Wrote partial results (split-half) to {OUT_JSON}")

    out["precision"] = run_precision()
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Wrote final results to {OUT_JSON}")


if __name__ == "__main__":
    main()
