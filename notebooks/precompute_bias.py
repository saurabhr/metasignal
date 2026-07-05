"""Compute metacognitive-bias (Xue recode) measures for Haddara, Maniscalco, Shekhar
and response-bias measures for Locke, then append/merge into the existing *_mle.npz
caches produced by 02_compute_measures.ipynb, so that 03/04/06/07/08 notebooks can
load ha_npz['bias'], ma_npz['bias'], shekhar_mle.npz['bias'] (per-contrast, unaveraged),
and locke_mle.npz['rb'] is already present from notebook 02.

This mirrors analysis_core.py's compute_bias / compute_bias_shekhar, but preserves the
per-contrast Shekhar shape (20,3,2,26) that 06_metacognitive_bias.ipynb expects (it
averages over contrasts itself), rather than analysis_core.py's pre-averaged (20,2,26).
"""
import sys, os, warnings
warnings.filterwarnings('ignore')

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT  = os.path.join(REPO, 'notebooks', 'precomputed')
sys.path.insert(0, os.path.join(REPO, 'src'))

import numpy as np
from metasignal.stdpy.compute_all import compute_all_measures

N_MEAS = 26


def xue_recode(conf, rtype):
    valid = conf[~np.isnan(conf)]
    if len(np.unique(valid)) < 3:
        return np.full_like(conf, np.nan, dtype=float)
    c = conf.copy().astype(float)
    if rtype == 1:
        c -= 1
        cmin = np.nanmin(c)
        c[c == cmin] = cmin + 1
    else:
        cmax = np.nanmax(c)
        c[c == cmax] = cmax - 1
    return c


def compute_bias(subjects):
    """Returns (n_sub, 2, N_MEAS): dim1 = [recode1 (high-conf bias), recode2 (low-conf bias)]."""
    out = []
    for s in subjects:
        nr = s['n_ratings']
        row = np.full((2, N_MEAS), np.nan)
        for ri, rtype in enumerate([1, 2]):
            conf_r = xue_recode(s['conf'], rtype)
            if not np.all(np.isnan(conf_r)):
                row[ri] = compute_all_measures(s['stim'], s['resp'], conf_r, nr - 1)
        out.append(row)
    return np.array(out)


def compute_bias_shekhar_full(subjects):
    """Returns (n_sub, 3, 2, N_MEAS): per-contrast, unaveraged (matches 06's expected shape)."""
    out = []
    for s in subjects:
        nr = s['n_ratings']
        per_contrast = np.full((3, 2, N_MEAS), np.nan)
        for ci, c in enumerate([1, 2, 3]):
            mask = s['contrast'] == c
            if mask.sum() < 10:
                continue
            for ri, rtype in enumerate([1, 2]):
                conf_r = xue_recode(s['conf'][mask], rtype)
                if not np.all(np.isnan(conf_r)):
                    per_contrast[ci, ri] = compute_all_measures(
                        s['stim'][mask], s['resp'][mask], conf_r, nr - 1)
        out.append(per_contrast)
    return np.array(out)


def merge_savez(path, **new_arrays):
    """Load existing npz (if any), merge in new_arrays, and re-save."""
    existing = {}
    if os.path.exists(path):
        d = np.load(path)
        existing = {k: d[k] for k in d.files}
    existing.update(new_arrays)
    np.savez(path, **existing)


if __name__ == '__main__':
    haddara = np.load(os.path.join(OUT, 'haddara.npz'), allow_pickle=True)['subjects']
    maniscalco = np.load(os.path.join(OUT, 'maniscalco.npz'), allow_pickle=True)['subjects']
    shekhar = np.load(os.path.join(OUT, 'shekhar.npz'), allow_pickle=True)['subjects']
    print(f"Loaded {len(haddara)} Haddara, {len(maniscalco)} Maniscalco, {len(shekhar)} Shekhar subjects")

    print("Computing Haddara bias...")
    ha_bias = compute_bias(haddara)
    merge_savez(os.path.join(OUT, 'haddara_mle.npz'), bias=ha_bias)
    print("  Saved Haddara bias:", ha_bias.shape)

    print("Computing Maniscalco bias...")
    ma_bias = compute_bias(maniscalco)
    merge_savez(os.path.join(OUT, 'maniscalco_mle.npz'), bias=ma_bias)
    print("  Saved Maniscalco bias:", ma_bias.shape)

    print("Computing Shekhar bias (per-contrast, unaveraged)...")
    sh_bias = compute_bias_shekhar_full(shekhar)
    merge_savez(os.path.join(OUT, 'shekhar_mle.npz'), bias=sh_bias)
    print("  Saved Shekhar bias:", sh_bias.shape)

    print("Done.")
