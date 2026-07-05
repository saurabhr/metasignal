"""Compute odd/even split-half reliability for Haddara and Maniscalco datasets and
merge 'split' key into their *_mle.npz caches (mirrors precompute_shekhar_split.py).

Required by 08_split_half_precision.ipynb, which loads ha_npz['split'] and
ma_npz['split'] with shape (n_sub, 2, 26) [odd=0, even=1].
"""
import sys, os, warnings
warnings.filterwarnings('ignore')

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT  = os.path.join(REPO, 'notebooks', 'precomputed')
sys.path.insert(0, os.path.join(REPO, 'src'))

import numpy as np
from metasignal.stdpy.compute_all import compute_all_measures

N_MEAS = 26


def compute_split(subjects):
    n_sub = len(subjects)
    split = np.full((n_sub, 2, N_MEAS), np.nan)
    for si, s in enumerate(subjects):
        print(f"  Subject {si+1}/{n_sub}...", flush=True)
        n = len(s['stim'])
        odd_idx  = np.arange(0, n, 2)
        even_idx = np.arange(1, n, 2)
        if len(odd_idx) < 10 or len(even_idx) < 10:
            continue
        try:
            split[si, 0] = compute_all_measures(
                s['stim'][odd_idx], s['resp'][odd_idx], s['conf'][odd_idx], s['n_ratings'])
            split[si, 1] = compute_all_measures(
                s['stim'][even_idx], s['resp'][even_idx], s['conf'][even_idx], s['n_ratings'])
        except Exception as e:
            print(f"    Error: {e}")
    return split


def merge_savez(path, **new_arrays):
    existing = {}
    if os.path.exists(path):
        d = np.load(path)
        existing = {k: d[k] for k in d.files}
    existing.update(new_arrays)
    np.savez(path, **existing)


if __name__ == '__main__':
    haddara = np.load(os.path.join(OUT, 'haddara.npz'), allow_pickle=True)['subjects']
    maniscalco = np.load(os.path.join(OUT, 'maniscalco.npz'), allow_pickle=True)['subjects']
    print(f"Loaded {len(haddara)} Haddara, {len(maniscalco)} Maniscalco subjects")

    print("Computing Haddara split-half...")
    ha_split = compute_split(haddara)
    merge_savez(os.path.join(OUT, 'haddara_mle.npz'), split=ha_split)
    print("  Saved Haddara split:", ha_split.shape)

    print("Computing Maniscalco split-half...")
    ma_split = compute_split(maniscalco)
    merge_savez(os.path.join(OUT, 'maniscalco_mle.npz'), split=ma_split)
    print("  Saved Maniscalco split:", ma_split.shape)

    print("Done.")
