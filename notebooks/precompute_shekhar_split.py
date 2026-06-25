"""Compute split-half reliability for Shekhar dataset and add to shekhar_mle.npz."""
import sys, os, warnings
warnings.filterwarnings('ignore')

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT  = os.path.join(REPO, 'notebooks', 'precomputed')
sys.path.insert(0, os.path.join(REPO, 'src'))

import numpy as np
from metasignal.stdpy.compute_all import compute_all_measures

N_MEAS = 20

subjects = np.load(os.path.join(OUT, 'shekhar.npz'), allow_pickle=True)['subjects']
print(f"Loaded {len(subjects)} Shekhar subjects")

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

print("Split shape:", split.shape)

existing = np.load(os.path.join(OUT, 'shekhar_mle.npz'), allow_pickle=True)
save_dict = {k: existing[k] for k in existing.files}
save_dict['split'] = split
np.savez(os.path.join(OUT, 'shekhar_mle.npz'), **save_dict)
print("Saved shekhar_mle.npz with 'split' key.")
