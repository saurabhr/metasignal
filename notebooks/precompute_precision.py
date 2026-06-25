"""Compute precision analysis for Haddara dataset (how fast measures degrade with corrupted conf)."""
import sys, os, warnings
warnings.filterwarnings('ignore')

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT  = os.path.join(REPO, 'notebooks', 'precomputed')
sys.path.insert(0, os.path.join(REPO, 'src'))

import numpy as np
from metasignal.stdpy.compute_all import compute_all_measures

# Proportions of trials to corrupt
PROPS = [0.02, 0.04, 0.06]
N_MEAS = 20
# MLE measures excluded — too slow for repeated fitting across ~200 corruption calls
MLE_INDICES = [0, 5, 10]  # meta-d', M-Ratio, M-Diff

subjects = np.load(os.path.join(OUT, 'haddara.npz'), allow_pickle=True)['subjects']
ha_raw   = np.load(os.path.join(OUT, 'haddara_mle.npz'))['raw']  # (70, 20)
print(f"Loaded {len(subjects)} Haddara subjects")

n_sub  = len(subjects)
n_props = len(PROPS)
drops_raw = np.full((n_sub, n_props, N_MEAS), np.nan)
rng = np.random.default_rng(42)

for si, s in enumerate(subjects):
    print(f"  Subject {si+1}/{n_sub}...", flush=True)
    stim, resp, conf = s['stim'], s['resp'], s['conf']
    n_trials = len(stim)
    correct  = (stim == resp).astype(bool)

    for pi, prop in enumerate(PROPS):
        conf_corrupt = conf.copy().astype(float)
        n_corrupt = max(1, int(round(prop * n_trials)))
        corrupt_idx = rng.choice(n_trials, size=n_corrupt, replace=False)
        for idx in corrupt_idx:
            if correct[idx]:
                conf_corrupt[idx] = max(1, conf[idx] - 1)   # correct → lower conf
            else:
                conf_corrupt[idx] = min(s['n_ratings'], conf[idx] + 1)  # incorrect → higher conf
        try:
            m_corrupt = compute_all_measures(stim, resp, conf_corrupt, s['n_ratings'])
            drop = ha_raw[si] - m_corrupt
            drop[MLE_INDICES] = np.nan
            drops_raw[si, pi] = drop
        except Exception as e:
            print(f"    Error at prop {prop}: {e}")

# Normalize each measure by its across-subject SD; average over subjects → (n_props, N_MEAS)
raw_sd = np.nanstd(ha_raw, axis=0, ddof=1)
raw_sd[raw_sd == 0] = np.nan
drops_norm = drops_raw / raw_sd[np.newaxis, np.newaxis, :]
drops = np.nanmean(drops_norm, axis=0)  # (n_props, N_MEAS)

print("Drops shape:", drops.shape)
np.savez(os.path.join(OUT, 'haddara_precision.npz'), drops=drops, drops_per_subject=drops_raw)
print("Saved haddara_precision.npz")
