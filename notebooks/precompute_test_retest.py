"""Compute per-day MLE measures for Haddara test-retest reliability (days 2–7)."""
import sys, os, warnings
warnings.filterwarnings('ignore')

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT  = os.path.join(REPO, 'notebooks', 'precomputed')
sys.path.insert(0, os.path.join(REPO, 'src'))

import numpy as np
from metasignal.stdpy.compute_all import compute_all_measures

DAYS   = [2, 3, 4, 5, 6, 7]
N_MEAS = 26

subjects = np.load(os.path.join(OUT, 'haddara.npz'), allow_pickle=True)['subjects']
print(f"Loaded {len(subjects)} Haddara subjects")

n_sub  = len(subjects)
n_days = len(DAYS)
tt = np.full((n_sub, n_days, N_MEAS), np.nan)

for si, s in enumerate(subjects):
    print(f"  Subject {si+1}/{n_sub}...", flush=True)
    for di, day in enumerate(DAYS):
        mask = s['day'] == day
        if mask.sum() < 10:
            continue
        try:
            tt[si, di] = compute_all_measures(
                s['stim'][mask], s['resp'][mask], s['conf'][mask], s['n_ratings'])
        except Exception as e:
            print(f"    Error at day {day}: {e}")

print("Test-retest shape:", tt.shape)
np.savez(os.path.join(OUT, 'haddara_testRetest.npz'), data=tt)
print("Saved haddara_testRetest.npz")
