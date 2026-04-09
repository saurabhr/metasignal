"""Replicate analysis_Rouault2.m dataset processing."""

import os
import scipy.io
import numpy as np
from metasignal.stdpy.compute_all import compute_all_measures

def process_subject(args):
    sub_idx, sub_data, n_ratings = args
    print(f"Processing subject {sub_idx + 1}...")

    stim = sub_data['stim'][0,0].flatten()
    resp = sub_data['resp'][0,0].flatten()
    conf = sub_data['conf'][0,0].flatten()
    contrast = sub_data['contrast'][0,0].flatten()

    filt_high = contrast > np.median(contrast)

    res = {}
    res['metas_diff'] = np.zeros((2, 20))
    res['metas_diff'][0, :] = compute_all_measures(stim[~filt_high], resp[~filt_high], conf[~filt_high], n_ratings)
    res['metas_diff'][1, :] = compute_all_measures(stim[filt_high], resp[filt_high], conf[filt_high], n_ratings)

    return res

if __name__ == '__main__':
    n_ratings = 6

    print("Loading Dataset Rouault 2...")
    mat = scipy.io.loadmat('matlab/metasignal_mat/Preprocess/dataset_Rouault_2018_Expt2.mat')
    data = mat['data'][0]

    args_list = [(i, data[i], n_ratings) for i in range(len(data))]

    print(f"Processing {len(args_list)} subjects sequentially (or with pool)...")
    results = [process_subject(a) for a in args_list[:2]]

    print("Saving sample output...")
    np.savez('analysis/results_Rouault2.npz', results=results)
    print("Done!")
