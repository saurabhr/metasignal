"""Replicate analysis_Maniscalco.m dataset processing."""

import os
import scipy.io
import numpy as np
from multiprocessing import Pool
from metasignal.stdpy.compute_all import compute_all_measures
from analysis.helpers import xue_recode, metas_altered_conf

def process_subject(args):
    sub_idx, sub_data, n_ratings, bin_size_sh, prop_altered = args
    print(f"Processing subject {sub_idx + 1}...")

    stim = sub_data['stim'][0,0].flatten()
    resp = sub_data['resp'][0,0].flatten()
    conf = sub_data['conf'][0,0].flatten()

    res = {}

    # Base
    res['metas_raw'] = compute_all_measures(stim, resp, conf, n_ratings)
    num_trials = len(conf)

    # Precision
    precision = []
    for i, bin_s in enumerate(bin_size_sh):
        n_bins = int(num_trials // bin_s)
        arr_pr = np.full((n_bins, len(prop_altered) + 1, 20), np.nan)
        for b in range(n_bins):
            start_idx = b * bin_s
            filt = np.zeros(num_trials, dtype=bool)
            if start_idx + bin_s <= num_trials:
                filt[start_idx:start_idx+bin_s] = True
                s_filt, r_filt, c_filt = stim[filt], resp[filt], conf[filt]

                arr_pr[b, 0, :] = compute_all_measures(s_filt, r_filt, c_filt, n_ratings)
                for a_idx, p_alt in enumerate(prop_altered):
                    arr_pr[b, a_idx+1, :] = metas_altered_conf(s_filt, r_filt, c_filt, n_ratings, p_alt)
        precision.append(arr_pr)
    res['metas_precision'] = precision

    # Xue recode
    res['metas_confRecode'] = np.zeros((2, 20))
    res['metas_confRecode'][0, :] = compute_all_measures(stim, resp, xue_recode(conf, 1), n_ratings - 1)
    res['metas_confRecode'][1, :] = compute_all_measures(stim, resp, xue_recode(conf, 2), n_ratings - 1)

    # Odd/Even
    res['metas_oddEven'] = np.zeros((2, 20))
    res['metas_oddEven'][0, :] = compute_all_measures(stim[0::2], resp[0::2], conf[0::2], n_ratings)
    res['metas_oddEven'][1, :] = compute_all_measures(stim[1::2], resp[1::2], conf[1::2], n_ratings)

    # Split Half
    split_half = []
    for i, bs_half in enumerate(bin_size_sh):
        bin_s = 2 * bs_half
        n_bins = int(num_trials // bin_s)
        arr_sh = np.full((n_bins, 2, 20), np.nan)
        for b in range(n_bins):
            odds = np.zeros(num_trials, dtype=bool)
            evens = np.zeros(num_trials, dtype=bool)
            start_idx = b * bin_s
            for j in range(bs_half):
                odds[start_idx + 2*j] = True
                evens[start_idx + 2*j + 1] = True

            arr_sh[b, 0, :] = compute_all_measures(stim[odds], resp[odds], conf[odds], n_ratings)
            arr_sh[b, 1, :] = compute_all_measures(stim[evens], resp[evens], conf[evens], n_ratings)
        split_half.append(arr_sh)
    res['metas_splitHalf'] = split_half

    return res

if __name__ == '__main__':
    n_ratings = 4
    bin_size_sh = [50, 100, 200, 400]
    prop_altered = [0.02, 0.04, 0.06]

    print("Loading Dataset Maniscalco...")
    mat = scipy.io.loadmat('matlab/metasignal_mat/Preprocess/dataset_Maniscalco_2017_expt1.mat')
    data = mat['data'][0]

    args_list = [(i, data[i], n_ratings, bin_size_sh, prop_altered) for i in range(len(data))]

    print(f"Processing {len(args_list)} subjects sequentially (or with pool)...")
    results = [process_subject(a) for a in args_list[:2]]

    print("Saving sample output...")
    np.savez('analysis/results_Maniscalco.npz', results=results)
    print("Done!")
