"""Replicate analysis_Shekhar.m dataset processing."""

import os
import scipy.io
import numpy as np
from metasignal.stdpy.compute_all import compute_all_measures
from analysis.helpers import xue_recode

def process_subject(args):
    sub_idx, sub_data, n_ratings, bin_size_sh, bin_size_tr = args
    print(f"Processing subject {sub_idx + 1}...")

    edges = np.linspace(50, 100, n_ratings + 1)
    num_contrasts = 3
    res = {}

    sub_contrast = sub_data['contrast'][0,0].flatten()
    sub_stim = sub_data['stim'][0,0].flatten()
    sub_resp = sub_data['resp'][0,0].flatten()
    sub_conf_raw = sub_data['conf'][0,0].flatten()
    sub_day = sub_data['day'][0,0].flatten()

    res['metas_raw'] = np.full((num_contrasts, 20), np.nan)
    res['metas_diff'] = np.full((num_contrasts, 20), np.nan)
    res['metas_confRecode'] = np.full((num_contrasts, 2, 20), np.nan)
    res['metas_oddEven'] = np.full((2, num_contrasts, 20), np.nan)

    split_half = []
    for bs in bin_size_sh:
        size = 2 * bs
        n_bins = int(800 // size)
        split_half.append(np.full((n_bins, num_contrasts, 2, 20), np.nan))
    res['metas_splitHalf'] = split_half

    test_retest = []
    for bs in bin_size_tr:
        n_bins = int(200 // bs)
        test_retest.append(np.full((num_contrasts, n_bins, num_contrasts, 20), np.nan))
    res['metas_testRetest'] = test_retest

    for contr in range(1, num_contrasts + 1):
        fc = sub_contrast == contr
        stim = sub_stim[fc]
        resp = sub_resp[fc]
        conf_raw = sub_conf_raw[fc]

        # Discretize confidence
        conf = np.digitize(conf_raw, edges, right=True)
        conf = np.clip(conf, 1, n_ratings)
        day_of_testing = sub_day[fc]

        stim_day = [stim[day_of_testing == c] for c in range(1, 4)]
        resp_day = [resp[day_of_testing == c] for c in range(1, 4)]
        conf_day = [conf[day_of_testing == c] for c in range(1, 4)]

        res['metas_raw'][contr-1, :] = compute_all_measures(stim, resp, conf, n_ratings)
        res['metas_diff'][contr-1, :] = compute_all_measures(stim, resp, conf, n_ratings)
        res['metas_confRecode'][contr-1, 0, :] = compute_all_measures(stim, resp, xue_recode(conf, 1), n_ratings-1)
        res['metas_confRecode'][contr-1, 1, :] = compute_all_measures(stim, resp, xue_recode(conf, 2), n_ratings-1)
        res['metas_oddEven'][0, contr-1, :] = compute_all_measures(stim[0::2], resp[0::2], conf[0::2], n_ratings)
        res['metas_oddEven'][1, contr-1, :] = compute_all_measures(stim[1::2], resp[1::2], conf[1::2], n_ratings)

        # Split Half
        for i, bs in enumerate(bin_size_sh):
            size = 2 * bs
            for bin_num in range(int(800 // size)):
                n_trials = len(stim)
                odds = np.zeros(n_trials, dtype=bool)
                evens = np.zeros(n_trials, dtype=bool)
                start_idx = bin_num * size
                limit = min(start_idx + size, n_trials)
                for j in range(start_idx, limit, 2):
                    odds[j] = True
                    if j+1 < limit:
                        evens[j+1] = True
                res['metas_splitHalf'][i][bin_num, contr-1, 0, :] = compute_all_measures(stim[odds], resp[odds], conf[odds], n_ratings)
                res['metas_splitHalf'][i][bin_num, contr-1, 1, :] = compute_all_measures(stim[evens], resp[evens], conf[evens], n_ratings)

        # Test Retest
        # Separated by contrast (1..3) and day (which corresponds to contrast here in MATLAB)
        for d_idx in range(1, 4):
            for i, bs in enumerate(bin_size_tr):
                for bin_num in range(int(200 // bs)):
                    start_idx = bin_num * bs
                    n_trials = len(stim_day[d_idx-1])
                    filt = np.zeros(n_trials, dtype=bool)
                    if start_idx + bs <= n_trials:
                        filt[start_idx:start_idx+bs] = True
                        s_f, r_f, c_f = stim_day[d_idx-1][filt], resp_day[d_idx-1][filt], conf_day[d_idx-1][filt]
                        res['metas_testRetest'][i][contr-1, bin_num, d_idx-1, :] = compute_all_measures(s_f, r_f, c_f, n_ratings)

    return res

if __name__ == '__main__':
    n_ratings = 6
    bin_size_sh = [50, 100, 200, 400]
    bin_size_tr = [50, 100, 200]

    print("Loading Dataset Shekhar...")
    mat = scipy.io.loadmat('matlab/metasignal_mat/Preprocess/dataset_Shekhar_2021.mat')
    data = mat['data'][0]

    args_list = [(i, data[i], n_ratings, bin_size_sh, bin_size_tr) for i in range(len(data))]

    print(f"Processing {len(args_list)} subjects sequentially (or with pool)...")
    results = [process_subject(a) for a in args_list[:2]]

    print("Saving sample output...")
    np.savez('analysis/results_Shekhar.npz', results=results)
    print("Done!")
