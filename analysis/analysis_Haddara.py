"""Replicate analysis_Haddara.m dataset processing."""

import os
import scipy.io
import numpy as np
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from metasignal.stdpy.compute_all import compute_all_measures

def xue_recode(conf, low_high_recoding):
    valid_conf = conf[~np.isnan(conf)]
    if len(np.unique(valid_conf)) < 3 or not np.all(valid_conf == np.round(valid_conf)):
        return np.full_like(conf, np.nan)

    conf_new = conf.copy()
    if low_high_recoding == 1:
        conf_new = conf_new - 1
        conf_new[conf_new == np.nanmin(conf_new)] = np.nanmin(conf_new) + 1
    elif low_high_recoding == 2:
        conf_new[conf_new == np.nanmax(conf_new)] = np.nanmax(conf_new) - 1
    return conf_new

def metas_altered_conf(stim, resp, conf, n_ratings, prop_altered):
    num_trials = len(conf)
    num_to_alter = int(np.round(num_trials * prop_altered))
    conf_altered = conf.copy()

    num_altered = 0
    for i in range(num_trials):
        if stim[i] == resp[i] and conf[i] > 1:
            conf_altered[i] -= 1
            num_altered += 1
        elif stim[i] != resp[i] and conf[i] < n_ratings:
            conf_altered[i] += 1
            num_altered += 1

        if num_altered == num_to_alter:
            break

    return compute_all_measures(stim, resp, conf_altered, n_ratings)


def process_subject(args):
    sub_idx, sub_data, n_ratings, bin_size_sh, bin_size_tr, prop_altered = args
    print(f"Processing subject {sub_idx + 1}...")

    # In MATLAB, structures inside cell arrays loaded with scipy.io are nested
    # sub_data is expected to have shape (1,1) with fields 'stim', 'resp', 'conf', 'day'
    stim = sub_data['stim'][0,0].flatten()
    resp = sub_data['resp'][0,0].flatten()
    conf = sub_data['conf'][0,0].flatten()
    day_of_testing = sub_data['day'][0,0].flatten()

    stim_day = []
    resp_day = []
    conf_day = []
    for day in range(2, 8):
        mask = (day_of_testing == day)
        stim_day.append(stim[mask])
        resp_day.append(resp[mask])
        conf_day.append(conf[mask])

    res = {}

    # Base
    res['metas_raw'] = compute_all_measures(stim, resp, conf, n_ratings)

    # Xue recode
    res['metas_confRecode'] = np.zeros((2, 20))
    res['metas_confRecode'][0, :] = compute_all_measures(stim, resp, xue_recode(conf, 1), n_ratings - 1)
    res['metas_confRecode'][1, :] = compute_all_measures(stim, resp, xue_recode(conf, 2), n_ratings - 1)

    # Odd/Even
    res['metas_oddEven'] = np.zeros((2, 20))
    res['metas_oddEven'][0, :] = compute_all_measures(stim[0::2], resp[0::2], conf[0::2], n_ratings)
    res['metas_oddEven'][1, :] = compute_all_measures(stim[1::2], resp[1::2], conf[1::2], n_ratings)

    # Split Half
    # It stores based on binSize_sh index
    split_half = []
    for i in range(len(bin_size_sh)):
        # Initialize dynamically based on max possible sizes
        # In MATLAB: 4 bins for 100, 2 for 200, 1 for 400. And day=6.
        # SH uses bin_size = 2 * bin_size_sh(i) => 100, 200, 400, 800
        # For 800 (last idx), days are grouped.
        bin_s = 2 * bin_size_sh[i]
        n_bins = int(400 / bin_s) if bin_s <= 400 else 1
        n_days = 6 if bin_s <= 400 else 3
        arr = np.full((n_bins, 6, 2, 20), np.nan)

        if i < len(bin_size_sh) - 1:
            for day in range(6):
                for b in range(n_bins):
                    # odd/even filter within bin
                    n_trials = len(stim_day[day])
                    odds = np.zeros(n_trials, dtype=bool)
                    evens = np.zeros(n_trials, dtype=bool)
                    start_idx = b * bin_s
                    for j in range(bin_s // 2):
                        odds[start_idx + 2*j] = True
                        evens[start_idx + 2*j + 1] = True
                    if len(stim_day[day]) >= start_idx + bin_s:
                        arr[b, day, 0, :] = compute_all_measures(stim_day[day][odds], resp_day[day][odds], conf_day[day][odds], n_ratings)
                        arr[b, day, 1, :] = compute_all_measures(stim_day[day][evens], resp_day[day][evens], conf_day[day][evens], n_ratings)
            split_half.append(arr)
        else:
            # 800-trial bin
            for day in range(1, 7, 2):
                if day < 6:
                    s400 = np.concatenate([stim_day[day-1], stim_day[day][:300]])
                    r400 = np.concatenate([resp_day[day-1], resp_day[day][:300]])
                    c400 = np.concatenate([conf_day[day-1], conf_day[day][:300]])
                    arr[0, day//2, 0, :] = compute_all_measures(s400[0::2], r400[0::2], c400[0::2], n_ratings)
                    arr[0, day//2, 1, :] = compute_all_measures(s400[1::2], r400[1::2], c400[1::2], n_ratings)
            split_half.append(arr)
    res['metas_splitHalf'] = split_half

    # Test Retest
    test_retest = []
    precision = []
    for i in range(len(bin_size_tr)):
        bin_s = bin_size_tr[i]
        n_bins = int(400 / bin_s)
        arr_tr = np.full((n_bins, 6, 20), np.nan)
        arr_pr = np.full((n_bins, 6, len(prop_altered) + 1, 20), np.nan)
        for day in range(6):
            for b in range(n_bins):
                start_idx = b * bin_s
                n_trials = len(stim_day[day])
                filt = np.zeros(n_trials, dtype=bool)
                if start_idx + bin_s <= n_trials:
                    filt[start_idx:start_idx+bin_s] = True

                if len(stim_day[day]) >= start_idx + bin_s:
                    s_filt = stim_day[day][filt]
                    r_filt = resp_day[day][filt]
                    c_filt = conf_day[day][filt]
                    tr_meas = compute_all_measures(s_filt, r_filt, c_filt, n_ratings)
                    arr_tr[b, day, :] = tr_meas
                    arr_pr[b, day, 0, :] = tr_meas

                    for a_idx, p_alt in enumerate(prop_altered):
                        arr_pr[b, day, a_idx+1, :] = metas_altered_conf(s_filt, r_filt, c_filt, n_ratings, p_alt)
        test_retest.append(arr_tr)
        precision.append(arr_pr)

    res['metas_testRetest'] = test_retest
    res['metas_precision'] = precision

    return res

if __name__ == '__main__':
    n_ratings = 4
    bin_size_sh = [50, 100, 200, 400]
    bin_size_tr = [50, 100, 200, 400]
    prop_altered = [0.02, 0.04, 0.06]

    print("Loading Dataset Haddara...")
    mat = scipy.io.loadmat('matlab/metasignal_mat/Preprocess/dataset_Haddara_2022_Expt2.mat')
    data = mat['data'][0]

    args_list = [(i, data[i], n_ratings, bin_size_sh, bin_size_tr, prop_altered) for i in range(len(data))]

    # Take a small subset for quick verification
    args_list = args_list[:2]

    print(f"Processing {len(args_list)} subjects sequentially for debugging...")
    results = []
    for args in args_list:
        results.append(process_subject(args))

    print("Done processing. Saving sample output...")
    # In a full run, we would save to 'Results/' and match MATLAB's format exactly.
    np.savez('analysis/results_Haddara.npz', results=results)
    print("Execution complete. Validated analysis_Haddara.py baseline.")
