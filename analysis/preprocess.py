"""Replicate step1_importDataToMatlab.m and step2_preprocessData.m."""

import os
import glob
import numpy as np
import pandas as pd
from scipy import stats

def preprocess_datasets():
    csv_dir = 'matlab/metasignal_mat/Preprocess/orig_csv_files'
    dataset_names = [
        'Haddara_2022_Expt2', 'Locke_2020', 'Maniscalco_2017_expt1',
        'Rouault_2018_Expt1', 'Rouault_2018_Expt2', 'Shekhar_2021'
    ]

    acc_thresholds = [0.6, 0.95]
    max_prop_same = 0.85

    stats_out = {}

    for dset in dataset_names:
        csv_file = os.path.join(csv_dir, f'data_{dset}.csv')
        if not os.path.exists(csv_file):
            print(f"Skipping {dset}: {csv_file} not found.")
            continue

        print(f"Processing {dset}...")
        df = pd.read_csv(csv_file)

        # Step 1: Read and organize base arrays
        subj_idx = df['Subj_idx'].to_numpy()
        stim = df['Stimulus'].to_numpy()
        resp = df['Response'].to_numpy()
        conf = df['Confidence'].to_numpy()

        # Dataset specifics
        condition = None
        contrast = None
        day = None

        if dset == 'Locke_2020':
            training = df['Training'].to_numpy()
            filt_trn = (training == 0)
            subj_idx = subj_idx[filt_trn]
            stim = stim[filt_trn]
            resp = resp[filt_trn]
            conf = conf[filt_trn] + 1
            condition = df['Condition'].to_numpy()[filt_trn]
        elif dset == 'Shekhar_2021':
            contrast = df['Contrast'].to_numpy()
            day = np.tile(np.concatenate([np.ones(800), 2*np.ones(1000), 3*np.ones(1000)]), 20)
        elif dset.startswith('Rouault'):
            contrast = df['DotDiff'].to_numpy()
        elif dset == 'Haddara_2022_Expt2':
            day = df['Day'].to_numpy()

        # Step 2: Preprocess & Filter
        num_excluded = 0
        total_trials = 0
        u_subjs = np.unique(subj_idx)

        data_clean = []
        for s in u_subjs:
            s_filt = (subj_idx == s)
            s_stim = stim[s_filt]
            s_resp = resp[s_filt]
            s_conf = conf[s_filt]
            s_correct = (s_stim == s_resp).astype(int)

            s_acc = np.mean(s_correct)
            mode_resp_prop = np.max(np.unique(s_resp, return_counts=True)[1]) / len(s_resp)
            mode_conf_prop = np.max(np.unique(s_conf, return_counts=True)[1]) / len(s_conf)

            if s_acc < acc_thresholds[0] or s_acc > acc_thresholds[1] or \
               mode_resp_prop > max_prop_same or mode_conf_prop > max_prop_same:
                num_excluded += 1
            else:
                s_dict = {
                    'stim': s_stim,
                    'resp': s_resp,
                    'conf': s_conf
                }
                if contrast is not None: s_dict['contrast'] = contrast[s_filt]
                if condition is not None: s_dict['condition'] = condition[s_filt]
                if day is not None: s_dict['day'] = day[s_filt]
                data_clean.append(s_dict)
                total_trials += len(s_stim)

        # Save output
        out_file = f'analysis/dataset_{dset}.npy'
        np.save(out_file, data_clean)

        stats_out[dset] = {
            'num_excluded': num_excluded,
            'percent_excluded': (num_excluded / len(u_subjs)) * 100,
            'number_good_subjects': len(data_clean),
            'trials_per_subj': total_trials / len(data_clean) if len(data_clean) > 0 else 0
        }

    print("\n--- Summary ---")
    for dset, s in stats_out.items():
        print(f"{dset}: Excluded {s['num_excluded']} ({s['percent_excluded']:.1f}%), Good Subjs: {s['number_good_subjects']}, Avg Trials: {s['trials_per_subj']:.1f}")

if __name__ == '__main__':
    preprocess_datasets()
