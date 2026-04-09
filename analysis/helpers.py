import numpy as np
from metasignal.stdpy.compute_all import compute_all_measures

def xue_recode(conf, low_high_recoding):
    """Xue recoding procedure for metacognitive bias."""
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
    """Altered confidence ratings for precision analyses."""
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
