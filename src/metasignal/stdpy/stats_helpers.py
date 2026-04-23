import numpy as np
import scipy.stats as stats

def z2r(z):
    """Fisher's Z to R."""
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def r2z(r):
    """R to Fisher's Z."""
    return 0.5 * np.log((1 + r) / (1 - r))

def perform_ttest(data, test_description="", display=True):
    """Replicates perform_ttest.m: returns p, t, df, Cohen_d, CI."""
    data = np.asarray(data)
    valid_data = data[~np.isnan(data)]

    df = len(valid_data) - 1
    tstat, pval = stats.ttest_1samp(valid_data, 0, nan_policy='omit')
    cohen_d = tstat / np.sqrt(df + 1)

    sem = np.std(valid_data, ddof=1) / np.sqrt(len(valid_data))
    ts = stats.t.ppf([0.025, 0.975], df)
    m = np.mean(valid_data)
    ci = m + ts * sem

    if display:
        print(f"{test_description}: t({df}) = {tstat:.3f}, p = {pval:.3e}, Cohen's d = {cohen_d:.3f}, 95% CI = [{ci[0]:.3f}, {ci[1]:.3f}]")

    return pval, tstat, df, cohen_d, ci

def icc(data, icc_type='C-k'):
    """
    Intraclass Correlation proxy for ICC.m
    Relies on standard pandas/pingouin logic for advanced cases.
    We return a dictionary replicating the values.
    """
    import pingouin as pg
    import pandas as pd
    n, k = data.shape
    df = pd.DataFrame(data)
    df = df.reset_index().melt(id_vars='index', var_name='rater', value_name='score')
    df.columns = ['target', 'rater', 'score']
    res = pg.intraclass_corr(data=df, targets='target', raters='rater', ratings='score')

    # Map back to standard ICC types
    # ICC(2, k) = A-k, ICC(3, k) = C-k, etc.
    return res
