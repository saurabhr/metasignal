"""Statistical helper functions for SDT and metacognition analyses."""
import numpy as np
from scipy import stats

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
        msg = (
            f"{test_description}: t({df}) = {tstat:.3f}, p = {pval:.3e}, "
            f"Cohen's d = {cohen_d:.3f}, 95% CI = [{ci[0]:.3f}, {ci[1]:.3f}]"
        )
        print(msg)

    return pval, tstat, df, cohen_d, ci

def icc(data, icc_type='C-k'):
    """Intraclass Correlation Coefficient — port of ICC.m (Salarian 2008).

    Args:
        data: 2-D array of shape (n_targets, n_raters).
        icc_type: ICC variant using the MATLAB naming convention:
            ``'1-1'``, ``'1-k'``, ``'C-1'``, ``'C-k'``, ``'A-1'``, ``'A-k'``.
            Default ``'C-k'`` (consistency, average of k raters — Cronbach's alpha).

    Returns:
        Single-row ``pandas.DataFrame`` from ``pingouin.intraclass_corr``
        for the requested type, with columns ``Type``, ``ICC``, ``F``,
        ``df1``, ``df2``, ``pval``, ``CI95%``.

    Raises:
        ImportError: If ``pingouin`` is not installed.
        ValueError: If ``icc_type`` is not one of the six recognised codes.
    """
    try:
        import pingouin as pg
    except ImportError as e:
        raise ImportError(
            "pingouin is not installed. Run:\n    pip install pingouin"
        ) from e
    import pandas as pd

    df = pd.DataFrame(data)
    df = df.reset_index().melt(id_vars='index', var_name='rater', value_name='score')
    df.columns = ['target', 'rater', 'score']
    res = pg.intraclass_corr(data=df, targets='target', raters='rater', ratings='score')

    pingouin_type = f"ICC({icc_type.replace('-', ',')})"
    match = res.loc[res["Type"] == pingouin_type]
    if match.empty:
        raise ValueError(
            f"icc_type '{icc_type}' ('{pingouin_type}') not found. "
            f"Available: {res['Type'].tolist()}"
        )
    return match
