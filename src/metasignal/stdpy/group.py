"""Group-level wrapper for all metasignal stdpy measures.

``fit_group(data, stimuli, responses, confidence, nRatings, subject, within, between)``
    Loops over every cell defined by subject × within × between factors,
    computes type-1 SDT and all type-2 / metacognitive measures for each cell,
    and returns a tidy DataFrame with one row per cell and one column per measure.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from metasignal.stdpy.compute_all import compute_all_measures
from metasignal.stdpy.core import compute_sdt_resp

# Type-1 SDT measures computed directly from stimulus/response arrays
_TYPE1_COLS = [
    "hit_rate", "fa_rate", "dprime", "criterion", "ln_beta",
    "n_trials", "n_hits", "n_misses", "n_fa", "n_cr",
]

# Type-2 / metacognitive measures — same ordering as compute_all_measures
_TYPE2_COLS = [
    "meta_d", "AUC2", "Gamma", "Phi", "DeltaConf",
    "M_ratio", "AUC2_ratio", "Gamma_ratio", "Phi_ratio", "DeltaConf_ratio",
    "M_diff", "AUC2_diff", "Gamma_diff", "Phi_diff", "DeltaConf_diff",
    "MetaNoise", "MetaUncertainty", "mean_conf",
    "logL", "AIC", "BIC", "AICc", "k", "n",
]

# All measures in display order: type-1 first, then type-2
MEASURE_COLS: List[str] = _TYPE1_COLS + _TYPE2_COLS


def fit_group(
    data: pd.DataFrame = None,
    stimuli: str = "Stimuli",
    responses: str = "Responses",
    confidence: str = "Confidence",
    nRatings: int = 4,
    subject: Optional[str] = None,
    within: Optional[Union[str, List[str]]] = None,
    between: Optional[Union[str, List[str]]] = None,
    measures: Optional[Union[str, List[str]]] = None,
    nR_S1: Optional[np.ndarray] = None,
    nR_S2: Optional[np.ndarray] = None,
    method: str = "mle",
    **bayes_kwargs,
) -> "pd.DataFrame | object":
    """Compute type-1 SDT and type-2 metacognitive measures for every group cell.

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level DataFrame with columns for stimuli, responses, and
        confidence. Subject / condition columns are optional.
    stimuli : str
        Column for stimulus identity (0 = S1, 1 = S2).
    responses : str
        Column for participant response (0 = S1, 1 = S2).
    confidence : str
        Column for confidence rating (integer, 1..nRatings).
    nRatings : int
        Number of confidence rating levels.
    subject : str or None
        Column identifying participants. ``None`` treats all rows as one cell.
    within : str or list of str or None
        Within-subjects factor column(s).  When a single within-subjects
        factor with exactly two levels is supplied together with
        ``method='bayesian'``, ``fit_within_subject_comparison`` is used
        automatically.
    between : str or list of str or None
        Between-subjects factor column(s).  When supplied together with
        ``method='bayesian'``, ``fit_full_metad_comparison`` is used.
    nR_S1 : array-like or None
        Count vector for S1 stimulus, length ``2 * nRatings``.
        When provided with ``nR_S2``, ``data`` is ignored and a single-cell
        fit is run directly from counts.
        Equivalent to metadpy's ``metad(nR_S1=..., nR_S2=...)``.
    nR_S2 : array-like or None
        Count vector for S2 stimulus (same layout as ``nR_S1``).
    method : str
        Fitting method.  One of:

        ``'mle'`` *(default)*
            Frequentist MLE via :func:`fit_meta_d_mle`.  Returns a tidy
            DataFrame with one row per cell and 28 measure columns.

        ``'bayesian'``
            Hierarchical Bayesian via ``metasignal.sdtbayes``.  Requires
            ``pip install metasignal[sdtbayes]`` and ``setup_runtime()``.
            The specific sdtbayes function is chosen automatically:

            - No ``subject`` → ``fit_subject_level`` (single participant)
            - ``subject`` only → ``fit_full_metad`` (group, no comparison)
            - ``between`` set → ``fit_full_metad_comparison`` (between-groups)
            - ``within`` with 2 levels → ``fit_within_subject_comparison``
            - ``within`` with >2 levels → ``fit_hierarchical_metad``

            Returns the raw ``FitResult`` from sdtbayes (use
            ``metasignal.sdtbayes.posterior_summary`` to inspect it).

        ``'two_stage'``
            Fast two-stage Bayesian: MLE per participant then Bayesian
            pooling via ``fit_two_stage_group`` / ``fit_two_stage_comparison``.

        ``'robust'``
            Full HMeta-d with Student-t hyperprior (downweights outliers).

        ``'hierarchical'``
            Ordered-logistic brms model (trial-level, extensible with
            ``items=`` for crossed effects).

    **bayes_kwargs
        Extra keyword arguments forwarded to the chosen sdtbayes function
        (e.g. ``draws=2000``, ``chains=4``, ``target_accept=0.9``,
        ``items=[...]``).

    Returns
    -------
    results : pd.DataFrame
        When ``method='mle'``: one row per group cell, 28 measure columns.
    result : FitResult
        When any Bayesian method is chosen: the raw posterior fit object
        from ``metasignal.sdtbayes``.

    Examples
    --------
    MLE (default):

    >>> fit_group(df, subject='Subject', within='Condition', nRatings=4)

    Subject-level Bayesian (metadpy ``hmetad`` equivalent):

    >>> fit_group(nR_S1=nR_S1, nR_S2=nR_S2, nRatings=4, method='bayesian')

    Group Bayesian — between-groups:

    >>> fit_group(df, subject='Subject', between='Group',
    ...           nRatings=4, method='bayesian', draws=2000)

    Within-subject Bayesian comparison:

    >>> fit_group(df, subject='Subject', within='Condition',
    ...           nRatings=4, method='bayesian')

    Returns
    -------
    results : pd.DataFrame or FitResult
        MLE → tidy DataFrame.  Bayesian → ``FitResult`` (use
        ``metasignal.sdtbayes.posterior_summary`` to inspect posteriors).

    Examples
    --------
    All measures, single subject:

    >>> from metasignal.stdpy import trialSimulation, fit_group
    >>> df = trialSimulation(d=1.5, metad=1.5, nTrials=300)
    >>> fit_group(df, nRatings=4).round(3)

    Only type-1 measures across subjects:

    >>> from metasignal.stdpy import responseSimulation, fit_group
    >>> df = responseSimulation(d=1.5, metad=1.5, nSubjects=10, nTrials=200)
    >>> fit_group(df, subject='Subject', nRatings=4,
    ...           measures=['dprime', 'criterion', 'hit_rate', 'fa_rate'])

    metadpy-style subset (meta-d', M_ratio, M_diff) with conditions:

    >>> from metasignal.stdpy import pairedResponseSimulation, fit_group
    >>> df = pairedResponseSimulation(nSubjects=10, nTrials=200)
    >>> fit_group(df, subject='Subject', within='Condition', nRatings=4,
    ...           measures=['dprime', 'meta_d', 'M_ratio', 'M_diff'])
    """
    keep_cols = _resolve_measures(measures)

    # --- Bayesian routing ---
    if method != "mle":
        return _fit_group_bayesian(
            data=data, stimuli=stimuli, responses=responses,
            confidence=confidence, nRatings=nRatings,
            subject=subject, within=within, between=between,
            nR_S1=nR_S1, nR_S2=nR_S2,
            method=method, **bayes_kwargs,
        )

    # --- Count-array shortcut (single cell, MLE, no grouping) ---
    if nR_S1 is not None and nR_S2 is not None:
        from metasignal.stdpy.simulate import ratings2df
        nr1 = np.asarray(nR_S1, dtype=float)
        nr2 = np.asarray(nR_S2, dtype=float)
        if len(nr1) != 2 * nRatings:
            raise ValueError(f"nR_S1 length {len(nr1)} != 2 * nRatings ({2 * nRatings})")
        df_counts = ratings2df(nr1, nr2)
        row = _fit_cell(df_counts, "Stimuli", "Responses", "Confidence", nRatings)
        return pd.DataFrame([{k: v for k, v in row.items() if k in keep_cols}])

    if data is None:
        raise ValueError("Provide either 'data' (DataFrame) or both 'nR_S1' and 'nR_S2'.")

    within_cols:  List[str] = _to_list(within)
    between_cols: List[str] = _to_list(between)

    group_cols: List[str] = []
    if subject is not None:
        group_cols.append(subject)
    group_cols.extend(within_cols)
    group_cols.extend(between_cols)

    if not group_cols:
        row = _fit_cell(data, stimuli, responses, confidence, nRatings)
        return pd.DataFrame([{k: v for k, v in row.items() if k in keep_cols}])

    rows = []
    for keys, grp in data.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_meta = dict(zip(group_cols, keys))
        cell = _fit_cell(grp, stimuli, responses, confidence, nRatings)
        rows.append({**group_meta, **{k: v for k, v in cell.items() if k in keep_cols}})

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_measures(measures) -> List[str]:
    if measures is None:
        return list(MEASURE_COLS)
    requested = [measures] if isinstance(measures, str) else list(measures)
    invalid = [m for m in requested if m not in MEASURE_COLS]
    if invalid:
        raise ValueError(
            f"Unknown measure(s): {invalid}.\n"
            f"Valid options: {MEASURE_COLS}"
        )
    return [m for m in MEASURE_COLS if m in set(requested)]


def _to_list(x) -> List[str]:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)


def _fit_cell(
    df: pd.DataFrame,
    stimuli: str,
    responses: str,
    confidence: str,
    nRatings: int,
) -> dict:
    """Compute all type-1 and type-2 measures for one cell."""
    stim = df[stimuli].to_numpy(dtype=float)
    resp = df[responses].to_numpy(dtype=float)
    conf = df[confidence].to_numpy(dtype=float)

    # --- Type-1 SDT ---
    valid = ~np.isnan(stim) & ~np.isnan(resp)
    stim_v = stim[valid]
    resp_v = resp[valid]

    n_trials = int(len(stim_v))
    s2 = stim_v == 1
    s1 = stim_v == 0
    n_s2 = int(s2.sum())
    n_s1 = int(s1.sum())

    n_hits   = int(((stim_v == 1) & (resp_v == 1)).sum())
    n_misses = int(((stim_v == 1) & (resp_v == 0)).sum())
    n_fa     = int(((stim_v == 0) & (resp_v == 1)).sum())
    n_cr     = int(((stim_v == 0) & (resp_v == 0)).sum())

    hit_rate = n_hits / n_s2 if n_s2 > 0 else np.nan
    fa_rate  = n_fa  / n_s1 if n_s1 > 0 else np.nan

    try:
        dprime, criterion, ln_beta = compute_sdt_resp(stim_v.astype(int), resp_v.astype(int))
    except Exception:
        dprime = criterion = ln_beta = np.nan

    type1 = {
        "hit_rate": hit_rate, "fa_rate": fa_rate,
        "dprime": dprime, "criterion": criterion, "ln_beta": ln_beta,
        "n_trials": n_trials, "n_hits": n_hits, "n_misses": n_misses,
        "n_fa": n_fa, "n_cr": n_cr,
    }

    # --- Type-2 / metacognitive (compute_all_measures handles NaN/edge cases) ---
    type2_vals = compute_all_measures(stim, resp, conf, nRatings)
    # compute_all_measures returns 26 values:
    #   [0-16]: meta_d…MetaUncertainty
    #   [17]: dprime (dup — already in type1), [18]: criterion (dup), [19]: mean_conf
    #   [20-25]: logL, AIC, BIC, AICc, k, n
    type2_values = list(type2_vals[:17]) + [type2_vals[19]] + list(type2_vals[20:])
    type2 = dict(zip(_TYPE2_COLS, type2_values))

    return {**type1, **type2}


# ---------------------------------------------------------------------------
# Bayesian routing
# ---------------------------------------------------------------------------

def _fit_group_bayesian(
    data, stimuli, responses, confidence, nRatings,
    subject, within, between, nR_S1, nR_S2, method, **kwargs,
):
    """Delegate to fit_meta_formula (cmdstanpy backend) for all Bayesian fitting."""
    try:
        from metasignal.sdtbayes.formula import fit_meta_formula
    except (ImportError, OSError, RuntimeError) as e:
        raise RuntimeError(
            "Bayesian fitting requires cmdstanpy.\n"
            "Install with: pip install cmdstanpy\n"
            "Then run: import cmdstanpy; cmdstanpy.install_cmdstan()\n"
            f"Original error: {e}"
        ) from e

    import pandas as pd

    within_cols  = _to_list(within)
    between_cols = _to_list(between)

    parameterization = kwargs.pop("parameterization", "mratio")
    backend = kwargs.pop("backend", "stan")

    # --- Single participant from count arrays: expand to trial arrays ---
    if nR_S1 is not None and nR_S2 is not None:
        nr1 = np.asarray(nR_S1, dtype=int)
        nr2 = np.asarray(nR_S2, dtype=int)
        # Inverse of trials_to_counts (see core.py):
        # First half  (idx 0..nR-1): S1 responses, conf = nRatings - idx (high→low)
        # Second half (idx nR..2nR-1): S2 responses, conf = idx - nR + 1 (low→high)
        stims, resps, confs = [], [], []
        for idx in range(nRatings):
            conf = nRatings - idx  # high confidence at idx=0
            n = int(nr1[idx])        # CR: S1 stim, S1 resp
            stims += [0]*n; resps += [0]*n; confs += [conf]*n
            n = int(nr2[idx])        # Miss: S2 stim, S1 resp
            stims += [1]*n; resps += [0]*n; confs += [conf]*n
        for idx in range(nRatings):
            conf = idx + 1           # low confidence at idx=0
            n = int(nr1[nRatings + idx])  # FA: S1 stim, S2 resp
            stims += [0]*n; resps += [1]*n; confs += [conf]*n
            n = int(nr2[nRatings + idx])  # Hit: S2 stim, S2 resp
            stims += [1]*n; resps += [1]*n; confs += [conf]*n
        participants = [(np.array(stims), np.array(resps), np.array(confs))]
        pred_df = pd.DataFrame({"participant": [0]})
        return fit_meta_formula(
            participants=participants,
            n_ratings=nRatings,
            formula="~ 1",
            data=pred_df,
            parameterization=parameterization,
            backend=backend,
            **kwargs,
        )

    if data is None:
        raise ValueError("Provide either 'data' (DataFrame) or both 'nR_S1' and 'nR_S2'.")

    def _trials(df):
        return (
            df[stimuli].to_numpy(dtype=int),
            df[responses].to_numpy(dtype=int),
            df[confidence].to_numpy(dtype=int),
        )

    # --- No subject column — single participant ---
    if subject is None:
        participants = [_trials(data)]
        pred_df = pd.DataFrame({"participant": [0]})
        return fit_meta_formula(
            participants=participants,
            n_ratings=nRatings,
            formula="~ 1",
            data=pred_df,
            parameterization=parameterization,
            backend=backend,
            **kwargs,
        )

    sub_ids = sorted(data[subject].unique())

    if within_cols:
        # Within-subject: one (subject × condition) cell per participant entry.
        # Between-subject factors are constant within a subject; within-subject
        # factors vary — both become predictors in the formula.
        within_col = within_cols[0]
        cond_vals = sorted(data[within_col].unique())
        pred_rows = []
        participants = []
        for s in sub_ids:
            sub_data = data[data[subject] == s]
            for cond in cond_vals:
                cell = sub_data[sub_data[within_col] == cond]
                if len(cell) == 0:
                    continue
                participants.append(_trials(cell))
                row = {"participant": s, within_col: cond}
                for col in between_cols:
                    row[col] = sub_data[col].iloc[0]
                for extra in within_cols[1:]:
                    row[extra] = cell[extra].iloc[0]
                pred_rows.append(row)
        pred_df = pd.DataFrame(pred_rows)
    else:
        # Between-subject or no conditions: one entry per subject.
        # Between columns are constant within subject — safe to take .iloc[0].
        pred_rows = []
        participants = []
        for s in sub_ids:
            sub_data = data[data[subject] == s]
            participants.append(_trials(sub_data))
            row = {"participant": s}
            for col in between_cols:
                row[col] = sub_data[col].iloc[0]
            pred_rows.append(row)
        pred_df = pd.DataFrame(pred_rows)

    predictor_cols = between_cols + within_cols
    formula_rhs = " + ".join(predictor_cols) if predictor_cols else "1"

    return fit_meta_formula(
        participants=participants,
        n_ratings=nRatings,
        formula=f"~ {formula_rhs}",
        data=pred_df,
        parameterization=parameterization,
        backend=backend,
        **kwargs,
    )
