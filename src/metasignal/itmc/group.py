"""Group-level wrapper for information-theoretic metacognition measures.

``fit_group(data, ...)`` loops over every cell defined by
subject × within × between factors, computes all five IT metacognition
measures for each cell, and returns a tidy DataFrame — mirroring the
interface of ``metasignal.stdpy.fit_group``.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from metasignal.itmc.measures import (
    Backend,
    _build_contingency_table,
    _estimate_dprime_statconfr,
    meta_I,
    meta_Ir1,
    meta_Ir1_acc,
    meta_Ir2,
    RMI,
)
from metasignal.stdpy.core import compute_sdt_resp

MEASURE_COLS: List[str] = ["meta_I", "meta_Ir1", "meta_Ir1_acc", "meta_Ir2", "RMI"]


def fit_group(
    data: pd.DataFrame,
    stimuli: str = "Stimuli",
    responses: str = "Responses",
    confidence: str = "Confidence",
    subject: Optional[str] = None,
    within: Optional[Union[str, List[str]]] = None,
    between: Optional[Union[str, List[str]]] = None,
    measures: Optional[Union[str, List[str]]] = None,
    backend: Backend = "simple",
    bias_correction: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute IT metacognition measures for every group cell.

    Parameters
    ----------
    data:
        Trial-level DataFrame with one row per trial.
    stimuli, responses, confidence:
        Column names for stimulus (0/1), response (0/1), and confidence
        rating (integer 1…n_ratings).
    subject:
        Column identifying participants. ``None`` treats all rows as one cell.
    within:
        Within-subjects factor column(s). Pass a single string or a list
        of strings for multiple factors (e.g. ``['condition', 'block']``).
    between:
        Between-subjects factor column(s). Pass a single string or a list
        of strings for multiple factors (e.g. ``['group', 'task']``).
    measures:
        Subset of ``['meta_I', 'meta_Ir1', 'meta_Ir1_acc', 'meta_Ir2', 'RMI']``
        to return.  ``None`` returns all.
    backend:
        ``'simple'``: H₂(acc) − H₂(acc|conf) formulation (fast).
        ``'statconfr'``: exact port of the R package using the full 2×2K
        contingency table and analytic bounds.
    bias_correction:
        Subtract permutation-estimated positive sampling bias from meta-I
        before deriving all normalised variants.
    seed:
        RNG seed for bias correction and Gaussian normalisation simulations.

    Returns
    -------
    pd.DataFrame
        One row per cell (subject × within × between combination).
        Grouping columns appear first, followed by measure columns.

    Examples
    --------
    Single participant, no grouping:

    >>> from metasignal.itmc import fit_group
    >>> fit_group(df, stimuli='stimulus', responses='response',
    ...           confidence='rating')

    Multi-participant:

    >>> fit_group(df, stimuli='stimulus', responses='response',
    ...           confidence='rating', subject='participant')

    With a single within-subjects condition:

    >>> fit_group(df, stimuli='stimulus', responses='response',
    ...           confidence='rating', subject='participant',
    ...           within='condition')

    Multiple within-subjects factors (e.g. condition × block):

    >>> fit_group(df, stimuli='stimulus', responses='response',
    ...           confidence='rating', subject='participant',
    ...           within=['condition', 'block'])

    Multiple between-subjects factors:

    >>> fit_group(df, stimuli='stimulus', responses='response',
    ...           confidence='rating', subject='participant',
    ...           between=['group', 'task'])

    Mixed multi-factor design:

    >>> fit_group(df, stimuli='stimulus', responses='response',
    ...           confidence='rating', subject='participant',
    ...           within=['condition', 'block'], between=['group', 'task'])
    """
    keep_cols = _resolve_measures(measures)

    within_cols  = _to_list(within)
    between_cols = _to_list(between)
    group_cols: List[str] = []
    if subject is not None:
        group_cols.append(subject)
    group_cols.extend(within_cols)
    group_cols.extend(between_cols)

    kw = dict(backend=backend, bias_correction=bias_correction, seed=seed)

    if not group_cols:
        row = _fit_cell(data, stimuli, responses, confidence, **kw)
        return pd.DataFrame([{k: v for k, v in row.items() if k in keep_cols}])

    rows = []
    for keys, grp in data.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_meta = dict(zip(group_cols, keys))
        cell = _fit_cell(grp, stimuli, responses, confidence, **kw)
        rows.append({**group_meta, **{k: v for k, v in cell.items() if k in keep_cols}})

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_cell(
    df: pd.DataFrame,
    stimuli: str,
    responses: str,
    confidence: str,
    backend: Backend,
    bias_correction: bool,
    seed: int,
) -> dict:
    stim = df[stimuli].to_numpy(dtype=int)
    resp = df[responses].to_numpy(dtype=int)
    conf = df[confidence].to_numpy(dtype=int)

    if backend == "statconfr":
        table = _build_contingency_table(stim, resp, conf)
        dp = _estimate_dprime_statconfr(table)
    else:
        try:
            dp, *_ = compute_sdt_resp(stim, resp)
        except Exception:
            dp = None

    kw = dict(backend=backend, bias_correction=bias_correction, seed=seed)
    return {
        "meta_I":       meta_I(stim, resp, conf, **kw),
        "meta_Ir1":     meta_Ir1(stim, resp, conf, dprime=dp, **kw),
        "meta_Ir1_acc": meta_Ir1_acc(stim, resp, conf, **kw),
        "meta_Ir2":     meta_Ir2(stim, resp, conf, **kw),
        "RMI":          RMI(stim, resp, conf, **kw),
    }


def _resolve_measures(measures) -> List[str]:
    if measures is None:
        return list(MEASURE_COLS)
    requested = [measures] if isinstance(measures, str) else list(measures)
    invalid = [m for m in requested if m not in MEASURE_COLS]
    if invalid:
        raise ValueError(
            f"Unknown measure(s): {invalid}. Valid: {MEASURE_COLS}"
        )
    return [m for m in MEASURE_COLS if m in set(requested)]


def _to_list(x) -> List[str]:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)
