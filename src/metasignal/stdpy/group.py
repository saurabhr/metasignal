"""Group-level wrapper for all metasignal stdpy measures.

``metad(data, stimuli, responses, confidence, nRatings, subject, within, between)``
    Loops over every cell defined by subject × within × between factors,
    calls ``compute_all_measures`` for each cell, and returns a tidy DataFrame
    with one row per cell and one column per measure.

    Interface mirrors metadpy's ``metad(data=df, subject=..., within=..., between=...)``.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from metasignal.stdpy.compute_all import compute_all_measures

# Output column names — same ordering as compute_all_measures return array
_MEASURE_COLS = [
    "meta_d", "AUC2", "Gamma", "Phi", "DeltaConf",
    "M_ratio", "AUC2_ratio", "Gamma_ratio", "Phi_ratio", "DeltaConf_ratio",
    "M_diff", "AUC2_diff", "Gamma_diff", "Phi_diff", "DeltaConf_diff",
    "MetaNoise", "MetaUncertainty", "dprime", "criterion", "mean_conf",
]


def metad(
    data: pd.DataFrame,
    stimuli: str = "Stimuli",
    responses: str = "Responses",
    confidence: str = "Confidence",
    nRatings: int = 4,
    subject: Optional[str] = None,
    within: Optional[Union[str, List[str]]] = None,
    between: Optional[Union[str, List[str]]] = None,
) -> pd.DataFrame:
    """Compute all metasignal sdtpy measures for every group cell.

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level DataFrame. Must contain columns for stimuli, responses,
        and confidence. Subject / condition columns are optional.
    stimuli : str
        Column name for stimulus identity (0 = S1, 1 = S2).
    responses : str
        Column name for participant response (0 = S1, 1 = S2).
    confidence : str
        Column name for confidence rating (integer, 1..nRatings).
    nRatings : int
        Number of confidence rating levels.
    subject : str or None
        Column name identifying participants. If ``None``, all rows are
        treated as a single participant.
    within : str, list of str, or None
        Within-subjects factor column(s). Each unique combination defines
        a separate cell per subject.
    between : str, list of str, or None
        Between-subjects factor column(s). These vary across subjects,
        not within them.

    Returns
    -------
    results : pd.DataFrame
        One row per group cell. Grouping columns (subject, within, between)
        appear first, followed by the 20 measure columns:

        meta_d, AUC2, Gamma, Phi, DeltaConf,
        M_ratio, AUC2_ratio, Gamma_ratio, Phi_ratio, DeltaConf_ratio,
        M_diff, AUC2_diff, Gamma_diff, Phi_diff, DeltaConf_diff,
        MetaNoise, MetaUncertainty, dprime, criterion, mean_conf

    Examples
    --------
    Single subject, no conditions:

    >>> from metasignal.stdpy import trialSimulation, metad
    >>> df = trialSimulation(d=1.5, metad=1.5, nTrials=300)
    >>> metad(df, nRatings=4).round(3)

    Multiple subjects, one within-subjects condition:

    >>> from metasignal.stdpy import pairedResponseSimulation, metad
    >>> df = pairedResponseSimulation(nSubjects=10, nTrials=200)
    >>> metad(df, subject='Subject', within='Condition', nRatings=4)
    """
    # Normalise factor lists
    within_cols:  List[str] = _to_list(within)
    between_cols: List[str] = _to_list(between)

    # Build grouping key: subject + within + between
    group_cols: List[str] = []
    if subject is not None:
        group_cols.append(subject)
    group_cols.extend(within_cols)
    group_cols.extend(between_cols)

    if not group_cols:
        # No grouping — treat entire dataset as one cell
        row = _fit_cell(data, stimuli, responses, confidence, nRatings)
        return pd.DataFrame([row])

    rows = []
    for keys, grp in data.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        measures = _fit_cell(grp, stimuli, responses, confidence, nRatings)
        rows.append({**meta, **measures})

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    """Run compute_all_measures on one cell and return a named dict."""
    stim = df[stimuli].to_numpy(dtype=float)
    resp = df[responses].to_numpy(dtype=float)
    conf = df[confidence].to_numpy(dtype=float)

    vals = compute_all_measures(stim, resp, conf, nRatings)
    return dict(zip(_MEASURE_COLS, vals))
