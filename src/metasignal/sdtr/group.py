"""Group-level wrapper for the base Gaussian SDT model.

``fit_group(data, ...)`` loops over every cell defined by
subject × within × between factors, aggregates trial-level rows into a
response-frequency table per cell, fits :func:`metasignal.sdtr.sdt.fit_sdt`,
and returns a tidy DataFrame — mirroring the interface of
``metasignal.stdpy.fit_group`` / ``metasignal.itmc.fit_group``.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from metasignal.sdtr.sdt import Restriction, fit_sdt


def fit_group(
    data: pd.DataFrame,
    signal: str = "signal",
    response: str = "response",
    n_signals: Optional[int] = None,
    n_categories: Optional[int] = None,
    subject: Optional[str] = None,
    within: Optional[Union[str, List[str]]] = None,
    between: Optional[Union[str, List[str]]] = None,
    restriction: Restriction = "no",
    n_starts: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Fit the base Gaussian SDT model for every group cell.

    Parameters
    ----------
    data:
        Trial-level DataFrame with one row per trial.
    signal:
        Column with the signal-class id (integer, 0 = reference/noise signal,
        1..n_signals-1 = the other signals, lowest to highest strength).
    response:
        Column with the response category (integer 1..n_categories, from
        most noise-like to most signal-like).
    n_signals, n_categories:
        Total counts across the whole ``data`` (not just one cell) — used to
        give every cell's response-frequency table the same shape, including
        signal/category values a given cell happens not to observe.
        ``None`` infers them from the full ``data`` (``max + 1`` / ``max``).
    subject, within, between:
        Grouping columns — see ``metasignal.itmc.fit_group`` for the exact
        semantics (same convention).
    restriction, n_starts, seed:
        Passed through to :func:`metasignal.sdtr.sdt.fit_sdt`.

    Returns
    -------
    pd.DataFrame
        One row per cell. Grouping columns first, then ``mean_<j>``,
        ``sd_<j>``, ``d_a_<j>``, ``d_e_<j>``, ``A_z_<j>`` for each
        non-reference signal ``j``, then ``threshold_<r>`` for each of the
        shared thresholds, then ``logL``, ``aic``, ``bic``, ``success``.
    """
    n_signals = n_signals or int(data[signal].max()) + 1
    n_categories = n_categories or int(data[response].max())

    within_cols = _to_list(within)
    between_cols = _to_list(between)
    group_cols: List[str] = []
    if subject is not None:
        group_cols.append(subject)
    group_cols.extend(within_cols)
    group_cols.extend(between_cols)

    kw = dict(restriction=restriction, n_starts=n_starts, seed=seed)

    if not group_cols:
        row = _fit_cell(data, signal, response, n_signals, n_categories, **kw)
        return pd.DataFrame([row])

    rows = []
    for keys, grp in data.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_meta = dict(zip(group_cols, keys))
        cell = _fit_cell(grp, signal, response, n_signals, n_categories, **kw)
        rows.append({**group_meta, **cell})

    return pd.DataFrame(rows).reset_index(drop=True)


def _fit_cell(
    df: pd.DataFrame,
    signal: str,
    response: str,
    n_signals: int,
    n_categories: int,
    **kw,
) -> dict:
    counts = np.zeros((n_signals, n_categories))
    table = pd.crosstab(df[signal], df[response])
    for s in table.index:
        for r in table.columns:
            counts[int(s), int(r) - 1] = table.loc[s, r]

    result = fit_sdt(counts, **kw)

    row: dict = {}
    for j in range(1, n_signals):
        row[f"mean_{j}"] = result.means[j - 1]
        row[f"sd_{j}"] = result.sds[j - 1]
        row[f"d_a_{j}"] = result.d_a[j - 1]
        row[f"d_e_{j}"] = result.d_e[j - 1]
        row[f"A_z_{j}"] = result.A_z[j - 1]
    for r, t in enumerate(result.thresholds, start=1):
        row[f"threshold_{r}"] = t
    row["logL"] = result.logL
    row["aic"] = result.aic
    row["bic"] = result.bic
    row["success"] = result.success
    return row


def _to_list(x) -> List[str]:
    if x is None:
        return []
    return [x] if isinstance(x, str) else list(x)
