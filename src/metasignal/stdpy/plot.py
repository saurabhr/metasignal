"""Plotting utilities for Type-1 SDT and Type-2 / metacognitive measures.

Functions
---------
``plot_confidence``
    Bar chart of response-count distributions (nR_S1 / nR_S2) by
    stimulus class, response, and rating level.

``plot_type2roc``
    Type-2 ROC curve with optional ideal-observer overlay.

``plot_sanity_check``
    Multi-panel diagnostic: confidence distributions, type-2 ROC,
    and key SDT parameter summary for a single participant.

``plot_forest``
    Forest plot of any scalar measure across subjects / conditions,
    with group mean ± CI.

``plot_measures``
    Bar chart comparing one or more measures across conditions or groups.

All functions return the ``matplotlib.axes.Axes`` object(s) so callers
can further customise the figure.  No metadpy dependency.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm, t as t_dist

from metasignal.stdpy.type2 import compute_type2_auc, sdt_expect_conf


# ---------------------------------------------------------------------------
# Colour palette (accessible, consistent across all plots)
# ---------------------------------------------------------------------------
_S1_COLOR  = "#2196F3"   # blue  — S1 stimulus
_S2_COLOR  = "#FF5722"   # orange — S2 stimulus
_CORR_ALPHA = 0.85
_INCORR_ALPHA = 0.40


# ---------------------------------------------------------------------------
# plot_confidence
# ---------------------------------------------------------------------------

def plot_confidence(
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Confidence rating distribution",
    xlabel: str = "Rating (S1 ← → S2)",
) -> plt.Axes:
    """Bar chart of response-count distributions split by stimulus class.

    Parameters
    ----------
    nR_S1, nR_S2 :
        Count vectors of length ``2 * nRatings``.
    ax :
        Existing axes to draw on. If ``None`` a new figure is created.
    title, xlabel :
        Axis labels.

    Returns
    -------
    ax : plt.Axes
    """
    nR_S1 = np.asarray(nR_S1, dtype=float)
    nR_S2 = np.asarray(nR_S2, dtype=float)
    nRatings = len(nR_S1) // 2

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(2 * nRatings)
    rating_labels = (
        [f"S1-r{i}" for i in range(nRatings, 0, -1)]
        + [f"S2-r{i}" for i in range(1, nRatings + 1)]
    )

    # S1 stim: correct (S1 response half) and incorrect (S2 response half)
    ax.bar(x[:nRatings],  nR_S1[:nRatings],  color=_S1_COLOR, alpha=_CORR_ALPHA,  label="S1 stim, S1 resp (correct)")
    ax.bar(x[nRatings:],  nR_S1[nRatings:],  color=_S1_COLOR, alpha=_INCORR_ALPHA, label="S1 stim, S2 resp (error)")
    # S2 stim: incorrect (S1 response half) and correct (S2 response half)
    ax.bar(x[:nRatings],  nR_S2[:nRatings],  color=_S2_COLOR, alpha=_INCORR_ALPHA, label="S2 stim, S1 resp (error)",
           bottom=nR_S1[:nRatings])
    ax.bar(x[nRatings:],  nR_S2[nRatings:],  color=_S2_COLOR, alpha=_CORR_ALPHA,  label="S2 stim, S2 resp (correct)",
           bottom=nR_S1[nRatings:])

    ax.axvline(nRatings - 0.5, color="black", linewidth=1.2, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(rating_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper center", ncol=2)
    return ax


# ---------------------------------------------------------------------------
# plot_type2roc
# ---------------------------------------------------------------------------

def plot_type2roc(
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
    ax: Optional[plt.Axes] = None,
    show_ideal: bool = True,
    label: str = "Observed",
    color: str = _S2_COLOR,
    title: str = "Type-2 ROC",
) -> plt.Axes:
    """Plot the Type-2 ROC curve.

    Parameters
    ----------
    nR_S1, nR_S2 :
        Count vectors of length ``2 * nRatings``.
    ax :
        Existing axes. ``None`` creates a new figure.
    show_ideal :
        Overlay the ideal SDT-predicted ROC curve.
    label :
        Legend label for the observed curve.
    color :
        Line / marker colour.
    title :
        Axes title.

    Returns
    -------
    ax : plt.Axes
    """
    nR_S1 = np.asarray(nR_S1, dtype=float)
    nR_S2 = np.asarray(nR_S2, dtype=float)
    nRatings = len(nR_S1) // 2

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    # Observed type-2 ROC points
    counts_c = nR_S2[nRatings:] + nR_S1[:nRatings][::-1]
    counts_i = nR_S1[nRatings:] + nR_S2[:nRatings][::-1]
    hr2  = np.concatenate([[0], np.cumsum(counts_c[::-1]) / (counts_c.sum() or 1)])
    far2 = np.concatenate([[0], np.cumsum(counts_i[::-1]) / (counts_i.sum() or 1)])
    auc2 = compute_type2_auc(nR_S1, nR_S2)

    ax.plot(far2, hr2, "o-", color=color, label=f"{label} (AUC2={auc2:.3f})")

    # Ideal SDT overlay
    if show_ideal:
        sdt = sdt_expect_conf(nR_S1, nR_S2)
        nr1e = np.array(sdt["nR_S1_exp"])
        nr2e = np.array(sdt["nR_S2_exp"])
        c_e  = nr2e[nRatings:] + nr1e[:nRatings][::-1]
        i_e  = nr1e[nRatings:] + nr2e[:nRatings][::-1]
        hr2e  = np.concatenate([[0], np.cumsum(c_e[::-1]) / (c_e.sum() or 1)])
        far2e = np.concatenate([[0], np.cumsum(i_e[::-1]) / (i_e.sum() or 1)])
        auc2e = compute_type2_auc(nr1e, nr2e)
        ax.plot(far2e, hr2e, "s--", color="grey", alpha=0.7,
                label=f"SDT ideal (AUC2={auc2e:.3f})")

    ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Chance")
    ax.set_xlabel("Type-2 False Alarm Rate")
    ax.set_ylabel("Type-2 Hit Rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    return ax


# ---------------------------------------------------------------------------
# plot_sanity_check
# ---------------------------------------------------------------------------

def plot_sanity_check(
    nR_S1: np.ndarray,
    nR_S2: np.ndarray,
    meta_d: Optional[float] = None,
    title: str = "Sanity check",
) -> Tuple[plt.Figure, np.ndarray]:
    """Three-panel diagnostic for a single participant.

    Panels: (1) confidence distribution, (2) type-2 ROC,
    (3) text summary of key SDT parameters.

    Parameters
    ----------
    nR_S1, nR_S2 :
        Count vectors of length ``2 * nRatings``.
    meta_d :
        Optional pre-computed meta-d' to include in the summary panel.
    title :
        Figure suptitle.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray of plt.Axes  (shape (3,))
    """
    nR_S1 = np.asarray(nR_S1, dtype=float)
    nR_S2 = np.asarray(nR_S2, dtype=float)
    nRatings = len(nR_S1) // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1 — confidence distribution
    plot_confidence(nR_S1, nR_S2, ax=axes[0], title="Confidence distribution")

    # Panel 2 — type-2 ROC
    plot_type2roc(nR_S1, nR_S2, ax=axes[1], title="Type-2 ROC")

    # Panel 3 — parameter summary text
    ax3 = axes[2]
    ax3.axis("off")
    sdt = sdt_expect_conf(nR_S1, nR_S2)
    dprime = sdt["dprime"]
    auc2   = compute_type2_auc(nR_S1, nR_S2)
    n_total = int(nR_S1.sum() + nR_S2.sum())

    lines = [
        f"N trials : {n_total}",
        f"d'       : {dprime:.3f}",
        f"AUC2     : {auc2:.3f}",
    ]
    if meta_d is not None:
        lines.append(f"meta-d'  : {meta_d:.3f}")
        lines.append(f"M_ratio  : {meta_d / dprime:.3f}" if dprime > 0 else "M_ratio  : N/A")

    text = "\n".join(lines)
    ax3.text(0.1, 0.55, text, transform=ax3.transAxes,
             fontsize=11, verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.8))
    ax3.set_title("Summary")

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# plot_forest
# ---------------------------------------------------------------------------

def plot_forest(
    data,
    measure: str,
    group_col: Optional[str] = None,
    subject_col: Optional[str] = "Subject",
    ci: float = 0.95,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    palette: Optional[List[str]] = None,
) -> plt.Axes:
    """Forest plot of a scalar measure across subjects / conditions.

    Parameters
    ----------
    data : pd.DataFrame
        Output of :func:`fit_group` — one row per subject (× condition).
    measure : str
        Column name of the measure to plot (e.g. ``'M_ratio'``).
    group_col : str or None
        Column to colour-code groups / conditions. ``None`` = single colour.
    subject_col : str or None
        Column identifying individual subjects (plotted as points).
    ci : float
        Confidence interval coverage for the group mean bar (default 95 %).
    ax :
        Existing axes. ``None`` creates a new figure.
    title : str or None
        Axes title (defaults to measure name).
    palette : list of str or None
        Colours per group level.

    Returns
    -------
    ax : plt.Axes
    """
    import pandas as _pd

    data = _pd.DataFrame(data)
    groups = [None] if group_col is None else sorted(data[group_col].unique())
    palette = palette or [_S1_COLOR, _S2_COLOR, "#4CAF50", "#9C27B0"]

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, len(groups) * 2.5), 4))

    for gi, grp in enumerate(groups):
        color = palette[gi % len(palette)]
        sub = data if grp is None else data[data[group_col] == grp]
        vals = sub[measure].dropna().values
        if len(vals) == 0:
            continue

        mean = vals.mean()
        se   = vals.std(ddof=1) / np.sqrt(len(vals))
        tcrit = t_dist.ppf((1 + ci) / 2, df=len(vals) - 1) if len(vals) > 1 else 0
        ci_lo, ci_hi = mean - tcrit * se, mean + tcrit * se

        x = gi
        label = str(grp) if grp is not None else measure

        # Individual subject points
        jitter = np.random.default_rng(gi).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals,
                   color=color, alpha=0.4, s=25, zorder=2)

        # Mean ± CI
        ax.errorbar(x, mean, yerr=[[mean - ci_lo], [ci_hi - mean]],
                    fmt="D", color=color, markersize=8, linewidth=2,
                    capsize=5, zorder=3, label=label)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(g) for g in groups] if groups[0] is not None else [""])
    ax.set_ylabel(measure)
    ax.set_title(title or measure)
    if group_col is not None:
        ax.legend(title=group_col, fontsize=8)
    return ax


# ---------------------------------------------------------------------------
# plot_measures
# ---------------------------------------------------------------------------

def plot_measures(
    data,
    measures: Union[str, List[str]],
    group_col: Optional[str] = None,
    ci: float = 0.95,
    palette: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Bar chart comparing one or more measures across conditions / groups.

    Parameters
    ----------
    data : pd.DataFrame
        Output of :func:`fit_group`.
    measures : str or list of str
        Measure column(s) to plot. One subplot per measure.
    group_col : str or None
        Column used to split bars by colour.
    ci : float
        Confidence interval coverage (default 95 %).
    palette : list of str or None
        Colours per group level.
    figsize : tuple or None
        Figure size override.

    Returns
    -------
    fig : plt.Figure
    axes : np.ndarray of plt.Axes
    """
    import pandas as _pd

    data = _pd.DataFrame(data)
    if isinstance(measures, str):
        measures = [measures]

    groups  = [None] if group_col is None else sorted(data[group_col].unique())
    palette = palette or [_S1_COLOR, _S2_COLOR, "#4CAF50", "#9C27B0"]
    n_meas  = len(measures)
    n_grp   = len(groups)

    fig, axes = plt.subplots(1, n_meas,
                             figsize=figsize or (max(4, n_meas * 3), 4),
                             sharey=False)
    if n_meas == 1:
        axes = np.array([axes])

    bar_width = 0.7 / max(n_grp, 1)

    for mi, meas in enumerate(measures):
        ax = axes[mi]
        for gi, grp in enumerate(groups):
            color = palette[gi % len(palette)]
            sub   = data if grp is None else data[data[group_col] == grp]
            vals  = sub[meas].dropna().values
            if len(vals) == 0:
                continue

            mean  = vals.mean()
            se    = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            tcrit = t_dist.ppf((1 + ci) / 2, df=max(len(vals) - 1, 1))
            err   = tcrit * se

            xpos = gi * bar_width - (n_grp - 1) * bar_width / 2
            label = str(grp) if grp is not None else meas
            ax.bar(xpos, mean, width=bar_width * 0.85,
                   color=color, alpha=0.8, label=label, zorder=2)
            ax.errorbar(xpos, mean, yerr=err,
                        fmt="none", color="black", linewidth=1.5, capsize=4, zorder=3)

        ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
        ax.set_title(meas)
        ax.set_xticks([])
        if mi == 0:
            ax.set_ylabel("Value")
        if group_col is not None and mi == n_meas - 1:
            ax.legend(title=group_col, fontsize=8)

    fig.tight_layout()
    return fig, axes
