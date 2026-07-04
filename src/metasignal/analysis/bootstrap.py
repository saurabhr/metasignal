"""Non-parametric bootstrap confidence intervals for SDT measures.

Usage example::

    import numpy as np
    from metasignal import stdpy
    from metasignal.analysis import bootstrap_measure

    stim = np.array([0, 1] * 50)
    resp = np.array([0, 1] * 50)
    conf = np.random.randint(1, 3, 100)

    ci_low, ci_high = bootstrap_measure(
        stim, resp, conf, n_ratings=2,
        measure_index=0,   # meta_d (index 0 of the 26-element array)
        n_boot=2000,
        ci=0.95,
    )
"""

from __future__ import annotations

import warnings

import numpy as np

from metasignal.stdpy.compute_all import compute_all_measures


def bootstrap_measure(
    stim: np.ndarray,
    resp: np.ndarray,
    conf: np.ndarray,
    n_ratings: int,
    measure_index: int,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return a bootstrap confidence interval for one element of the 26-measure array.

    Resamples trials with replacement ``n_boot`` times and computes the
    requested measure on each resample. The interval is percentile-based.

    Args:
        stim: Stimulus array (binary, same encoding as ``compute_all_measures``).
        resp: Response array (binary).
        conf: Confidence rating array (1 to n_ratings).
        n_ratings: Number of confidence rating categories.
        measure_index: Index into the 26-element output of ``compute_all_measures``.
            See that function's docstring for the index→measure mapping.
        n_boot: Number of bootstrap resamples. Default 2000.
        ci: Coverage of the interval, e.g. 0.95 for a 95% CI. Default 0.95.
        rng: Optional ``numpy.random.Generator`` for reproducibility.

    Returns:
        Tuple ``(lower, upper)`` confidence bounds.

    Raises:
        ValueError: If ``measure_index`` is outside [0, 25].
    """
    if not 0 <= measure_index <= 25:
        msg = f"measure_index must be in [0, 25], got {measure_index}"
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng()

    stim = np.asarray(stim)
    resp = np.asarray(resp)
    conf = np.asarray(conf)
    n = len(stim)

    boot_vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        result = compute_all_measures(stim[idx], resp[idx], conf[idx], n_ratings)
        val = result[measure_index]
        if not np.isnan(val):
            boot_vals.append(float(val))

    if len(boot_vals) == 0:
        warnings.warn(
            f"bootstrap_measure: all {n_boot} resamples produced NaN for "
            f"measure_index={measure_index}; returning (nan, nan).",
            stacklevel=2,
        )
        return (float("nan"), float("nan"))

    if len(boot_vals) < 0.5 * n_boot:
        warnings.warn(
            f"bootstrap_measure: only {len(boot_vals)} of {n_boot} requested "
            f"resamples produced a finite value for measure_index={measure_index} "
            "(the rest were NaN, e.g. due to MLE non-convergence); the returned "
            "CI is based on this smaller effective sample.",
            stacklevel=2,
        )

    alpha = 1.0 - ci
    lo = float(np.percentile(boot_vals, 100 * alpha / 2))
    hi = float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))
    return lo, hi
