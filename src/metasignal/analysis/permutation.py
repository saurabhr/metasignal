"""Permutation test for condition differences in SDT measures.

Tests the null hypothesis that two conditions produce the same value of a
measure by shuffling condition labels and recomputing the difference.

Usage example::

    import numpy as np
    from metasignal.analysis import permutation_test

    # condition A
    stim_a = np.array([0, 1] * 40)
    resp_a = np.array([0, 1] * 40)
    conf_a = np.random.randint(1, 3, 80)

    # condition B (slightly worse metacognition)
    stim_b = np.array([0, 1] * 40)
    resp_b = np.array([0, 1, 1, 0] * 20)
    conf_b = np.random.randint(1, 3, 80)

    p_val, obs_diff = permutation_test(
        stim_a, resp_a, conf_a,
        stim_b, resp_b, conf_b,
        n_ratings=2,
        measure_index=5,   # M_ratio
        n_perm=5000,
    )
"""

from __future__ import annotations

import numpy as np

from metasignal.stdpy.compute_all import compute_all_measures


def permutation_test(
    stim_a: np.ndarray,
    resp_a: np.ndarray,
    conf_a: np.ndarray,
    stim_b: np.ndarray,
    resp_b: np.ndarray,
    conf_b: np.ndarray,
    n_ratings: int,
    measure_index: int,
    n_perm: int = 5000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Two-sided permutation test for a difference in one SDT measure between conditions.

    Shuffles trial condition labels ``n_perm`` times and computes the
    difference in the measure on each shuffle to build a null distribution.
    Returns the two-sided p-value and the observed difference.

    Args:
        stim_a: Stimulus array for condition A.
        resp_a: Response array for condition A.
        conf_a: Confidence array for condition A.
        stim_b: Stimulus array for condition B.
        resp_b: Response array for condition B.
        conf_b: Confidence array for condition B.
        n_ratings: Number of confidence rating categories (shared across conditions).
        measure_index: Index into the 26-element output of ``compute_all_measures``.
        n_perm: Number of permutations. Default 5000.
        rng: Optional ``numpy.random.Generator`` for reproducibility.

    Returns:
        Tuple ``(p_value, observed_difference)``.  ``observed_difference`` is
        ``measure(A) - measure(B)``.

    Raises:
        ValueError: If ``measure_index`` is outside [0, 25].
    """
    if not 0 <= measure_index <= 25:
        msg = f"measure_index must be in [0, 25], got {measure_index}"
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng()

    stim_a, resp_a, conf_a = (np.asarray(x) for x in (stim_a, resp_a, conf_a))
    stim_b, resp_b, conf_b = (np.asarray(x) for x in (stim_b, resp_b, conf_b))

    def _measure(s: np.ndarray, r: np.ndarray, c: np.ndarray) -> float:
        return float(compute_all_measures(s, r, c, n_ratings)[measure_index])

    obs_a = _measure(stim_a, resp_a, conf_a)
    obs_b = _measure(stim_b, resp_b, conf_b)
    obs_diff = obs_a - obs_b

    # Pool all trials for shuffling
    stim_all = np.concatenate([stim_a, stim_b])
    resp_all = np.concatenate([resp_a, resp_b])
    conf_all = np.concatenate([conf_a, conf_b])
    n_a = len(stim_a)
    n_total = len(stim_all)

    null_diffs: list[float] = []
    for _ in range(n_perm):
        perm = rng.permutation(n_total)
        idx_a, idx_b = perm[:n_a], perm[n_a:]
        val_a = _measure(stim_all[idx_a], resp_all[idx_a], conf_all[idx_a])
        val_b = _measure(stim_all[idx_b], resp_all[idx_b], conf_all[idx_b])
        diff = val_a - val_b
        if not np.isnan(diff):
            null_diffs.append(diff)

    if len(null_diffs) == 0:
        return (float("nan"), obs_diff)

    null = np.array(null_diffs)
    p_value = float(np.mean(np.abs(null) >= abs(obs_diff)))
    return p_value, obs_diff
