"""Information-theoretic metacognition measures.

Two backends are available via the ``backend`` parameter:

``'simple'`` (default)
    Directly computes MI(accuracy; confidence) = H₂(acc) − H₂(acc|conf).
    Fast, intuitive, follows the notation in Dayan (2023) text.

``'statconfr'``
    Exact port of statConfR (Rausch et al., 2025, JOSS doi:10.21105/joss.06966).
    Builds the full 2×2K joint contingency table over stimulus × signed-response,
    computes I(stimulus; graded_response) − I_min(prior, accuracy) with analytic
    bounds and a continuous Gaussian reference.  More faithful to the original
    Dayan (2023) information-theoretic derivation; values differ when priors are
    unequal or the criterion is biased.

References
----------
Dayan P (2023). Metacognitive Information Theory. Open Mind, 7, 392-411.
  https://doi.org/10.1162/opmi_a_00091
Rausch et al. (2025). statConfR. JOSS. https://doi.org/10.21105/joss.06966
  Source: https://github.com/ManuelRausch/StatConfR

Data conventions
----------------
- stimulus / stim_id : 0 = S1 (noise), 1 = S2 (signal)
- response           : 0 = "S1", 1 = "S2"
- rating             : integer 1 … n_ratings (1 = lowest confidence)
- accuracy           : 1 if response == stimulus, else 0  (derived)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import xlogy

Backend = Literal["simple", "statconfr"]


# ============================================================================
# SHARED LOW-LEVEL ENTROPY HELPERS
# ============================================================================

def _h2(p: float | np.ndarray) -> float | np.ndarray:
    """Binary entropy h₂(p) = −p log₂p − (1−p) log₂(1−p) in bits."""
    p = np.asarray(p, dtype=float)
    return -xlogy(p, p) / np.log(2) - xlogy(1 - p, 1 - p) / np.log(2)


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) in bits; zeros are handled by convention 0·log0 = 0."""
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ============================================================================
# BACKEND: simple  (H₂(acc) − H₂(acc|conf))
# ============================================================================

def _simple_conditional_entropy(acc: np.ndarray, conf: np.ndarray) -> float:
    """Σ_c P(c)·h₂(P(acc=1|c))."""
    H_rc = 0.0
    for c in np.unique(conf):
        mask = conf == c
        p_c = mask.mean()
        H_rc += p_c * float(_h2(acc[mask].mean()))
    return H_rc


def _simple_meta_I(acc: np.ndarray, conf: np.ndarray) -> float:
    """MI(accuracy; confidence) = H₂(acc) − H₂(acc|conf)."""
    return float(_h2(acc.mean())) - _simple_conditional_entropy(acc, conf)


def _simple_bias(acc: np.ndarray, conf: np.ndarray,
                 n_bootstrap: int = 2000, seed: int = 42) -> float:
    """Positive sampling bias estimated by permuting confidence ratings.

    Permuting rating (not accuracy) keeps H₂(acc) constant and breaks the
    confidence–accuracy link, giving a clean null where MI ≈ 0.
    """
    rng = np.random.default_rng(seed)
    null = np.array([_simple_meta_I(acc, rng.permutation(conf))
                     for _ in range(n_bootstrap)])
    return float(null.mean())


# ============================================================================
# BACKEND: statconfr
# (port of estimateMetaI + helpers from ManuelRausch/StatConfR)
# ============================================================================

# --- contingency table ---

def _build_contingency_table(
    stimulus: np.ndarray,
    response: np.ndarray,
    rating: np.ndarray,
) -> np.ndarray:
    """Build the 2 × 2K joint table used by statConfR.

    Rows  = stimulus class (0=S1, 1=S2).
    Cols  = signed response: [−K,…,−1, +1,…,+K] where sign encodes
            correctness (+ = correct) and magnitude = confidence level.
    """
    n_ratings = int(rating.max())
    # signed response column index in [−K…−1, +1…+K]
    # correct → sign = +1; incorrect → sign = −1
    # mapped to 0-based col: negative cols are 0…K-1, positive are K…2K-1
    correct = (stimulus == response).astype(int)
    sign = 2 * correct - 1          # −1 or +1
    # true stimulus label: S1→−1, S2→+1
    true_label = 2 * stimulus - 1   # −1 or +1
    # signed response (as in statConfR): decision·label·rating
    # decision = correct→+1, incorrect→−1; true_label = −1 or +1
    # So: signed_resp = sign * true_label * rating
    # = (+1*(-1)*r) for correct S1 → −r  (left/S1 response, correct)
    # = (−1*(+1)*r) for incorrect S2 → −r  (left/S1 response, wrong)
    # = (+1*(+1)*r) for correct S2 → +r  (right/S2 response, correct)
    # etc.
    # Map to column index: negative cols [0…K-1], positive [K…2K-1]
    signed_resp = sign * true_label * rating.astype(int)

    table = np.zeros((2, 2 * n_ratings), dtype=float)
    for i, (s, sr) in enumerate(zip(stimulus, signed_resp)):
        row = int(s)           # 0 or 1
        # col: sr in {-n_ratings…-1, +1…+n_ratings}
        if sr < 0:
            col = n_ratings + sr   # -K→0, -1→K-1
        else:
            col = n_ratings + sr - 1   # +1→K, +K→2K-1
        table[row, col] += 1
    return table


def _get_info(table: np.ndarray) -> float:
    """I(stimulus; graded_response) = H(cols) + H(rows) − H(joint)."""
    total = table.sum()
    if total == 0:
        return 0.0
    joint = table / total
    return _entropy(joint.sum(axis=0)) + _entropy(joint.sum(axis=1)) - _entropy(joint)


def _get_accuracy_from_table(table: np.ndarray) -> float:
    """P(correct) = sum over columns of max(P(S1|col), P(S2|col))·P(col)."""
    total = table.sum()
    if total == 0:
        return float("nan")
    col_sums = table.sum(axis=0)
    col_max  = table.max(axis=0)
    return float((col_max / total).sum())


# --- analytic information bounds (statConfR: int_get_analytic_information_bounds.R) ---

def _lower_info_bound(prior: np.ndarray, accuracy: float) -> float:
    """Analytic lower bound on I(S; R) at given accuracy (Fano-based)."""
    p = np.sort(prior)[::-1]   # descending
    L = len(p)
    a = float(accuracy)

    if abs(a - 1.0) < 1e-9:
        return _entropy(p)

    if abs(a - float(p[0])) < 1e-9:
        return 0.0

    # Find m3: last index where the condition holds.
    # R uses denom = 1:L - 1 = [0, 1, ..., L-1].
    # At i=0 denom=0: (cumsum[0]-a)/0 → -∞ when a > p[0], so s[0] is always TRUE.
    # At i>0: threshold = (cumsum[i] - a) / i  (i in 1..L-1).
    cumsum = np.cumsum(p)
    s = np.empty(L, dtype=bool)
    s[0] = True  # denominator 0 → threshold -∞ (since a > p[0] in general case)
    if L > 1:
        idx_denom = np.arange(1, L, dtype=float)   # [1, ..., L-1]
        s[1:] = p[1:] >= (cumsum[1:] - a) / idx_denom
    idxs = np.where(s)[0]
    if len(idxs) == 0:
        return float("nan")
    m3 = int(idxs[-1]) + 1   # 1-indexed count

    q = cumsum[m3 - 1]
    pl = p[:m3]
    H_Y  = _entropy(pl / pl.sum()) + np.log2(pl.sum()) * (pl.sum() > 0)
    # Re-derive: statConfR's H_Y = sum(pl * log(1/pl))  (unnormalized Shannon)
    H_Y  = float(np.sum(-xlogy(pl, pl) / np.log(2)))
    rem  = q - a
    if rem <= 0 or m3 <= 1:
        H_YC = a * np.log2(1.0 / a) if a > 0 else 0.0
    else:
        H_YC = (a * np.log2(1.0 / a)
                + rem * np.log2((m3 - 1) / rem))
    return float(H_Y - H_YC)


def _upper_info_bound(prior: np.ndarray, accuracy: float) -> float:
    """Analytic upper bound on I(S; R) at given accuracy."""
    a = float(accuracy)
    if a <= 0:
        return 0.0
    m1 = int(np.floor(1.0 / a))
    m2 = m1 + 1
    H_Y = _entropy(prior)
    if abs(1.0 / m1 - 1.0 / m2) < 1e-15:
        return float(H_Y)
    H_YC = ((1.0 / m1 - a) * np.log2(m2) + (a - 1.0 / m2) * np.log2(m1)
            ) / (1.0 / m1 - 1.0 / m2)
    return float(H_Y - H_YC)


# --- continuous Gaussian reference (statConfR: int_get_normal_noise_information.R) ---

def _gaussian_info_statconfr(dprime: float, n_grid: int = 5000) -> float:
    """I(S; R) for an ideal Gaussian observer with given d′.

    Uses a continuous Gaussian classifier over a fine grid (n=5000), matching
    statConfR's ``get_normal_noise_information(sensitivity=dprime)``.
    """
    if dprime == 0:
        return 0.0
    if np.isinf(dprime):
        return 1.0   # perfect discrimination → 1 bit

    d = abs(float(dprime))
    x = np.linspace(-7 - d, 7 + d, n_grid)
    dx = x[1] - x[0]

    # classifier[0,:] = P(x, S1)·dx = 0.5 · N(x; -d/2, 1) · dx
    # classifier[1,:] = P(x, S2)·dx = 0.5 · N(x; +d/2, 1) · dx
    f1 = norm.pdf(x, -d / 2) * dx * 0.5
    f2 = norm.pdf(x, +d / 2) * dx * 0.5
    table = np.vstack([f1, f2])

    # I(S; R) = H(col marginals) + H(row marginals) - H(joint)
    total = table.sum()
    if total == 0:
        return 0.0
    p_joint = table / total
    return _entropy(p_joint.sum(axis=0)) + _entropy(p_joint.sum(axis=1)) - _entropy(p_joint)


# --- bias reduction (statConfR: int_get_bias_reduced_meta_I_measures.R) ---

def _statconfr_bias(
    table: np.ndarray,
    fn,          # callable(table) → float
    n_sim: int = 1000,
    seed: int = 42,
) -> float:
    """Estimate sampling bias via multinomial row-resampling (statConfR convention).

    Matches statConfR's ``get_bias_reduced_meta_I_measures``:
      1. Resample each row of the table from its observed proportions (n_sim times).
      2. Compute the measure on each resampled table → mean(simulated).
      3. Return  observed − mean(simulated)  (called ``estimated_bias`` in R).

    Callers then compute:
        fn(table) − _statconfr_bias(table, fn) = mean(simulated)

    which is exactly statConfR's ``bias_reduced`` value.  The double subtraction
    is intentional: observed − (observed − mean(sim)) = mean(sim).
    """
    rng = np.random.default_rng(seed)
    observed = fn(table)
    null_vals = []
    for _ in range(n_sim):
        sim = np.zeros_like(table)
        for r in range(table.shape[0]):
            n_r = int(table[r].sum())
            probs = table[r] / n_r if n_r > 0 else np.ones(table.shape[1]) / table.shape[1]
            sim[r] = rng.multinomial(n_r, probs)
        acc = _get_accuracy_from_table(sim)
        if acc is None or np.isnan(acc) or abs(acc - 1.0) < 1e-6 or abs(acc) < 1e-6:
            continue
        null_vals.append(fn(sim))
    if not null_vals:
        return 0.0
    return observed - float(np.mean(null_vals))


# --- statconfr single-value computations ---

def _sc_meta_I(table: np.ndarray) -> float:
    prior = table.sum(axis=1) / table.sum()
    acc   = _get_accuracy_from_table(table)
    info  = _get_info(table)
    lb    = _lower_info_bound(prior, acc)
    return float(info - lb)


def _sc_meta_Ir1(table: np.ndarray, dprime: float) -> float:
    # I_min uses observed accuracy as the lower bound for BOTH numerator and
    # denominator (matching statConfR), even though a Gaussian observer at this
    # d' would have a slightly different accuracy.
    prior  = table.sum(axis=1) / table.sum()
    acc    = _get_accuracy_from_table(table)
    lb     = _lower_info_bound(prior, acc)
    info_g = _gaussian_info_statconfr(dprime)
    denom  = info_g - lb
    if denom == 0:
        return float("nan")
    return float(_sc_meta_I(table) / denom)


def _sc_meta_Ir1_acc(table: np.ndarray) -> float:
    prior  = table.sum(axis=1) / table.sum()
    acc    = _get_accuracy_from_table(table)
    dp_acc = 2.0 * norm.ppf(acc) if 0 < acc < 1 else float("nan")
    if np.isnan(dp_acc):
        return float("nan")
    lb     = _lower_info_bound(prior, acc)
    info_g = _gaussian_info_statconfr(dp_acc)
    denom  = info_g - lb
    if denom == 0:
        return float("nan")
    return float(_sc_meta_I(table) / denom)


def _sc_meta_Ir2(table: np.ndarray) -> float:
    acc = _get_accuracy_from_table(table)
    H_a = float(_h2(acc))
    if H_a == 0:
        return float("nan")
    return float(_sc_meta_I(table) / H_a)


def _sc_RMI(table: np.ndarray) -> float:
    prior  = table.sum(axis=1) / table.sum()
    acc    = _get_accuracy_from_table(table)
    info   = _get_info(table)
    lb     = _lower_info_bound(prior, acc)
    ub     = _upper_info_bound(prior, acc)
    denom  = ub - lb
    if abs(denom) < 1e-15:
        return float("nan")
    return float((info - lb) / denom)


# ============================================================================
# SENSITIVITY ESTIMATION (statConfR convention)
# (uses response fractions, not raw stimulus/response — matches R source)
# ============================================================================

def _estimate_dprime_statconfr(table: np.ndarray) -> float:
    """d' from contingency table using statConfR's estimate_sensitivity().

    Note: the formula divides by total S1/S2 *response* counts rather than
    stimulus counts, which is only exact when N_S1 = N_S2 (balanced design).
    Unbalanced designs will give a biased d' estimate; this matches statConfR.
    """
    n_resp = table.shape[1]
    half   = n_resp // 2
    # Cols 0…half-1 are "S1" responses (CR + Miss); half…2K-1 are "S2" responses (Hit + FA)
    n_correct_s1 = table[0, :half].sum()   # CR: S1 stim, S1 resp
    n_s1_resp    = table[:, :half].sum()   # all S1 responses
    n_correct_s2 = table[1, half:].sum()   # S2 stim, S2 resp (correct)
    n_s2_resp    = table[:, half:].sum()   # all S2 responses

    # 0.5-correction (Miller/Murdock-Ogilvie convention)
    if n_correct_s1 == n_s1_resp:
        n_correct_s1 -= 0.5
    if n_correct_s1 == 0:
        n_correct_s1 = 0.5
    if n_correct_s2 == n_s2_resp:
        n_correct_s2 -= 0.5
    if n_correct_s2 == 0:
        n_correct_s2 = 0.5

    hr  = n_correct_s1 / n_s1_resp if n_s1_resp > 0 else 0.5
    far = 1 - n_correct_s2 / n_s2_resp if n_s2_resp > 0 else 0.5
    return float(norm.ppf(hr) - norm.ppf(far))


# ============================================================================
# PUBLIC API — all measures accept backend='simple' | 'statconfr'
# ============================================================================

def meta_I(
    stimulus: np.ndarray,
    response: np.ndarray,
    rating: np.ndarray,
    *,
    backend: Backend = "simple",
    bias_correction: bool = False,
    seed: int = 42,
) -> float:
    """Mutual information between confidence and accuracy (bits).

    Parameters
    ----------
    stimulus, response, rating:
        Trial-level arrays. stimulus/response are 0/1; rating is 1…n_ratings.
    backend:
        ``'simple'``: H₂(acc) − H₂(acc|conf).  Fast, directly measures how
        much confidence predicts correctness.

        ``'statconfr'``: I(stimulus; graded_response) − I_min(prior, acc).
        Builds the full 2×2K joint table, computes Shannon MI, then subtracts
        the analytic minimum achievable at this accuracy.  Matches the R package
        exactly.
    bias_correction:
        Subtract the estimated positive sampling bias.
        ``'simple'``: permutation of accuracy labels (2000 shuffles).
        ``'statconfr'``: multinomial resampling of each table row (1000 sims),
        matching statConfR's ``bias_reduction=TRUE``.  Returns mean(simulated),
        which can be higher or lower than the observed value.
    seed:
        RNG seed for bias reduction.
    """
    stimulus = np.asarray(stimulus)
    response = np.asarray(response)
    rating   = np.asarray(rating)

    if backend == "simple":
        acc = (stimulus == response).astype(int)
        mi  = _simple_meta_I(acc, rating)
        if bias_correction:
            mi -= _simple_bias(acc, rating, seed=seed)
        return float(mi)

    if backend == "statconfr":
        table = _build_contingency_table(stimulus, response, rating)
        if bias_correction:
            bias = _statconfr_bias(table, _sc_meta_I, seed=seed)
            return float(_sc_meta_I(table) - bias)
        return _sc_meta_I(table)

    raise ValueError(f"Unknown backend {backend!r}. Choose 'simple' or 'statconfr'.")


def meta_Ir1(
    stimulus: np.ndarray,
    response: np.ndarray,
    rating: np.ndarray,
    dprime: float | None = None,
    *,
    backend: Backend = "simple",
    bias_correction: bool = False,
    seed: int = 42,
) -> float:
    """Gaussian-normalised relative metacognitive efficiency (meta-I₁ʳ).

    meta-I₁ʳ = meta-I / meta-I_Gaussian(d′)

    Parameters
    ----------
    dprime:
        First-order sensitivity. If None, estimated from stimulus/response.
    backend:
        ``'simple'``: Gaussian reference via Monte Carlo simulation with
        equal-count confidence bins.

        ``'statconfr'``: Gaussian reference via continuous 5000-point grid
        integration of the ideal Gaussian classifier, matching the R package.
    """
    stimulus = np.asarray(stimulus)
    response = np.asarray(response)
    rating   = np.asarray(rating)

    if backend == "simple":
        from metasignal.stdpy.core import compute_sdt_resp
        acc = (stimulus == response).astype(int)
        mi  = _simple_meta_I(acc, rating)
        if bias_correction:
            mi -= _simple_bias(acc, rating, seed=seed)
        if dprime is None:
            dprime, *_ = compute_sdt_resp(stimulus, response)
        n_ratings = int(np.unique(rating).size)
        expected  = _gaussian_meta_I_simple(abs(dprime), n_ratings=n_ratings)
        return float("nan") if expected == 0 else float(mi / expected)

    if backend == "statconfr":
        table = _build_contingency_table(stimulus, response, rating)
        if dprime is None:
            dprime = _estimate_dprime_statconfr(table)
        if bias_correction:
            fn   = lambda t: _sc_meta_Ir1(t, dprime)
            bias = _statconfr_bias(table, fn, seed=seed)
            return float(_sc_meta_Ir1(table, dprime) - bias)
        return _sc_meta_Ir1(table, dprime)

    raise ValueError(f"Unknown backend {backend!r}.")


def meta_Ir1_acc(
    stimulus: np.ndarray,
    response: np.ndarray,
    rating: np.ndarray,
    *,
    backend: Backend = "simple",
    bias_correction: bool = False,
    seed: int = 42,
) -> float:
    """Accuracy-normalised relative efficiency (meta-I₁ʳ_acc).

    Normalises by Gaussian meta-I at d′ derived from observed accuracy,
    avoiding SDT assumptions about criterion placement.
    """
    stimulus = np.asarray(stimulus)
    response = np.asarray(response)
    rating   = np.asarray(rating)

    if backend == "simple":
        acc = (stimulus == response).astype(int)
        mi  = _simple_meta_I(acc, rating)
        if bias_correction:
            mi -= _simple_bias(acc, rating, seed=seed)
        p_acc = float(acc.mean())
        if p_acc <= 0 or p_acc >= 1:
            return float("nan")
        dp_acc    = 2.0 * norm.ppf(p_acc)
        n_ratings = int(np.unique(rating).size)
        expected  = _gaussian_meta_I_simple(abs(dp_acc), n_ratings=n_ratings)
        return float("nan") if expected == 0 else float(mi / expected)

    if backend == "statconfr":
        table = _build_contingency_table(stimulus, response, rating)
        if bias_correction:
            bias = _statconfr_bias(table, _sc_meta_Ir1_acc, seed=seed)
            return float(_sc_meta_Ir1_acc(table) - bias)
        return _sc_meta_Ir1_acc(table)

    raise ValueError(f"Unknown backend {backend!r}.")


def meta_Ir2(
    stimulus: np.ndarray,
    response: np.ndarray,
    rating: np.ndarray,
    *,
    backend: Backend = "simple",
    bias_correction: bool = False,
    seed: int = 42,
) -> float:
    """Entropy-normalised absolute metacognitive efficiency (meta-I₂ʳ).

    meta-I₂ʳ = meta-I / H₂(accuracy).  Range [0, 1].

    Both backends use H₂(accuracy) as the denominator; the numerator
    (meta-I) differs between backends.
    """
    stimulus = np.asarray(stimulus)
    response = np.asarray(response)
    rating   = np.asarray(rating)

    if backend == "simple":
        acc = (stimulus == response).astype(int)
        H_r = float(_h2(acc.mean()))
        if H_r == 0:
            return float("nan")
        mi = _simple_meta_I(acc, rating)
        if bias_correction:
            mi -= _simple_bias(acc, rating, seed=seed)
        return float(mi / H_r)

    if backend == "statconfr":
        table = _build_contingency_table(stimulus, response, rating)
        if bias_correction:
            bias = _statconfr_bias(table, _sc_meta_Ir2, seed=seed)
            return float(_sc_meta_Ir2(table) - bias)
        return _sc_meta_Ir2(table)

    raise ValueError(f"Unknown backend {backend!r}.")


def RMI(
    stimulus: np.ndarray,
    response: np.ndarray,
    rating: np.ndarray,
    *,
    backend: Backend = "simple",
    bias_correction: bool = False,
    seed: int = 42,
) -> float:
    """Relative Meta-Information, range [0, 1].

    RMI = (meta-I − I_min) / (I_max − I_min)

    Parameters
    ----------
    backend:
        ``'simple'``: bounds are H_s − H_r (lower) and H_s (upper) where
        H_s = H₂(P(S2)) and H_r = H₂(accuracy).  These bounds assume
        I(S;R) as the numerator; since the simple backend uses MI(acc;conf)
        instead, RMI can fall outside [0, 1] — treat as approximate.
        Prefer the ``'statconfr'`` backend for RMI.

        ``'statconfr'``: analytic Fano-inequality bounds
        (``get_lower_info_for_one`` / ``get_upper_info_for_one`` in statConfR)
        applied to I(stimulus; graded_response).  RMI is properly bounded in
        [0, 1] and matches the R package exactly.
    """
    stimulus = np.asarray(stimulus)
    response = np.asarray(response)
    rating   = np.asarray(rating)

    if backend == "simple":
        acc       = (stimulus == response).astype(int)
        p_correct = float(acc.mean())
        p_plus    = float((stimulus == 1).mean())
        H_s       = float(_h2(p_plus))
        H_r       = float(_h2(p_correct))
        min_info  = H_s - H_r
        max_info  = H_s
        denom     = max_info - min_info
        if denom <= 0:
            return float("nan")
        mi = _simple_meta_I(acc, rating)
        if bias_correction:
            mi -= _simple_bias(acc, rating, seed=seed)
        return float((mi - min_info) / denom)

    if backend == "statconfr":
        table = _build_contingency_table(stimulus, response, rating)
        if bias_correction:
            bias = _statconfr_bias(table, _sc_RMI, seed=seed)
            return float(_sc_RMI(table) - bias)
        return _sc_RMI(table)

    raise ValueError(f"Unknown backend {backend!r}.")


# ============================================================================
# HYPOTHESIS TEST
# ============================================================================

def permtest_meta_I(
    stimulus: np.ndarray,
    response: np.ndarray,
    rating: np.ndarray,
    *,
    backend: Backend = "simple",
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    """Test whether meta-I is significantly greater than chance.

    Builds a permutation null by shuffling confidence ratings across trials,
    which breaks the confidence–accuracy link while preserving everything else.
    Returns the observed value, bias-corrected value (observed − mean(null)),
    a one-tailed p-value, and null distribution summary statistics.

    This is the IT analogue of asking "is metacognition above chance?" without
    fitting a parametric model.

    Parameters
    ----------
    stimulus, response, rating :
        Trial-level arrays (same conventions as ``meta_I``).
    backend :
        ``'simple'`` or ``'statconfr'``.  Both use rating permutation for the
        null; the backend controls how meta-I is computed on each permuted sample.
    n_perm :
        Number of permutations for the null distribution.
    seed :
        RNG seed.

    Returns
    -------
    dict with keys:
        ``observed``   — raw meta-I
        ``corrected``  — bias-corrected meta-I (observed − mean(null))
        ``p_value``    — proportion of null samples ≥ observed (one-tailed)
        ``null_mean``  — mean of null distribution
        ``null_std``   — std of null distribution
        ``null``       — full null array (length n_perm)
        ``backend``    — backend used
        ``n_perm``     — number of permutations used
    """
    stimulus = np.asarray(stimulus)
    response = np.asarray(response)
    rating   = np.asarray(rating)
    rng      = np.random.default_rng(seed)

    if backend == "simple":
        acc      = (stimulus == response).astype(int)
        observed = _simple_meta_I(acc, rating)
        null     = np.array([_simple_meta_I(acc, rng.permutation(rating))
                             for _ in range(n_perm)])

    elif backend == "statconfr":
        observed = _sc_meta_I(_build_contingency_table(stimulus, response, rating))
        null_vals = []
        for _ in range(n_perm):
            perm_rating = rng.permutation(rating)
            t = _build_contingency_table(stimulus, response, perm_rating)
            acc_t = _get_accuracy_from_table(t)
            if acc_t is None or np.isnan(acc_t) or abs(acc_t - 1.0) < 1e-6:
                continue
            null_vals.append(_sc_meta_I(t))
        null = np.array(null_vals)

    else:
        raise ValueError(f"Unknown backend {backend!r}. Choose 'simple' or 'statconfr'.")

    null_mean = float(null.mean()) if len(null) else float("nan")
    null_std  = float(null.std())  if len(null) else float("nan")
    p_value   = float((null >= observed).mean()) if len(null) else float("nan")
    corrected = observed - null_mean

    return {
        "observed":  float(observed),
        "corrected": float(corrected),
        "p_value":   p_value,
        "null_mean": null_mean,
        "null_std":  null_std,
        "null":      null,
        "backend":   backend,
        "n_perm":    len(null),
    }


# ============================================================================
# GROUP-LEVEL WRAPPER
# ============================================================================

def estimate_meta_I(
    data: pd.DataFrame,
    *,
    stimulus_col: str = "stimulus",
    response_col: str = "response",
    rating_col: str = "rating",
    participant_col: str = "participant",
    backend: Backend = "simple",
    bias_correction: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute all five IT metacognition measures per participant.

    Mirrors ``estimateMetaI()`` from statConfR (Rausch et al., 2025).

    Parameters
    ----------
    data:
        DataFrame with one row per trial.
    stimulus_col, response_col, rating_col, participant_col:
        Column name overrides.
    backend:
        ``'simple'`` or ``'statconfr'`` — see module docstring.
    bias_correction:
        Subtract estimated positive sampling bias.
    seed:
        RNG seed.

    Returns
    -------
    pd.DataFrame
        One row per participant with columns:
        ``participant``, ``meta_I``, ``meta_Ir1``, ``meta_Ir1_acc``,
        ``meta_Ir2``, ``RMI``.
    """
    if backend == "statconfr":
        global_max = int(data[rating_col].max())
        offenders = [
            (pid, int(grp[rating_col].max()))
            for pid, grp in data.groupby(participant_col, sort=False)
            if int(grp[rating_col].max()) != global_max
        ]
        if offenders:
            import warnings

            warnings.warn(
                f"estimate_meta_I: confidence scale (max rating) is {global_max} "
                f"overall, but {len(offenders)} participant(s) never used the top "
                f"rating (e.g. participant {offenders[0][0]}). 'statconfr' infers "
                "n_ratings per participant from their own observed max, so these "
                "participants' analytic bounds are not comparable to the rest of "
                "the group.",
                stacklevel=2,
            )

    records = []
    for pid, grp in data.groupby(participant_col, sort=False):
        stim = grp[stimulus_col].to_numpy()
        resp = grp[response_col].to_numpy()
        rat  = grp[rating_col].to_numpy()

        kw = dict(backend=backend, bias_correction=bias_correction, seed=seed)

        if backend == "statconfr":
            table  = _build_contingency_table(stim, resp, rat)
            dp     = _estimate_dprime_statconfr(table)
        else:
            try:
                from metasignal.stdpy.core import compute_sdt_resp
                dp, *_ = compute_sdt_resp(stim, resp)
            except Exception:
                dp = None

        rec = {
            participant_col: pid,
            "meta_I":        meta_I(stim, resp, rat, **kw),
            "meta_Ir1":      meta_Ir1(stim, resp, rat, dprime=dp, **kw),
            "meta_Ir1_acc":  meta_Ir1_acc(stim, resp, rat, **kw),
            "meta_Ir2":      meta_Ir2(stim, resp, rat, **kw),
            "RMI":           RMI(stim, resp, rat, **kw),
        }
        records.append(rec)

    cols = [participant_col, "meta_I", "meta_Ir1", "meta_Ir1_acc", "meta_Ir2", "RMI"]
    return pd.DataFrame(records, columns=cols)


# ============================================================================
# SIMPLE-BACKEND GAUSSIAN REFERENCE (kept separate from statconfr version)
# ============================================================================

def _gaussian_meta_I_simple(dprime: float, n_ratings: int = 4) -> float:
    """Gaussian meta-I for the simple backend via Monte Carlo + equal-count bins."""
    rng    = np.random.default_rng(0)
    n_sim  = 200_000
    stim   = rng.choice([0, 1], size=n_sim)
    mean   = np.where(stim == 1, dprime / 2, -dprime / 2)
    x      = rng.normal(mean, 1)
    resp   = (x > 0).astype(int)
    acc    = (resp == stim).astype(int)
    conf_c = norm.cdf(np.abs(x))
    qs     = np.linspace(0, 1, n_ratings + 1)
    bdry   = np.quantile(conf_c, qs)
    bdry[-1] += 1e-9
    conf_b = np.digitize(conf_c, bdry[1:])
    return _simple_meta_I(acc, conf_b)
