"""Group × sessions state-space model for time-varying meta-d'.

Models how a group's metacognitive efficiency evolves across repeated
measurement sessions.  The group-level log M-ratio follows a random walk
over sessions, with separate per-participant stable offsets.

Data shape
----------
``sessions : list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]]``

    sessions[t]    — list of (stim, resp, conf) tuples for each participant
                     at session t.  All sessions must contain the same number
                     of participants in the same order.
    sessions[t][i] — (stim, resp, conf) for participant i at time t.

Stage-1 MLE per (participant, session) produces an observed log M-ratio
matrix ``log_mr_obs[i, t]`` which serves as the outcome for the Stan model.
Sessions where MLE fails are treated as missing data.

Stan model
----------
::

    mu_logMr[1]  ~ Normal(0, 1)               # initial group mean
    mu_logMr[t]  ~ Normal(mu_logMr[t-1], sigma_process)   # random walk
    sigma_process ~ Exponential(1)            # session-to-session drift

    subj_z[i]    ~ Normal(0, 1)               # per-participant stable offset
    sigma_subj   ~ Exponential(1)             # between-subject SD

    log_mr_obs[i, t] ~ Normal(mu_logMr[t] + sigma_subj * subj_z[i], sigma_obs)
    sigma_obs    ~ Exponential(1)             # observation noise (MLE uncertainty)

Key posterior parameters
------------------------
- ``mu_logMr[t]`` — group-level log M-ratio trajectory over sessions.
  Convert to M-ratio: ``exp(mu_logMr[t])``.
- ``sigma_process`` — session-to-session drift of the group mean.
  Small values → stable; large values → rapid change.
- ``sigma_subj`` — stable between-subject SD (constant across sessions).
- ``sigma_obs`` — observation noise (captures MLE estimation error).
- ``subj_z[i]`` — per-participant stable deviation (non-centred).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from metasignal.sdtbayes.diagnostics import FitResult

_BRMSPY_MSG = "brmspy is not installed. Run:\n    pip install metasignal[sdtbayes]"

# ---------------------------------------------------------------------------
# Stan model blocks
# ---------------------------------------------------------------------------

_SS_DATA = """\
int<lower=1> T;               // number of sessions
int<lower=1> N;               // number of participants
matrix[N, T] log_mr_obs;      // observed MLE log M-ratio; 0 where invalid
matrix[N, T] is_valid;        // 1.0 = valid observation, 0.0 = missing
"""

_SS_PARAMETERS = """\
vector[T] mu_logMr;              // group-level trajectory
real<lower=0> sigma_process;     // session-to-session drift (process noise)
real<lower=0> sigma_subj;        // stable between-subject scale
real<lower=0> sigma_obs;         // observation noise
vector[N] subj_z;                // per-participant stable offset (non-centred)
"""

_SS_TRANSFORMED_PARAMETERS = """\
vector[N] subj_effect;
subj_effect = sigma_subj * subj_z;
"""

_SS_MODEL = """\
// Process priors
sigma_process ~ exponential(1);
sigma_subj    ~ exponential(1);
sigma_obs     ~ exponential(1);
subj_z        ~ std_normal();

// Random walk prior on group trajectory
mu_logMr[1]   ~ normal(0, 1);
for (t in 2:T)
    mu_logMr[t] ~ normal(mu_logMr[t - 1], sigma_process);

// Observation model (skip missing sessions)
for (n in 1:N) {
    for (t in 1:T) {
        if (is_valid[n, t] > 0.5)
            log_mr_obs[n, t] ~ normal(mu_logMr[t] + subj_effect[n], sigma_obs);
    }
}
"""

_SS_GENERATED = """\
// Posterior predictive trajectory on the M-ratio scale
vector[T] group_mratio;
for (t in 1:T)
    group_mratio[t] = exp(mu_logMr[t]);
"""


# ---------------------------------------------------------------------------
# Stage-1 helper
# ---------------------------------------------------------------------------

def _mle_matrix(
    sessions: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    n_ratings: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (log_mr_obs, is_valid) arrays of shape (N, T).

    Invalid sessions (MLE failed or log M-ratio undefined) are marked with
    is_valid=0 and a placeholder value of 0.0 in log_mr_obs.
    """
    from metasignal.stdpy.core import compute_sdt_resp, trials_to_counts
    from metasignal.stdpy.metad import fit_meta_d_mle

    n_sess = len(sessions)
    n_subj = len(sessions[0])
    log_mr = np.zeros((n_subj, n_sess), dtype=float)
    valid = np.zeros((n_subj, n_sess), dtype=float)

    for t, session in enumerate(sessions):
        if len(session) != n_subj:
            msg = (
                f"Session {t} has {len(session)} participants but session 0 "
                f"has {n_subj}.  All sessions must have the same number of participants."
            )
            raise ValueError(msg)
        for i, (stim, resp, conf) in enumerate(session):
            stim = np.asarray(stim)
            resp = np.asarray(resp)
            conf = np.asarray(conf)
            try:
                dp, _, _ = compute_sdt_resp(stim, resp)
                nr_s1, nr_s2 = trials_to_counts(stim, resp, conf, n_ratings)
                res = fit_meta_d_mle(nr_s1, nr_s2)
                m_ratio = res["M_ratio"]
                if np.isnan(m_ratio) or m_ratio <= 0 or dp < 0.2:
                    raise ValueError("invalid M-ratio")
                log_mr[i, t] = float(np.log(m_ratio))
                valid[i, t] = 1.0
            except (ValueError, RuntimeError) as exc:
                warnings.warn(
                    f"Participant {i}, session {t}: MLE failed ({exc}). "
                    "Treating as missing.",
                    stacklevel=3,
                )

    return log_mr, valid


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_statespace_metad(
    sessions: list[list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    n_ratings: int,
    chains: int = 4,
    n_iter: int = 2000,
    warmup: int = 1000,
    seed: int = 42,
    **kwargs: Any,
) -> FitResult:
    """Group × sessions state-space model for time-varying meta-d'.

    Fits a random-walk model to the group-level log M-ratio trajectory
    across sessions.  Per-participant stable offsets capture consistent
    individual differences; session-to-session drift is captured by
    ``sigma_process``.

    Args:
        sessions: ``sessions[t][i]`` = ``(stim, resp, conf)`` for participant
            ``i`` at session ``t``.  All sessions must have the same number of
            participants in the same order.
        n_ratings: Number of confidence rating categories.
        chains: MCMC chains (default 4).
        n_iter: Total iterations per chain including warmup (default 2000).
        warmup: Warmup iterations (default 1000).
        seed: Random seed (default 42).
        **kwargs: Forwarded to ``brmspy.brms.brm``.

    Returns:
        ``FitResult``.  Key posterior parameters:

        - ``mu_logMr[t]`` — group-level log M-ratio at session ``t``.
            Use ``exp(mu_logMr)`` for M-ratio trajectory.
        - ``group_mratio[t]`` — same on M-ratio scale (generated quantity).
        - ``sigma_process`` — session-to-session drift.
        - ``sigma_subj`` — stable between-subject SD.
        - ``sigma_obs`` — observation (MLE estimation) noise.
        - ``subj_effect[i]`` — per-participant stable offset.

    Raises:
        RuntimeError: Always — see Notes.
        ValueError: If sessions are inconsistent in participant count, or
            fewer than 2 sessions are provided.

    Notes:
        This function's brmspy stanvar-injection path is currently
        **non-functional**: the brmspy/rpy2 bridge cannot reliably convert a
        list of multiple ``stanvar()`` objects into R.  A ``block="data"``
        stanvar (carrying a real data value via ``x=``) round-trips to Python
        as a ``dict``, while a ``block != "data"`` stanvar (code-only, e.g.
        ``block="parameters"``) round-trips as a ``list`` — mixing the two
        shapes in one Python list breaks rpy2's homogeneous-type dispatch, and
        no combination method (``c()``, ``+``, dummy data values) was found
        to fix this from calling code.  This is an upstream brmspy limitation.

        Unlike :func:`~metasignal.sdtbayes.fit_full_metad` (whose functionality
        has an equivalent cmdstanpy-backed implementation via
        :func:`~metasignal.sdtbayes.fit_meta_formula`), the state-space
        random-walk-over-sessions model has no cmdstanpy port yet — there is
        currently no working replacement for this specific model.

    Example::

        import numpy as np
        import arviz as az
        from metasignal.sdtbayes import fit_statespace_metad

        rng = np.random.default_rng(0)
        N, T = 20, 5
        # 5 sessions, 20 participants each
        sessions = [
            [
                (rng.integers(0, 2, 150), rng.integers(0, 2, 150), rng.integers(1, 5, 150))
                for _ in range(N)
            ]
            for _ in range(T)
        ]
        fit = fit_statespace_metad(sessions, n_ratings=4)

        post = az.extract(fit.idata)
        trajectory = np.exp(post["mu_logMr"].values)   # shape (T, draws)
        print("Group M-ratio trajectory (session means):")
        print(trajectory.mean(axis=-1))
        print(f"Estimated drift (sigma_process): "
              f"{post['sigma_process'].values.mean():.3f}")
    """
    n_sess = len(sessions)
    if n_sess < 2:
        msg = "Need at least 2 sessions for the state-space model."
        raise ValueError(msg)

    raise RuntimeError(
        "fit_statespace_metad is currently unavailable: its brmspy stanvar-injection "
        "path is blocked by an upstream brmspy/rpy2 conversion limitation (a list "
        "mixing data-carrying and code-only stanvar() objects cannot be converted "
        "to R), and there is no cmdstanpy-backed equivalent for this state-space "
        "model yet. If you need a time-varying trajectory of M-ratio, compute "
        "per-session MLE estimates via metasignal.stdpy.fit_meta_d_mle and fit a "
        "trend model directly, or use fit_full_metad(...) per session as an "
        "approximation."
    )
