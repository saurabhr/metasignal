# Alternative SDT Models (sdtr)

`metasignal.sdtr` is a pure-Python port of the model family from
[Macho (2020), *SDT-Models in R*](https://www.unifr.ch/psycho/fr/assets/public/Forschungseinheiten/sdt/SDT.pdf).
None of these models overlap with the 26 measures in `stdpy` or the
hierarchical models in `sdtbayes`. It's in the base install — no optional
extra, and no R/rpy2 dependency at any point.

This is a pre-1.0, experimental subpackage. **Phase 1** (current) implements
the shared MLE estimation engine and the base Gaussian SDT model (manual
Ch. 3.1–3.2). See [Roadmap](roadmap.md) for the planned model family
(mixture, dual-process, high-threshold, bivariate, ranking, and IRF models)
and phasing.

## The base Gaussian SDT model

`n_signals` Gaussian distributions share a single set of
`n_categories - 1` decision thresholds. Signal 0 is the fixed
reference/noise distribution, `N(0, 1)`; signals `1 .. n_signals - 1` have
free `(mean, sd)` parameters.

```python
import numpy as np
from metasignal.sdtr import fit_sdt

# Yes/No recognition data: rows = signals (noise, then old-item strength),
# columns = response categories (respond "new", respond "old").
counts = np.array([
    [1780, 763],   # NEW items
    [883, 1025],   # OLD items
])

result = fit_sdt(counts, restriction="equalvar")
print(result.means, result.thresholds, result.d_a, result.A_z)
```

`restriction="no"` (default) lets each non-reference signal's standard
deviation vary freely (unequal-variance SDT). `restriction="equalvar"` fixes
every SD to 1 (the classic equal-variance model) — use this for rating data
with more than two response categories per signal, where `counts` has one
row per signal and one column per rating category.

## Group-level fitting

```python
from metasignal.sdtr import fit_group

# Trial-level DataFrame: one row per trial, with a signal-class column
# (0 = reference/noise) and a response-category column (1..n_categories).
result = fit_group(df, subject="participant", restriction="equalvar")
```

Mirrors `metasignal.stdpy.fit_group` / `metasignal.itmc.fit_group`: one row
per participant (or per subject × within × between cell), with `mean_<j>`,
`sd_<j>`, `d_a_<j>`, `d_e_<j>`, `A_z_<j>` columns for each non-reference
signal `j`, `threshold_<r>` for each shared threshold, and `logL`/`aic`/`bic`/
`success`.

## Validation

Since there's no R runtime available to cross-check against interactively,
`fit_sdt` is validated against Macho's own worked-example numbers from the
manual (Ch. 5.1.1: the equal-variance Yes/No example above) — see
`tests/test_sdtr.py`.

## API Reference

::: metasignal.sdtr.fit_sdt

::: metasignal.sdtr.fit_group

::: metasignal.sdtr.SDTModelFit
