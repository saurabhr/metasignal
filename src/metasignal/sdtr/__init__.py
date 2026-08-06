"""Alternative Signal Detection Theory models (sdtr).

Python port of the model family from:
  Macho S (2020). SDT-Models in R.
  https://www.unifr.ch/psycho/fr/assets/public/Forschungseinheiten/sdt/SDT.pdf

Phase 1 implements the base Gaussian SDT model (manual Ch. 3.1-3.2: the
``"SDT"`` model, and the ``"Gaussian"`` model restricted to a single shared
threshold set) and the shared MLE estimation engine every later model in
this subpackage will reuse. See ``docs/roadmap.md`` for the full planned
model family (mixture, dual-process, high-threshold, bivariate, ranking,
and IRF models) and phasing.

This is a pre-1.0, experimental component; its API may change between
releases.
"""

from metasignal.sdtr._optimize import SDTFitResult, expand_params, fit_mle
from metasignal.sdtr.group import fit_group
from metasignal.sdtr.sdt import SDTModelFit, fit_sdt

__all__ = [
    "fit_sdt",
    "fit_group",
    "SDTModelFit",
    "SDTFitResult",
    "fit_mle",
    "expand_params",
]
