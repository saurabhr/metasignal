#!/usr/bin/env python3
"""Recompute only meta-uncertainty (index 16) in Python precomputed caches.

After the multi-start deterministic fix in ``src/metasignal/stdpy/uncertainty.py``,
this patches column 16 of existing ``notebooks/precomputed/*_mle.npz`` arrays
using the same trial subsets as ``analysis_core`` / ``refresh_metanoise_caches.py``.

Replicability
-------------
Same preprocessing, same masks, same merge-into-existing-npz pattern.
Only the meta-uncertainty estimator changes (multi-start NLL).
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

# Avoid BLAS oversubscription during many serial fits.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

warnings.filterwarnings("ignore")

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "notebooks"))

import numpy as np
from analysis_core import (
    preprocess_haddara,
    preprocess_maniscalco,
    preprocess_rouault,
    preprocess_shekhar,
    preprocess_locke,
    xue_recode,
)
from metasignal.stdpy.uncertainty import compute_meta_uncertainty

OUT = os.path.join(REPO, "notebooks", "precomputed")
N_MEAS = 26
META_UNCERT_IDX = 16
N_STARTS = 5


def log(msg: str) -> None:
    print(msg, flush=True)


def overlay_column(existing: np.ndarray, new_partial: np.ndarray) -> np.ndarray:
    out = np.array(existing, dtype=float, copy=True)
    out[..., META_UNCERT_IDX] = new_partial[..., META_UNCERT_IDX]
    return out


def merge_overlay(path: str, key: str, new_partial: np.ndarray) -> None:
    with np.load(path) as data:
        existing = {k: data[k] for k in data.files}
    if key not in existing:
        raise KeyError(f"{path} missing key {key}")
    existing[key] = overlay_column(existing[key], new_partial)
    np.savez(path, **existing)
    col = existing[key][..., META_UNCERT_IDX]
    log(
        f"  updated {path}[{key}] meta-uncertainty "
        f"mean={np.nanmean(col):.4f} n_finite={int(np.isfinite(col).sum())}"
    )


def unc(stim, resp, conf, n_ratings) -> float:
    return float(
        compute_meta_uncertainty(
            stim, resp, conf, int(n_ratings), n_starts=N_STARTS
        )
    )


def compute_raw(subjects) -> np.ndarray:
    out = np.full((len(subjects), N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        out[i, META_UNCERT_IDX] = unc(s["stim"], s["resp"], s["conf"], s["n_ratings"])
        if (i + 1) % 5 == 0 or i + 1 == len(subjects):
            log(f"    raw {i+1}/{len(subjects)}")
    return out


def compute_bias(subjects) -> np.ndarray:
    out = np.full((len(subjects), 2, N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        nr = int(s["n_ratings"])
        for ri, rtype in enumerate([1, 2]):
            conf_r = xue_recode(s["conf"], rtype)
            if np.all(np.isnan(conf_r)):
                continue
            out[i, ri, META_UNCERT_IDX] = unc(s["stim"], s["resp"], conf_r, nr - 1)
        if (i + 1) % 5 == 0 or i + 1 == len(subjects):
            log(f"    bias {i+1}/{len(subjects)}")
    return out


def compute_split(subjects) -> np.ndarray:
    out = np.full((len(subjects), 2, N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        nr = int(s["n_ratings"])
        for hi, sl in enumerate([slice(None, None, 2), slice(1, None, 2)]):
            stim, resp, conf = s["stim"][sl], s["resp"][sl], s["conf"][sl]
            if len(stim) < 10:
                continue
            out[i, hi, META_UNCERT_IDX] = unc(stim, resp, conf, nr)
        if (i + 1) % 5 == 0 or i + 1 == len(subjects):
            log(f"    split {i+1}/{len(subjects)}")
    return out


def compute_rouault_diff(subjects) -> np.ndarray:
    """Low/high contrast median split — same as analysis_core.compute_difficulty_rouault."""
    out = np.full((len(subjects), 2, N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        med = np.median(s["contrast"])
        lo = s["contrast"] <= med
        hi = s["contrast"] > med
        nr = int(s["n_ratings"])
        if lo.sum() >= 10:
            out[i, 0, META_UNCERT_IDX] = unc(
                s["stim"][lo], s["resp"][lo], s["conf"][lo], nr
            )
        if hi.sum() >= 10:
            out[i, 1, META_UNCERT_IDX] = unc(
                s["stim"][hi], s["resp"][hi], s["conf"][hi], nr
            )
        if (i + 1) % 10 == 0 or i + 1 == len(subjects):
            log(f"    rouault diff {i+1}/{len(subjects)}")
    return out


def compute_shekhar_diff(subjects) -> np.ndarray:
    out = np.full((len(subjects), 3, N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        nr = int(s["n_ratings"])
        for ci, c in enumerate([1, 2, 3]):
            mask = s["contrast"] == c
            if mask.sum() < 10:
                continue
            out[i, ci, META_UNCERT_IDX] = unc(
                s["stim"][mask], s["resp"][mask], s["conf"][mask], nr
            )
        if (i + 1) % 5 == 0 or i + 1 == len(subjects):
            log(f"    shekhar diff {i+1}/{len(subjects)}")
    return out


def compute_shekhar_bias(subjects) -> np.ndarray:
    """Unaveraged (n, 3, 2, 26) as stored in shekhar_mle.npz."""
    out = np.full((len(subjects), 3, 2, N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        nr = int(s["n_ratings"])
        for ci, c in enumerate([1, 2, 3]):
            mask = s["contrast"] == c
            if mask.sum() < 10:
                continue
            for ri, rtype in enumerate([1, 2]):
                conf_r = xue_recode(s["conf"][mask], rtype)
                if np.all(np.isnan(conf_r)):
                    continue
                out[i, ci, ri, META_UNCERT_IDX] = unc(
                    s["stim"][mask], s["resp"][mask], conf_r, nr - 1
                )
        if (i + 1) % 5 == 0 or i + 1 == len(subjects):
            log(f"    shekhar bias {i+1}/{len(subjects)}")
    return out


def compute_locke_rb(subjects) -> np.ndarray:
    out = np.full((len(subjects), 7, N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        nr = int(s["n_ratings"])
        for ci, cond in enumerate(range(1, 8)):
            mask = s["condition"] == cond
            if mask.sum() < 5:
                continue
            out[i, ci, META_UNCERT_IDX] = unc(
                s["stim"][mask], s["resp"][mask], s["conf"][mask], nr
            )
        if (i + 1) % 5 == 0 or i + 1 == len(subjects):
            log(f"    locke rb {i+1}/{len(subjects)}")
    return out


def main() -> None:
    global N_STARTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-starts",
        type=int,
        default=5,
        help="Deterministic multi-start count (default 5; MATLAB uses 1 random start)",
    )
    args = parser.parse_args()
    N_STARTS = max(1, int(args.n_starts))

    os.makedirs(OUT, exist_ok=True)
    log(f"Refreshing meta-uncertainty with n_starts={N_STARTS}")

    log("Haddara...")
    ha = preprocess_haddara()
    path = os.path.join(OUT, "haddara_mle.npz")
    merge_overlay(path, "raw", compute_raw(ha))
    merge_overlay(path, "bias", compute_bias(ha))
    merge_overlay(path, "split", compute_split(ha))

    log("Maniscalco...")
    ma = preprocess_maniscalco()
    path = os.path.join(OUT, "maniscalco_mle.npz")
    merge_overlay(path, "raw", compute_raw(ma))
    merge_overlay(path, "bias", compute_bias(ma))
    merge_overlay(path, "split", compute_split(ma))

    log("Rouault1...")
    merge_overlay(
        os.path.join(OUT, "rouault1_mle.npz"),
        "diff",
        compute_rouault_diff(preprocess_rouault(1)),
    )

    log("Rouault2...")
    merge_overlay(
        os.path.join(OUT, "rouault2_mle.npz"),
        "diff",
        compute_rouault_diff(preprocess_rouault(2)),
    )

    log("Shekhar...")
    sh = preprocess_shekhar()
    path = os.path.join(OUT, "shekhar_mle.npz")
    merge_overlay(path, "diff", compute_shekhar_diff(sh))
    merge_overlay(path, "bias", compute_shekhar_bias(sh))
    merge_overlay(path, "split", compute_split(sh))

    log("Locke...")
    merge_overlay(
        os.path.join(OUT, "locke_mle.npz"),
        "rb",
        compute_locke_rb(preprocess_locke()),
    )

    log("Done.")


if __name__ == "__main__":
    main()
