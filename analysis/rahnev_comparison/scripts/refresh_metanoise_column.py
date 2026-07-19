#!/usr/bin/env python3
"""Patch only meta-noise (index 15) in Python MLE caches after Inf-criteria fix.

Same trial masks as ``refresh_metanoise_caches.py`` / ``analysis_core``.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

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
from metasignal.stdpy.metanoise import compute_meta_noise

OUT = os.path.join(REPO, "notebooks", "precomputed")
N_MEAS = 26
META_NOISE_IDX = 15


def log(msg: str) -> None:
    print(msg, flush=True)


def mn(stim, resp, conf, n_ratings) -> float:
    return float(
        compute_meta_noise(stim, resp, conf, int(n_ratings))["meta_noise"]
    )


def overlay(path: str, key: str, partial: np.ndarray) -> None:
    with np.load(path) as data:
        existing = {k: data[k] for k in data.files}
    out = np.array(existing[key], dtype=float, copy=True)
    out[..., META_NOISE_IDX] = partial[..., META_NOISE_IDX]
    existing[key] = out
    np.savez(path, **existing)
    col = out[..., META_NOISE_IDX]
    log(
        f"  updated {path}[{key}] meta-noise "
        f"mean={np.nanmean(col):.4f} n_finite={int(np.isfinite(col).sum())}"
    )


def fill_raw(subjects) -> np.ndarray:
    out = np.full((len(subjects), N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        out[i, META_NOISE_IDX] = mn(s["stim"], s["resp"], s["conf"], s["n_ratings"])
        if (i + 1) % 5 == 0 or i + 1 == len(subjects):
            log(f"    raw {i+1}/{len(subjects)}")
    return out


def fill_bias(subjects) -> np.ndarray:
    out = np.full((len(subjects), 2, N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        nr = int(s["n_ratings"])
        for ri, rtype in enumerate([1, 2]):
            conf_r = xue_recode(s["conf"], rtype)
            if np.all(np.isnan(conf_r)):
                continue
            out[i, ri, META_NOISE_IDX] = mn(s["stim"], s["resp"], conf_r, nr - 1)
        if (i + 1) % 5 == 0 or i + 1 == len(subjects):
            log(f"    bias {i+1}/{len(subjects)}")
    return out


def fill_split(subjects) -> np.ndarray:
    out = np.full((len(subjects), 2, N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        nr = int(s["n_ratings"])
        for hi, sl in enumerate([slice(None, None, 2), slice(1, None, 2)]):
            if len(s["stim"][sl]) < 10:
                continue
            out[i, hi, META_NOISE_IDX] = mn(
                s["stim"][sl], s["resp"][sl], s["conf"][sl], nr
            )
        if (i + 1) % 5 == 0 or i + 1 == len(subjects):
            log(f"    split {i+1}/{len(subjects)}")
    return out


def fill_rouault(subjects) -> np.ndarray:
    out = np.full((len(subjects), 2, N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        med = np.median(s["contrast"])
        lo = s["contrast"] <= med
        hi = s["contrast"] > med
        nr = int(s["n_ratings"])
        if lo.sum() >= 10:
            out[i, 0, META_NOISE_IDX] = mn(
                s["stim"][lo], s["resp"][lo], s["conf"][lo], nr
            )
        if hi.sum() >= 10:
            out[i, 1, META_NOISE_IDX] = mn(
                s["stim"][hi], s["resp"][hi], s["conf"][hi], nr
            )
        if (i + 1) % 10 == 0 or i + 1 == len(subjects):
            log(f"    rouault {i+1}/{len(subjects)}")
    return out


def fill_shekhar_diff(subjects) -> np.ndarray:
    out = np.full((len(subjects), 3, N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        nr = int(s["n_ratings"])
        for ci, c in enumerate([1, 2, 3]):
            mask = s["contrast"] == c
            if mask.sum() < 10:
                continue
            out[i, ci, META_NOISE_IDX] = mn(
                s["stim"][mask], s["resp"][mask], s["conf"][mask], nr
            )
        if (i + 1) % 5 == 0 or i + 1 == len(subjects):
            log(f"    shekhar diff {i+1}/{len(subjects)}")
    return out


def fill_shekhar_bias(subjects) -> np.ndarray:
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
                out[i, ci, ri, META_NOISE_IDX] = mn(
                    s["stim"][mask], s["resp"][mask], conf_r, nr - 1
                )
        if (i + 1) % 5 == 0 or i + 1 == len(subjects):
            log(f"    shekhar bias {i+1}/{len(subjects)}")
    return out


def fill_locke(subjects) -> np.ndarray:
    out = np.full((len(subjects), 7, N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        nr = int(s["n_ratings"])
        for ci, cond in enumerate(range(1, 8)):
            mask = s["condition"] == cond
            if mask.sum() < 5:
                continue
            out[i, ci, META_NOISE_IDX] = mn(
                s["stim"][mask], s["resp"][mask], s["conf"][mask], nr
            )
        if (i + 1) % 5 == 0 or i + 1 == len(subjects):
            log(f"    locke {i+1}/{len(subjects)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=["all", "rouault", "core"],
        default="all",
        help="rouault=Rouault1/2 only (fastest check of Inf fix); core=HA/MA/Shekhar/Locke",
    )
    args = parser.parse_args()

    if args.only in ("all", "core"):
        log("Haddara...")
        ha = preprocess_haddara()
        path = os.path.join(OUT, "haddara_mle.npz")
        overlay(path, "raw", fill_raw(ha))
        overlay(path, "bias", fill_bias(ha))
        overlay(path, "split", fill_split(ha))

        log("Maniscalco...")
        ma = preprocess_maniscalco()
        path = os.path.join(OUT, "maniscalco_mle.npz")
        overlay(path, "raw", fill_raw(ma))
        overlay(path, "bias", fill_bias(ma))
        overlay(path, "split", fill_split(ma))

        log("Shekhar...")
        sh = preprocess_shekhar()
        path = os.path.join(OUT, "shekhar_mle.npz")
        overlay(path, "diff", fill_shekhar_diff(sh))
        overlay(path, "bias", fill_shekhar_bias(sh))
        overlay(path, "split", fill_split(sh))

        log("Locke...")
        overlay(os.path.join(OUT, "locke_mle.npz"), "rb", fill_locke(preprocess_locke()))

    if args.only in ("all", "rouault"):
        log("Rouault1...")
        overlay(
            os.path.join(OUT, "rouault1_mle.npz"),
            "diff",
            fill_rouault(preprocess_rouault(1)),
        )
        log("Rouault2...")
        overlay(
            os.path.join(OUT, "rouault2_mle.npz"),
            "diff",
            fill_rouault(preprocess_rouault(2)),
        )

    log("Done.")


if __name__ == "__main__":
    main()
