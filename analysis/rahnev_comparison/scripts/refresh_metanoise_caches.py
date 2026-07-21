#!/usr/bin/env python3
"""Recompute Python precomputed caches after meta-noise fix.

Updates notebooks/precomputed/*_mle.npz (and related) by re-running
``compute_all_measures`` on the same trial subsets used by the comparison.
"""

from __future__ import annotations

import os
import sys
import warnings

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
    compute_difficulty_shekhar,
    compute_difficulty_rouault,
    compute_bias,
    compute_bias_shekhar,
    compute_response_bias_locke,
    compute_splithalf,
)
from metasignal.stdpy.compute_all import compute_all_measures

OUT = os.path.join(REPO, "notebooks", "precomputed")
N_MEAS = 26
META_NOISE_IDX = 15


def merge_savez(path: str, **new_arrays) -> None:
    existing = {}
    if os.path.exists(path):
        with np.load(path) as data:
            existing = {k: data[k] for k in data.files}
    existing.update(new_arrays)
    np.savez(path, **existing)
    print(f"  wrote {path} keys={sorted(existing)}")


def compute_raw(subjects):
    out = np.full((len(subjects), N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        out[i] = compute_all_measures(s["stim"], s["resp"], s["conf"], int(s["n_ratings"]))
        if (i + 1) % 10 == 0:
            print(f"    raw {i+1}/{len(subjects)}")
    return out


def compute_split(subjects):
    out = np.full((len(subjects), 2, N_MEAS), np.nan)
    for i, s in enumerate(subjects):
        out[i, 0], out[i, 1] = compute_splithalf(
            s["stim"], s["resp"], s["conf"], int(s["n_ratings"])
        )
        if (i + 1) % 10 == 0:
            print(f"    split {i+1}/{len(subjects)}")
    return out


def compute_bias_shekhar_full(subjects):
    """Unaveraged (n, 3, 2, 26) as expected by shekhar_mle bias key."""
    out = []
    for s in subjects:
        nr = s["n_ratings"]
        per_contrast = np.full((3, 2, N_MEAS), np.nan)
        for ci, c in enumerate([1, 2, 3]):
            mask = s["contrast"] == c
            if mask.sum() < 10:
                continue
            for ri, rtype in enumerate([1, 2]):
                conf = s["conf"][mask].astype(float).copy()
                valid = conf[~np.isnan(conf)]
                if len(np.unique(valid)) < 3:
                    continue
                if rtype == 1:
                    conf = conf - 1
                    cmin = np.nanmin(conf)
                    conf[conf == cmin] = cmin + 1
                else:
                    cmax = np.nanmax(conf)
                    conf[conf == cmax] = cmax - 1
                per_contrast[ci, ri] = compute_all_measures(
                    s["stim"][mask], s["resp"][mask], conf, nr - 1
                )
        out.append(per_contrast)
    return np.asarray(out)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)

    print("Haddara...")
    ha = preprocess_haddara()
    merge_savez(
        os.path.join(OUT, "haddara_mle.npz"),
        raw=compute_raw(ha),
        bias=compute_bias(ha),
        split=compute_split(ha),
    )

    print("Maniscalco...")
    ma = preprocess_maniscalco()
    merge_savez(
        os.path.join(OUT, "maniscalco_mle.npz"),
        raw=compute_raw(ma),
        bias=compute_bias(ma),
        split=compute_split(ma),
    )

    print("Rouault1...")
    r1 = preprocess_rouault(1)
    merge_savez(os.path.join(OUT, "rouault1_mle.npz"), diff=compute_difficulty_rouault(r1))

    print("Rouault2...")
    r2 = preprocess_rouault(2)
    merge_savez(os.path.join(OUT, "rouault2_mle.npz"), diff=compute_difficulty_rouault(r2))

    print("Shekhar...")
    sh = preprocess_shekhar()
    # Comparison scripts expect all 3 contrasts (n, 3, 26), not just hard/easy.
    shekhar_diff = []
    for s in sh:
        row = np.full((3, N_MEAS), np.nan)
        for ci, c in enumerate([1, 2, 3]):
            mask = s["contrast"] == c
            if mask.sum() < 10:
                continue
            row[ci] = compute_all_measures(
                s["stim"][mask], s["resp"][mask], s["conf"][mask], int(s["n_ratings"])
            )
        shekhar_diff.append(row)
    merge_savez(
        os.path.join(OUT, "shekhar_mle.npz"),
        diff=np.asarray(shekhar_diff),
        bias=compute_bias_shekhar_full(sh),
        split=compute_split(sh),
    )

    print("Locke...")
    lk = preprocess_locke()
    merge_savez(os.path.join(OUT, "locke_mle.npz"), rb=compute_response_bias_locke(lk))

    # Summarize meta-noise means
    for name in ("haddara", "maniscalco"):
        z = np.load(os.path.join(OUT, f"{name}_mle.npz"))
        mn = z["raw"][:, META_NOISE_IDX]
        print(f"{name} raw meta-noise mean={np.nanmean(mn):.4f} median={np.nanmedian(mn):.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
