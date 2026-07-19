#!/usr/bin/env python3
"""Rebuild precision / split-half / test–retest caches to paper MATLAB protocols.

Mirrors ``analysis_Haddara.m`` and ``live_scripts/02_analysis_Maniscalco.m``:

* Bin sizes ``[50, 100, 200, 400]``
* Haddara: days 2–7; first 400 trials/day for TR & precision
* Haddara split-half: odd/even within ``2*bin`` blocks; special 400-trial packing
* Precision alterations via sequential ``metasAlteredConf`` (not random sampling)
* Maniscalco: single session, floor(n/bin) windows

Outputs (under ``notebooks/precomputed/``):

* ``haddara_protocol.npz`` — keys ``precision``, ``splitHalf``, ``testRetest``
  (each an object array of length 4, one array per bin size)
* ``maniscalco_protocol.npz`` — keys ``precision``, ``splitHalf``
* Also refreshes ``haddara_testRetest.npz`` / ``haddara_precision.npz`` summary
  views used by older plot code.

Usage::

    python analysis/rahnev_comparison/scripts/rebuild_protocol_caches.py
    python analysis/rahnev_comparison/scripts/rebuild_protocol_caches.py --dataset haddara --only test_retest
    python analysis/rahnev_comparison/scripts/rebuild_protocol_caches.py --max-subjects 5  # smoke
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

import numpy as np
from metasignal.stdpy.compute_all import compute_all_measures

OUT = os.path.join(REPO, "notebooks", "precomputed")
BIN_SIZES = [50, 100, 200, 400]
PROP_ALTERED = [0.02, 0.04, 0.06]
N_MEAS = 26
DAYS = [2, 3, 4, 5, 6, 7]  # recoded to 0..5 internally


def log(msg: str) -> None:
    print(msg, flush=True)


def measures(stim, resp, conf, n_ratings) -> np.ndarray:
    try:
        return compute_all_measures(stim, resp, conf, int(n_ratings))
    except Exception:
        return np.full(N_MEAS, np.nan)


def metas_altered_conf(stim, resp, conf, n_ratings, prop_altered) -> np.ndarray:
    """MATLAB ``metasAlteredConf`` — sequential alterations until count reached."""
    conf = np.asarray(conf, dtype=float).copy()
    stim = np.asarray(stim)
    resp = np.asarray(resp)
    num_to_alter = int(round(len(conf) * prop_altered))
    n_altered = 0
    for trial in range(len(conf)):
        if stim[trial] == resp[trial] and conf[trial] > 1:
            conf[trial] -= 1
            n_altered += 1
        elif stim[trial] != resp[trial] and conf[trial] < n_ratings:
            conf[trial] += 1
            n_altered += 1
        if n_altered == num_to_alter:
            break
    return measures(stim, resp, conf, n_ratings)


def _day_arrays(subject) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return list of (stim, resp, conf) for days 2..7 (length 6)."""
    out = []
    for day in DAYS:
        mask = subject["day"] == day
        out.append(
            (
                subject["stim"][mask],
                subject["resp"][mask],
                subject["conf"][mask],
            )
        )
    return out


def build_haddara_test_retest(subjects, max_subjects: int | None) -> list[np.ndarray]:
    """metas_testRetest{bs}(sub, bin, day, measure) — first 400 trials/day."""
    n_sub = len(subjects) if max_subjects is None else min(max_subjects, len(subjects))
    cells: list[np.ndarray] = []
    for bin_size in BIN_SIZES:
        n_bins = 400 // bin_size
        arr = np.full((n_sub, n_bins, 6, N_MEAS), np.nan)
        cells.append(arr)

    for si in range(n_sub):
        s = subjects[si]
        nr = int(s["n_ratings"])
        days = _day_arrays(s)
        log(f"  Haddara TR subject {si+1}/{n_sub}")
        for di, (stim, resp, conf) in enumerate(days):
            n_use = min(400, len(stim))
            stim, resp, conf = stim[:n_use], resp[:n_use], conf[:n_use]
            for bi, bin_size in enumerate(BIN_SIZES):
                n_bins = 400 // bin_size
                for bn in range(n_bins):
                    start = bn * bin_size
                    end = start + bin_size
                    if end > len(stim):
                        continue
                    cells[bi][si, bn, di] = measures(
                        stim[start:end], resp[start:end], conf[start:end], nr
                    )
    return cells


def build_haddara_precision(
    subjects, tr_cells: list[np.ndarray], max_subjects: int | None
) -> list[np.ndarray]:
    """metas_precision{bs}(sub, bin, day, alter, measure); alter0 = original TR."""
    n_sub = len(subjects) if max_subjects is None else min(max_subjects, len(subjects))
    cells: list[np.ndarray] = []
    for bi, bin_size in enumerate(BIN_SIZES):
        n_bins = 400 // bin_size
        arr = np.full((n_sub, n_bins, 6, 1 + len(PROP_ALTERED), N_MEAS), np.nan)
        arr[:, :, :, 0, :] = tr_cells[bi]
        cells.append(arr)

    for si in range(n_sub):
        s = subjects[si]
        nr = int(s["n_ratings"])
        days = _day_arrays(s)
        log(f"  Haddara precision subject {si+1}/{n_sub}")
        for di, (stim, resp, conf) in enumerate(days):
            n_use = min(400, len(stim))
            stim, resp, conf = stim[:n_use], resp[:n_use], conf[:n_use]
            for bi, bin_size in enumerate(BIN_SIZES):
                n_bins = 400 // bin_size
                for bn in range(n_bins):
                    start = bn * bin_size
                    end = start + bin_size
                    if end > len(stim):
                        continue
                    st, re, co = stim[start:end], resp[start:end], conf[start:end]
                    for ai, prop in enumerate(PROP_ALTERED):
                        cells[bi][si, bn, di, ai + 1] = metas_altered_conf(
                            st, re, co, nr, prop
                        )
    return cells


def build_haddara_split(subjects, max_subjects: int | None) -> list[np.ndarray]:
    """Port of analysis_Haddara.m split-half block (incl. 400-trial packing)."""
    n_sub = len(subjects) if max_subjects is None else min(max_subjects, len(subjects))
    # bin sizes 50,100,200 → shape (sub, n_bins, day, half, meas)
    # bin 400 → (sub, 1, 3, half, meas) via day pairs
    cells: list[np.ndarray] = [
        np.full((n_sub, 400 // (2 * 50), 6, 2, N_MEAS), np.nan),   # 50 → block 100
        np.full((n_sub, 400 // (2 * 100), 6, 2, N_MEAS), np.nan),  # 100
        np.full((n_sub, 400 // (2 * 200), 6, 2, N_MEAS), np.nan),  # 200
        np.full((n_sub, 1, 3, 2, N_MEAS), np.nan),                 # 400 packed
    ]

    for si in range(n_sub):
        s = subjects[si]
        nr = int(s["n_ratings"])
        days = _day_arrays(s)
        log(f"  Haddara split subject {si+1}/{n_sub}")
        for di, (stim_d, resp_d, conf_d) in enumerate(days):
            n_use = min(400, len(stim_d))
            stim_d, resp_d, conf_d = stim_d[:n_use], resp_d[:n_use], conf_d[:n_use]
            for bi, half_bin in enumerate([50, 100, 200]):
                block = 2 * half_bin
                n_bins = 400 // block
                for bn in range(n_bins):
                    # MATLAB builds a leading-false pad then a block-length
                    # odd/even pattern; equivalent to a contiguous block window.
                    start = bn * block
                    if start + block > len(stim_d):
                        continue
                    odds = np.zeros(len(stim_d), dtype=bool)
                    evens = np.zeros(len(stim_d), dtype=bool)
                    pattern_odd = np.array([True, False] * (block // 2))
                    pattern_even = np.array([False, True] * (block // 2))
                    odds[start : start + block] = pattern_odd
                    evens[start : start + block] = pattern_even
                    cells[bi][si, bn, di, 0] = measures(
                        stim_d[odds], resp_d[odds], conf_d[odds], nr
                    )
                    cells[bi][si, bn, di, 1] = measures(
                        stim_d[evens], resp_d[evens], conf_d[evens], nr
                    )

            # 400-trial packed bins on odd days (1-indexed day 1,3,5 → di 0,2,4)
            if di % 2 == 0 and di + 1 < 6:
                stim400 = np.concatenate([days[di][0], days[di + 1][0][:300]])[:800]
                resp400 = np.concatenate([days[di][1], days[di + 1][1][:300]])[:800]
                conf400 = np.concatenate([days[di][2], days[di + 1][2][:300]])[:800]
                if len(stim400) < 800:
                    continue
                pack_day = di // 2
                cells[3][si, 0, pack_day, 0] = measures(
                    stim400[0::2], resp400[0::2], conf400[0::2], nr
                )
                cells[3][si, 0, pack_day, 1] = measures(
                    stim400[1::2], resp400[1::2], conf400[1::2], nr
                )
    return cells


def build_maniscalco_precision(subjects, max_subjects: int | None) -> list[np.ndarray]:
    n_sub = len(subjects) if max_subjects is None else min(max_subjects, len(subjects))
    cells: list[np.ndarray] = []
    # Pre-allocate with max possible bins per subject then trim? Variable bins —
    # use max floor(n/bin) across subjects.
    max_bins = []
    for bin_size in BIN_SIZES:
        mb = max(len(s["stim"]) // bin_size for s in subjects[:n_sub])
        max_bins.append(max(mb, 1))
        cells.append(
            np.full((n_sub, max_bins[-1], 1 + len(PROP_ALTERED), N_MEAS), np.nan)
        )

    for si in range(n_sub):
        s = subjects[si]
        nr = int(s["n_ratings"])
        stim, resp, conf = s["stim"], s["resp"], s["conf"]
        log(f"  Maniscalco precision subject {si+1}/{n_sub}")
        for bi, bin_size in enumerate(BIN_SIZES):
            n_bins = len(stim) // bin_size
            for bn in range(n_bins):
                start = bn * bin_size
                end = start + bin_size
                st, re, co = stim[start:end], resp[start:end], conf[start:end]
                cells[bi][si, bn, 0] = measures(st, re, co, nr)
                for ai, prop in enumerate(PROP_ALTERED):
                    cells[bi][si, bn, ai + 1] = metas_altered_conf(
                        st, re, co, nr, prop
                    )
    return cells


def build_maniscalco_split(subjects, max_subjects: int | None) -> list[np.ndarray]:
    n_sub = len(subjects) if max_subjects is None else min(max_subjects, len(subjects))
    cells: list[np.ndarray] = []
    for half_bin in BIN_SIZES:
        block = 2 * half_bin
        mb = max(len(s["stim"]) // block for s in subjects[:n_sub])
        cells.append(np.full((n_sub, max(mb, 1), 2, N_MEAS), np.nan))

    for si in range(n_sub):
        s = subjects[si]
        nr = int(s["n_ratings"])
        stim, resp, conf = s["stim"], s["resp"], s["conf"]
        log(f"  Maniscalco split subject {si+1}/{n_sub}")
        for bi, half_bin in enumerate(BIN_SIZES):
            block = 2 * half_bin
            n_bins = len(stim) // block
            for bn in range(n_bins):
                start = bn * block
                odds = np.zeros(len(stim), dtype=bool)
                evens = np.zeros(len(stim), dtype=bool)
                odds[start : start + block] = [True, False] * (block // 2)
                evens[start : start + block] = [False, True] * (block // 2)
                cells[bi][si, bn, 0] = measures(
                    stim[odds], resp[odds], conf[odds], nr
                )
                cells[bi][si, bn, 1] = measures(
                    stim[evens], resp[evens], conf[evens], nr
                )
    return cells


def pack_cells(cells: list[np.ndarray]) -> np.ndarray:
    """Pack differently shaped bin-size arrays into a 1-D object array."""
    out = np.empty(len(cells), dtype=object)
    for i, arr in enumerate(cells):
        out[i] = arr
    return out


def unpack_cells(obj: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(obj[i]) for i in range(len(obj))]


def _summary_precision_drops(prec_cells: list[np.ndarray]) -> np.ndarray:
    """Legacy (n_props, n_meas) mean drop for haddara_precision.npz compatibility."""
    # Average absolute drop original−altered across bins/days/subjects/bin-sizes
    drops = []
    for arr in prec_cells:
        # (sub, bin, day, alter, meas) or squeeze
        if arr.ndim == 5:
            base = arr[:, :, :, 0, :]
            for ai in range(1, arr.shape[3]):
                drops.append(np.nanmean(base - arr[:, :, :, ai, :], axis=(0, 1, 2)))
        else:
            base = arr[:, :, 0, :]
            for ai in range(1, arr.shape[2]):
                drops.append(np.nanmean(base - arr[:, :, ai, :], axis=(0, 1)))
    # Group by alteration index across bin sizes
    n_alter = len(PROP_ALTERED)
    out = np.full((n_alter, N_MEAS), np.nan)
    for ai in range(n_alter):
        vals = [drops[i] for i in range(ai, len(drops), n_alter)]
        out[ai] = np.nanmean(vals, axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["all", "haddara", "maniscalco"],
        default="all",
    )
    parser.add_argument(
        "--only",
        choices=["all", "test_retest", "precision", "split"],
        default="all",
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(OUT, exist_ok=True)
    ha = np.load(os.path.join(OUT, "haddara.npz"), allow_pickle=True)["subjects"]
    ma = np.load(os.path.join(OUT, "maniscalco.npz"), allow_pickle=True)["subjects"]

    if args.dataset in ("all", "haddara"):
        log("=== Haddara protocol caches ===")
        protocol_path = os.path.join(OUT, "haddara_protocol.npz")
        tr = None
        if args.only in ("all", "test_retest"):
            tr = build_haddara_test_retest(ha, args.max_subjects)
            tt400 = tr[3][:, 0, :, :]  # (sub, day, meas) — 400-trial bin
            np.savez(os.path.join(OUT, "haddara_testRetest.npz"), data=tt400)
            log(f"  wrote haddara_testRetest.npz {tt400.shape}")
        elif args.only == "precision" and os.path.exists(protocol_path):
            with np.load(protocol_path, allow_pickle=True) as z:
                if "testRetest" in z.files:
                    tr = unpack_cells(z["testRetest"])
                    log("  reused testRetest from haddara_protocol.npz")
        if args.only == "precision" and tr is None:
            log("  no saved testRetest; rebuilding TR first (required for precision)")
            tr = build_haddara_test_retest(ha, args.max_subjects)

        prec = None
        if args.only in ("all", "precision"):
            assert tr is not None
            prec = build_haddara_precision(ha, tr, args.max_subjects)
            drops = _summary_precision_drops(prec)
            np.savez(
                os.path.join(OUT, "haddara_precision.npz"),
                drops=drops,
                protocol="paper_matlab",
            )
            log(f"  wrote haddara_precision.npz drops {drops.shape}")

        split = None
        if args.only in ("all", "split"):
            split = build_haddara_split(ha, args.max_subjects)

        payload = {}
        if tr is not None:
            payload["testRetest"] = pack_cells(tr)
        if prec is not None:
            payload["precision"] = pack_cells(prec)
        if split is not None:
            payload["splitHalf"] = pack_cells(split)
        if payload:
            path = protocol_path
            if os.path.exists(path):
                with np.load(path, allow_pickle=True) as old:
                    merged = {k: old[k] for k in old.files}
                merged.update(payload)
                np.savez(path, **merged)
            else:
                np.savez(path, **payload)
            log(f"  wrote {path} keys={sorted(payload)}")

    if args.dataset in ("all", "maniscalco"):
        log("=== Maniscalco protocol caches ===")
        payload = {}
        if args.only in ("all", "precision"):
            payload["precision"] = pack_cells(
                build_maniscalco_precision(ma, args.max_subjects)
            )
        if args.only in ("all", "split"):
            payload["splitHalf"] = pack_cells(
                build_maniscalco_split(ma, args.max_subjects)
            )
        if payload:
            path = os.path.join(OUT, "maniscalco_protocol.npz")
            if os.path.exists(path):
                with np.load(path, allow_pickle=True) as old:
                    merged = {k: old[k] for k in old.files}
                merged.update(payload)
                np.savez(path, **merged)
            else:
                np.savez(path, **payload)
            log(f"  wrote {path} keys={sorted(payload)}")

    log("Done.")


if __name__ == "__main__":
    main()
