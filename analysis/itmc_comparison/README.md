# itmc validation: statConfR (R) cross-check + direct Dayan (2023) reproduction

Validates `metasignal.itmc` two ways:
1. **Cross-validation against the real R `statConfR` package** (Rausch et al.,
   2025, JOSS) it is ported from — mirroring what
   [`analysis/rahnev_comparison/`](../rahnev_comparison/) does for `stdpy` vs.
   the MATLAB pipeline.
2. **Direct reproduction of Dayan (2023)'s own hand-worked numerical example**
   (`Dayan_2023.pdf`, Table 1, p.397) — the strongest check, since it validates
   against the theory paper itself rather than a third-party implementation.

**[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)** — full write-up: verdicts, per-measure
agreement tables, the low-d' ratio-instability finding, and the Dayan Table 1 check.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/generate_shared_data.py` | Simulate 12 synthetic participants spanning d', bias, rating-scale, and sample-size combinations |
| `scripts/run_python_itmc.py` | Compute all 5 itmc measures (Python, `backend='statconfr'`) |
| `scripts/run_r_statconfr.R` | Compute the same 5 measures via the real R `statConfR::estimateMetaI()` |
| `scripts/compare.py` | Merge both sides, compute agreement, write `ANALYSIS_REPORT.md` |
| `scripts/check_dayan_table1.py` | Reproduce Dayan (2023)'s own hand-worked meta-I example (Table 1) directly |

## Reproduce

Requires R with `statConfR` installed (`install.packages("statConfR")`; on CRAN).

```bash
python analysis/itmc_comparison/scripts/generate_shared_data.py
python analysis/itmc_comparison/scripts/run_python_itmc.py
Rscript analysis/itmc_comparison/scripts/run_r_statconfr.R
python analysis/itmc_comparison/scripts/compare.py
python analysis/itmc_comparison/scripts/check_dayan_table1.py
```

## One-line verdict

Deterministic core math matches R to machine precision (r=1.0000, MAE=0.0000,
all 5 measures), **and** independently reproduces Dayan (2023)'s own
published meta-I value exactly (0.0943 vs. his reported 0.094). Bias-corrected
(Monte Carlo) R↔Python values agree well; the one notable spread (`meta_Ir1`
at very low d') is a known ratio-measure instability present in both
implementations, not an R↔Python discrepancy — see the report for detail.
`meta_Ir1_acc` and `RMI` are confirmed (from Dayan's own paper text) to be
Rausch et al.'s additions, not measures Dayan himself proposed.
