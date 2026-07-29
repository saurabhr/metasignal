# itmc vs. statConfR (R) comparison

Cross-validates `metasignal.itmc` against the R `statConfR` package
(Rausch et al., 2025, JOSS) it is ported from — mirroring what
[`analysis/rahnev_comparison/`](../rahnev_comparison/) does for `stdpy` vs.
the MATLAB pipeline.

**[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)** — full write-up: verdict, per-measure
agreement tables, and the low-d' ratio-instability finding.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/generate_shared_data.py` | Simulate 12 synthetic participants spanning d', bias, rating-scale, and sample-size combinations |
| `scripts/run_python_itmc.py` | Compute all 5 itmc measures (Python, `backend='statconfr'`) |
| `scripts/run_r_statconfr.R` | Compute the same 5 measures via the real R `statConfR::estimateMetaI()` |
| `scripts/compare.py` | Merge both sides, compute agreement, write `ANALYSIS_REPORT.md` |

## Reproduce

Requires R with `statConfR` installed (`install.packages("statConfR")`; on CRAN).

```bash
python analysis/itmc_comparison/scripts/generate_shared_data.py
python analysis/itmc_comparison/scripts/run_python_itmc.py
Rscript analysis/itmc_comparison/scripts/run_r_statconfr.R
python analysis/itmc_comparison/scripts/compare.py
```

## One-line verdict

Deterministic core math matches R to machine precision (r=1.0000, MAE=0.0000,
all 5 measures). Bias-corrected (Monte Carlo) values agree well; the one
notable spread (`meta_Ir1` at very low d') is a known ratio-measure
instability present in both implementations, not an R↔Python discrepancy —
see the report for detail.
