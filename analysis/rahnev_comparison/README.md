# Rahnev (2025) MATLAB · Python · Paper comparison

Comparative analysis of Rahnev *Nat Commun* 2025 figure replication using **this** repo’s MATLAB Results and Python (`metasignal.stdpy`) caches.

## Paper

- `Rahnev_2025_NatureCommunications.pdf` (this folder — not tracked in git, add your own copy)

## Start here

**[ANALYSIS_REPORT.md](ANALYSIS_REPORT.md)** — full write-up of methods, agreement, divergences, and root causes.  
**[REPLICABILITY.md](REPLICABILITY.md)** — step-by-step protocol (fixes → caches → gates → plots).

## Scripts

| Script | Purpose |
|---|---|
| `scripts/compare_paper_matlab_python.py` | Numeric three-way gate (arrays, Supp *t*, paper scalars) → JSON |
| `scripts/generate_rahnev_comparison_plots.py` | Fig. 1–7 style Paper/MATLAB/Python overlays |
| `scripts/make_validation_figures.py` | Publication validation panels |
| `scripts/plot_full_matlab_python_comparison.py` | Subject-level MATLAB↔Python scatters |
| `scripts/refresh_metanoise_caches.py` | Rebuild MLE caches after meta-noise fix |
| `scripts/refresh_metauncertainty_caches.py` | Patch meta-uncertainty (col 16) after multi-start fix |

```bash
cd metasignal  # repo root
python analysis/rahnev_comparison/scripts/compare_paper_matlab_python.py --repo .
python analysis/rahnev_comparison/scripts/generate_rahnev_comparison_plots.py --repo . \
  --out analysis/rahnev_comparison/figures/rahnev_style
python analysis/rahnev_comparison/scripts/make_validation_figures.py --repo . \
  --out analysis/rahnev_comparison/figures/validation
python analysis/rahnev_comparison/scripts/plot_full_matlab_python_comparison.py --repo . \
  --out analysis/rahnev_comparison/figures/full_comparison
```

## Outputs

- `comparison_report.json` / `comparison_console.txt` — numeric gate (**OVERALL: PASS**)
- `figures/rahnev_style/` — Rahnev-style comparison plots + CSV + PDF
- `figures/validation/` — main validation figure set
- `figures/full_comparison/` — per-analysis identity scatters

## One-line verdict

Task performance, metacognitive bias, response-bias profiles, and most measures match paper/MATLAB closely. Fixed: **meta-noise** search and **SDT-expect proportions** (Locke/Rouault Ratio–Diff). Remaining soft spots: Rouault meta-noise on small difficulty splits, and reliability protocol/packaging.
