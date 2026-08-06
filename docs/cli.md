# CLI Reference

The `metasignal` command-line tool gives access to the pure-Python backend
without writing any Python code. The optional `bayes` sub-group requires the
`sdtbayes` extra:

```bash
pip install "metasignal[sdtbayes] @ git+https://github.com/saurabhr/metasignal.git"  # from GitHub
pip install ".[sdtbayes]"       # from a local clone
```

## Commands

::: mkdocs-click
    :module: metasignal.cli
    :command: cli
    :prog_name: metasignal
    :depth: 2

---

## `compute` — all 26 measures

Compute all 26 SDT and metacognitive measures from a single participant's
trial-level data.

```bash
metasignal compute \
  --stim  "0,1,0,1,1,0,1,0,0,1" \
  --resp  "0,1,1,1,1,0,0,0,0,1" \
  --conf  "2,3,1,4,4,3,2,1,3,4" \
  --n-ratings 4
```

Output:

```
Measure                   Value
--------------------------------
meta_d                   1.6571
AUC2                     0.8750
gamma                    0.8571
phi                      0.5455
deltaConf                1.5000
M_ratio                  1.2284
AUC2_ratio               1.2755
gamma_ratio              1.7820
phi_ratio                1.8885
deltaConf_ratio          2.0248
M_diff                   0.3082
AUC2_diff                0.1890
gamma_diff               0.3761
phi_diff                 0.2566
deltaConf_diff           0.7592
metaNoise                0.1115
metaUncertainty          0.0100
dprime                   1.6832
c                       -0.0000
mean_conf                2.7000
logL                   -13.0667
AIC                     40.1334
BIC                     43.5278
AICc                    68.1334
k                        7.0000
n                       12.0000
```

### Input format

| Option | Format | Example |
|---|---|---|
| `--stim` | Comma-separated 0/1 integers | `"0,1,0,1"` |
| `--resp` | Comma-separated 0/1 integers | `"0,1,1,0"` |
| `--conf` | Comma-separated integers (1 to n-ratings) | `"1,3,2,4"` |
| `--n-ratings` | Integer — number of confidence categories | `4` |

`--n-ratings` is always required. Provide either all three of
`--stim`/`--resp`/`--conf`, or `--csv` — not both. Arrays (or CSV columns)
must be the same length.

### Reading trial data from a CSV

As an alternative to typing values inline, `--csv` reads one trial per row:

```bash
metasignal compute --csv trials.csv --n-ratings 4
```

Column names default to `stim`, `resp`, `conf` and can be overridden with
`--stim-col`, `--resp-col`, `--conf-col` for CSVs with different headers.

---

## `itmc` — information-theoretic metacognition

Computes meta-I, meta-Ir1, meta-Ir1_acc, meta-Ir2, and RMI (Dayan, 2023) for
each participant in a **long-format CSV** — one trial per row.

```bash
metasignal itmc --csv trials.csv
```

Column names default to `participant`, `stim`, `resp`, `conf` and can be
overridden with `--participant-col`, `--stim-col`, `--resp-col`,
`--conf-col`. `--backend` selects `simple` (default, fast) or `statconfr`
(exact port of the statConfR R package); `--bias-correction` subtracts the
estimated positive sampling bias.

Output — one row per participant:

```
participant   meta_I  meta_Ir1  meta_Ir1_acc  meta_Ir2     RMI
         s1   0.1421    0.3384        0.3384    0.2116  0.2841
         s2   0.1198    0.2872        0.2872    0.1874  0.2460
```

See [ITMC](api.md#itmc-information-theoretic-metacognition-experimental) for
the underlying measures and the Python API.

---

## `sdtr` — alternative SDT models (Macho, 2020)

Fits the base Gaussian SDT model — mean/SD per non-reference signal and a
shared set of decision thresholds — for each participant in a **long-format
CSV** — one trial per row, with a signal-class column (`0` = reference/noise
signal) and a response-category column (`1..n_categories`).

```bash
metasignal sdtr --csv trials.csv --restriction equalvar
```

Column names default to `participant`, `signal`, `response` and can be
overridden with `--participant-col`, `--signal-col`, `--response-col`.
`--restriction` selects `no` (default, free SD per non-reference signal) or
`equalvar` (all SDs fixed to 1 — use for rating data with more than two
response categories per signal). `--n-starts` runs multiple optimizer
starts, keeping the lowest-NLL result.

Output — one row per participant:

```
participant   mean_1  sd_1    d_a_1    d_e_1    A_z_1  threshold_1        logL         aic         bic  success
         p1 0.617699   1.0 0.617699 0.617699 0.668864     0.524287 -2870.748505 5745.497010 5758.298779     True
```

See [SDT-R](sdtr.md) for the model description and the Python API. This is
Phase 1 of a larger planned model family — see [Roadmap](roadmap.md).

---

## `bayes` — Bayesian hierarchical meta-d'

Requires `pip install "metasignal[sdtbayes] @ git+https://github.com/saurabhr/metasignal.git"` and a one-time Stan runtime setup
(see [SDT Bayes](sdtbayes.md) for installation instructions).

Both sub-commands take a **long-format CSV** — one trial per row — with
columns for participant ID, stimulus, response, and confidence rating.

### CSV format

```
participant,stim,resp,conf
P001,0,0,2
P001,1,1,4
P001,0,1,1
...
P002,0,0,3
P002,1,1,4
...
```

Column names default to `participant`, `stim`, `resp`, `conf` and can be
overridden with `--participant-col`, `--stim-col`, `--resp-col`, `--conf-col`.

---

### `bayes two-stage` — group-level M-ratio

Fits a two-stage Bayesian model: MLE per participant (Stage 1) then a
hierarchical Bayesian model over log M-ratio (Stage 2).

```bash
metasignal bayes two-stage \
  --csv participants.csv \
  --n-ratings 4
```

Key output parameters:

| Parameter | Interpretation |
|---|---|
| `b_Intercept` | Group mean log M-ratio; `exp(b_Intercept)` gives the M-ratio |
| `sigma` | Between-subject SD on the log scale |

Use `--var-names "b_Intercept,sigma"` to restrict the summary to these two
parameters.

---

### `bayes compare` — two-group M-ratio comparison

Adds a `group` column to the CSV (exactly two unique values). Groups are
sorted alphabetically — the first becomes group A, the second group B.

```
participant,group,stim,resp,conf
P001,control,0,0,2
P001,control,1,1,4
...
P020,patient,0,1,1
...
```

```bash
metasignal bayes compare \
  --csv study.csv \
  --n-ratings 4 \
  --group-col group
```

Key output parameter:

| Parameter | Interpretation |
|---|---|
| `b_group1` | Posterior difference in log M-ratio (group B − group A). `exp(b_group1) > 1` means group B has higher metacognitive efficiency. |

To compute the posterior probability that group B has lower M-ratio than
group A, extract the posterior in Python:

```python
import arviz as az
post = az.extract(fit.idata)["b_group1"].values
print(f"P(B < A): {(post < 0).mean():.3f}")
```
