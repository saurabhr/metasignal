# CLI Reference

The `metasignal` command-line tool gives access to the pure-Python backend
without writing any Python code. The optional `bayes` sub-group requires
`pip install metasignal[sdtbayes]`.

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
meta_d                   1.4769
AUC2                     0.7412
gamma                    0.5795
phi                      0.3493
deltaConf                0.8941
M_ratio                  0.9934
AUC2_ratio               1.0467
gamma_ratio              1.0558
phi_ratio                1.1168
deltaConf_ratio          1.1586
M_diff                  -0.0098
AUC2_diff                0.0331
gamma_diff               0.0306
phi_diff                 0.0365
deltaConf_diff           0.1224
metaNoise                0.0000
metaUncertainty          0.0100
dprime                   1.6732
c                       -0.0719
mean_conf                2.6000
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

## `bayes` — Bayesian hierarchical meta-d'

Requires `pip install metasignal[sdtbayes]` and a one-time Stan runtime setup
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
