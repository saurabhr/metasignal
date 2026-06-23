# CLI Reference

The `metasignal` command-line tool gives access to the pure-Python backend without writing any Python code.

## Commands

::: mkdocs-click
    :module: metasignal.cli
    :command: cli
    :prog_name: metasignal
    :depth: 1

## Example

Compute all twenty measures from a small set of trials:

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

## Input format

| Option | Format | Example |
|---|---|---|
| `--stim` | Comma-separated 0/1 integers | `"0,1,0,1"` |
| `--resp` | Comma-separated 0/1 integers | `"0,1,1,0"` |
| `--conf` | Comma-separated integers (1 to n-ratings) | `"1,3,2,4"` |
| `--n-ratings` | Integer — number of confidence categories | `4` |

All four options are required. Arrays must be the same length.
