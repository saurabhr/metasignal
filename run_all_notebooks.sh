#!/bin/bash
# run_all_notebooks.sh
# Runs all 10 tutorial notebooks end-to-end using env_metasignal.
# Deletes cached NPZ files first so everything is recomputed from scratch
# (including meta-noise, meta-uncertainty, test-retest, precision).
#
# Usage: bash run_all_notebooks.sh
# Estimated time: 2-4 hours (MLE fitting is slow)

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$REPO/env_metasignal/bin/python"
NBDIR="$REPO/notebooks"
PRECOMP="$NBDIR/precomputed"
LOGDIR="$REPO/notebooks/logs"
mkdir -p "$LOGDIR"

echo "=============================================="
echo "  metasignal full replication run"
echo "  $(date)"
echo "  Python: $PYTHON"
echo "  Notebooks: $NBDIR"
echo "=============================================="

# ── Step 1: Clear all cached NPZ files ──────────────────────────
echo ""
echo "[1/3] Clearing precomputed cache..."
rm -f "$PRECOMP"/*.npz
echo "  Cleared: $PRECOMP"

# ── Step 2: Extract each notebook to a .py script ───────────────
echo ""
echo "[2/3] Extracting notebook code cells..."

extract_nb() {
    local nb="$1"
    local out="$2"
    "$PYTHON" - << PYEOF
import json, sys
with open('$nb') as f:
    nb = json.load(f)
src = 'import matplotlib; matplotlib.use("Agg")\n'
for c in nb['cells']:
    if c['cell_type'] == 'code':
        src += ''.join(c['source']) + '\n'
with open('$out', 'w') as f:
    f.write(src)
print(f"  Extracted: $out")
PYEOF
}

extract_nb "$NBDIR/00_setup.ipynb"                   "$LOGDIR/run_00.py"
extract_nb "$NBDIR/01_preprocessing.ipynb"           "$LOGDIR/run_01.py"
extract_nb "$NBDIR/02_compute_measures.ipynb"        "$LOGDIR/run_02.py"
extract_nb "$NBDIR/03_statistical_tables.ipynb"      "$LOGDIR/run_03.py"
extract_nb "$NBDIR/04_figures.ipynb"                 "$LOGDIR/run_04.py"
extract_nb "$NBDIR/05_difficulty_dependence.ipynb"   "$LOGDIR/run_05.py"
extract_nb "$NBDIR/06_metacognitive_bias.ipynb"      "$LOGDIR/run_06.py"
extract_nb "$NBDIR/07_response_bias.ipynb"           "$LOGDIR/run_07.py"
extract_nb "$NBDIR/08_split_half_precision.ipynb"    "$LOGDIR/run_08.py"
extract_nb "$NBDIR/09_test_retest_correlations.ipynb" "$LOGDIR/run_09.py"

# ── Step 3: Patch NB08 and NB09 to compute fully (no skip) ──────
# NB08: enable precision computation
python3 - << 'PATCH'
path = '/Users/saurabhext/Documents/metasignal/notebooks/logs/run_08.py'
with open(path) as f:
    src = f.read()
# Remove the skip guard so precision actually runs
src = src.replace(
    "ha_prec = None\nif os.path.exists(PREC_CACHE):\n    prec_npz = np.load(PREC_CACHE)\n    drops = prec_npz['drops']\n    ha_prec = drops\n    print(\"Loaded precision from cache.\")\nelse:\n    print(\"Precision cache not found. Skipping live computation (would take ~7 min).\")\n    print(\"To generate: run ana_precision section with SKIP_PRECISION=False in a long session.\")",
    """ha_prec = None
PREC_CACHE_PATH = os.path.join(OUT, 'haddara_precision.npz')
if os.path.exists(PREC_CACHE_PATH):
    prec_npz = np.load(PREC_CACHE_PATH)
    drops = prec_npz['drops']
    ha_prec = drops
    print("Loaded precision from cache.")
else:
    print("Computing precision for Haddara (all measures, may take 10-20 min)...")
    raw_arr = ha_npz['raw']
    sd_raw = np.nanstd(raw_arr, axis=0)
    ha = preprocess_haddara()
    PROPS = [0.02, 0.04, 0.06]
    drops = np.full((N_MEAS, len(PROPS)), np.nan)
    for ai, prop in enumerate(PROPS):
        print(f"  Corruption proportion {prop}...")
        alt = np.array([corrupt_conf(s['stim'], s['resp'], s['conf'], s['n_ratings'], prop)
                        for s in ha])
        drop = raw_arr - alt
        drops[:, ai] = np.nanmean(drop, axis=0) / (sd_raw + 1e-10)
    np.savez(PREC_CACHE_PATH, drops=drops)
    ha_prec = drops
    print("  Precision done and cached.")"""
)
with open(path, 'w') as f:
    f.write(src)
print("Patched NB08 precision")
PATCH

# NB09: enable test-retest computation
python3 - << 'PATCH'
path = '/Users/saurabhext/Documents/metasignal/notebooks/logs/run_09.py'
with open(path) as f:
    src = f.read()
src = src.replace(
    "tt = None\nif os.path.exists(TT_PATH):\n    tt = np.load(TT_PATH)['data']   # (70, 6, 20)\n    print(\"Loaded test-retest data:\", tt.shape)\nelse:\n    print(\"Test-retest cache not found.\")\n    print(\"Requires ~14 min (70 subs x 6 days x MLE). Run precompute_test_retest.py to generate cache.\")\n    print(\"Skipping ICC computation — showing MATLAB reference values instead.\")",
    """tt = None
if os.path.exists(TT_PATH):
    tt = np.load(TT_PATH)['data']
    print("Loaded test-retest data:", tt.shape)
else:
    print("Computing test-retest measures per day (70 subs x 6 days, may take 15-20 min)...")
    tt = np.full((len(ha), len(DAYS), N_MEAS), np.nan)
    for si, s in enumerate(ha):
        if si % 10 == 0:
            print(f"  Subject {si+1}/{len(ha)}...")
        for di, day in enumerate(DAYS):
            mask = s['day'] == day
            if mask.sum() < 10: continue
            tt[si, di] = compute_all_measures(
                s['stim'][mask], s['resp'][mask], s['conf'][mask], s['n_ratings'])
    np.savez(TT_PATH, data=tt)
    print("  Test-retest done and cached:", tt.shape)"""
)
with open(path, 'w') as f:
    f.write(src)
print("Patched NB09 test-retest")
PATCH

# ── Step 4: Run each notebook, logging output ────────────────────
echo ""
echo "[3/3] Running all notebooks..."
echo ""

run_nb() {
    local num="$1"
    local script="$LOGDIR/run_0${num}.py"
    local log="$LOGDIR/nb0${num}.log"
    echo "──────────────────────────────────────────"
    echo "  NB0${num}: $(date '+%H:%M:%S')"
    "$PYTHON" "$script" > "$log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "  ✓ PASS  (log: $log)"
        tail -5 "$log"
    else
        echo "  ✗ FAIL  (log: $log)"
        tail -20 "$log"
    fi
    echo ""
}

run_nb 0
run_nb 1
run_nb 2   # ~30-60 min: computes all 20 measures for all datasets from scratch
run_nb 3
run_nb 4
run_nb 5
run_nb 6
run_nb 7
run_nb 8   # ~20 min: precision analysis
run_nb 9   # ~20 min: test-retest ICC

echo "=============================================="
echo "  All notebooks complete: $(date)"
echo "  Logs: $LOGDIR"
echo "=============================================="
