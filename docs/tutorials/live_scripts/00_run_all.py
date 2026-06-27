# %% [markdown]
# # Run All Tutorial Live Scripts
#
# Execute every tutorial in sequence. Each script is self-contained and
# re-seeds its RNG, so they can also be run individually.

# %%
import runpy, pathlib, sys

_here = pathlib.Path(__file__).parent

scripts = [
    '01_getting_started.py',
    '02_computing_measures.py',
    '03_statistical_inference.py',
    '04_difficulty_dependence.py',
    '05_metacognitive_bias.py',
    '06_split_half_reliability.py',
    '07_bayesian_hierarchical.py',
]

for script in scripts:
    path = _here / script
    print(f'\n{"="*60}')
    print(f'Running: {script}')
    print('='*60)
    try:
        runpy.run_path(str(path), run_name='__main__')
        print(f'[OK] {script}')
    except Exception as exc:
        print(f'[FAILED] {script}: {exc}')
