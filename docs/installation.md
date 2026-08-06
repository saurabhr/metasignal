# Installation

## Requirements

- Python 3.10 or later
- NumPy, SciPy, matplotlib, pandas, click (installed automatically)

## Install

The base install includes `stdpy` (SDT + all 26 measures), `analysis` (bootstrap CIs, permutation
tests, group summaries), `itmc` (information-theoretic metacognition), `sdtr` (alternative SDT
models), and the CLI.

### Directly from GitHub, no clone needed

```bash
pip install git+https://github.com/saurabhr/metasignal.git
```

### From a downloaded/cloned copy of the source

```bash
git clone https://github.com/saurabhr/metasignal.git
cd metasignal
pip install .
```

### Optional subpackages

Installed as extras on top of either method above:

```bash
pip install "metasignal[sdtbayes] @ git+https://github.com/saurabhr/metasignal.git"  # from GitHub
pip install ".[sdtbayes]"       # from a local clone — adds metasignal.sdtbayes (hierarchical Bayesian models)
pip install ".[matlab]"         # adds the deprecated MATLAB engine wrapper
pip install ".[sdtbayes,matlab]"  # everything
```

New to git and GitHub? See [Installing with GitHub Desktop](#installing-with-github-desktop-no-command-line-experience-needed)
below for a point-and-click walkthrough that avoids the command line except for a few
copy-pasteable setup commands.

## Development install

To install metasignal in editable mode with all development tools (tests, docs):

```bash
pip install -e ".[docs]"
```

Or use [nox](https://nox.thea.codes) to create a complete dev environment:

```bash
pip install nox
nox -s dev
```

## Installing with GitHub Desktop (no command line experience needed)

If you don't use git or the command line regularly, [GitHub Desktop](https://desktop.github.com)
gives you a point-and-click way to get the code. You'll still need a terminal for a handful of
short, copy-pasteable commands to set up Python and install the package itself. Every step below
has been run start-to-finish to confirm it works.

1. **Create a GitHub account**, if you don't have one — sign up free at
   [github.com/join](https://github.com/join).

2. **Install GitHub Desktop** from [desktop.github.com](https://desktop.github.com), open it, and
   sign in with your GitHub account. New to it? See
   [Getting started with GitHub Desktop](https://docs.github.com/en/desktop/overview/getting-started-with-github-desktop).

3. **Fork the repository** — go to the
   [metasignal GitHub page](https://github.com/saurabhr/metasignal) and click **Fork** (top
   right). This creates your own copy under your account. Background:
   [Fork a repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo).

4. **Clone your fork with GitHub Desktop** — on your fork's page, click the green **Code** button,
   then **Open with GitHub Desktop**. Choose where to save it locally and click **Clone**. GitHub
   Desktop treats forks specially: if it asks **"How are you planning to use this fork?"**, choose
   **To contribute to the parent project**. If it doesn't ask, you can set (or change) this later
   via **Repository > Repository Settings > Fork Behavior**. Background:
   [Cloning and forking repositories from GitHub Desktop](https://docs.github.com/en/desktop/adding-and-cloning-repositories/cloning-and-forking-repositories-from-github-desktop).

5. **Install conda**, if you don't already have it — download
   [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your OS and run the installer
   with the default options. Background:
   [Getting started with conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

6. **Open a terminal.** On Mac, open *Terminal*; on Windows, open *Anaconda Prompt*. You should
   see `(base)` at the start of the prompt line — that's conda's default environment. To open a
   terminal already inside your cloned copy, without typing `cd`, use GitHub Desktop's
   **Repository > Open in Terminal** menu instead.

7. **Create a dedicated environment** for metasignal (only needs to be done once):

   ```bash
   conda create -n metasignal_env python=3.13
   ```

   Answer `y` when prompted.

8. **Activate it:**

   ```bash
   conda activate metasignal_env
   ```

   The prompt now starts with `(metasignal_env)` instead of `(base)`. Repeat this step every time
   you open a new terminal to work with metasignal — the rest of these commands assume it's
   active.

9. **Install [uv](https://docs.astral.sh/uv/getting-started/installation/):**

   ```bash
   pip install uv
   ```

10. **Install metasignal**, run from inside your cloned folder:

    ```bash
    uv pip install -e .
    ```

    The `-e` (editable) install means updates you later pull from GitHub take effect without
    reinstalling. uv installs into whichever conda environment is active — `metasignal_env`, from
    step 8.

Then continue to [Verify the installation](#verify-the-installation) below, running it from the
same terminal.

## Verify the installation

```python
import metasignal
import numpy as np

stim = np.array([0, 1] * 50)                    # 100 trials, equal S1/S2
resp = np.array([0, 1] * 40 + [1, 0] * 10)     # 80% accuracy
conf = np.array([2, 2] * 40 + [1, 1] * 10)     # high conf when correct

results = metasignal.stdpy.compute_all_measures(stim, resp, conf, n_ratings=2)
print(results.shape)  # (26,)
```

If you see `(26,)` printed, the installation is working correctly.
