"""Bundled example datasets for metasignal.

Real trial-level confidence/perceptual-decision datasets from published
studies, packaged for quick loading during development, analysis, and
tutorials — no separate download step. Available in a development checkout
(``git clone`` + editable install); see :func:`dataset_path` for the error
raised if the data files aren't present (e.g. a wheel install that excludes
them).

>>> from metasignal.data_library import list_datasets, load_dataset
>>> list_datasets()
['haddara_2020', 'locke_2020', ...]
>>> df = load_dataset("shekhar_2021")
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for one bundled dataset."""

    file: str
    description: str
    citation: str
    readme_file: str | None = None


DATASETS: Dict[str, DatasetInfo] = {
    "haddara_2020": DatasetInfo(
        file="haddara_2020.csv",
        description=(
            "X-vs-O grid perceptual task, 75 MTurk subjects over 7 days, "
            "4-point confidence, trial-by-trial feedback manipulation "
            "(feedback vs. no-feedback groups), 3500 trials/subject."
        ),
        citation=(
            "Haddara, N., & Rahnev, D. (2020). The impact of feedback on "
            "perceptual decision making and metacognition. PsyArXiv. "
            "https://doi.org/10.31234/OSF.IO/P8ZYW"
        ),
        readme_file="haddara_2020_readme.txt",
    ),
    "locke_2020": DatasetInfo(
        file="locke_2020.csv",
        description=(
            "Gabor tilt discrimination, 10 subjects, 7 prior/reward "
            "conditions, binary confidence, ~4900 trials in the main "
            "experiment."
        ),
        citation=(
            "Locke, S.M.*, Gaffin-Cahn, E.*, Hosseinizaveh, N., Mamassian, "
            "P., & Landy, M.S. (2020). Priors and payoffs in confidence "
            "judgments. Attention, Perception, & Psychophysics. "
            "doi:10.3758/s13414-020-02018-x"
        ),
        readme_file="locke_2020_readme.txt",
    ),
    "maniscalco_2017": DatasetInfo(
        file="maniscalco_2017.csv",
        description=(
            "Left/right noise-patch grating detection, 30 subjects, "
            "4-point confidence, 10 blocks x 100 trials."
        ),
        citation=(
            "Maniscalco, B., McCurdy, L. Y., Odegaard, B., & Lau, H. "
            "(2017). Limited Cognitive Resources Explain a Trade-Off "
            "between Perceptual and Metacognitive Vigilance. J. Neurosci., "
            "37(5), 1213-1224. https://doi.org/10.1523/JNEUROSCI.2271-13.2016"
        ),
        readme_file="maniscalco_2017_readme.txt",
    ),
    "rouault_2018_expt1": DatasetInfo(
        file="rouault_2018_expt1.csv",
        description=(
            "Web-based dot-counting task, N=498 subjects, 11-point "
            "probabilistic confidence scale, 210 trials, transdiagnostic "
            "psychopathology study (Experiment 1)."
        ),
        citation=(
            "Rouault, M.*, Seow, T.*, Gillan, C. M., & Fleming, S. M. "
            "(2018). Psychiatric symptom dimensions are associated with "
            "dissociable shifts in metacognition but not task performance. "
            "Biol Psychiatry, 84(6), 443-451. "
            "doi:10.1016/j.biopsych.2017.12.017"
        ),
        readme_file="rouault_2018_expt1_readme.txt",
    ),
    "rouault_2018_expt2": DatasetInfo(
        file="rouault_2018_expt2.csv",
        description=(
            "Same study/team as rouault_2018_expt1, N=497, "
            "staircase-adjusted difficulty, 6-point verbal confidence "
            "scale, 210 trials (Experiment 2)."
        ),
        citation=(
            "Rouault, M.*, Seow, T.*, Gillan, C. M., & Fleming, S. M. "
            "(2018). Psychiatric symptom dimensions are associated with "
            "dissociable shifts in metacognition but not task performance. "
            "Biol Psychiatry, 84(6), 443-451. "
            "doi:10.1016/j.biopsych.2017.12.017"
        ),
        readme_file="rouault_2018_expt2_readme.txt",
    ),
    "shekhar_2021": DatasetInfo(
        file="shekhar_2021.csv",
        description=(
            "Gabor orientation discrimination at 3 contrast levels, young "
            "adults, continuous 50-100% confidence scale, 3 days x "
            "~800-1000 trials/day."
        ),
        citation=(
            "Shekhar, M. & Rahnev, D. The nature of metacognitive "
            "imperfection in perceptual decision making. PsyArXiv "
            "preprint. DOI: 10.31234/osf.io/g9qzh"
        ),
        readme_file="shekhar_2021_readme.txt",
    ),
    "metadpy_rm": DatasetInfo(
        file="metadpy_rm.txt",
        description=(
            "Repeated-measures example dataset bundled with metadpy: 20 "
            "subjects x 2 conditions, ~100 trials each. Used here to "
            "validate group-level MLE/Bayesian fitting against metadpy."
        ),
        citation=(
            "metadpy (embodied-computation-group/metadpy), "
            "metadpy/datasets/rm.txt."
        ),
    ),
}


class DatasetNotAvailableError(FileNotFoundError):
    """Raised when a bundled dataset's data file isn't present on disk."""


def list_datasets() -> list[str]:
    """Return the names of all bundled datasets."""
    return sorted(DATASETS)


def describe_dataset(name: str) -> str:
    """Return a human-readable description and citation for a dataset."""
    info = _get_info(name)
    return f"{info.description}\n\nCitation: {info.citation}"


def dataset_path(name: str) -> "resources.abc.Traversable":
    """Return the path to a bundled dataset's data file.

    Raises
    ------
    DatasetNotAvailableError:
        If the data file isn't present — e.g. a package install that
        excludes ``data_library/data`` (it's a development-checkout
        convenience, not required at runtime). Clone
        https://github.com/saurabhr/metasignal and install from the
        checkout (``pip install -e .``) to get the bundled datasets.
    """
    info = _get_info(name)
    path = resources.files("metasignal.data_library") / "data" / info.file
    if not path.is_file():
        raise DatasetNotAvailableError(
            f"Dataset '{name}' data file not found at {path}. Bundled "
            "datasets are only available in a development checkout of "
            "metasignal — clone https://github.com/saurabhr/metasignal "
            "and install from the checkout (`pip install -e .`)."
        )
    return path


def load_dataset(name: str) -> pd.DataFrame:
    """Load a bundled dataset as a DataFrame.

    Parameters
    ----------
    name:
        One of :func:`list_datasets`.
    """
    path = dataset_path(name)
    with resources.as_file(path) as p:
        return pd.read_csv(p)


def _get_info(name: str) -> DatasetInfo:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset {name!r}. Available: {list_datasets()}"
        ) from None


__all__ = [
    "DATASETS",
    "DatasetInfo",
    "DatasetNotAvailableError",
    "list_datasets",
    "describe_dataset",
    "dataset_path",
    "load_dataset",
]
