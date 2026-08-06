"""Tests for metasignal.data_library."""

from __future__ import annotations

import pandas as pd
import pytest

from metasignal.data_library import (
    DATASETS,
    DatasetInfo,
    DatasetNotAvailableError,
    dataset_path,
    describe_dataset,
    list_datasets,
    load_dataset,
)


class TestListDatasets:
    def test_matches_registry_keys(self):
        assert list_datasets() == sorted(DATASETS)

    def test_nonempty(self):
        assert len(list_datasets()) >= 7


class TestDescribeDataset:
    def test_includes_description_and_citation(self):
        text = describe_dataset("shekhar_2021")
        assert DATASETS["shekhar_2021"].description in text
        assert DATASETS["shekhar_2021"].citation in text

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            describe_dataset("not_a_real_dataset")


class TestDatasetPath:
    @pytest.mark.parametrize("name", list(DATASETS))
    def test_path_exists_for_every_dataset(self, name):
        assert dataset_path(name).is_file()

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            dataset_path("not_a_real_dataset")

    def test_missing_data_file_raises_dataset_not_available(self, monkeypatch):
        monkeypatch.setitem(
            DATASETS, "_fixture_missing",
            DatasetInfo(file="does_not_exist.csv", description="", citation=""),
        )
        with pytest.raises(DatasetNotAvailableError, match="development checkout"):
            dataset_path("_fixture_missing")


class TestLoadDataset:
    @pytest.mark.parametrize("name", list(DATASETS))
    def test_loads_nonempty_dataframe(self, name):
        df = load_dataset(name)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert len(df.columns) > 1

    def test_shekhar_2021_columns(self):
        df = load_dataset("shekhar_2021")
        for col in ("Subj_idx", "Stimulus", "Response", "Confidence"):
            assert col in df.columns

    def test_metadpy_rm_columns(self):
        df = load_dataset("metadpy_rm")
        for col in ("Stimuli", "Responses", "Accuracy", "Confidence", "Subject"):
            assert col in df.columns

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_dataset("not_a_real_dataset")
