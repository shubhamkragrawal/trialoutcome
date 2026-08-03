"""Fast, dependency-free unit tests for core/dataset_builder_base.py's
temporal_split / random_split / controlled_leakage_ablation -- the leakage
guard itself, and the negative control / isolation tool built to demonstrate
it. No DB, no MLflow required (see tests/test_calibration.py's module
docstring for why CI needs these fast, no-fixture tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.dataset_builder_base import SplitDates, TemporalDatasetBuilder


class _DummyBuilder(TemporalDatasetBuilder):
    """Minimal concrete subclass -- fetch_raw/build_features are pharma-
    specific and irrelevant to testing the split logic itself."""

    def fetch_raw(self) -> pd.DataFrame:
        raise NotImplementedError

    def build_features(self, raw: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


@pytest.fixture
def builder() -> _DummyBuilder:
    return _DummyBuilder()


def test_temporal_split_respects_dates(builder):
    split_dates = SplitDates(
        train_end=pd.Timestamp("2020-01-01"),
        calib_end=pd.Timestamp("2022-01-01"),
    )
    df = pd.DataFrame(
        {
            "nct_id": [f"NCT{i}" for i in range(6)],
            "start_date": pd.to_datetime(
                [
                    "2015-06-01",  # train
                    "2019-12-31",  # train
                    "2020-01-01",  # calib (boundary, inclusive on the low end)
                    "2021-06-15",  # calib
                    "2022-01-01",  # test (boundary, inclusive on the low end)
                    "2024-03-01",  # test
                ]
            ),
        }
    )

    out = builder.temporal_split(df, date_col="start_date", split_dates=split_dates)

    assert out["split"].tolist() == ["train", "train", "calib", "calib", "test", "test"]


def test_temporal_split_raises_on_null_dates(builder):
    split_dates = SplitDates(
        train_end=pd.Timestamp("2020-01-01"),
        calib_end=pd.Timestamp("2022-01-01"),
    )
    df = pd.DataFrame(
        {
            "nct_id": ["NCT0", "NCT1", "NCT2"],
            "start_date": pd.to_datetime(["2019-01-01", None, "2023-01-01"]),
        }
    )

    with pytest.raises(ValueError, match="null"):
        builder.temporal_split(df, date_col="start_date", split_dates=split_dates)


def test_random_split_is_deterministic(builder):
    df = pd.DataFrame({"nct_id": [f"NCT{i}" for i in range(200)], "value": range(200)})

    out1 = builder.random_split(df, random_state=42)
    out2 = builder.random_split(df, random_state=42)

    assert out1["split"].tolist() == out2["split"].tolist()
    # sanity: all three buckets actually populated at roughly the requested proportions
    counts = out1["split"].value_counts()
    assert counts["train"] == 120
    assert counts["calib"] == 40
    assert counts["test"] == 40


def test_random_split_different_seed_gives_different_assignment(builder):
    df = pd.DataFrame({"nct_id": [f"NCT{i}" for i in range(200)], "value": range(200)})

    out1 = builder.random_split(df, random_state=42)
    out2 = builder.random_split(df, random_state=1)

    assert out1["split"].tolist() != out2["split"].tolist()


def test_controlled_leakage_ablation_uses_fixed_window(builder):
    test_window_start = pd.Timestamp("2020-01-01")
    test_window_end = pd.Timestamp("2022-01-01")

    rng = np.random.default_rng(0)
    n = 1000
    # dates spread from 1990 to 2026 so the leaky pool has real post-cutoff rows to draw from
    offsets_days = rng.integers(0, 365 * 36, size=n)
    dates = pd.Timestamp("1990-01-01") + pd.to_timedelta(offsets_days, unit="D")
    df = pd.DataFrame({"nct_id": [f"NCT{i}" for i in range(n)], "start_date": dates})

    fixed_test, honest_train, leaky_train = builder.controlled_leakage_ablation(
        df,
        date_col="start_date",
        test_window_start=test_window_start,
        test_window_end=test_window_end,
        random_state=42,
    )

    # fixed_test is exactly the window, nothing more
    assert (fixed_test["start_date"] >= test_window_start).all()
    assert (fixed_test["start_date"] < test_window_end).all()

    # honest_train contains ONLY rows strictly before the test window --
    # the fixed-window filter that makes "honest" honest.
    assert (honest_train["start_date"] < test_window_start).all()

    # leaky_train is drawn from outside the window and, given real post-cutoff
    # data exists in the input, actually contains some of it -- otherwise this
    # ablation wouldn't isolate anything.
    assert (leaky_train["start_date"] < test_window_start).any()
    assert (leaky_train["start_date"] >= test_window_end).any()
    assert (
        not leaky_train["start_date"]
        .between(test_window_start, test_window_end, inclusive="left")
        .any()
    )

    # equal size by construction, so any metric delta isn't a sample-size artifact
    assert len(honest_train) == len(leaky_train)
