"""Fast, dependency-free unit tests for M9-8: imputation constants frozen at
training time (PharmaDatasetBuilder.compute_imputation_constants) and loaded
at serving (domains/pharma/serving/api.py's _row_from_trial_features), instead
of being recomputed from a single-row request. No DB, no MLflow, no running
API required -- PharmaDatasetBuilder() itself is safe to construct without a
live Postgres connection (SQLAlchemy's create_engine() is lazy, see decisions.md
M8), and compute_imputation_constants only ever touches the `raw` DataFrame
argument passed to it.
"""

from __future__ import annotations
import os
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5433")
os.environ.setdefault("POSTGRES_DB", "pharmapulse")
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")

import numpy as np
import pandas as pd

from domains.pharma.dataset_builder import PharmaDatasetBuilder
from domains.pharma.serving.api import TrialFeatures, _PharmaModelBundle, _row_from_trial_features


def _raw_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_primary_outcomes": [1, 2, np.nan, 3, 5],
            "num_sites": [1, 1, 2, np.nan, 10],
            "sponsor_prior_termination_rate": [0.0, 0.1, np.nan, 0.2, 0.5],
        }
    )


_FEATURE_COLS = [
    "phase",
    "allocation",
    "masking",
    "has_dmc_str",
    "sponsor_class",
    "num_primary_outcomes",
    "num_sites",
    "has_results",
    "eligibility_criteria_length",
    "exclusion_keyword_count",
    "sponsor_prior_trial_count",
    "sponsor_prior_termination_rate",
    "condition_rarity",
    "start_year",
    "start_quarter",
]


def _bundle(imputation_constants: dict[str, float]) -> _PharmaModelBundle:
    return _PharmaModelBundle(
        calibrated_model=None,
        conformal_wrapper=None,
        xgb_pipeline=None,
        shap_explainer=None,
        feature_cols=_FEATURE_COLS,
        condition_cols=[],
        top_conditions=[],
        threshold=0.16,
        feature_pipeline_version="test",
        training_date="2026-08-03",
        pr_auc=0.6,
        ece=0.03,
        model_version="45",
        db_engine=None,
        imputation_constants=imputation_constants,
        empirical_coverage=0.931,
    )


def _trial_features(**overrides) -> TrialFeatures:
    defaults = {
        "phase": "PHASE3",
        "num_primary_outcomes": 2,
        "num_sites": 3,
        "has_results": True,
        "eligibility_criteria_length": 100,
        "exclusion_keyword_count": 2,
        "sponsor_prior_trial_count": 5,
        "sponsor_prior_termination_rate": None,
        "condition_rarity": 1,
        "start_year": 2024,
        "start_quarter": 1,
    }
    defaults.update(overrides)
    return TrialFeatures(**defaults)


def test_compute_imputation_constants_matches_median_skipping_nan():
    builder = PharmaDatasetBuilder()
    raw = _raw_fixture()

    constants = builder.compute_imputation_constants(raw)

    assert constants["num_primary_outcomes"] == raw["num_primary_outcomes"].median()
    assert constants["num_sites"] == raw["num_sites"].median()
    assert constants["sponsor_prior_termination_rate"] == (
        raw["sponsor_prior_termination_rate"].median()
    )


def test_serving_uses_frozen_constant_not_hardcoded_zero_when_field_missing():
    b = _bundle({"sponsor_prior_termination_rate": 0.1234})
    trial = _trial_features(sponsor_prior_termination_rate=None)

    row = _row_from_trial_features(trial, b)

    assert row.iloc[0]["sponsor_prior_termination_rate"] == 0.1234


def test_serving_constant_matches_value_computed_at_training_time():
    """The exact fixture-equality check the M9 fix plan's DoD calls for:
    the constant _row_from_trial_features falls back to, when loaded from
    what compute_imputation_constants would have produced at training time
    on the same raw data, must equal that training-time value exactly --
    not a value independently recomputed from the live (single-row) request.
    """
    builder = PharmaDatasetBuilder()
    raw = _raw_fixture()
    training_time_constants = builder.compute_imputation_constants(raw)

    # Simulate _load_bundle() loading this exact dict back from MLflow.
    b = _bundle(training_time_constants)
    trial = _trial_features(sponsor_prior_termination_rate=None)

    row = _row_from_trial_features(trial, b)

    assert (
        row.iloc[0]["sponsor_prior_termination_rate"]
        == (training_time_constants["sponsor_prior_termination_rate"])
    )


def test_serving_still_uses_request_value_when_field_supplied():
    b = _bundle({"sponsor_prior_termination_rate": 0.1234})
    trial = _trial_features(sponsor_prior_termination_rate=0.02)

    row = _row_from_trial_features(trial, b)

    assert row.iloc[0]["sponsor_prior_termination_rate"] == 0.02
