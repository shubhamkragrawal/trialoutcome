"""Integration test for M7's end-to-end retrain-trigger mechanism: a
synthetic drift breach must (a) make the drift job flag drifted=True and
(b) make check_and_trigger_retrain() actually retrain the champion and
register a new Staging (never Production) model version, logging a row to
ml.retrain_log. Requires a live Postgres with a real, already-built
ml.training_dataset (train/calib/test all populated -- see
domains/pharma/dataset_builder.py's M1 build) and the local file-backed
MLflow store -- mirrors tests/test_api_contract.py's pattern of an
integration test that needs already-built state, not a fast DB-free unit
test (see tests/test_drift_base.py for that kind).

PERTURBATION CHOICE (deviates from the M7 brief's literal suggestion -- see
decisions.md's M7 entry): the brief suggested "multiply log_enrollment_count
by 3" as the synthetic perturbation. Empirically verified against the real
ml.training_dataset this does NOT work: the real baseline is already 11/38
features (28.9%) drifted, and log_enrollment_count's drift status doesn't
even change under a 3x (or +20 additive) shift -- checked directly with
PharmaDriftMonitor.score_batch/check_thresholds before writing this test.
Shifting every NUMERIC_FEATURES column only reaches 34.2%, still short of
the 0.5 threshold. 22 of the 38 compared columns are the one-hot
`condition_*` indicators, and those dominate drift_share -- flipping even
5% of each one's boolean values (a small, seeded perturbation, not a
wholesale corruption) reliably crosses 0.5 (~63% in practice). This test
perturbs both: log_enrollment_count x3 (kept for continuity with the
brief's intent, verified non-load-bearing on its own) plus the condition
one-hot flip (the part that actually forces the breach).

NLTK_DISABLE_IMPORT_SECURITY=1 is required to import evidently transitively
via drift_job.py -- see core/monitoring/drift_base.py's module docstring.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sqlalchemy import text

from domains.pharma.dataset_builder import PharmaDatasetBuilder
from domains.pharma.monitoring.drift_job import PharmaDriftMonitor
from domains.pharma.monitoring.retrain_trigger import (
    REGISTERED_MODEL_NAME,
    check_and_trigger_retrain,
)

CONDITION_FLIP_PROB = 0.10  # empirically landed exactly at drift_share=0.5 (19/38) at 0.05 --
# bumped for margin above the boundary rather than a knife-edge pass (verified: still
# far short of the ~92-100% seen when also touching numeric/categorical features).
SEED = 42


@pytest.fixture
def perturbed_test_split():
    """Multiply log_enrollment_count by 3x and flip a small seeded fraction
    of each condition_* one-hot column for every 'test'-split row in
    ml.training_dataset, forcing a real drift breach against 'train'.
    Restores every perturbed row's original features JSONB on teardown,
    regardless of test outcome."""
    builder = PharmaDatasetBuilder()
    engine = builder.engine

    with engine.connect() as conn:
        original_rows = conn.execute(
            text("SELECT nct_id, features FROM ml.training_dataset WHERE split = 'test'")
        ).mappings().all()
    assert original_rows, "ml.training_dataset has no 'test'-split rows -- build it first (make build-dataset)"

    rng = np.random.default_rng(SEED)
    with engine.begin() as conn:
        for row in original_rows:
            features = dict(row["features"])
            if "log_enrollment_count" in features and features["log_enrollment_count"] is not None:
                features["log_enrollment_count"] = features["log_enrollment_count"] * 3.0
            for key in features:
                if key.startswith("condition_") and isinstance(features[key], bool):
                    if rng.random() < CONDITION_FLIP_PROB:
                        features[key] = not features[key]
            conn.execute(
                text(
                    "UPDATE ml.training_dataset SET features = CAST(:features AS JSONB) "
                    "WHERE nct_id = :nct_id"
                ),
                {"features": json.dumps(features), "nct_id": row["nct_id"]},
            )

    try:
        yield builder
    finally:
        with engine.begin() as conn:
            for row in original_rows:
                conn.execute(
                    text(
                        "UPDATE ml.training_dataset SET features = CAST(:features AS JSONB) "
                        "WHERE nct_id = :nct_id"
                    ),
                    {"features": json.dumps(dict(row["features"])), "nct_id": row["nct_id"]},
                )


def test_synthetic_drift_breach_triggers_staging_registration(real_dev_state, perturbed_test_split):
    builder = perturbed_test_split

    drift_summary = PharmaDriftMonitor().run()
    assert drift_summary["drifted"] is True, (
        f"perturbation did not force a breach: {drift_summary} -- "
        f"see this file's module docstring for the empirically-verified perturbation"
    )

    result = check_and_trigger_retrain()
    assert result is not None

    client = MlflowClient()
    try:
        new_version_info = client.get_model_version(REGISTERED_MODEL_NAME, result["new_version"])
        assert new_version_info.current_stage == "Staging"

        prod_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        assert all(v.version != result["new_version"] for v in prod_versions), (
            "retrain candidate must never land in Production automatically"
        )

        with builder.engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM ml.retrain_log WHERE new_run_id = :run_id"),
                {"run_id": result["new_run_id"]},
            ).mappings().first()
        assert row is not None
        assert row["promoted"] is False
    finally:
        client.delete_model_version(REGISTERED_MODEL_NAME, result["new_version"])
