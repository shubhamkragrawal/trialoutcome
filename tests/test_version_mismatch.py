"""Integration test for M7's version-mismatch check: a retrain candidate
tagged with a `feature_pipeline_version` that differs from the real current
Production tag must flag version_mismatch=True, print a WARNING, and log a
row to ml.retrain_log with promoted=False. Deliberately exercises
`_version_mismatch_and_log` directly (rather than a full
`check_and_trigger_retrain()` run) so this test doesn't pay for a real
XGBoost retrain -- tests/test_retrain_trigger.py covers the full path.

Requires a live Postgres (ml.retrain_log is created idempotently by the
function under test) and the local file-backed MLflow store, same as
test_retrain_trigger.py and test_rollback.py -- see those files' module
docstrings for why these three are integration tests rather than fast
DB-free unit tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sqlalchemy import text

from domains.pharma.dataset_builder import PharmaDatasetBuilder
from domains.pharma.monitoring.retrain_trigger import (
    REGISTERED_MODEL_NAME,
    _current_production_version_and_tag,
    _set_tracking_uri,
    _version_mismatch_and_log,
)

FAKE_FEATURE_PIPELINE_VERSION = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"


def _register_fake_staging_run(client: MlflowClient) -> tuple[str, str]:
    """Register a trivial DummyClassifier as a new model version tagged with
    a bogus feature_pipeline_version, transitioned to Staging -- simulates
    "a test Staging run" whose training-time feature logic diverged from
    current Production's, without paying for a real retrain."""
    import mlflow

    X = pd.DataFrame({"x": np.arange(10.0)})
    y = pd.Series([0, 1] * 5)
    model = DummyClassifier(strategy="stratified", random_state=0).fit(X, y)

    with mlflow.start_run(run_name="test_version_mismatch_fake_candidate") as run:
        mlflow.set_tag("feature_pipeline_version", FAKE_FEATURE_PIPELINE_VERSION)
        mlflow.set_tag("run_type", "test_fake_candidate")
        mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME
        )
        run_id = run.info.run_id

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    version = next(v.version for v in versions if v.run_id == run_id)
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME, version=version, stage="Staging"
    )
    return run_id, version


def test_mismatched_feature_pipeline_version_is_flagged_and_logged(real_dev_state, capsys):
    _set_tracking_uri()
    client = MlflowClient()

    _, current_prod_fpv = _current_production_version_and_tag(client)
    assert current_prod_fpv != FAKE_FEATURE_PIPELINE_VERSION, (
        "test fixture's fake tag collided with the real current Production tag -- "
        "pick a different placeholder"
    )

    run_id, version = _register_fake_staging_run(client)
    try:
        builder = PharmaDatasetBuilder()
        version_mismatch = _version_mismatch_and_log(
            builder.engine,
            client,
            drift_report_uri=None,
            new_run_id=run_id,
            new_version=version,
            new_feature_pipeline_version=FAKE_FEATURE_PIPELINE_VERSION,
        )
        assert version_mismatch is True

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "mismatch" in captured.out.lower()

        with builder.engine.connect() as conn:
            row = (
                conn.execute(
                    text("SELECT * FROM ml.retrain_log WHERE new_run_id = :run_id"),
                    {"run_id": run_id},
                )
                .mappings()
                .first()
            )
        assert row is not None
        assert row["version_mismatch"] is True
        assert row["promoted"] is False
        assert row["new_feature_pipeline_version"] == FAKE_FEATURE_PIPELINE_VERSION
    finally:
        client.delete_model_version(REGISTERED_MODEL_NAME, version)
