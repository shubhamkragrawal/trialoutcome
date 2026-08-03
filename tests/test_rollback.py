"""Integration test for M7's rollback procedure: demonstrates the exact M7
acceptance criteria -- promote a deliberately bad model to Production, verify
it's live, then roll back to the known-good version via
rollback_production(). Requires the local file-backed MLflow store
(mlruns/) with a real version already registered in stage "Production" --
this project's actual M5 champion, so this test reads real production
state rather than a fixture, mirroring tests/test_api_contract.py's pattern
of an integration test that depends on already-built state (not a fast
DB-free unit test -- see tests/test_drift_base.py for that kind).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier

from domains.pharma.monitoring.rollback import (
    REGISTERED_MODEL_NAME,
    _set_tracking_uri,
    rollback_production,
)

KNOWN_GOOD_PR_AUC_FLOOR = 0.85  # per decisions.md M3/M5: real champion's temporal PR-AUC is ~0.888


def _register_and_promote_bad_model(client: MlflowClient) -> tuple[str, str, float]:
    """Fit a DummyClassifier (stratified, no real signal) on a tiny 10-row
    synthetic subset -- clearly worse than the real champion's ~0.888
    temporal PR-AUC -- and promote it straight to Production, simulating a
    bad promotion that already happened before this test runs."""
    import mlflow
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    X_train = pd.DataFrame({"x": rng.normal(size=10)})
    y_train = pd.Series(rng.integers(0, 2, size=10))
    bad_model = DummyClassifier(strategy="stratified", random_state=0).fit(X_train, y_train)

    X_eval = pd.DataFrame({"x": rng.normal(size=200)})
    y_eval = pd.Series(rng.integers(0, 2, size=200))
    bad_pr_auc = float(average_precision_score(y_eval, bad_model.predict_proba(X_eval)[:, 1]))

    with mlflow.start_run(run_name="test_bad_promotion_simulated") as run:
        mlflow.set_tag("run_type", "test_bad_promotion")
        mlflow.log_metric("pr_auc_temporal", bad_pr_auc)
        mlflow.sklearn.log_model(
            bad_model, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME
        )
        run_id = run.info.run_id

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    bad_version = next(v.version for v in versions if v.run_id == run_id)
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=bad_version,
        stage="Production",
        archive_existing_versions=True,
    )
    return run_id, bad_version, bad_pr_auc


def test_bad_promotion_is_rolled_back_to_known_good_version(real_dev_state):
    _set_tracking_uri()
    client = MlflowClient()

    live_before = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    assert live_before, "no version currently in Production -- run `make register-model` first"
    known_good_version = live_before[0].version

    run_id, bad_version, bad_pr_auc = _register_and_promote_bad_model(client)
    assert bad_pr_auc < KNOWN_GOOD_PR_AUC_FLOOR, (
        f"fixture's 'bad' model (PR-AUC={bad_pr_auc:.4f}) wasn't clearly worse than the "
        f"real champion's ~0.888 -- re-seed the DummyClassifier fixture"
    )

    try:
        live_bad = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        assert live_bad[0].version == bad_version
        assert live_bad[0].version != known_good_version

        rollback_production(int(known_good_version))

        live_restored = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        assert live_restored[0].version == known_good_version
    finally:
        client.delete_model_version(REGISTERED_MODEL_NAME, bad_version)
