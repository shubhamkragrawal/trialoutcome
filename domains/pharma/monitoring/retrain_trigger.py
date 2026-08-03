"""Pharma-domain M7 retraining trigger: reacts to a drift breach logged by
domains/pharma/monitoring/drift_job.py by retraining the known-best XGBoost
champion (same fixed hyperparameters register_model.py uses -- this is a
retrain, not a re-search, so the 32-trial Optuna sweep in train_pipeline.py
is never re-run) and registering the result to MLflow stage "Staging". Run as
`python -m domains.pharma.monitoring.retrain_trigger` (or `make
check-drift-retrain`; pass --force to bypass the drift check for testing).

WHY THIS NEVER AUTO-PROMOTES TO PRODUCTION (the core M7 design point, not a
minor detail): a drift breach means the model's INPUT feature distributions
have shifted versus training time. It says nothing about whether the model's
LABEL quality or predictive relationship still holds -- a retrained candidate
could easily be worse (overfit to a small recent window, trained on stale or
partially-mislabeled trials, etc.). Auto-promoting on drift alone would
silently swap Production for a model no human has ever evaluated. Every
retrain this function produces lands in stage "Staging" and is logged to
`ml.retrain_log` for a human to review the metrics and promote by hand (see
domains/pharma/monitoring/rollback.py's `rollback_production()`, which is
the one function anywhere in this codebase that ever transitions a version
to stage "Production" via an automated code path -- reused here for manual
promotion too, since "make the target version live" is the same operation
whether the target was a previous Production version (rollback) or a
reviewed Staging candidate (forward promotion). See decisions.md's M7 entry:
the spec's literal `mlflow models transition-stage` CLI snippet does not
exist in mlflow 2.22.5 (`mlflow models --help` has no such subcommand --
registry stage transitions are Python/REST-API only), so `make rollback
VERSION=N` is the actual one-command procedure, for both directions.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sqlalchemy import text
from xgboost import XGBClassifier

from core.calibration import expected_calibration_error
from core.dataset_builder_base import SplitDates
from domains.pharma.dataset_builder import PharmaDatasetBuilder, feature_pipeline_version
from domains.pharma.register_model import (
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    THRESHOLD,
    XGBOOST_BEST_PARAMS,
)
from domains.pharma.train_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    _apply_condition_one_hot,
    _fit_condition_vocab,
    _make_preprocessor,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

_CREATE_RETRAIN_LOG_SQL = """
CREATE TABLE IF NOT EXISTS ml.retrain_log (
    id SERIAL PRIMARY KEY,
    drift_report_uri TEXT,
    triggered_at TIMESTAMPTZ DEFAULT now(),
    new_run_id TEXT,
    new_feature_pipeline_version TEXT,
    current_production_version TEXT,
    version_mismatch BOOL,
    promoted BOOL DEFAULT FALSE,
    promoted_at TIMESTAMPTZ
);
"""

_INSERT_RETRAIN_LOG_SQL = """
INSERT INTO ml.retrain_log
    (drift_report_uri, new_run_id, new_feature_pipeline_version,
     current_production_version, version_mismatch)
VALUES
    (:drift_report_uri, :new_run_id, :new_feature_pipeline_version,
     :current_production_version, :version_mismatch)
"""


def _set_tracking_uri() -> None:
    """
    Purpose: Point MLflow at this repo's own file-backed tracking store,
        exactly like drift_job.py's _current_production_version and
        register_model.py -- respecting MLFLOW_TRACKING_URI if a caller has
        set it, otherwise resolving to THIS checkout's mlruns/.
    Leakage guard: N/A.
    Failure mode: Without this, a caller whose environment doesn't set
        MLFLOW_TRACKING_URI would silently read/write whatever mlruns/ the
        current working directory happens to resolve to instead.
    """
    import os

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or f"file:{REPO_ROOT / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)


def _latest_drift_row(engine) -> dict | None:
    """
    Purpose: Read the single most recent ml.drift_log row, the same table
        domains/pharma/monitoring/drift_job.py writes to -- this is the only
        thing that decides whether a retrain is warranted (short of --force).
    Leakage guard: N/A.
    Failure mode: If ml.drift_log has never been written to (fresh
        environment, drift job never run), returns None rather than raising
        -- the caller treats "no drift history" the same as "not drifted".
    """
    with engine.connect() as conn:
        row = (
            conn.execute(
                text(
                    "SELECT drift_share, n_features_drifted, drifted, report_path, model_version "
                    "FROM ml.drift_log ORDER BY id DESC LIMIT 1"
                )
            )
            .mappings()
            .first()
        )
    return dict(row) if row else None


def _current_production_version_and_tag(client: MlflowClient) -> tuple[str, str]:
    """
    Purpose: Look up the currently-registered Production model version and
        the feature_pipeline_version tag its run was logged with, so a
        retrain candidate's own tag can be compared against it.
    Leakage guard: N/A.
    Failure mode: If no version is in stage "Production" (fresh environment),
        returns ("unknown", "unknown") rather than raising -- mirrors
        drift_job.py's _current_production_version.
    """
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    if not versions:
        return "unknown", "unknown"
    version = versions[0]
    run = client.get_run(version.run_id)
    tag = run.data.tags.get("feature_pipeline_version", "unknown")
    return version.version, tag


def _retrain_and_stage(builder: PharmaDatasetBuilder) -> tuple[str, str, str]:
    """
    Purpose: Refit the M2 XGBoost champion on a fresh pull of
        ml.training_dataset using the SAME fixed hyperparameters
        register_model.py uses (XGBOOST_BEST_PARAMS) -- a retrain, not a
        re-search -- calibrate on CALIB exactly like register_model.py, and
        register the result to stage "Staging". Returns
        (run_id, version, feature_pipeline_version) for the caller to log
        and compare.
    Leakage guard: Reuses PharmaDatasetBuilder.temporal_split and the same
        train-only condition vocabulary fit as register_model.py/
        train_pipeline.py -- no leakage-relevant logic is reimplemented here.
    Failure mode: If ml.training_dataset's calib split is empty (e.g. a
        fixture that only seeds train/test rows), CalibratedClassifierCV.fit
        raises on zero calibration rows -- deliberate: a retrain that cannot
        be calibrated should not silently register an uncalibrated model.
    """
    raw = builder.fetch_raw()
    feat = builder.build_features(raw)

    split_cfg = builder.config["split"]
    dates = SplitDates(
        train_end=pd.Timestamp(split_cfg["train_end"]),
        calib_end=pd.Timestamp(split_cfg["calib_end"]),
    )
    temporal = builder.temporal_split(feat, date_col="start_date", split_dates=dates)

    top_conditions = _fit_condition_vocab(temporal[temporal["split"] == "train"])
    temporal, condition_cols = _apply_condition_one_hot(temporal, top_conditions)
    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES + condition_cols

    train = temporal[temporal["split"] == "train"]
    calib = temporal[temporal["split"] == "calib"]
    test = temporal[temporal["split"] == "test"]

    scale_pos_weight = float((train["label"] == 0).sum() / (train["label"] == 1).sum())
    clf = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        **XGBOOST_BEST_PARAMS,
    )
    xgb_pipeline = Pipeline([("pre", _make_preprocessor(condition_cols)), ("clf", clf)])
    xgb_pipeline.fit(train[feature_cols], train["label"])

    calibrated_pipeline = CalibratedClassifierCV(
        estimator=FrozenEstimator(xgb_pipeline), method="isotonic"
    )
    calibrated_pipeline.fit(calib[feature_cols], calib["label"])

    proba_test = calibrated_pipeline.predict_proba(test[feature_cols])[:, 1]
    ece_test, _ = expected_calibration_error(test["label"].to_numpy(), proba_test)
    pr_auc_test = float(average_precision_score(test["label"], proba_test))
    roc_auc_test = float(roc_auc_score(test["label"], proba_test))

    version = feature_pipeline_version()

    _set_tracking_uri()
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="retrain_trigger_candidate") as run:
        mlflow.set_tag("feature_pipeline_version", version)
        mlflow.set_tag("training_date", date.today().isoformat())
        mlflow.set_tag("run_type", "retrain_trigger")
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_param("calibration_method", "isotonic")
        for k, v in XGBOOST_BEST_PARAMS.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("ece_test", ece_test)
        mlflow.log_metric("pr_auc_temporal", pr_auc_test)
        mlflow.log_metric("roc_auc_temporal", roc_auc_test)
        mlflow.sklearn.log_model(
            calibrated_pipeline, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME
        )
        run_id = run.info.run_id

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    this_version = next(v.version for v in versions if v.run_id == run_id)
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME, version=this_version, stage="Staging"
    )
    return run_id, this_version, version


def _version_mismatch_and_log(
    engine,
    client: MlflowClient,
    drift_report_uri: str | None,
    new_run_id: str,
    new_version: str,
    new_feature_pipeline_version: str,
) -> bool:
    """
    Purpose: Compare a retrain candidate's feature_pipeline_version tag
        against current Production's, log exactly one row to
        ml.retrain_log (promoted always defaults to FALSE here -- this
        function never promotes anything), print a WARNING when they
        differ, and print the staged-for-review message a human uses to
        promote by hand. Split out from _retrain_and_stage so
        tests/test_version_mismatch.py can exercise this comparison+logging
        logic directly against a fake Staging run, without paying for a
        real retrain.
    Leakage guard: N/A.
    Failure mode: If current Production's tag is "unknown" (fresh
        environment, nothing registered yet), any real candidate's tag will
        compare as a mismatch -- a false positive in that specific case,
        acceptable since a fresh environment has no real Production baseline
        to protect yet.
    """
    current_prod_version, current_prod_fpv = _current_production_version_and_tag(client)
    version_mismatch = new_feature_pipeline_version != current_prod_fpv

    with engine.begin() as conn:
        conn.execute(text(_CREATE_RETRAIN_LOG_SQL))
        conn.execute(
            text(_INSERT_RETRAIN_LOG_SQL),
            {
                "drift_report_uri": drift_report_uri,
                "new_run_id": new_run_id,
                "new_feature_pipeline_version": new_feature_pipeline_version,
                "current_production_version": current_prod_version,
                "version_mismatch": version_mismatch,
            },
        )

    if version_mismatch:
        print(
            f"WARNING: feature_pipeline_version mismatch -- candidate run {new_run_id} "
            f"(version {new_version}) was trained with version="
            f"{new_feature_pipeline_version!r}, but current Production (version "
            f"{current_prod_version}) was trained with version={current_prod_fpv!r}. "
            "Surfaced for human review only -- never auto-blocks or auto-proceeds "
            "(drift is not a label-quality guarantee, and neither is a version match)."
        )

    print(
        f"Staged for review: run_id={new_run_id}, version={new_version}. "
        f"Promote with: make rollback VERSION={new_version}"
    )
    return version_mismatch


def check_and_trigger_retrain(force: bool = False) -> dict | None:
    """
    Purpose: M7 entrypoint. Reads the latest ml.drift_log verdict; if
        drifted (or --force), retrains the champion, registers it to
        Staging, compares its feature_pipeline_version against current
        Production's, and logs one ml.retrain_log row. Returns None if no
        retrain was triggered, else a dict with the new run's identifiers.
    Leakage guard: See _retrain_and_stage.
    Failure mode: See module docstring for why this never promotes to
        Production under any condition, including a version match.
    """
    builder = PharmaDatasetBuilder()
    _set_tracking_uri()
    client = MlflowClient()

    latest_drift = _latest_drift_row(builder.engine)
    drifted = force or bool(latest_drift and latest_drift["drifted"])
    if not drifted:
        print("No drift breach detected (and --force not set) -- nothing to do.")
        return None

    new_run_id, new_version, new_fpv = _retrain_and_stage(builder)
    drift_report_uri = latest_drift["report_path"] if latest_drift else None
    version_mismatch = _version_mismatch_and_log(
        builder.engine, client, drift_report_uri, new_run_id, new_version, new_fpv
    )

    return {
        "new_run_id": new_run_id,
        "new_version": new_version,
        "new_feature_pipeline_version": new_fpv,
        "version_mismatch": version_mismatch,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M7 drift-triggered retrain check")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if the latest drift check is clean (testing)",
    )
    args = parser.parse_args()
    check_and_trigger_retrain(force=args.force)
