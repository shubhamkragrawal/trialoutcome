"""M5 Step 0b: refit the M2 XGBoost champion, wrap it with an isotonic
calibrator, and register the whole calibrated pipeline as one MLflow model
artifact in stage "Production". Run as `python -m domains.pharma.register_model`
(or `make register-model`).

M2 never called mlflow.sklearn.log_model (see decisions.md) -- this script is
what finally closes that gap, using the exact same refit-from-logged-
hyperparameters procedure M3/M4 already validated (assert PR-AUC matches to
6dp before proceeding).
"""

from __future__ import annotations

import json
from datetime import date

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from core.calibration import expected_calibration_error
from core.dataset_builder_base import SplitDates
from domains.pharma.dataset_builder import PharmaDatasetBuilder, feature_pipeline_version
from domains.pharma.train_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    REPO_ROOT,
    _apply_condition_one_hot,
    _fit_condition_vocab,
    _make_preprocessor,
)

EXPERIMENT_NAME = "trialoutcome_m2"
REGISTERED_MODEL_NAME = "trialoutcome_xgb_calibrated"
THRESHOLD = 0.22

# Logged hyperparams for MLflow run c4a4d0300bd949f8bb07b7c48417be4d (M2 XGBoost
# champion, CV-selected). Reproduced exactly here per decisions.md's M3 refit
# convention -- see that run's params in the MLflow UI for provenance.
XGBOOST_BEST_PARAMS = dict(
    n_estimators=452,
    max_depth=3,
    learning_rate=0.03442632871923141,
    subsample=0.6455669343594874,
    colsample_bytree=0.7423738358796874,
    min_child_weight=3,
)
XGBOOST_BEST_RUN_ID = "c4a4d0300bd949f8bb07b7c48417be4d"
EXPECTED_PR_AUC_TEMPORAL = 0.8877613220680413


def main() -> None:
    builder = PharmaDatasetBuilder()
    raw = builder.fetch_raw()
    feat = builder.build_features(raw)

    split_cfg = builder.config["split"]
    dates = SplitDates(
        train_end=pd.Timestamp(split_cfg["train_end"]), calib_end=pd.Timestamp(split_cfg["calib_end"])
    )
    temporal = builder.temporal_split(feat, date_col="start_date", split_dates=dates)

    top_conditions = _fit_condition_vocab(temporal[temporal["split"] == "train"])
    temporal, condition_cols = _apply_condition_one_hot(temporal, top_conditions)
    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES + condition_cols

    train = temporal[temporal["split"] == "train"]
    calib = temporal[temporal["split"] == "calib"]
    test = temporal[temporal["split"] == "test"]
    print(f"train={len(train)}  calib={len(calib)}  test={len(test)}")

    # --- Refit the XGBoost champion on TRAIN only, verify it reproduces the
    # logged M2 run exactly (same procedure M3/M4 used) before calibrating.
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

    proba_test_raw = xgb_pipeline.predict_proba(test[feature_cols])[:, 1]
    pr_check = float(average_precision_score(test["label"], proba_test_raw))
    roc_check = float(roc_auc_score(test["label"], proba_test_raw))
    print(f"Refit sanity check -- pr_auc_temporal={pr_check:.6f} (expected {EXPECTED_PR_AUC_TEMPORAL:.6f})")
    assert abs(pr_check - EXPECTED_PR_AUC_TEMPORAL) < 1e-6, (
        "refit does not match logged run c4a4d0300bd949f8bb07b7c48417be4d -- feature pipeline drifted"
    )
    print("Refit reproduces the logged run exactly.")

    # --- Wrap the already-fitted pipeline with sklearn's CalibratedClassifierCV
    # in isotonic mode, fit on CALIB only. NOTE (deviation from the M5 prompt's
    # literal snippet): sklearn 1.9 removed `cv="prefit"` and renamed
    # `base_estimator` -> `estimator` (see decisions.md M5 entry) -- an
    # already-fitted estimator must instead be wrapped in
    # sklearn.frozen.FrozenEstimator so CalibratedClassifierCV treats all
    # supplied data as the calibration set, mirroring the old cv="prefit"
    # behavior with the current sklearn API.
    calibrated_pipeline = CalibratedClassifierCV(
        estimator=FrozenEstimator(xgb_pipeline), method="isotonic"
    )
    calibrated_pipeline.fit(calib[feature_cols], calib["label"])

    proba_test_calibrated = calibrated_pipeline.predict_proba(test[feature_cols])[:, 1]
    ece_test, _ = expected_calibration_error(test["label"].to_numpy(), proba_test_calibrated)
    pr_auc_calibrated = float(average_precision_score(test["label"], proba_test_calibrated))
    print(f"Calibrated TEST ECE={ece_test:.4f}  PR-AUC={pr_auc_calibrated:.4f}")

    version = feature_pipeline_version()
    print(f"feature_pipeline_version = {version}")
    if version == "unknown":
        print(
            "WARNING: feature_pipeline_version is still 'unknown' -- commit the repo "
            "before treating this run as the real production record."
        )

    mlflow.set_tracking_uri(f"file:{REPO_ROOT / 'mlruns'}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="xgboost_calibrated_production") as run:
        mlflow.set_tag("feature_pipeline_version", version)
        mlflow.set_tag("training_date", date.today().isoformat())
        mlflow.set_tag("source_run_id", XGBOOST_BEST_RUN_ID)
        mlflow.log_param("threshold", THRESHOLD)
        mlflow.log_param("calibration_method", "isotonic")
        for k, v in XGBOOST_BEST_PARAMS.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("ece_test", ece_test)
        mlflow.log_metric("pr_auc_temporal", pr_auc_calibrated)
        mlflow.log_metric("roc_auc_temporal", roc_check)

        mlflow.log_dict({"condition_cols": condition_cols, "top_conditions": top_conditions}, "condition_vocab.json")
        mlflow.log_dict(
            {
                "categorical_features": CATEGORICAL_FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "feature_cols": feature_cols,
            },
            "feature_schema.json",
        )

        mlflow.sklearn.log_model(
            calibrated_pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        run_id = run.info.run_id

    client = MlflowClient()
    # New registration -> version number is deterministic only after the
    # log_model call above actually creates it; look it up rather than
    # hardcoding "1" so re-runs (which create version 2, 3, ...) still work.
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    this_version = next(v.version for v in versions if v.run_id == run_id)

    # `transition_model_version_stage` does NOT archive other versions
    # already in "Production" by default -- without this, re-running
    # `make register-model` accumulates multiple simultaneous "Production"
    # versions, and `models:/{name}/Production` becomes an ambiguous URI for
    # the serving layer to resolve. Archive any existing Production version(s)
    # first so exactly one version holds that stage at a time.
    for v in versions:
        if v.current_stage == "Production" and v.version != this_version:
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME, version=v.version, stage="Archived"
            )
            print(f"Archived previous Production version {v.version}")

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME, version=this_version, stage="Production"
    )
    print(f"Registered {REGISTERED_MODEL_NAME} version {this_version} (run {run_id}) -> Production")


if __name__ == "__main__":
    main()
