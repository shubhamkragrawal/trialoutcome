"""Pharma-domain FastAPI implementation (TrialOutcome M5, spec Section 6).
Loads the calibrated + conformal model artifacts from the MLflow registry's
"Production" stage at startup and implements the LOCKED CROSS-PROJECT
CONTRACT /predict routes on top of core/serving/api_base.py's generic shell.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from core.explain import SHAPExplainer
from core.serving.api_base import (
    PredictionResponse,
    ServingState,
    SHAPContributor,
    build_base_router,
)
from domains.pharma.dataset_builder import PharmaDatasetBuilder
from domains.pharma.plain_english import generate_summary
from domains.pharma.train_pipeline import CATEGORICAL_FEATURES

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTERED_MODEL_NAME = "trialoutcome_xgb_calibrated"


class TrialFeatures(BaseModel):
    """
    Input schema for POST /api/v1/predict.

    DEVIATION FROM THE M5 PROMPT'S LITERAL SCHEMA (flagged, not silent --
    see decisions.md M5 entry "TrialFeatures input schema corrected to match
    the actually-trained feature set"): the prompt's `TrialFeatures` included
    `intervention_model`, which was never part of the trained feature set
    (train_pipeline.py's CATEGORICAL_FEATURES/NUMERIC_FEATURES do not include
    it -- the M1 spec-table note calling it a "bonus feature" turned out to
    be stale) and OMITTED `has_results`, which the model was actually trained
    on (rank-7 global SHAP importance -- see M4's error analysis). Serving
    with the prompt's literal schema would either silently ignore a client-
    supplied field with no effect (intervention_model) or force a fabricated
    default for a real, importance-ranked feature (has_results) -- both worse
    than fixing the schema to match reality. `intervention_model` is dropped;
    `has_results` is added as a required field.

    M9-1 DEPRECATION: `log_enrollment_count` is still ACCEPTED but is now
    IGNORED. enrollment_count was dropped from the feature set in M9 as target
    leakage (see decisions.md M9-1). The field is kept as an optional no-op
    rather than deleted so RegIntel's `trial_risk` tool wrapper -- which builds
    request bodies against this schema -- keeps working without a coordinated
    release. It should be removed from both sides at the next contract
    revision. Requests that omit it behave identically to requests that
    supply it.
    """

    phase: str
    log_enrollment_count: Optional[float] = None  # M9-1: accepted, ignored.
    num_primary_outcomes: int
    num_sites: int
    has_dmc: Optional[bool] = None
    masking: Optional[str] = None
    allocation: Optional[str] = None
    has_results: bool
    eligibility_criteria_length: int
    exclusion_keyword_count: int
    sponsor_prior_trial_count: int
    sponsor_prior_termination_rate: Optional[float] = None
    sponsor_class: Optional[str] = None
    condition_name: Optional[str] = None
    condition_rarity: int
    start_year: int
    start_quarter: int


class ModelInfoResponse(BaseModel):
    model_version: str
    training_date: str
    pr_auc: float
    ece: float
    feature_pipeline_version: str


@dataclass
class _PharmaModelBundle:
    calibrated_model: object
    conformal_wrapper: object
    xgb_pipeline: object
    shap_explainer: SHAPExplainer
    feature_cols: list[str]
    condition_cols: list[str]
    top_conditions: list[str]
    threshold: float
    feature_pipeline_version: str
    training_date: str
    pr_auc: float
    ece: float
    model_version: str
    db_engine: object
    extras: dict = field(default_factory=dict)


bundle: Optional[_PharmaModelBundle] = None
state = ServingState()


def _load_bundle() -> _PharmaModelBundle:
    # NOTE: MLflow's local file-backed store records each run's artifact_uri
    # as an ABSOLUTE path at logging time (e.g.
    # "file:///Users/.../trialoutcome/mlruns/<exp>/<run>/artifacts"). That
    # path must resolve identically wherever `mlflow.sklearn.load_model` runs
    # -- inside a container, `REPO_ROOT` is `/app`, which does NOT match the
    # host path baked into every existing run's metadata. Docker Compose
    # therefore sets MLFLOW_TRACKING_URI to the *host's* absolute mlruns path
    # and bind-mounts mlruns/ at that same absolute path inside the container
    # (see docker-compose.yml) -- respected here if set, falling back to the
    # REPO_ROOT-relative default for local (non-container) runs.
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or f"file:{REPO_ROOT / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    mv = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])[0]
    run = client.get_run(mv.run_id)

    calibrated_model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/Production")
    conformal_wrapper = mlflow.sklearn.load_model(f"runs:/{mv.run_id}/conformal")

    feature_schema = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/feature_schema.json")
    condition_vocab = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/condition_vocab.json")

    # Unwrap CalibratedClassifierCV(estimator=FrozenEstimator(xgb_pipeline))
    # to get back the raw fitted Pipeline([("pre", ColumnTransformer), ("clf",
    # XGBClassifier)]) -- needed for SHAP (TreeExplainer wants the raw
    # booster, not the calibration wrapper).
    xgb_pipeline = calibrated_model.calibrated_classifiers_[0].estimator.estimator
    shap_explainer = SHAPExplainer(xgb_pipeline.named_steps["clf"])

    config = yaml.safe_load((REPO_ROOT / "domains/pharma/config.yaml").read_text())
    threshold = config["model"]["threshold_decision"]["value"]

    builder = PharmaDatasetBuilder()

    return _PharmaModelBundle(
        calibrated_model=calibrated_model,
        conformal_wrapper=conformal_wrapper,
        xgb_pipeline=xgb_pipeline,
        shap_explainer=shap_explainer,
        feature_cols=feature_schema["feature_cols"],
        condition_cols=condition_vocab["condition_cols"],
        top_conditions=condition_vocab["top_conditions"],
        threshold=threshold,
        feature_pipeline_version=run.data.tags.get("feature_pipeline_version", "unknown"),
        training_date=run.data.tags.get("training_date", "unknown"),
        pr_auc=run.data.metrics.get("pr_auc_temporal", float("nan")),
        ece=run.data.metrics.get("ece_test", float("nan")),
        model_version=mv.version,
        db_engine=builder.engine,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    bundle = _load_bundle()
    state.model_loaded = True
    state.conformal_loaded = True
    yield
    bundle = None
    state.model_loaded = False
    state.conformal_loaded = False


app = FastAPI(title="TrialOutcome API", lifespan=lifespan)
app.include_router(build_base_router(state))


def _require_bundle() -> _PharmaModelBundle:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return bundle


def _condition_bucket(condition_name: Optional[str], top_conditions: list[str]) -> str:
    value = condition_name if condition_name is not None else "unknown"
    if value in top_conditions:
        return value
    if value == "unknown":
        return "unknown"
    return "other"


def _row_from_trial_features(trial: TrialFeatures, b: _PharmaModelBundle) -> pd.DataFrame:
    """
    Purpose: Turn a raw TrialFeatures request body into the one-hot-expanded
        feature row the Production pipeline expects.
    Leakage guard: N/A -- inference only.
    Failure mode (documented limitation): `sponsor_prior_termination_rate=None`
        defaults to 0.0 rather than the
        TRAIN-split median `config.yaml`'s missingness_policy specifies --
        replicating that exact median in live serving would require
        persisting it as a training-time artifact, deferred past M5's DoD.
    """
    has_dmc_str = "unknown" if trial.has_dmc is None else ("true" if trial.has_dmc else "false")
    bucket = _condition_bucket(trial.condition_name, b.top_conditions)

    row: dict = {
        "phase": trial.phase,
        "allocation": trial.allocation if trial.allocation is not None else "unknown",
        "masking": trial.masking if trial.masking is not None else "unknown",
        "has_dmc_str": has_dmc_str,
        "sponsor_class": trial.sponsor_class if trial.sponsor_class is not None else "unknown",
        # M9-1: trial.log_enrollment_count is deliberately NOT read here --
        # the field is accepted for backward compatibility and ignored.
        "num_primary_outcomes": trial.num_primary_outcomes,
        "num_sites": trial.num_sites,
        "has_results": trial.has_results,
        "eligibility_criteria_length": trial.eligibility_criteria_length,
        "exclusion_keyword_count": trial.exclusion_keyword_count,
        "sponsor_prior_trial_count": trial.sponsor_prior_trial_count,
        "sponsor_prior_termination_rate": trial.sponsor_prior_termination_rate
        if trial.sponsor_prior_termination_rate is not None
        else 0.0,
        "condition_rarity": trial.condition_rarity,
        "start_year": trial.start_year,
        "start_quarter": trial.start_quarter,
    }
    for col in b.condition_cols:
        row[col] = False
    row[f"condition_{bucket}"] = True

    df = pd.DataFrame([row])
    return df.reindex(columns=b.feature_cols, fill_value=False)


def _row_from_jsonb(features: dict, b: _PharmaModelBundle) -> pd.DataFrame:
    """
    Purpose: Turn the already-engineered feature dict persisted in
        `ml.training_dataset.features` into the Production pipeline's exact
        input row.
    Leakage guard: N/A -- read-only inference lookup.
    Failure mode: `reindex(fill_value=False)` fills any column the Production
        model expects but this row's JSONB lacks (in practice only ever
        `condition_unknown`, which never appears in this dataset --
        `condition_name` null rate is 0% -- see notebooks/04_shap_analysis.ipynb's
        equivalent note) with False, matching the M4 notebook's identical
        reconciliation between two independently-fit one-hot vocabularies.
    """
    df = pd.DataFrame([features]).reindex(columns=b.feature_cols, fill_value=False)
    for col in b.condition_cols:
        df[col] = df[col].astype(bool)
    return df


def _original_feature_of(expanded_name: str) -> str:
    """Reverses ColumnTransformer's `cat__`/`num__` prefixing back to the
    original feature name -- identical logic to notebooks/04_shap_analysis.ipynb's
    `original_feature_of`, reused here so a one-hot categorical's SHAP
    contribution is reported once, on its parent feature, rather than as
    fragments per dummy column."""
    if expanded_name.startswith("cat__"):
        rest = expanded_name[len("cat__") :]
        for f in CATEGORICAL_FEATURES:
            if rest.startswith(f + "_"):
                return f
        return rest
    if expanded_name.startswith("num__"):
        return expanded_name[len("num__") :]
    return expanded_name


def _top_shap_contributors(row_df: pd.DataFrame, b: _PharmaModelBundle, top_n: int = 5) -> list[dict]:
    pre = b.xgb_pipeline.named_steps["pre"]
    expanded = pre.transform(row_df[b.feature_cols])
    expanded_names = list(pre.get_feature_names_out())
    expanded_df = pd.DataFrame(np.asarray(expanded), columns=expanded_names).astype(float)

    shap_vals = b.shap_explainer.compute_shap_values(expanded_df)  # shape (1, n_expanded)

    orig_map = [_original_feature_of(n) for n in expanded_names]
    seen: dict[str, list[int]] = {}
    for i, of in enumerate(orig_map):
        seen.setdefault(of, []).append(i)
    agg_names = list(seen.keys())
    agg_shap = np.column_stack([shap_vals[:, idxs].sum(axis=1) for idxs in seen.values()])[0]

    order = np.argsort(-np.abs(agg_shap))[:top_n]
    return [
        {
            "feature": agg_names[i],
            # .item() unwraps numpy scalars (int64/float64/bool_) to native
            # Python types -- pydantic's `Any` field can't serialize numpy
            # scalars to JSON otherwise.
            "value": _to_native(row_df.iloc[0][agg_names[i]]),
            "shap_contribution": float(agg_shap[i]),
        }
        for i in order
    ]


def _to_native(value):
    return value.item() if hasattr(value, "item") else value


def _predict_from_row(row_df: pd.DataFrame, b: _PharmaModelBundle) -> PredictionResponse:
    proba_arr = b.calibrated_model.predict_proba(row_df[b.feature_cols])[:, 1]
    proba = float(proba_arr[0])

    _, intervals = b.conformal_wrapper.predict_with_interval(row_df[b.feature_cols])
    conformal_interval = intervals[0]

    threshold_decision = "high_risk" if proba >= b.threshold else "low_risk"
    top_shap = _top_shap_contributors(row_df, b, top_n=5)
    summary = generate_summary(top_shap, threshold_decision, proba)

    return PredictionResponse(
        proba=proba,
        conformal_interval=conformal_interval,
        threshold_decision=threshold_decision,
        top_shap=[SHAPContributor(**c) for c in top_shap],
        plain_english_summary=summary,
        feature_pipeline_version=b.feature_pipeline_version,
    )


@app.get("/api/v1/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    b = _require_bundle()
    return ModelInfoResponse(
        model_version=str(b.model_version),
        training_date=b.training_date,
        pr_auc=b.pr_auc,
        ece=b.ece,
        feature_pipeline_version=b.feature_pipeline_version,
    )


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(trial: TrialFeatures) -> PredictionResponse:
    b = _require_bundle()
    row_df = _row_from_trial_features(trial, b)
    return _predict_from_row(row_df, b)


@app.api_route(
    "/api/v1/predict/nct/{nct_id}", methods=["GET", "POST"], response_model=PredictionResponse
)
def predict_by_nct_id(nct_id: str) -> PredictionResponse:
    """
    NOTE: the M5 spec text lists this route as POST-only, but its own
    tests/test_api_contract.py calls it with `requests.get(...)` -- an
    internal inconsistency in the milestone prompt. Since the route takes no
    request body (only a path param), GET is arguably the more correct verb
    for a read-only fetch-by-ID anyway; both methods are registered so the
    literal test file passes and the documented contract (POST) still works.
    """
    b = _require_bundle()
    with b.db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT features FROM ml.training_dataset WHERE nct_id = :nct_id"),
            {"nct_id": nct_id},
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"nct_id '{nct_id}' not found in ml.training_dataset")

    row_df = _row_from_jsonb(row[0], b)
    return _predict_from_row(row_df, b)
