"""Pharma-domain FastAPI implementation (TrialOutcome M5, spec Section 6).
Loads the calibrated + conformal model artifacts from the MLflow registry's
"Production" stage at startup and implements the LOCKED CROSS-PROJECT
CONTRACT /predict routes on top of core/serving/api_base.py's generic shell.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sqlalchemy import text

from core.explain import SHAPExplainer
from core.serving.api_base import (
    CoverageGuarantee,
    PredictionResponse,
    ServingState,
    SHAPContributor,
    build_base_router,
)
from domains.pharma.dataset_builder import CATEGORICAL_FEATURES, PharmaDatasetBuilder
from domains.pharma.plain_english import generate_summary

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTERED_MODEL_NAME = "trialoutcome_xgb_calibrated"

# M9-7: idempotent create so /predict's background write works even on an
# environment where db-init hasn't been re-run since ml.prediction_log was
# added -- identical pattern to drift_job.py's _CREATE_DRIFT_LOG_SQL.
_CREATE_PREDICTION_LOG_SQL = """
CREATE TABLE IF NOT EXISTS ml.prediction_log (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT NOT NULL,
    nct_id TEXT,
    proba NUMERIC(6,5) NOT NULL,
    threshold_decision TEXT NOT NULL,
    feature_pipeline_version TEXT NOT NULL,
    model_version INT NOT NULL,
    features_hash TEXT NOT NULL,
    conformal_low NUMERIC(6,5),
    conformal_high NUMERIC(6,5),
    top_shap_feature TEXT,
    latency_ms INT,
    created_at TIMESTAMPTZ DEFAULT now()
);
"""


class TrialFeatures(BaseModel):
    """
    Input schema for POST /api/v1/predict.

    DEVIATION FROM THE M5 SPEC'S LITERAL SCHEMA (flagged, not silent --
    see decisions.md M5 entry "TrialFeatures input schema corrected to match
    the actually-trained feature set"): the spec's `TrialFeatures` included
    `intervention_model`, which was never part of the trained feature set
    (train_pipeline.py's CATEGORICAL_FEATURES/NUMERIC_FEATURES do not include
    it -- the M1 spec-table note calling it a "bonus feature" turned out to
    be stale) and OMITTED `has_results`, which the model was actually trained
    on (rank-7 global SHAP importance -- see M4's error analysis). Serving
    with the spec's literal schema would either silently ignore a client-
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
    log_enrollment_count: float | None = None  # M9-1: accepted, ignored.
    num_primary_outcomes: int
    num_sites: int
    has_dmc: bool | None = None
    masking: str | None = None
    allocation: str | None = None
    has_results: bool
    eligibility_criteria_length: int
    exclusion_keyword_count: int
    sponsor_prior_trial_count: int
    sponsor_prior_termination_rate: float | None = None
    sponsor_class: str | None = None
    condition_name: str | None = None
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
    imputation_constants: dict[str, float]
    empirical_coverage: float
    extras: dict = field(default_factory=dict)


bundle: _PharmaModelBundle | None = None
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
    # M9-15: loaded as its own artifact (see register_model.py) instead of
    # reached for via calibrated_model.calibrated_classifiers_[0].estimator.estimator
    # -- see the git history of this line for what that looked like.
    xgb_pipeline = mlflow.sklearn.load_model(f"runs:/{mv.run_id}/raw_pipeline")

    feature_schema = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/feature_schema.json")
    condition_vocab = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/condition_vocab.json")
    # M9-8: frozen at training time (build_features()'s exact medians -- see
    # PharmaDatasetBuilder.compute_imputation_constants) and loaded here, not
    # recomputed from a single-row request at serving time. See
    # _row_from_trial_features's use of this below.
    imputation_constants = mlflow.artifacts.load_dict(
        f"runs:/{mv.run_id}/imputation_constants.json"
    )

    # xgb_pipeline (loaded above from its own "raw_pipeline" artifact) is the
    # raw fitted Pipeline([("pre", ColumnTransformer), ("clf", XGBClassifier)])
    # -- needed for SHAP (TreeExplainer wants the raw booster, not the
    # calibration wrapper).
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
        imputation_constants=imputation_constants,
        # M9-9: loaded once at startup (not recomputed per-request) so
        # coverage_guarantee.empirical reflects the TEST-split verification
        # MAPIEConformalWrapper.verify_coverage() actually measured for THIS
        # run, not a value hardcoded in the API layer.
        empirical_coverage=run.data.metrics.get("empirical_coverage", float("nan")),
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
# M9-17: /metrics -- standard request-count/latency-histogram instrumentation
# (per-route, per-status-code) in Prometheus text format. Nothing scrapes it
# in this project today (no traffic to monitor yet, same honest state M9-7's
# prediction_log entry already documents for drift) -- it exists so the
# surface is real for "what would you monitor in production" rather than a
# claim with nothing backing it.
Instrumentator().instrument(app).expose(app)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """
    Purpose: Tag every request with a uuid4 request_id (M9-7), used to
        correlate a served prediction with its ml.prediction_log row.
    Leakage guard: N/A.
    Failure mode: N/A -- request.state is per-request; nothing here can leak
        across concurrent requests.
    """
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


def _require_bundle() -> _PharmaModelBundle:
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return bundle


def _condition_bucket(condition_name: str | None, top_conditions: list[str]) -> str:
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
    Failure mode: `sponsor_prior_termination_rate=None` is filled with
        `b.imputation_constants["sponsor_prior_termination_rate"]` -- the
        actual median build_features() computed and imputed with at training
        time (frozen and loaded in _load_bundle(), see M9-8), not a fresh
        "median" of this single-row request (meaningless for n=1) and not the
        pre-M9-8 hardcoded 0.0.
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
        else b.imputation_constants["sponsor_prior_termination_rate"],
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


def _top_shap_contributors(
    row_df: pd.DataFrame, b: _PharmaModelBundle, top_n: int = 5
) -> list[dict]:
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


def _features_hash(row_df: pd.DataFrame, b: _PharmaModelBundle) -> str:
    """Deterministic sha256 of the exact feature row scored, for
    ml.prediction_log dedup/audit (M9-7) -- not the full row is logged, only
    its hash, so this is what lets two requests be recognized as scoring
    identical inputs without persisting the full feature vector per row."""
    payload = {col: _to_native(row_df.iloc[0][col]) for col in b.feature_cols}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _log_prediction_background(
    b: _PharmaModelBundle,
    request_id: str,
    nct_id: str | None,
    response: PredictionResponse,
    row_df: pd.DataFrame,
    latency_ms: int,
) -> None:
    """
    Purpose: Best-effort write of one served prediction to ml.prediction_log
        (M9-7), run as a FastAPI BackgroundTask so it adds zero latency to
        the response actually returned to the caller.
    Leakage guard: N/A -- write-only logging of an already-computed response.
    Failure mode (by design, not a gap): any exception here (DB unreachable,
        schema not yet migrated, etc.) is caught and printed to stdout, never
        re-raised -- a client must always get their prediction even if the
        prediction-log database is down. This is the deliberate opposite of
        _require_bundle()'s fail-loud contract: serving must not degrade,
        logging is allowed to.
    """
    try:
        with b.db_engine.begin() as conn:
            conn.execute(text(_CREATE_PREDICTION_LOG_SQL))
            conn.execute(
                text(
                    """
                    INSERT INTO ml.prediction_log
                        (request_id, nct_id, proba, threshold_decision,
                         feature_pipeline_version, model_version, features_hash,
                         conformal_low, conformal_high, top_shap_feature, latency_ms)
                    VALUES
                        (:request_id, :nct_id, :proba, :threshold_decision,
                         :feature_pipeline_version, :model_version, :features_hash,
                         :conformal_low, :conformal_high, :top_shap_feature, :latency_ms)
                    """
                ),
                {
                    "request_id": request_id,
                    "nct_id": nct_id,
                    "proba": response.proba,
                    "threshold_decision": response.threshold_decision,
                    "feature_pipeline_version": response.feature_pipeline_version,
                    "model_version": int(b.model_version),
                    "features_hash": _features_hash(row_df, b),
                    # ml.prediction_log's columns are conformal_low/conformal_high
                    # (unchanged, not part of the M9-9 API rename -- these are
                    # internal DB column names, not the locked HTTP contract).
                    "conformal_low": response.uncertainty_band[0],
                    "conformal_high": response.uncertainty_band[1],
                    "top_shap_feature": response.top_shap[0].feature if response.top_shap else None,
                    "latency_ms": latency_ms,
                },
            )
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        print(f"[prediction_log] write failed (request_id={request_id}): {exc}")


def _predict_from_row(row_df: pd.DataFrame, b: _PharmaModelBundle) -> PredictionResponse:
    proba_arr = b.calibrated_model.predict_proba(row_df[b.feature_cols])[:, 1]
    # M9-18: clip before anything downstream (threshold decision, SHAP
    # summary, the served response) reads it -- an isotonic calibrator can
    # legitimately output exactly 0.0 or 1.0 at the extremes of its training
    # range, and "100% termination probability" / "0% termination
    # probability" is a string no stakeholder reading this API's output will
    # accept at face value, however well-calibrated the number actually is.
    # See plain_english.py's _format_probability for the matching "at least
    # 99%" / "less than 1%" prose treatment.
    proba = float(np.clip(proba_arr[0], 0.001, 0.999))

    _, intervals = b.conformal_wrapper.predict_with_interval(row_df[b.feature_cols])
    uncertainty_band = intervals[0]

    threshold_decision = "high_risk" if proba >= b.threshold else "low_risk"
    top_shap = _top_shap_contributors(row_df, b, top_n=5)
    summary = generate_summary(top_shap, threshold_decision, proba)

    return PredictionResponse(
        proba=proba,
        uncertainty_band=uncertainty_band,
        coverage_guarantee=CoverageGuarantee(
            type="label_set",
            target=b.conformal_wrapper.target_coverage,
            empirical=b.empirical_coverage,
            note=("Guarantee is on set membership of the true label, not on the probability band."),
        ),
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
def predict(
    trial: TrialFeatures, request: Request, background_tasks: BackgroundTasks
) -> PredictionResponse:
    b = _require_bundle()
    start = time.perf_counter()
    row_df = _row_from_trial_features(trial, b)
    response = _predict_from_row(row_df, b)
    latency_ms = int((time.perf_counter() - start) * 1000)
    background_tasks.add_task(
        _log_prediction_background,
        b,
        request.state.request_id,
        None,
        response,
        row_df,
        latency_ms,
    )
    return response


@app.api_route(
    "/api/v1/predict/nct/{nct_id}", methods=["GET", "POST"], response_model=PredictionResponse
)
def predict_by_nct_id(
    nct_id: str, request: Request, background_tasks: BackgroundTasks
) -> PredictionResponse:
    """
    NOTE: the M5 spec text lists this route as POST-only, but its own
    tests/test_api_contract.py calls it with `requests.get(...)` -- an
    internal inconsistency in the milestone spec. Since the route takes no
    request body (only a path param), GET is arguably the more correct verb
    for a read-only fetch-by-ID anyway; both methods are registered so the
    literal test file passes and the documented contract (POST) still works.
    """
    b = _require_bundle()
    start = time.perf_counter()
    with b.db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT features FROM ml.training_dataset WHERE nct_id = :nct_id"),
            {"nct_id": nct_id},
        ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"nct_id '{nct_id}' not found in ml.training_dataset"
        )

    row_df = _row_from_jsonb(row[0], b)
    response = _predict_from_row(row_df, b)
    latency_ms = int((time.perf_counter() - start) * 1000)
    background_tasks.add_task(
        _log_prediction_background,
        b,
        request.state.request_id,
        nct_id,
        response,
        row_df,
        latency_ms,
    )
    return response
