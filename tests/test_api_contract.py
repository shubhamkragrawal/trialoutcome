"""Fast, CI-safe rewrite of the LOCKED CROSS-PROJECT CONTRACT tests
(TrialOutcome M5, spec Section 6) against fastapi.testclient.TestClient and
the FastAPI app object directly -- no Docker container, no live Postgres, no
real MLflow registry required. See tests/test_api_contract_e2e.py for the
original Docker-based version (kept for local end-to-end verification
against the real Production model).

M9-14: the fix plan's own text asked to "mock the model bundle or use a tiny
fixture model" if loading the real Production MLflow model fails in CI (no
mlruns/ on a fresh checkout). A tiny fixture model was chosen over a mock:
`domains/pharma/serving/api._load_bundle()` is monkeypatched to return a
_PharmaModelBundle built from a REAL (if tiny) fitted XGBoost pipeline +
CalibratedClassifierCV + MAPIEConformalWrapper + SHAPExplainer, trained on
~5 lines of synthetic data mirroring register_model.py's own procedure --
not a Mock object standing in for predict_proba's return value. This means
every route under test genuinely exercises the real prediction path
(preprocessing, calibration, conformal interval, SHAP aggregation,
plain-English templating), not a hand-stubbed response shape that could
silently drift from what the real bundle actually produces. The `/predict/
nct/{id}` routes still need something standing in for `ml.training_dataset`
lookups -- a fake SQLAlchemy engine (`_FakeEngine` below) fills that role,
since there is no live Postgres in this job either.
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
import pytest
from fastapi.testclient import TestClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import domains.pharma.serving.api as api
from core.conformal import MAPIEConformalWrapper
from core.explain import SHAPExplainer
from domains.pharma.dataset_builder import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from domains.pharma.train_pipeline import _make_preprocessor

CONDITION_COLS = ["condition_other", "condition_unknown"]
FEATURE_COLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + CONDITION_COLS
THRESHOLD = 0.22
KNOWN_NCT_ID = "NCT00000001"
UNKNOWN_NCT_ID = "NCT99999999"

# NOTE (deviation from the M5 spec's literal sample, unchanged from the
# original test_api_contract.py -- see that file's history): `intervention_model`
# was dropped and `has_results` was added -- see domains/pharma/serving/api.py's
# TrialFeatures docstring for why.
SAMPLE_FEATURES = {
    "phase": "PHASE3",
    "log_enrollment_count": 6.215,
    "num_primary_outcomes": 2,
    "num_sites": 45,
    "has_dmc": True,
    "masking": "DOUBLE",
    "allocation": "RANDOMIZED",
    "has_results": False,
    "eligibility_criteria_length": 2840,
    "exclusion_keyword_count": 12,
    "sponsor_prior_trial_count": 47,
    "sponsor_prior_termination_rate": 0.085,
    "sponsor_class": "INDUSTRY",
    "condition_name": "Diabetes Mellitus, Type 2",
    "condition_rarity": 1842,
    "start_year": 2023,
    "start_quarter": 2,
}


def _synthetic_frame(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """A tiny, deterministic stand-in for a real TRAIN/CALIB split -- shaped
    exactly like PharmaDatasetBuilder.build_features()'s output (same
    columns _make_preprocessor expects), but fabricated, not queried from
    marts. Good enough to fit a real (if not remotely accurate) pipeline;
    this file tests response SHAPE and plumbing, not model quality."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "phase": rng.choice(["PHASE2", "PHASE3"], n),
            "allocation": rng.choice(["RANDOMIZED", "NON_RANDOMIZED"], n),
            "masking": rng.choice(["DOUBLE", "OPEN"], n),
            "has_dmc_str": rng.choice(["true", "false", "unknown"], n),
            "sponsor_class": rng.choice(["INDUSTRY", "OTHER"], n),
            "num_primary_outcomes": rng.integers(1, 5, n),
            "num_sites": rng.integers(1, 50, n),
            "has_results": rng.choice([True, False], n),
            "eligibility_criteria_length": rng.integers(50, 500, n),
            "exclusion_keyword_count": rng.integers(0, 20, n),
            "sponsor_prior_trial_count": rng.integers(0, 50, n),
            "sponsor_prior_termination_rate": rng.uniform(0, 0.5, n),
            "condition_rarity": rng.integers(1, 100, n),
            "start_year": rng.integers(2015, 2023, n),
            "start_quarter": rng.integers(1, 5, n),
        }
    )
    df["condition_other"] = False
    df["condition_unknown"] = True
    label = (df["sponsor_prior_termination_rate"] > 0.25).astype(int)
    return df, label


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConnection:
    """Purpose: stand in for a SQLAlchemy Connection so
    `/predict/nct/{id}` (SELECT) and the background prediction-log write
    (CREATE TABLE / INSERT, M9-7) both have something to call `.execute()`
    on without a live Postgres. Only understands the one SELECT this API
    actually issues; anything else (the prediction-log writes) is a no-op
    that returns cleanly, mirroring what a real INSERT's return value would
    never be inspected for anyway (see api.py's _log_prediction_background,
    which discards the execute() result)."""

    def __init__(self, nct_rows: dict[str, dict]):
        self._nct_rows = nct_rows

    def execute(self, stmt, params=None):
        if "SELECT features FROM ml.training_dataset" in str(stmt):
            row = self._nct_rows.get((params or {}).get("nct_id"))
            return _FakeResult((row,) if row is not None else None)
        return _FakeResult(None)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeEngine:
    """M9-14: stands in for the real SQLAlchemy engine `_load_bundle()`
    would normally build against a live Postgres -- the fixture bundle below
    has no live DB to query, so `/predict/nct/{id}` and the background
    prediction-log write (M9-7) both go through this instead."""

    def __init__(self, nct_rows: dict[str, dict]):
        self._nct_rows = nct_rows

    def connect(self):
        return _FakeConnection(self._nct_rows)

    def begin(self):
        return _FakeConnection(self._nct_rows)


def _build_fixture_bundle() -> api._PharmaModelBundle:
    x_train, y_train = _synthetic_frame(80, seed=1)
    x_calib, y_calib = _synthetic_frame(40, seed=2)

    clf = XGBClassifier(n_estimators=10, max_depth=2, random_state=42, eval_metric="logloss")
    xgb_pipeline = Pipeline([("pre", _make_preprocessor(CONDITION_COLS)), ("clf", clf)])
    xgb_pipeline.fit(x_train[FEATURE_COLS], y_train)

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(xgb_pipeline), method="isotonic"
    )
    calibrated_model.fit(x_calib[FEATURE_COLS], y_calib)

    conformal_wrapper = MAPIEConformalWrapper(target_coverage=0.9)
    conformal_wrapper.fit_conformal(calibrated_model, x_calib[FEATURE_COLS], y_calib)

    shap_explainer = SHAPExplainer(xgb_pipeline.named_steps["clf"])

    known_row = x_train.iloc[0][FEATURE_COLS].to_dict()

    return api._PharmaModelBundle(
        calibrated_model=calibrated_model,
        conformal_wrapper=conformal_wrapper,
        xgb_pipeline=xgb_pipeline,
        shap_explainer=shap_explainer,
        feature_cols=FEATURE_COLS,
        condition_cols=CONDITION_COLS,
        top_conditions=[],
        threshold=THRESHOLD,
        feature_pipeline_version="test-fixture",
        training_date="2026-08-03",
        pr_auc=0.6,
        ece=0.03,
        model_version="1",
        db_engine=_FakeEngine({KNOWN_NCT_ID: known_row}),
        imputation_constants={
            "sponsor_prior_termination_rate": float(
                x_train["sponsor_prior_termination_rate"].median()
            )
        },
        empirical_coverage=0.9,
    )


@pytest.fixture(scope="module")
def client():
    original_load_bundle = api._load_bundle
    api._load_bundle = _build_fixture_bundle
    try:
        with TestClient(api.app) as c:
            yield c
    finally:
        api._load_bundle = original_load_bundle


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["conformal_loaded"] is True


def test_predict_response_schema(client):
    """Validates the LOCKED cross-project contract shape.
    RegIntel's trial_risk tool wrapper depends on exactly these fields."""
    r = client.post("/api/v1/predict", json=SAMPLE_FEATURES)
    assert r.status_code == 200
    body = r.json()
    # Locked field names -- do not rename without updating RegIntel's tool wrapper.
    # M9-9: conformal_interval -> uncertainty_band (renamed) + coverage_guarantee
    # (added) -- locked-contract change, see decisions.md M9-9.
    assert "proba" in body
    assert "uncertainty_band" in body
    assert "coverage_guarantee" in body
    assert "threshold_decision" in body
    assert "top_shap" in body
    assert "plain_english_summary" in body
    assert "feature_pipeline_version" in body


def test_predict_field_types(client):
    r = client.post("/api/v1/predict", json=SAMPLE_FEATURES)
    body = r.json()
    assert isinstance(body["proba"], float)
    assert 0.0 <= body["proba"] <= 1.0
    assert isinstance(body["uncertainty_band"], list)
    assert len(body["uncertainty_band"]) == 2
    guarantee = body["coverage_guarantee"]
    assert guarantee["type"] == "label_set"
    assert 0.0 <= guarantee["target"] <= 1.0
    assert 0.0 <= guarantee["empirical"] <= 1.0
    assert isinstance(guarantee["note"], str) and len(guarantee["note"]) > 0
    assert body["threshold_decision"] in ("high_risk", "low_risk")
    assert isinstance(body["top_shap"], list)
    assert len(body["top_shap"]) <= 5
    assert isinstance(body["plain_english_summary"], str)
    assert len(body["plain_english_summary"]) > 0
    assert isinstance(body["feature_pipeline_version"], str)


def test_predict_threshold_applied(client):
    """Fixture bundle's threshold is THRESHOLD (0.22) -- verify decision matches proba."""
    r = client.post("/api/v1/predict", json=SAMPLE_FEATURES)
    body = r.json()
    if body["proba"] >= THRESHOLD:
        assert body["threshold_decision"] == "high_risk"
    else:
        assert body["threshold_decision"] == "low_risk"


def test_predict_nct_found(client):
    r = client.get(f"/api/v1/predict/nct/{KNOWN_NCT_ID}")
    assert r.status_code == 200
    body = r.json()
    assert "proba" in body


def test_predict_nct_not_found(client):
    r = client.get(f"/api/v1/predict/nct/{UNKNOWN_NCT_ID}")
    assert r.status_code == 404


def test_predict_missing_features(client):
    bad = {"phase": "PHASE3"}  # missing most features
    r = client.post("/api/v1/predict", json=bad)
    assert r.status_code == 422


def test_proba_is_clipped_to_avoid_100_or_0_percent(monkeypatch):
    """M9-18: an isotonic calibrator can legitimately return exactly 0.0 or
    1.0 at the extremes of its training range -- _predict_from_row must clip
    before that value reaches the response (and, downstream, plain_english's
    headline sentence). Builds its own bundle (not the module-scoped
    `client` fixture) so monkeypatching one bundle's calibrated_model
    doesn't leak into the other tests in this file."""
    bundle = _build_fixture_bundle()
    row_df = api._row_from_trial_features(api.TrialFeatures(**SAMPLE_FEATURES), bundle)

    monkeypatch.setattr(bundle.calibrated_model, "predict_proba", lambda x: np.array([[0.0, 1.0]]))
    response_high = api._predict_from_row(row_df, bundle)
    assert response_high.proba == 0.999
    assert "100%" not in response_high.plain_english_summary

    monkeypatch.setattr(bundle.calibrated_model, "predict_proba", lambda x: np.array([[1.0, 0.0]]))
    response_low = api._predict_from_row(row_df, bundle)
    assert response_low.proba == 0.001
    assert "0%" not in response_low.plain_english_summary
