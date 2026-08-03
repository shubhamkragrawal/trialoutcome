"""Integration tests for the LOCKED CROSS-PROJECT CONTRACT (TrialOutcome M5,
spec Section 6). RegIntel's `trial_risk` tool depends on this exact response
shape -- these tests validate the shape, not model quality.

Requires the API to be running first: `make serve` (or a local uvicorn run),
then `make test-api`.
"""

import pytest
import requests

BASE_URL = "http://localhost:8000"

# NOTE (deviation from the M5 prompt's literal sample, flagged in
# decisions.md): `intervention_model` was dropped and `has_results` was
# added -- see domains/pharma/serving/api.py's TrialFeatures docstring for
# why (the trained model never used intervention_model; it was trained on
# has_results, which the prompt's literal payload omitted).
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


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready():
    r = requests.get(f"{BASE_URL}/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["conformal_loaded"] is True


def test_predict_response_schema():
    """Validates the LOCKED cross-project contract shape.
    RegIntel's trial_risk tool wrapper depends on exactly these fields."""
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=SAMPLE_FEATURES)
    assert r.status_code == 200
    body = r.json()
    # Locked field names -- do not rename without updating RegIntel's tool wrapper
    assert "proba" in body
    assert "conformal_interval" in body
    assert "threshold_decision" in body
    assert "top_shap" in body
    assert "plain_english_summary" in body
    assert "feature_pipeline_version" in body


def test_predict_field_types():
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=SAMPLE_FEATURES)
    body = r.json()
    assert isinstance(body["proba"], float)
    assert 0.0 <= body["proba"] <= 1.0
    assert isinstance(body["conformal_interval"], list)
    assert len(body["conformal_interval"]) == 2
    assert body["threshold_decision"] in ("high_risk", "low_risk")
    assert isinstance(body["top_shap"], list)
    assert len(body["top_shap"]) <= 5
    assert isinstance(body["plain_english_summary"], str)
    assert len(body["plain_english_summary"]) > 0
    assert isinstance(body["feature_pipeline_version"], str)


def test_predict_threshold_applied():
    """Threshold=0.22: verify decision matches proba"""
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=SAMPLE_FEATURES)
    body = r.json()
    if body["proba"] >= 0.22:
        assert body["threshold_decision"] == "high_risk"
    else:
        assert body["threshold_decision"] == "low_risk"


def test_predict_nct_found():
    # Confirmed present in ml.training_dataset (see decisions.md M5 entry).
    r = requests.get(f"{BASE_URL}/api/v1/predict/nct/NCT05062889")
    assert r.status_code == 200
    body = r.json()
    assert "proba" in body


def test_predict_nct_not_found():
    r = requests.get(f"{BASE_URL}/api/v1/predict/nct/NCT99999999")
    assert r.status_code == 404


def test_predict_missing_features():
    bad = {"phase": "PHASE3"}  # missing most features
    r = requests.post(f"{BASE_URL}/api/v1/predict", json=bad)
    assert r.status_code == 422
