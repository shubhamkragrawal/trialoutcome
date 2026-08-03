"""Fast, dependency-free unit tests for domains/pharma/plain_english.py
(Section 4a templating). No DB, no MLflow, no running API required -- these
exist to give the CI `test-unit` job something real to run (see
.github/workflows/ci.yml's comment on why tests/test_api_contract.py alone
isn't enough: it requires a live Docker container and is excluded from CI).
"""

from domains.pharma.plain_english import generate_summary


def test_generate_summary_includes_decision_and_probability():
    contributors = [
        {"feature": "log_enrollment_count", "value": 3.9, "shap_contribution": 1.2},
    ]
    summary = generate_summary(contributors, "HIGH RISK", 0.87)
    assert "HIGH RISK" in summary
    assert "87%" in summary


def test_generate_summary_uses_feature_specific_template():
    contributors = [
        {"feature": "sponsor_prior_termination_rate", "value": 0.42, "shap_contribution": 0.5},
    ]
    summary = generate_summary(contributors, "HIGH RISK", 0.9)
    assert "sponsor's prior termination rate is 42%" in summary
    assert "high relative to the training population" in summary


def test_generate_summary_falls_back_for_unknown_feature():
    contributors = [
        {"feature": "some_future_feature", "value": 1.0, "shap_contribution": 0.3},
    ]
    summary = generate_summary(contributors, "LOW RISK", 0.1)
    assert "some_future_feature is 1.0" in summary
    assert "above average" in summary


def test_generate_summary_handles_no_contributors():
    summary = generate_summary([], "LOW RISK", 0.05)
    assert "no single feature stood out as a dominant driver" in summary


def test_generate_summary_caps_at_three_reasons():
    contributors = [
        {"feature": "phase", "value": "PHASE3", "shap_contribution": 0.1},
        {"feature": "num_sites", "value": 10, "shap_contribution": 0.2},
        {"feature": "eligibility_criteria_length", "value": 500, "shap_contribution": 0.3},
        {"feature": "condition_rarity", "value": 20, "shap_contribution": 0.4},
    ]
    summary = generate_summary(contributors, "HIGH RISK", 0.6)
    assert "(1)" in summary and "(2)" in summary and "(3)" in summary
    assert "(4)" not in summary
