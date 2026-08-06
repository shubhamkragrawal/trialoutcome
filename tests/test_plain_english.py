"""Fast, dependency-free unit tests for domains/pharma/plain_english.py
(Section 4a templating). No DB, no MLflow, no running API required -- these
exist to give the CI `test-unit` job something real to run (see
.github/workflows/ci.yml's comment on why tests/test_api_contract.py alone
isn't enough: it requires a live Docker container and is excluded from CI).
"""

from domains.pharma.plain_english import _format_probability, generate_summary


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


# --- M9-3: sign-correctness and casing (review section 1.6) ------------------


def test_high_risk_never_cites_a_risk_decreasing_factor_as_a_reason():
    """The bug this test exists for: a HIGH RISK summary citing a *good*
    sponsor track record as a reason FOR the flag."""
    contributors = [
        # Largest magnitude, but it pushes risk DOWN -- must not be a "reason".
        {"feature": "sponsor_prior_termination_rate", "value": 0.02, "shap_contribution": -2.5},
        {"feature": "num_sites", "value": 1, "shap_contribution": 0.4},
    ]
    summary = generate_summary(contributors, "high_risk", 0.81)

    reasons_clause = summary.split("One factor argues against the flag")[0]
    assert "low relative to the training population" not in reasons_clause
    assert "the trial runs at 1 sites" in reasons_clause
    # The opposing factor is surfaced, not silently dropped.
    assert "One factor argues against the flag" in summary
    assert "sponsor's prior termination rate is 2%" in summary


def test_low_risk_cites_only_risk_decreasing_factors_as_reasons():
    contributors = [
        {"feature": "sponsor_prior_termination_rate", "value": 0.01, "shap_contribution": -1.8},
        {"feature": "num_sites", "value": 1, "shap_contribution": 0.9},
    ]
    summary = generate_summary(contributors, "low_risk", 0.06)

    reasons_clause = summary.split("One factor to watch")[0]
    assert "sponsor's prior termination rate is 1%" in reasons_clause
    assert "the trial runs at 1 sites" not in reasons_clause
    assert "One factor to watch" in summary


def test_human_readable_string_uses_uppercase_not_the_enum():
    contributors = [{"feature": "num_sites", "value": 1, "shap_contribution": 0.4}]
    high = generate_summary(contributors, "high_risk", 0.81)
    assert "HIGH RISK" in high
    assert "high_risk" not in high

    low = generate_summary(
        [{"feature": "num_sites", "value": 40, "shap_contribution": -0.4}], "low_risk", 0.04
    )
    assert "LOW RISK" in low
    assert "low_risk" not in low


def test_reasons_are_ordered_by_absolute_magnitude_within_direction():
    contributors = [
        {"feature": "num_sites", "value": 1, "shap_contribution": 0.2},
        {"feature": "condition_rarity", "value": 2, "shap_contribution": 1.5},
    ]
    summary = generate_summary(contributors, "high_risk", 0.7)
    assert summary.index("prior trials on record") < summary.index("the trial runs at 1 sites")


def test_high_risk_with_no_risk_increasing_factors_says_so():
    """Rather than inventing a justification from wrong-signed factors."""
    contributors = [
        {"feature": "num_sites", "value": 40, "shap_contribution": -0.6},
    ]
    summary = generate_summary(contributors, "high_risk", 0.55)
    assert "none of its top factors point that way" in summary
    assert "primarily because" not in summary


def test_format_probability_normal_values_unchanged():
    assert _format_probability(0.87) == "87%"
    assert _format_probability(0.5) == "50%"
    assert _format_probability(0.22) == "22%"


def test_format_probability_high_extreme_never_says_100_percent():
    """M9-18: api.py clips proba to [0.001, 0.999] before this is ever
    called, but 0.999 still rounds to "100%" under plain :.0% formatting --
    this is the fix for that display-level rounding, independent of the
    clip itself."""
    assert _format_probability(0.999) == "at least 99%"
    assert _format_probability(0.996) == "at least 99%"
    assert "100%" not in _format_probability(0.999)


def test_format_probability_low_extreme_never_says_0_percent():
    assert _format_probability(0.001) == "less than 1%"
    assert _format_probability(0.004) == "less than 1%"


def test_generate_summary_never_contains_100_percent_or_0_percent():
    contributors = [
        {"feature": "num_sites", "value": 40, "shap_contribution": 0.6},
    ]
    high = generate_summary(contributors, "high_risk", 0.999)
    low = generate_summary(contributors, "low_risk", 0.001)
    assert "100%" not in high
    assert "at least 99%" in high
    assert "0%" not in low
    assert "less than 1%" in low
