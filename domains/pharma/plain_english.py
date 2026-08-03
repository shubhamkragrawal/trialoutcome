"""Pharma-domain plain-English risk-summary templating (TrialOutcome M4,
spec Section 4a). Maps a prediction's top SHAP contributors into a
human-readable sentence for a non-technical (BA/DA) audience. Pharma-
specific by design -- stays in domains/pharma/, not core/ (see
core/explain.py for the domain-agnostic SHAP plumbing this builds on).
"""

from __future__ import annotations

import math


def _sponsor_prior_termination_rate(value, contribution: float) -> str:
    rate = float(value)
    if rate >= 0.35:
        qualifier = "high relative to the training population"
    elif rate <= 0.10:
        qualifier = "low relative to the training population"
    else:
        qualifier = "roughly typical for the training population"
    return f"the sponsor's prior termination rate is {rate:.0%}, which is {qualifier}"


def _enrollment(value, contribution: float) -> str:
    # value is log1p(enrollment_count) -- invert to report the real headcount.
    enrollment = max(0, round(math.expm1(float(value))))
    if enrollment < 50:
        qualifier = "unusually small"
    elif enrollment > 1000:
        qualifier = "unusually large"
    else:
        qualifier = "within a typical range"
    return f"enrollment is {qualifier} at {enrollment:,} participants"


def _has_dmc(value, contribution: float) -> str:
    val = str(value).strip().lower()
    if val in ("true", "1", "1.0"):
        return "a Data Monitoring Committee is registered"
    if val in ("false", "0", "0.0"):
        return "no Data Monitoring Committee is registered"
    return "it is unknown whether a Data Monitoring Committee is registered"


def _condition_rarity(value, contribution: float) -> str:
    count = int(float(value))
    if count < 5:
        qualifier = "a rarely-studied condition in the training history"
    elif count < 50:
        qualifier = "a moderately-studied condition"
    else:
        qualifier = "a well-studied, common condition"
    return f"this trial's condition has {count} prior trials on record -- {qualifier}"


def _sponsor_prior_trial_count(value, contribution: float) -> str:
    count = int(float(value))
    if count < 3:
        qualifier = "a newer sponsor with a limited track record"
    else:
        qualifier = "an experienced sponsor with an established track record"
    return f"the sponsor has run {count} prior trials -- {qualifier}"


_PHASE_LABELS = {
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE2|PHASE3": "Phase 2/3",
}


def _phase(value, contribution: float) -> str:
    label = _PHASE_LABELS.get(str(value), str(value))
    return f"this is a {label} trial"


def _eligibility_criteria_length(value, contribution: float) -> str:
    words = int(float(value))
    if words < 50:
        qualifier = "unusually short"
    elif words > 400:
        qualifier = "unusually long"
    else:
        qualifier = "typical in length"
    return f"eligibility criteria are {qualifier} at {words} words"


def _num_sites(value, contribution: float) -> str:
    sites = int(float(value))
    if sites < 3:
        qualifier = "a small number of sites"
    elif sites > 50:
        qualifier = "a large, highly multi-site trial"
    else:
        qualifier = "a typical number of sites"
    return f"the trial runs at {sites} sites -- {qualifier}"


# Alias both the spec's example feature names (enrollment_count_log, has_dmc)
# and the pipeline's actual column names (log_enrollment_count, has_dmc_str)
# to the same template, so this resolves correctly regardless of which
# naming convention the caller's SHAP output actually uses.
_TEMPLATES = {
    "sponsor_prior_termination_rate": _sponsor_prior_termination_rate,
    "enrollment_count_log": _enrollment,
    "log_enrollment_count": _enrollment,
    "has_dmc": _has_dmc,
    "has_dmc_str": _has_dmc,
    "condition_rarity": _condition_rarity,
    "sponsor_prior_trial_count": _sponsor_prior_trial_count,
    "phase": _phase,
    "eligibility_criteria_length": _eligibility_criteria_length,
    "num_sites": _num_sites,
}


def _describe_contribution(contribution: dict) -> str:
    """
    Purpose: Render one SHAP contributor as a human-readable clause, via a
        feature-specific template if one exists, else a generic fallback.
    Leakage guard: N/A.
    Failure mode: If `feature` isn't in _TEMPLATES, falls back to
        "[feature] is [value] (above/below average)", using the SHAP
        contribution's sign to approximate "above/below average" -- not a
        true training-population comparison, since generate_summary's
        fixed 3-argument signature has no way to receive training stats.
        Documented here rather than silently mislabeled as a real
        population comparison.
    """
    feature = contribution["feature"]
    value = contribution["value"]
    shap_contribution = contribution.get("shap_contribution", 0.0)
    template = _TEMPLATES.get(feature)
    if template is not None:
        return template(value, shap_contribution)
    direction = "above average" if shap_contribution > 0 else "below average"
    return f"{feature} is {value} ({direction})"


def generate_summary(
    top_shap_contributors: list[dict], threshold_decision: str, proba: float
) -> str:
    """
    Purpose: Turn a prediction's top SHAP contributors into one
        human-readable risk-summary sentence for a non-technical (BA/DA)
        audience -- TrialOutcome spec Section 4a.
    Leakage guard: N/A -- pure string templating over already-computed
        SHAP output; touches no training data or labels.
    Failure mode: If top_shap_contributors is empty, returns a summary with
        no "primarily because" clause rather than raising -- an edge case
        (a prediction with no dominant driver) that should be visible in
        output, not crash the demo loop.
    """
    reasons = [_describe_contribution(c) for c in top_shap_contributors[:3]]
    if not reasons:
        reason_text = "no single feature stood out as a dominant driver"
    else:
        numbered = [f"({i + 1}) {reason}" for i, reason in enumerate(reasons)]
        reason_text = "primarily because: " + ", ".join(numbered)
    return (
        f"This trial is flagged {threshold_decision} "
        f"({proba:.0%} estimated termination probability) {reason_text}."
    )
