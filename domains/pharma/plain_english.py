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


# Human-readable renderings of the machine-readable `threshold_decision` enum.
# M9-3: the API contract field itself is UNCHANGED (`high_risk` / `low_risk` --
# it is part of the locked cross-project contract RegIntel builds against);
# only the prose embedded in `plain_english_summary` is humanized. Any other
# string (e.g. the "LOW RISK (missed)" label the error-analysis demo passes)
# falls through unchanged.
_DECISION_LABELS = {
    "high_risk": "HIGH RISK",
    "low_risk": "LOW RISK",
}


def _humanize_decision(threshold_decision: str) -> str:
    return _DECISION_LABELS.get(threshold_decision, threshold_decision)


def _partition_by_direction(
    top_shap_contributors: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Purpose: Split SHAP contributors into those that push risk UP and those
        that push risk DOWN, each ordered by absolute magnitude.
    Leakage guard: N/A.
    Failure mode: A contributor missing `shap_contribution` defaults to 0.0
        and lands in the risk-decreasing bucket. That is the conservative
        side to fail toward: a zero-signed factor is never cited as a
        *reason for* a high-risk flag, so a malformed contributor can weaken
        a summary but cannot fabricate a justification for one.
    """
    increasing = [c for c in top_shap_contributors if c.get("shap_contribution", 0.0) > 0]
    decreasing = [c for c in top_shap_contributors if c.get("shap_contribution", 0.0) <= 0]
    increasing.sort(key=lambda c: abs(c.get("shap_contribution", 0.0)), reverse=True)
    decreasing.sort(key=lambda c: abs(c.get("shap_contribution", 0.0)), reverse=True)
    return increasing, decreasing


def generate_summary(
    top_shap_contributors: list[dict], threshold_decision: str, proba: float
) -> str:
    """
    Purpose: Turn a prediction's top SHAP contributors into one
        human-readable risk-summary sentence for a non-technical (BA/DA)
        audience -- TrialOutcome spec Section 4a.
    Leakage guard: N/A -- pure string templating over already-computed
        SHAP output; touches no training data or labels.

    M9-3 CORRECTION (review section 1.6): this function previously cited the
        top-3 contributors by magnitude as "reasons for the flag" REGARDLESS
        of the sign of their SHAP contribution. A trial flagged high-risk
        could therefore cite a *good* sponsor track record as a reason for
        the high-risk flag -- incoherent in front of any stakeholder, and the
        kind of thing that discredits every other number on the page. Factors
        are now partitioned by sign: only risk-INCREASING factors justify a
        HIGH RISK flag, only risk-DECREASING factors justify a LOW RISK one,
        and the strongest factor pointing the other way is surfaced
        separately as a mitigating/watch factor rather than silently dropped
        (dropping it would overstate the model's confidence).

    Failure mode: If no contributor points in the direction the decision
        implies -- which would mean the flag disagrees with every one of its
        own top drivers -- the summary says so explicitly rather than
        inventing a justification from the wrong-signed factors. That case is
        a genuine signal that the threshold and the explanation have come
        apart, and it should be visible in the output, not smoothed over.
    """
    increasing, decreasing = _partition_by_direction(top_shap_contributors)
    is_high_risk = threshold_decision == "high_risk" or str(threshold_decision).upper().startswith(
        "HIGH"
    )

    if is_high_risk:
        supporting, opposing, opposing_label = increasing, decreasing, "One factor argues against the flag"
    else:
        supporting, opposing, opposing_label = decreasing, increasing, "One factor to watch"

    reasons = [_describe_contribution(c) for c in supporting[:3]]
    if not top_shap_contributors:
        reason_text = "-- no single feature stood out as a dominant driver"
    elif not reasons:
        reason_text = (
            "though notably none of its top factors point that way -- "
            "the flag rests on the threshold, not on any single driver"
        )
    else:
        numbered = [f"({i + 1}) {reason}" for i, reason in enumerate(reasons)]
        reason_text = "primarily because: " + ", ".join(numbered)

    summary = (
        f"This trial is flagged {_humanize_decision(threshold_decision)} "
        f"({proba:.0%} estimated termination probability) {reason_text}."
    )
    if opposing:
        summary += f" {opposing_label}: {_describe_contribution(opposing[0])}."
    return summary
