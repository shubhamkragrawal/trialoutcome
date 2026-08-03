"""Domain-agnostic FastAPI shell: the LOCKED CROSS-PROJECT CONTRACT response
schema, generic liveness/readiness state tracking, and the /health + /ready
routes every domain implementation shares. No pharma column names, pharma
thresholds, or pharma model paths belong in this file -- see
domains/pharma/serving/api.py for the concrete TrialOutcome implementation
(model loading, /predict routes, SHAP, plain-English summaries).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel


class SHAPContributor(BaseModel):
    feature: str
    value: Any
    shap_contribution: float


class CoverageGuarantee(BaseModel):
    """
    Purpose: Make explicit what `uncertainty_band`'s conformal coverage
        target actually applies to (M9-9) -- MAPIE's guarantee is on SET
        MEMBERSHIP of the true label (was the true class inside the
        predicted {0}/{1}/{0,1} set at the target rate), not on the
        probability band itself. Before M9-9 this field didn't exist and the
        band was named `conformal_interval`, which a reader could reasonably
        (but incorrectly) assume meant a coverage-guaranteed probability
        interval.
    Leakage guard: N/A.
    Failure mode: N/A (plain value container); `empirical` is measured once
        per training run via MAPIEConformalWrapper.verify_coverage() on TEST
        and loaded from that run's logged `empirical_coverage` metric, not
        recomputed at serving time.
    """

    type: str
    target: float
    empirical: float
    note: str


class PredictionResponse(BaseModel):
    """LOCKED CROSS-PROJECT CONTRACT -- see 02_TRIALOUTCOME_SPEC.md Section 6.
    RegIntel's `trial_risk` tool is built against this exact field set and
    these exact names. Do not rename/add/remove fields without flagging it
    and updating RegIntel's tool wrapper in lockstep.

    M9-9 (locked-contract change, coordinated with 01_REGINTEL_SPEC.md in the
    same session): `conformal_interval` renamed to `uncertainty_band`, and
    `coverage_guarantee` added, documenting that the coverage target/empirical
    numbers apply to label-set membership, not to the probability band a
    reader would otherwise assume `uncertainty_band` promises. See
    decisions.md M9-9 and 02_TRIALOUTCOME_SPEC.md Section 12 for the
    rationale.
    """

    proba: float
    uncertainty_band: tuple[float, float]
    coverage_guarantee: CoverageGuarantee
    threshold_decision: str  # "high_risk" or "low_risk"
    top_shap: list[SHAPContributor]
    plain_english_summary: str
    feature_pipeline_version: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str


class ReadyResponse(BaseModel):
    status: str  # "ready" or "not_ready"
    model_loaded: bool
    conformal_loaded: bool


@dataclass
class ServingState:
    """
    Purpose: Track whether the domain-specific model + conformal wrapper are
        actually loaded in memory, independent of whether the process is
        alive -- the distinction /health (liveness) vs /ready (readiness)
        exists to make.
    Leakage guard: N/A.
    Failure mode: N/A (plain state container); domain code is responsible
        for flipping model_loaded/conformal_loaded to True only after the
        corresponding artifact has actually finished loading, not before.
    """

    model_loaded: bool = False
    conformal_loaded: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def build_base_router(state: ServingState) -> APIRouter:
    """
    Purpose: Build the /health and /ready routes shared by every domain
        implementation of this serving shell.
    Leakage guard: N/A.
    Failure mode: /health always returns 200 once the process can serve HTTP
        at all -- even before the model has loaded -- by design: a K8s
        liveness probe hitting /health must never kill a pod that is still
        loading a large model artifact. /ready is the only route that
        reflects model_loaded/conformal_loaded state, and returns 503 (not
        200-with-a-false-field) when either is not yet loaded, so a
        readiness probe or load balancer can correctly withhold traffic
        without parsing the response body.
    """
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", timestamp=datetime.now(UTC).isoformat())

    @router.get("/ready", response_model=ReadyResponse)
    def ready(response: Response) -> ReadyResponse:
        is_ready = state.model_loaded and state.conformal_loaded
        if not is_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(
            status="ready" if is_ready else "not_ready",
            model_loaded=state.model_loaded,
            conformal_loaded=state.conformal_loaded,
        )

    return router
