"""Pharma-domain M7 rollback procedure: make any registered version of
`trialoutcome_xgb_calibrated` the sole Production version, archiving whatever
is currently there. Run as `python -m domains.pharma.monitoring.rollback
--version N` (or `make rollback VERSION=N`).

NOTE (spec deviation, see decisions.md's M7 entry): the M7 requirements' literal
`mlflow models transition-stage --name ... --version ... --stage Production`
CLI snippet does not exist in the installed mlflow (2.22.5) -- `mlflow
models --help` lists build-docker/generate-dockerfile/predict/prepare-env/
serve/update-pip-requirements only; there is no `transition-stage`
subcommand anywhere in the mlflow CLI, because model-registry stage
transitions are a Python-client/REST-API-only operation, not exposed via
CLI. This module wraps `MlflowClient.transition_model_version_stage`
directly (exactly the call the requirements themselves specify inside this
function's body) and exposes it as a one-command Makefile target instead,
which is what actually satisfies "a one-command, tested rollback procedure."

WHY rollback_production() IS ALSO REUSED FOR MANUAL FORWARD PROMOTION:
this function is deliberately agnostic about target_version's stage before
the call -- it doesn't care whether target_version was a previous
Production version (a genuine rollback) or a freshly retrained Staging
candidate a human has reviewed and decided to promote (see
domains/pharma/monitoring/retrain_trigger.py's "Staged for review" message,
which prints this exact command). That keeps exactly one function in this
entire codebase that ever transitions a version to stage "Production" via
code -- everything else either registers to "Staging" or only reads stage
state -- which is what M7's acceptance criteria grep check for auto-promotion is actually
protecting.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTERED_MODEL_NAME = "trialoutcome_xgb_calibrated"


def _set_tracking_uri() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or f"file:{REPO_ROOT / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)


def rollback_production(target_version: int) -> None:
    """
    Purpose: Make `target_version` the sole Production version of
        trialoutcome_xgb_calibrated, archiving whichever version currently
        holds that stage. This is the ONE human-invoked escape hatch for
        both a genuine rollback (undoing a bad promotion) and a manual
        forward promotion (accepting a reviewed Staging candidate) -- see
        module docstring for why both share this single code path.
    Leakage guard: N/A.
    Failure mode: If target_version does not exist in the registry,
        MlflowClient raises a RestException loudly rather than silently
        no-oping -- a rollback command that appears to succeed but changed
        nothing would be worse than a crash here.
    """
    _set_tracking_uri()
    client = MlflowClient()

    previous = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    previous_version = previous[0].version if previous else None

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=target_version,
        stage="Production",
        archive_existing_versions=True,
    )

    if previous_version and previous_version != str(target_version):
        print(f"Archived previous Production version {previous_version}.")
    print(f"{REGISTERED_MODEL_NAME} version {target_version} is now active in Production.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M7 rollback / manual promotion procedure")
    parser.add_argument(
        "--version", type=int, required=True, help="Model version to make Production"
    )
    args = parser.parse_args()
    rollback_production(args.version)
