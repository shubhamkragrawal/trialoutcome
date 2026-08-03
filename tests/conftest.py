"""Shared pytest fixtures for TrialOutcome's test suite.

`real_dev_state` is used only by tests/test_retrain_trigger.py,
tests/test_version_mismatch.py, and tests/test_rollback.py -- all three
exercise the real M7 retrain-trigger/rollback machinery against this
developer's actual marts-backed ml.training_dataset (via
domains/pharma/dataset_builder.py's fetch_raw()/cache) and this developer's
actual registered MLflow model registry, not a fixture (see each of those
files' own module docstring for why -- they demonstrably work, per
decisions.md's M7 entry, but need real state to run against).

That's the same reason .github/workflows/ci.yml's test-unit job excludes
tests/test_api_contract.py (needs a live Docker container CI doesn't stand
up): CI's ephemeral Postgres service gets the `ml` schema (via `make
db-init`) but never the `marts` schema these tests' retrain path reads from,
and a fresh mlruns/ store has no registered Production model to compare or
roll back against. Building a synthetic marts-schema fixture (and a way to
bootstrap a fixture Production registration without register_model.py's
real-data-calibrated reproducibility assertion) was considered and
deliberately deferred as scope beyond M7's actual ask -- see decisions.md.

Rather than silently exclude these three files from CI collection entirely,
each explicitly requests this fixture, which skips with a clear, visible
reason in CI logs when dev-environment prerequisites are missing, instead of
either failing or vanishing silently.
"""

from __future__ import annotations

import pytest
from mlflow.tracking import MlflowClient

from domains.pharma.monitoring.retrain_trigger import REGISTERED_MODEL_NAME, REPO_ROOT, _set_tracking_uri


def _has_real_dev_state() -> bool:
    _set_tracking_uri()
    client = MlflowClient()
    has_production_model = bool(client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"]))
    has_raw_cache = (REPO_ROOT / "data" / "raw_trials_cache.parquet").exists()
    return has_production_model and has_raw_cache


@pytest.fixture
def real_dev_state():
    """
    Purpose: Skip (not fail) a test that needs real dev-environment state --
        a registered Production model and a marts-backed raw-features cache
        -- neither of which a fresh CI checkout or ephemeral Postgres has.
    Leakage guard: N/A.
    Failure mode: N/A -- a clear pytest.skip with its reason printed is the
        intended behavior in an environment lacking these prerequisites, not
        a bug to fix.
    """
    if not _has_real_dev_state():
        pytest.skip(
            "requires real dev-environment state: a registered Production "
            "model version and data/raw_trials_cache.parquet -- not available "
            "on a fresh checkout or CI's ephemeral Postgres (no `marts` schema, "
            "no prior model registration). See .github/workflows/ci.yml's "
            "retrain-trigger-test job comment and decisions.md's M7 entry."
        )
