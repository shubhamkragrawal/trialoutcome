"""Pharma-domain batch drift-monitoring job (TrialOutcome M6). Compares the
Production model's training population against a proxy "current batch" using
Evidently's DataDriftPreset, writes an HTML report, and logs the verdict to
`ml.drift_log`. Run as `python -m domains.pharma.monitoring.drift_job`
(or `make drift`).

HONESTY NOTE (see decisions.md M6 entry, config.yaml's `drift` section, and
README.md's Drift Monitoring section): in production this job would score
last week's newly-registered trials against the training population. No live
scoring traffic exists yet -- the API has never been called outside of local
testing -- so `current` here is `ml.training_dataset WHERE split='test'`, the
same held-out TEST split M3-M5 already evaluated against. This is a stand-in
for "a batch of trials the model hasn't seen," not a claim that real drift
has (or hasn't) happened in production.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
from sqlalchemy import text

from core.monitoring.drift_base import DriftMonitorBase, DriftResult
from domains.pharma.dataset_builder import PharmaDatasetBuilder

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTERED_MODEL_NAME = "trialoutcome_xgb_calibrated"

_CREATE_DRIFT_LOG_SQL = """
CREATE TABLE IF NOT EXISTS ml.drift_log (
    id SERIAL PRIMARY KEY,
    run_date DATE,
    drift_share FLOAT,
    n_features_drifted INT,
    drifted BOOL,
    report_path TEXT,
    model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
"""


def _load_split_features(engine, split: str) -> pd.DataFrame:
    """
    Purpose: Load one split of `ml.training_dataset` and flatten its
        `features` JSONB column into a plain DataFrame -- identical parsing
        logic to `notebooks/04_shap_analysis.ipynb`'s
        `load_from_training_dataset`, reused here rather than reimplemented.
    Leakage guard: N/A -- read-only against an already-built, already-split
        table; no split boundary is being crossed by this query itself.
    Failure mode: If a future row's `features` JSONB is missing a key another
        row has, `pd.json_normalize` fills the gap with NaN rather than
        raising -- Evidently treats a mostly-NaN column as its own drift
        signal (an increase in missingness IS a form of drift), which is the
        correct behavior here, not a bug to guard against.
    """
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT features FROM ml.training_dataset WHERE split = :split"),
            conn,
            params={"split": split},
        )
    return pd.json_normalize(df["features"])


def _current_production_version() -> str:
    """
    Purpose: Look up the currently-registered Production model version, so
        every `ml.drift_log` row records which model this drift check was
        actually evaluating against.
    Leakage guard: N/A.
    Failure mode: If no version is currently in the "Production" stage
        (e.g. a fresh environment before `make register-model` has ever run),
        this returns "unknown" rather than raising -- the drift job's own
        job (checking feature distributions) doesn't depend on a registered
        model existing, so it shouldn't hard-fail over a missing tag.
    """
    try:
        # NOTE: respects MLFLOW_TRACKING_URI if set, exactly like
        # domains/pharma/serving/api.py's _load_bundle() -- otherwise this
        # would always resolve to THIS repo checkout's own mlruns/ path
        # regardless of what a caller's environment points at (verified
        # while building this: without this line, a CI run against a fresh
        # checkout still happens to work by accident, since no mlruns/
        # exists there, but a local test run against a throwaway tracking
        # dir would silently read the real dev mlruns/ instead).
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or f"file:{REPO_ROOT / 'mlruns'}"
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        return versions[0].version if versions else "unknown"
    except Exception:
        return "unknown"


class PharmaDriftMonitor(DriftMonitorBase):
    """
    Purpose: Concrete TrialOutcome drift monitor -- reference and current
        batch both come from `ml.training_dataset`, split by the `split`
        column per `config.yaml`'s `drift.reference_split`/`current_split`.
    Leakage guard: N/A -- both frames are read-only queries against an
        already-built table; nothing here computes a feature or a label.
    Failure mode: N/A (class-level).
    """

    def __init__(self):
        self.builder = PharmaDatasetBuilder()
        self.config = self.builder.config["drift"]

    def load_reference(self) -> pd.DataFrame:
        return _load_split_features(self.builder.engine, self.config["reference_split"])

    def load_current(self) -> pd.DataFrame:
        return _load_split_features(self.builder.engine, self.config["current_split"])

    def log_to_db(self, result: DriftResult, report_path: Path, model_version: str) -> None:
        """
        Purpose: Persist one row per drift-job run to `ml.drift_log`,
            creating the table first if it doesn't exist yet (so `make
            drift` works on a fresh environment without a separate db-init
            step, even though `schema.sql` also defines this table for
            `make db-init` consistency).
        Leakage guard: N/A.
        Failure mode: N/A -- a plain insert; a connection failure raises
            loudly rather than silently dropping the log row.
        """
        with self.builder.engine.begin() as conn:
            conn.execute(text(_CREATE_DRIFT_LOG_SQL))
            conn.execute(
                text(
                    """
                    INSERT INTO ml.drift_log
                        (run_date, drift_share, n_features_drifted, drifted, report_path, model_version)
                    VALUES (:run_date, :drift_share, :n_features_drifted, :drifted, :report_path, :model_version)
                    """
                ),
                {
                    "run_date": date.today().isoformat(),
                    "drift_share": result.drift_share,
                    "n_features_drifted": result.n_features_drifted,
                    "drifted": result.drifted,
                    "report_path": str(report_path),
                    "model_version": model_version,
                },
            )

    def run(self) -> dict:
        """
        Purpose: End-to-end M6 batch job: load reference + current ->
            score with Evidently -> write HTML report -> evaluate against
            `config.yaml`'s `drift.feature_drift_threshold` -> log to
            `ml.drift_log`. Returns the same summary dict it logs, for the
            notebook/CLI caller to print.
        Leakage guard: N/A.
        Failure mode: If Evidently's Snapshot parsing fails (see
            core/monitoring/drift_base.py's check_thresholds docstring),
            this raises before anything is written to `ml.drift_log` --
            a partially-logged run would be worse than a loud failure here.
        """
        reference = self.load_reference()
        current = self.load_current()
        print(f"Reference (split={self.config['reference_split']}): {reference.shape}")
        print(f"Current   (split={self.config['current_split']}):   {current.shape}")

        snapshot = self.score_batch(reference, current)
        report_path = REPO_ROOT / "reports" / f"drift_{date.today().isoformat()}.html"
        self.generate_report(snapshot, report_path)

        result = self.check_thresholds(snapshot, self.config["feature_drift_threshold"])
        model_version = _current_production_version()
        self.log_to_db(result, report_path, model_version)

        summary = {
            "drifted": result.drifted,
            "n_features_drifted": result.n_features_drifted,
            "drift_share": result.drift_share,
            "report_path": str(report_path),
        }
        print(f"Drift summary: {summary}")
        return summary


if __name__ == "__main__":
    PharmaDriftMonitor().run()
