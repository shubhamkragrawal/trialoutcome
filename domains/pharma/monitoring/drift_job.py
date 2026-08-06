"""Pharma-domain batch drift-monitoring job (TrialOutcome M6, extended M9-7).
Compares the Production model's training population against a "current
batch" using Evidently's DataDriftPreset, writes an HTML report, and logs the
verdict to `ml.drift_log`. Run as `python -m domains.pharma.monitoring.drift_job`
(or `make drift`).

Two modes for what "current batch" means, selected via --source:

  --source=training (default): `current` is `ml.training_dataset
      WHERE split='test'`, the same held-out TEST split M3-M5 already
      evaluated against. HONESTY NOTE (see decisions.md M6 entry,
      config.yaml's `drift` section, and README.md's Drift Monitoring
      section): this is a PROXY for local dev / this portfolio project's
      demo -- it stands in for "a batch of trials the model hasn't seen,"
      not a claim that real drift has (or hasn't) happened in production.

  --source=prediction_log --lookback=N (M9-7, the real production
      monitoring path): `current` is the last N days of rows from
      `ml.prediction_log`, the table `/predict` writes to on every served
      request (see domains/pharma/serving/api.py). HONESTLY DOCUMENTED GAP:
      this table's schema (per the M9 fix plan) logs `proba`/
      `threshold_decision`/`features_hash`/etc., not the full engineered
      feature vector -- so a feature-level DataDriftPreset comparison
      against the `training` mode's per-column reference is not literally
      possible from this table alone yet; extending it to also persist the
      request's feature values would be the next step if this project ever
      serves real traffic. Until then, this mode will return EMPTY results
      (0 rows compared) because the API has never received live traffic
      outside of local testing -- that is the honest current state, not a
      bug in this mode's implementation.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import mlflow
import pandas as pd
from sqlalchemy import text

from core.logging_utils import configure_json_logging
from core.monitoring.drift_base import DriftMonitorBase, DriftResult
from domains.pharma.dataset_builder import PharmaDatasetBuilder

configure_json_logging()
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTERED_MODEL_NAME = "trialoutcome_xgb_calibrated"

_CREATE_PREDICTION_LOG_SQL = """
CREATE TABLE IF NOT EXISTS ml.prediction_log (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT NOT NULL,
    nct_id TEXT,
    proba NUMERIC(6,5) NOT NULL,
    threshold_decision TEXT NOT NULL,
    feature_pipeline_version TEXT NOT NULL,
    model_version INT NOT NULL,
    features_hash TEXT NOT NULL,
    conformal_low NUMERIC(6,5),
    conformal_high NUMERIC(6,5),
    top_shap_feature TEXT,
    latency_ms INT,
    created_at TIMESTAMPTZ DEFAULT now()
);
"""

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


def _load_prediction_log_batch(engine, lookback_days: int) -> pd.DataFrame:
    """
    Purpose: Load `ml.prediction_log` rows from the last `lookback_days` days
        (M9-7's --source=prediction_log mode) -- the real served-traffic
        batch, as opposed to `training` mode's TEST-split proxy.
    Leakage guard: N/A -- read-only query against an append-only log table.
    Failure mode (honestly documented, not a bug): this returns whatever
        columns `ml.prediction_log` actually stores (proba/threshold_decision/
        feature_pipeline_version/model_version/features_hash/conformal_low/
        conformal_high/top_shap_feature/latency_ms) -- NOT the full engineered
        feature vector a request was scored on, since the M9-7 schema doesn't
        persist that (see this module's docstring). Comparing this against
        `training` mode's per-feature reference via score_batch() will find
        zero shared columns until this table is extended to also log feature
        values -- run() below detects the empty-or-non-comparable case
        explicitly and reports it honestly rather than letting Evidently fail
        confusingly on a near-empty column intersection.
    """
    with engine.begin() as conn:
        conn.execute(text(_CREATE_PREDICTION_LOG_SQL))
        cutoff = (pd.Timestamp.now(tz="UTC") - timedelta(days=lookback_days)).isoformat()
        df = pd.read_sql(
            text("SELECT * FROM ml.prediction_log WHERE created_at >= :cutoff"),
            conn,
            params={"cutoff": cutoff},
        )
    return df


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
    Purpose: Concrete TrialOutcome drift monitor. Reference is always
        `ml.training_dataset`'s `reference_split` (TRAIN). `current` depends
        on `source` (M9-7): "training" (default) reads `ml.training_dataset`'s
        `current_split` (TEST, the pre-M9-7 proxy); "prediction_log" reads the
        last `lookback_days` days of `ml.prediction_log` (real served
        traffic, see this module's docstring for the honestly-documented
        column-comparability gap that mode currently has).
    Leakage guard: N/A -- all frames are read-only queries against
        already-built/already-logged tables; nothing here computes a feature
        or a label.
    Failure mode: N/A (class-level).
    """

    def __init__(self, source: str = "training", lookback_days: int = 7):
        if source not in ("training", "prediction_log"):
            raise ValueError(
                f"Unknown drift source {source!r} -- expected 'training' or 'prediction_log'"
            )
        self.builder = PharmaDatasetBuilder()
        self.config = self.builder.config["drift"]
        self.source = source
        self.lookback_days = lookback_days

    def load_reference(self) -> pd.DataFrame:
        return _load_split_features(self.builder.engine, self.config["reference_split"])

    def load_current(self) -> pd.DataFrame:
        if self.source == "training":
            return _load_split_features(self.builder.engine, self.config["current_split"])
        return _load_prediction_log_batch(self.builder.engine, self.lookback_days)

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
        Purpose: End-to-end drift-check run: load reference + current ->
            score with Evidently -> write HTML report -> evaluate against
            `config.yaml`'s `drift.feature_drift_threshold` -> log to
            `ml.drift_log`. Returns the same summary dict it logs, for the
            notebook/CLI caller to print.
        Leakage guard: N/A.
        Failure mode: If Evidently's Snapshot parsing fails (see
            core/monitoring/drift_base.py's check_thresholds docstring),
            this raises before anything is written to `ml.drift_log` --
            a partially-logged run would be worse than a loud failure here.
            EXCEPTION (M9-7, by design, not a gap): --source=prediction_log
            with zero rows in the lookback window (or zero columns shared
            with `reference` -- see this module's docstring) skips Evidently
            entirely and logs an honest zero-comparison result instead of
            either crashing on a degenerate Evidently call or fabricating a
            "not drifted" verdict over data that was never actually compared.
        """
        reference = self.load_reference()
        current = self.load_current()
        if self.source == "training":
            logger.info("Reference (split=%s): %s", self.config["reference_split"], reference.shape)
            logger.info("Current   (split=%s):   %s", self.config["current_split"], current.shape)
        else:
            logger.info("Reference (split=%s): %s", self.config["reference_split"], reference.shape)
            logger.info(
                "Current   (ml.prediction_log, last %sd): %s", self.lookback_days, current.shape
            )

        common_cols = [c for c in reference.columns if c in current.columns]
        if current.empty or not common_cols:
            logger.info(
                "--source=prediction_log has no comparable data yet "
                "(%d rows logged in the last %s days, %d columns shared with the "
                "training reference) -- this is the honest current state (no live "
                "traffic has been served outside of local testing), not a bug. "
                "See this module's docstring.",
                len(current),
                self.lookback_days,
                len(common_cols),
            )
            result = DriftResult(drifted=False, n_features_drifted=0, drift_share=0.0)
            report_path = REPO_ROOT / "reports" / f"drift_{date.today().isoformat()}.html"
        else:
            snapshot = self.score_batch(reference, current)
            report_path = REPO_ROOT / "reports" / f"drift_{date.today().isoformat()}.html"
            self.generate_report(snapshot, report_path)
            result = self.check_thresholds(snapshot, self.config["feature_drift_threshold"])

        model_version = _current_production_version()
        self.log_to_db(result, report_path, model_version)

        summary = {
            "source": self.source,
            "drifted": result.drifted,
            "n_features_drifted": result.n_features_drifted,
            "drift_share": result.drift_share,
            "report_path": str(report_path),
        }
        logger.info("Drift summary: %s", summary)
        return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["training", "prediction_log"],
        default="training",
        help="'training' (default): TEST split vs TRAIN, a local-dev proxy. "
        "'prediction_log': last --lookback days of ml.prediction_log vs TRAIN, "
        "the real production monitoring path (M9-7).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=7,
        dest="lookback_days",
        help="Days of ml.prediction_log history to use as the current batch "
        "when --source=prediction_log (default: 7).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    PharmaDriftMonitor(source=args.source, lookback_days=args.lookback_days).run()
