-- The `ml` schema itself pre-exists on the shared PharmaPulse Postgres instance
-- this project normally targets (created out-of-band by that project), but a
-- fresh CI Postgres service container has neither `marts` nor `ml` -- IF NOT
-- EXISTS makes this file self-sufficient for both cases.
CREATE SCHEMA IF NOT EXISTS ml;

-- ml.training_dataset DDL (per 02_TRIALOUTCOME_SPEC.md Section 5).
-- Pharma-specific: primary key is nct_id, not a generic entity id.
CREATE TABLE IF NOT EXISTS ml.training_dataset (
    nct_id TEXT PRIMARY KEY,
    features JSONB,
    label BOOL,
    split TEXT CHECK (split IN ('train', 'calib', 'test')),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ml.drift_log DDL (M6 -- drift monitoring). Also created idempotently by
-- domains/pharma/monitoring/drift_job.py itself, so `make drift` works even
-- if db-init hasn't been re-run since this table was added.
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

-- ml.retrain_log DDL (M7 -- retraining trigger). Also created idempotently by
-- domains/pharma/monitoring/retrain_trigger.py itself, so `make check-drift-retrain`
-- works even if db-init hasn't been re-run since this table was added.
-- `promoted`/`promoted_at` default to FALSE/NULL and are never written by
-- retrain_trigger.py itself -- promotion is a separate, human-run action
-- (see decisions.md's M7 entry: drift is not a label-quality guarantee, so
-- nothing in this codebase auto-promotes a Staging candidate to Production).
CREATE TABLE IF NOT EXISTS ml.retrain_log (
    id SERIAL PRIMARY KEY,
    drift_report_uri TEXT,
    triggered_at TIMESTAMPTZ DEFAULT now(),
    new_run_id TEXT,
    new_feature_pipeline_version TEXT,
    current_production_version TEXT,
    version_mismatch BOOL,
    promoted BOOL DEFAULT FALSE,
    promoted_at TIMESTAMPTZ
);

-- ml.prediction_log DDL (M9-7 -- makes drift monitoring's
-- --source=prediction_log mode real instead of a proxy; see
-- domains/pharma/monitoring/drift_job.py's module docstring and
-- decisions.md M9-7 for what "real" means given this table's schema does
-- not persist the full engineered feature vector, only proba/decision/hash).
-- Also created idempotently by domains/pharma/serving/api.py's background
-- write path itself, so `/predict` works even if db-init hasn't been re-run
-- since this table was added -- same pattern ml.drift_log/ml.retrain_log use.
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

CREATE INDEX IF NOT EXISTS idx_prediction_log_created_at
    ON ml.prediction_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_prediction_log_feature_version
    ON ml.prediction_log(feature_pipeline_version);
