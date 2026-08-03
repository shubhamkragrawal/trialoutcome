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
