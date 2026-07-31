-- ml.training_dataset DDL (per 02_TRIALOUTCOME_SPEC.md Section 5).
-- Pharma-specific: primary key is nct_id, not a generic entity id.
CREATE TABLE IF NOT EXISTS ml.training_dataset (
    nct_id TEXT PRIMARY KEY,
    features JSONB,
    label BOOL,
    split TEXT CHECK (split IN ('train', 'calib', 'test')),
    created_at TIMESTAMPTZ DEFAULT now()
);
