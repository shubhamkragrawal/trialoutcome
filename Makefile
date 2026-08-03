.PHONY: db-init build-dataset train audit register-model serve test-api drift check-drift-retrain rollback

include .env
export

db-init:
	psql "host=$(POSTGRES_HOST) port=$(POSTGRES_PORT) dbname=$(POSTGRES_DB) user=$(POSTGRES_USER) password=$(POSTGRES_PASSWORD)" \
		-f domains/pharma/schema.sql

build-dataset:
	uv run python -m domains.pharma.dataset_builder

train:
	uv run python -m domains.pharma.train_pipeline

audit:
	uv run jupyter nbconvert --execute --to notebook --inplace notebooks/01_dataset_audit.ipynb

register-model:
	uv run python -m domains.pharma.register_model

serve:
	docker compose up --build

test-api:
	uv run pytest tests/test_api_contract.py -v

# NLTK_DISABLE_IMPORT_SECURITY=1: evidently imports nltk transitively, and
# nltk 2026's CWD-import security hook false-positives whenever the venv
# lives inside the project root (true here) -- see
# core/monitoring/drift_base.py's module docstring.
# M9-7: SOURCE=training (default) or SOURCE=prediction_log LOOKBACK=7 (days).
# Usage: make drift / make drift SOURCE=prediction_log LOOKBACK=14
drift:
	NLTK_DISABLE_IMPORT_SECURITY=1 uv run python -m domains.pharma.monitoring.drift_job \
		--source $(or $(SOURCE),training) $(if $(LOOKBACK),--lookback $(LOOKBACK),)

# M7: reacts to the latest ml.drift_log verdict -- retrains + registers to
# Staging on a real breach; no-ops (prints and exits 0) otherwise. Pass
# FORCE=1 to bypass the drift check (testing only).
check-drift-retrain:
	NLTK_DISABLE_IMPORT_SECURITY=1 uv run python -m domains.pharma.monitoring.retrain_trigger $(if $(FORCE),--force,)

# M7: one-command rollback (or manual Staging->Production promotion --
# see rollback.py's module docstring for why both share this command).
# Usage: make rollback VERSION=2
rollback:
	uv run python -m domains.pharma.monitoring.rollback --version $(VERSION)
