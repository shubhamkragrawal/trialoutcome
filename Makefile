.PHONY: db-init build-dataset train audit register-model serve test-api drift check-drift-retrain rollback export-requirements

include .env
export

# M9-13: requirements.txt is a generated courtesy export for non-uv tooling,
# not a hand-maintained source of truth -- uv.lock is that. Nothing in this
# repo's own Dockerfile or CI reads requirements.txt anymore (both use
# `uv sync --frozen` directly against uv.lock); re-run this after any
# pyproject.toml dependency change if you want the exported file to stay
# current. `--no-dev` matches the servable dependency set (see pyproject.toml's
# [project.dependencies] comment), not the full dev environment.
export-requirements:
	uv export --no-dev --no-hashes --format requirements.txt > requirements.txt

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

# M9-14: real Docker container + real Production model required first --
# `make serve` (separate terminal), then `make test-api`. The fast,
# CI-run TestClient version is just tests/test_api_contract.py, part of the
# regular `pytest tests/` run -- no separate make target needed for it.
test-api:
	uv run pytest tests/test_api_contract_e2e.py -v

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
