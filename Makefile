.PHONY: db-init build-dataset train audit

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
