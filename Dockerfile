# M9-19: pinned by digest, not the floating `python:3.11-slim` tag -- a tag
# can point at a different image tomorrow (a base-image supply-chain risk);
# a digest can't. Re-pull and update this if a security patch to the base
# image is ever needed: `docker pull python:3.11-slim && docker inspect
# python:3.11-slim --format='{{index .RepoDigests 0}}'`.
FROM python@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93
WORKDIR /app

# libgomp1: XGBoost's compiled binary links against libgomp.so.1 (OpenMP
# runtime), which python:3.11-slim omits -- this API always serves an
# XGBoost pipeline (register_model.py hardcodes the champion family), so
# this is a real serving-time need, not a training-only one.
# M9-13 correction: this comment previously blamed LightGBM, imported
# transitively via domains/pharma/train_pipeline.py's top-level `from
# lightgbm import LGBMClassifier` -- that import path is gone (api.py no
# longer imports train_pipeline.py at all, see dataset_builder.py's
# CATEGORICAL_FEATURES comment and decisions.md M9-13), so LightGBM itself
# is no longer part of this image regardless. libgomp1 stays because
# XGBoost needs it on its own.
# curl: required by the HEALTHCHECK below (M9-19).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# M9-19: run the server as a non-root user, not the image's default root.
RUN adduser --disabled-password --gecos "" appuser

# M9-13: installs uv itself (build-time tooling only, matching
# .github/workflows/ci.yml's `pip install uv` convention) so the serving
# image can be built the same way CI validates the lockfile, instead of a
# second, potentially-drifting `pip install -r requirements.txt` path.
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
# M9-13: --no-dev installs [project.dependencies] only -- the serving group
# (fastapi/uvicorn/pydantic/scikit-learn/xgboost/sqlalchemy/psycopg2-binary/
# mlflow/mapie/numpy/pandas/shap/pyyaml/python-dotenv/
# prometheus-fastapi-instrumentator; see pyproject.toml's own comment).
# jupyterlab/optuna/evidently/matplotlib*/seaborn/nbconvert/ipykernel/
# lightgbm/pytest/ruff/mypy -- the [dependency-groups] dev set -- never land
# in this image. (*mlflow itself pulls in matplotlib transitively --
# unavoidable without dropping mlflow, see decisions.md M9-13's measured
# image-size delta.) --frozen refuses to run if uv.lock has drifted from
# pyproject.toml, matching every other uv.lock-consuming CI job in this repo.
RUN uv sync --frozen --no-dev

COPY . .
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
# Mount mlruns/ as a volume -- do not bake model artifacts into the image
# (see decisions.md M5 entry: image vs. mlruns/ artifact lifecycle).
ENV MLFLOW_TRACKING_URI=file:./mlruns
ENV PATH="/app/.venv/bin:${PATH}"

# M9-19: liveness probe for orchestrators (Docker/K8s) that don't already
# poll /health themselves -- /health always returns 200 once the process can
# serve HTTP at all (see core/serving/api_base.py's build_base_router
# docstring), so this only ever fails if the server process itself is wedged
# or dead, never merely "model still loading" (that's what /ready is for).
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "domains.pharma.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
