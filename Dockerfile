FROM python:3.11-slim
WORKDIR /app
# libgomp1: python:3.11-slim omits it, but LightGBM's compiled binary links
# against libgomp.so.1 (OpenMP runtime) and fails to import without it --
# pulled in transitively via domains/pharma/train_pipeline.py's top-level
# `from lightgbm import LGBMClassifier`, even though this API never trains
# an LGBM model itself.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# Mount mlruns/ as a volume -- do not bake model artifacts into the image
# (see decisions.md M5 entry: image vs. mlruns/ artifact lifecycle).
ENV MLFLOW_TRACKING_URI=file:./mlruns
CMD ["uvicorn", "domains.pharma.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
