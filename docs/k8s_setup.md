# TrialOutcome M8 — K8s Setup

Scoped K8s deploy: this wraps **only** the `/predict` FastAPI service
(`domains/pharma/serving/api.py`). The rest of the portfolio (PharmaPulse,
other projects) stays on Docker Compose — see README's "Scoped Kubernetes
deploy" section for why.

## Prerequisites

```bash
brew install k3d hey   # k3d for the local cluster, hey for the M8 load test
```

Docker Desktop must be running (`docker ps` should succeed) — k3d nodes are
themselves Docker containers.

## 1. Create the cluster

**Deviation from the milestone brief's literal command, flagged explicitly:**
the brief's suggested command is

```bash
k3d cluster create trialoutcome-demo --agents 2
```

This alone does **not** make `mlruns/` visible inside the pods. k3d nodes are
Docker containers, not the Mac host itself — a `hostPath` volume in a pod
spec resolves inside that node container's own filesystem unless the real
host directory was bind-mounted into every node *at cluster-creation time*.
Without it, the API pods would crash-loop on startup (`_load_bundle()` in
`domains/pharma/serving/api.py` cannot find the MLflow-registered model).
The command actually used:

```bash
cd /path/to/trialoutcome   # repo root
k3d cluster create trialoutcome-demo --agents 2 \
  --volume "$(pwd)/mlruns:/Users/shubhamagrawal/Documents/MS_fall_25/MS_fall_25_uni/ds_projects/trialoutcome/mlruns@all"
```

The `@all` node-filter suffix mounts it into the server node and both agent
nodes, since the scheduler may place a pod on any of them. The container
path is deliberately identical to the repo's real absolute path — same
reason `docker-compose.yml` bind-mounts `mlruns/` at an identical absolute
path (see `decisions.md`'s M5 entry): MLflow's local file-backed tracking
store bakes each run's absolute host path into its own metadata at logging
time, so `mlflow.sklearn.load_model()` can only resolve it if the path
matches exactly, everywhere it's mounted.

Verify the mount landed before going further:

```bash
docker exec k3d-trialoutcome-demo-agent-0 \
  ls /Users/shubhamagrawal/Documents/MS_fall_25/MS_fall_25_uni/ds_projects/trialoutcome/mlruns
# expect: 0  757151125609562455  models
```

## 2. metrics-server — correction to the brief's assumption

The brief assumed metrics-server "usually" needs a separate install on k3d.
**Verified otherwise on this build:** k3d 5.9.0 (k3s `v1.35.5+k3s1`) ships
metrics-server bundled by default — `kubectl get deployment metrics-server -n
kube-system` shows it present and becoming `1/1` Ready within ~15s of
cluster creation, no extra install step needed. Still worth verifying on any
future cluster/version rather than assuming:

```bash
kubectl wait --for=condition=available --timeout=90s \
  deployment/metrics-server -n kube-system
kubectl top nodes   # should return real numbers once available
```

If it's ever missing on a different k3d/k3s version, install with:

```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

## 3. Build and import the image

```bash
docker build -t trialoutcome-api:m8 .
k3d image import trialoutcome-api:m8 -c trialoutcome-demo
```

`k3d image import` ships the image straight into every node's containerd
image store — no registry involved, and `imagePullPolicy: IfNotPresent` in
`k8s/deployment.yaml` means the kubelet won't try to pull it from anywhere
else.

## 4. Postgres credential — Secret, not plaintext, and not created by Claude

`k8s/deployment.yaml` sources `POSTGRES_PASSWORD` from a Kubernetes Secret
(`trialoutcome-postgres-secret`), not a plaintext env var — the same
credential `docker-compose.yml` gets from `.env` via `env_file:`. Per this
project's standing rule, Claude never reads the real `.env` file, so it
cannot embed the real password into any command or manifest. Two options:

**For this M8 demo** (HPA scaling / probe demo do not need real Postgres —
see below), a placeholder secret was created directly:

```bash
kubectl create secret generic trialoutcome-postgres-secret \
  --from-literal=POSTGRES_PASSWORD=changeme
```

**To exercise the real `/api/v1/predict/nct/{nct_id}` route** against the
live warehouse, run this yourself (never pasted to or run by Claude):

```bash
kubectl delete secret trialoutcome-postgres-secret
kubectl create secret generic trialoutcome-postgres-secret \
  --from-literal=POSTGRES_PASSWORD="$(grep '^POSTGRES_PASSWORD=' .env | cut -d= -f2)"
kubectl rollout restart deployment/trialoutcome-api
```

**Why the placeholder is fine for M8's actual DoD items:** SQLAlchemy's
`create_engine()` (`domains/pharma/dataset_builder.py`) is lazy — it never
opens a real connection until a query runs. `/health`, `/ready`, and
`POST /api/v1/predict` (the route used for the load test) never touch
Postgres, so the pods reach `Ready` and serve real predictions regardless of
whether the Postgres password is correct. Only `GET/POST
/api/v1/predict/nct/{nct_id}` would fail (500, connection refused/auth
error) against the placeholder.

## 5. Apply manifests

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

Verify:

```bash
kubectl get pods -o wide
kubectl wait --for=condition=Ready pod -l app=trialoutcome-api --timeout=90s
kubectl get deployment trialoutcome-api
kubectl get hpa trialoutcome-api
```

Observed on this build: both pods reach `1/1 Running` within ~14s of
`kubectl apply` (well inside the liveness probe's 15s `initialDelaySeconds`
margin — see the model-load-time measurement below).

## 6. Local demo access — port-forward, not an Ingress

```bash
kubectl port-forward svc/trialoutcome-api 8000:8000
```

Then, in another terminal:

```bash
curl localhost:8000/health
curl localhost:8000/ready
curl -X POST localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"phase":"PHASE3","log_enrollment_count":6.215,"num_primary_outcomes":2,
       "num_sites":45,"has_dmc":true,"masking":"DOUBLE","allocation":"RANDOMIZED",
       "has_results":false,"eligibility_criteria_length":2840,
       "exclusion_keyword_count":12,"sponsor_prior_trial_count":47,
       "sponsor_prior_termination_rate":0.085,"sponsor_class":"INDUSTRY",
       "condition_name":"Diabetes Mellitus, Type 2","condition_rarity":1842,
       "start_year":2023,"start_quarter":2}'
```

Verified: `/health` → `{"status":"ok",...}`, `/ready` →
`{"status":"ready","model_loaded":true,"conformal_loaded":true}`, `/predict`
→ full locked-contract response (`proba=0.146`, `threshold_decision="low_risk"`,
real SHAP contributors, plain-English summary, `feature_pipeline_version`
matching the same git hash M5/M7 verified). This is a real port-forwarded
K8s pod serving the request, not the docker-compose container.

## Model load time (used to tune the liveness probe)

Timed directly against the `trialoutcome-api:m8` image (container start →
`/ready` returning `200`, includes MLflow artifact download + model
deserialization): **~5 seconds**. `k8s/deployment.yaml`'s liveness probe
uses `initialDelaySeconds: 15` — roughly 3x margin, so liveness never fires
mid-load and kills a pod that's simply still starting. Readiness uses a
shorter `initialDelaySeconds: 5` since a `503` there only withholds traffic,
never kills the pod.

## Teardown

```bash
k3d cluster delete trialoutcome-demo
```
