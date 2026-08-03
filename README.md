# TrialOutcome 🎯

> Will this clinical trial finish? Calibrated, uncertainty-aware prediction with honest evaluation.

TrialOutcome predicts whether a Phase 2/3 clinical trial will complete or terminate
(terminate/withdraw/suspend), using only information available at trial start. It's the
most domain-agnostic project in a 6-project portfolio (see
[`ai_portfolio/00_PORTFOLIO_OVERVIEW.md`](ai_portfolio/00_PORTFOLIO_OVERVIEW.md)) — the
`core/` vs `domains/pharma/` split throughout this repo is a deliberate demonstration that
temporal-safe tabular outcome prediction (calibration, conformal prediction, drift
monitoring, CI) is a reusable pattern, not a pharma-specific one.

**Status: all 8 milestones complete (M1–M8).** Dataset builder + temporal split →
Optuna/MLflow model selection → calibration + cost-based threshold → SHAP + error
analysis → conformal prediction + FastAPI/Docker → drift monitoring + CI → automated
retraining trigger + rollback → scoped Kubernetes deploy. Full milestone-by-milestone
log: [`decisions.md`](decisions.md); spec with per-milestone Definition of Done:
[`ai_portfolio/02_TRIALOUTCOME_SPEC.md`](ai_portfolio/02_TRIALOUTCOME_SPEC.md).

## Headline numbers

| Metric | Value |
|---|---|
| PR-AUC, temporal test (XGBoost, CV-selected, uncalibrated) | **0.8878** |
| PR-AUC, temporal test (same model, after isotonic calibration) | 0.8775 |
| PR-AUC, random split (same features/model — leakage contrast, see below) | 0.6823 |
| ROC-AUC, temporal test (calibrated production model) | 0.9178 |
| ECE, after isotonic calibration (TEST) | **0.0238** (raw: 0.1897) |
| Conformal coverage (target 90%) | **94.6%** |
| Cost-optimal threshold (5:1 FN:FP cost matrix) | **0.22** |
| Majority-class baseline PR-AUC (temporal test) | 0.3167 |

Calibration and calibrated-model PR-AUC differ slightly (0.8878 vs 0.8775) because isotonic
regression is a monotonic but non-linear remapping of scores, which can shuffle tie-breaks
near the decision boundary — the calibrated number is what the Production model and this
README's other calibration/conformal/threshold numbers all use.

## The leakage experiment ← interviewers will read this

Trained the identical `LogisticRegression` pipeline (same features, same hyperparameters)
under two split strategies:

- **Temporal split** (train < 2020, calib 2020–2022, test ≥ 2022): PR-AUC **0.8657**
- **Random split** (60/20/20 shuffle, same proportions): PR-AUC **0.6823**

The temporal split scores *higher*, not lower — the opposite of the naive "random split
leaks future sponsor history and inflates performance" story this experiment was designed
to test for. Two confounds explain the gap, and neither is leakage:

1. **Base-rate confound.** The temporal test set is entirely 2022+, where termination rates
   have risen to ~31.7% (see the Drift Monitoring section below); the random test set mirrors
   the whole-dataset average (~20%). PR-AUC is mechanically higher against a higher-prevalence
   test set for an equally-good model.
2. **History-richness confound.** The random test set spans the entire 1990–2026 window,
   including early trials with thin point-in-time sponsor/condition history; the temporal
   test set is entirely 2022+, where 30+ years of history has already accumulated, making
   those predictions systematically easier regardless of split methodology.

A **controlled ablation** isolates the actual leakage mechanism directly: fix one test
window (2020–2022), and compare a same-size "honest" training set (only rows strictly
before the window) against a "leaky" one (allowed to include rows *after* the window). If
random-split leakage were real, the leaky model should win. It doesn't — across all three
model families checked (LogReg, XGBoost, LightGBM), the honest/leaky PR-AUC delta is
**≤0.002**, noise-level:

| Model | Honest PR-AUC | Leaky PR-AUC | Delta |
|---|---|---|---|
| LogReg | 0.785 | 0.786 | +0.0006 |
| XGBoost | 0.8031 | 0.8042 | +0.0011 |
| LightGBM | 0.8033 | 0.8049 | +0.0016 |

**Why it matters:** this is a property of *this specific feature set* — every feature
(sponsor history, condition rarity) is computed via a point-in-time self-join
(`hist.start_date < t.start_date`) in `domains/pharma/dataset_builder.py`, so no row's
features ever encode information from after its own start date. That's the leakage guard
working as intended, not proof that random splits are safe in general — a future feature
computed carelessly (e.g. a static lifetime aggregate instead of a point-in-time join) would
still leak hard under a random split, and this same controlled-ablation methodology is
exactly how you'd catch it. The temporal split stays mandatory regardless of this negative
result. Full writeup: `notebooks/02_leakage_demo.ipynb`, `decisions.md`'s M1/M2 entries.

**Documentation note:** `ai_portfolio/02_TRIALOUTCOME_SPEC.md`'s M1 milestone row states the
random-split PR-AUC as "0.891" — that figure doesn't match either `decisions.md`'s M1
finding or `notebooks/02_leakage_demo.ipynb`'s actual executed output (0.6823), both of which
agree with each other. Flagged here rather than silently reconciled; the notebook's number is
the one used throughout this README since it's the directly-executed artifact.

## Try it

```bash
docker compose up
curl -X POST localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
        "phase": "PHASE3",
        "log_enrollment_count": 6.215,
        "num_primary_outcomes": 2,
        "num_sites": 45,
        "has_dmc": true,
        "masking": "DOUBLE",
        "allocation": "RANDOMIZED",
        "has_results": false,
        "eligibility_criteria_length": 2840,
        "exclusion_keyword_count": 12,
        "sponsor_prior_trial_count": 47,
        "sponsor_prior_termination_rate": 0.085,
        "sponsor_class": "INDUSTRY",
        "condition_name": "Diabetes Mellitus, Type 2",
        "condition_rarity": 1842,
        "start_year": 2023,
        "start_quarter": 2
      }'

curl localhost:8000/api/v1/predict/nct/NCT05062889
```

Both return the locked cross-project contract shape:
`{proba, conformal_interval, threshold_decision, top_shap, plain_english_summary, feature_pipeline_version}`
— this exact shape is what RegIntel's `trial_risk` tool wrapper (Project 4) is built against.

## Model decisions worth reading

Full log: [`decisions.md`](decisions.md). Three entries an interviewer would find most
interesting:

- **Cost-optimal threshold = 0.22, not F1-max (0.37) or default (0.5)** — chosen by
  minimizing expected cost against the pharma domain's 5:1 FN:FP cost ratio, not by
  optimizing a metric that treats a missed termination and a false alarm as equally costly.
- **Conformal coverage 94.6% vs a 90% target** — MAPIE's LAC (least-ambiguous-set) conformal
  score is finite-sample conservative at this calibration-set size; a comfortable pass, not a
  razor-thin one, with per-request interval widths that still vary meaningfully by predicted
  probability (`docs/conformal_width_vs_proba.png`).
- **`FrozenEstimator`/MAPIE API changes** — this project's pinned `scikit-learn==1.9.0` and
  `mapie==1.4.1` both shipped breaking API changes since most tutorials were written
  (`cv="prefit"` → `sklearn.frozen.FrozenEstimator`; `MapieClassifier(method="score")` →
  `SplitConformalClassifier(conformity_score="lac")`) — resolved by inspecting the installed
  package directly rather than downgrading to match stale documentation.

## Error analysis

Full essay: [`docs/error_analysis.md`](docs/error_analysis.md). Three failure themes across
the 20 worst false negatives (all share one precondition: `sponsor_prior_termination_rate <
6%`):

1. **"Mega-sponsors"** (5/20) — a sponsor with hundreds of prior trials has a termination
   rate averaged over enough history that one failing trial barely moves it.
2. **Single/near-single-site trials** (9/20) — `num_sites` nudges risk up but stays modest;
   the feature set has no direct site-execution-risk signal.
3. **No individually extreme feature** (9/20) — **the honest feature ceiling.** These trials
   have no outlier design, sponsor, or site profile in any feature group this project builds;
   whatever drove termination (drug efficacy, funding withdrawal, protocol amendment) simply
   isn't observed by any feature here. This is the current feature set's explanatory floor,
   not a modeling bug.

## Calibration

Reliability curve (before vs after, isotonic vs Platt): `notebooks/03_calibration.ipynb`.

- ECE before calibration (raw XGBoost, TEST): **0.1897**
- ECE after isotonic calibration (TEST): **0.0238**
- ECE after Platt scaling (TEST, for comparison): 0.0405 (isotonic won)

Isotonic beat Platt because it makes no parametric shape assumption — it can correct
whatever miscalibration pattern the raw XGBoost scores actually have, while Platt is
constrained to a single-parameter sigmoid family. The fitted calibrator was evaluated on the
held-out TEST split (never used to fit or select it) specifically so a near-zero CALIB-split
ECE couldn't be mistaken for a calibrator that memorized its own fitting data.

## Drift monitoring

`domains/pharma/monitoring/drift_job.py` runs Evidently's `DataDriftPreset` comparing the
Production model's training population (`ml.training_dataset WHERE split='train'`,
n=66,105) against a current batch (`WHERE split='test'`, n=5,700), writes an HTML report to
`reports/`, and logs the verdict to `ml.drift_log`.

**Honesty note:** in production, "current batch" would be last week's newly-registered
trials scored by the live API. No live scoring traffic exists yet, so the held-out TEST
split is used as a stand-in for "a batch the model hasn't seen" — this validates the
monitoring *mechanism*, not a real production drift claim.

**Latest run:** 11/38 features drifted (`drift_share=0.289`), below the
`feature_drift_threshold=0.5` dataset-level alert — **not flagged**. The top-5 most-drifted
features are `start_year`, `condition_rarity`, `sponsor_prior_trial_count`,
`eligibility_criteria_length`, and `log_enrollment_count` — every one is either a temporal
feature (drifted by construction: train is entirely pre-2020, test is entirely 2022+), a
cumulative point-in-time count that mechanically grows over calendar time, or a real secular
trend in trial design over 30+ years. None of this is surprising; it's exactly what you'd
expect comparing a pre-2020 population to a 2022+ batch. Full writeup with per-feature
scores: `notebooks/06_drift_report.ipynb`.

**Label-drift stretch check (PSI on the rolling termination base rate): attempted, not
completed meaningfully — two distinct structural reasons, not a coding failure.**

1. The literal construction (bins = individual `start_year` values, TRAIN years =
   "expected," TEST years = "actual") is degenerate for *any* dataset built with this
   project's temporal split: TRAIN spans 1990–2019, TEST spans 2022–2026, and the split is
   defined by `start_date` — so TRAIN and TEST share **zero** `start_year` bins by
   construction, not because of anything specific to this data.
2. A reinterpretation (bin the yearly termination-*rate value* into deciles instead of
   calendar years, so bins can overlap) is technically computable but wildly unstable: PSI
   ranges from **0.64** (3 bins) to **11.30** (10 bins) depending purely on bin-count choice,
   because there are only 30 TRAIN "observations" and 5 TEST "observations" (one rate per
   `start_year`) to bin — nowhere near enough for PSI's binning approach to be a stable
   statistic at this granularity.

**What IS real** (directly from the data, not from PSI): TRAIN termination rate **17.8%**
vs TEST termination rate **31.7%** — a genuine ~14-point absolute increase, already
documented in `decisions.md`'s M1 finding and already accounted for by evaluating
calibration and the cost-optimal threshold on TEST rather than assuming they'd transfer from
TRAIN's lower base rate. Full attempt, both bin-count sensitivity tables, and the reasoning
above: `notebooks/06_drift_report.ipynb`.

**How to run:** `make drift`.

## Scoped Kubernetes deploy (M8)

**Explicitly scoped:** K8s wraps *only* the `/predict` FastAPI service
(`domains/pharma/serving/api.py`). The rest of the portfolio — PharmaPulse, every other
project — stays on Docker Compose. This is deliberate, not an oversight: K8s earns its
vocabulary honestly on the one service where autoscaling/probes are a real concern, not
because the whole platform needs an orchestrator it doesn't yet require.

- **Cluster:** local `k3d` (2 agent nodes), full setup + every command documented in
  [`docs/k8s_setup.md`](docs/k8s_setup.md), including two corrections to the milestone
  brief's own assumptions worth reading if you're evaluating this section: (1) a plain
  `k3d cluster create` does **not** make `mlruns/` visible inside pods — the real host
  directory has to be bind-mounted into every k3d node at cluster-creation time via
  `--volume ...@all`; (2) k3d 5.9.0 ships `metrics-server` bundled by default (verified
  directly), contradicting the brief's assumption that it usually needs a separate install.
- **Manifests:** [`k8s/deployment.yaml`](k8s/deployment.yaml) (2 replicas, resource
  requests/limits, hostPath `mlruns/` mount, distinct liveness/readiness probes),
  [`k8s/service.yaml`](k8s/service.yaml) (ClusterIP), [`k8s/hpa.yaml`](k8s/hpa.yaml)
  (CPU-based autoscaling, 2→4 replicas).
- **HPA scaling — real result, not assumed:** a 60-second `hey` load test at concurrency
  20 against `/api/v1/predict` drove CPU from 1% to 100% of the 50% target and the HPA
  scaled the deployment **2 → 4 pods** (hit `maxReplicas`), with zero dropped requests
  across all 1,796 requests during the scale-up. Real, unedited `kubectl` output:
  [`docs/k8s_load_test_output.txt`](docs/k8s_load_test_output.txt).
- **Probe distinction — real result, not hypothetical:** `/health` (liveness) and
  `/ready` (readiness) were demonstrated gating traffic independently — a pod held
  `Running` + `NotReady` for 57+ seconds with zero restarts, while its actual `/ready`
  endpoint (port-forwarded directly, bypassing the Service) reported `ready: true` the
  entire time. Full write-up, including why the milestone brief's literal suggested
  mechanism doesn't work against this app's real (crash-on-load-failure) lifespan code
  and what was used instead: [`docs/k8s_probe_demo.md`](docs/k8s_probe_demo.md).

## What I'd change at production scale

Honest gaps in this demo, not hidden:

- **Ingress.** This demo uses `kubectl port-forward` for local access — no real Ingress
  controller, no TLS termination, no path-based routing. Production needs an actual
  Ingress (nginx-ingress, Traefik configured properly, or a cloud load balancer) sitting
  in front of the Service.
- **Secrets management.** `POSTGRES_PASSWORD` is sourced from a Kubernetes `Secret`
  rather than a plaintext env var in the manifest — better than nothing, but a raw K8s
  `Secret` is only base64-encoded, not encrypted at rest, by default. Production needs a
  real secrets store (HashiCorp Vault, K8s Secrets with encryption-at-rest enabled, or a
  cloud provider's secret manager — AWS Secrets Manager / GCP Secret Manager) with
  rotation, not a static value created once by `kubectl create secret`.
- **The hostPath `mlruns/` volume — flagged explicitly, does not generalize.** This only
  works because k3d is a single logical host (every "node" is a Docker container on the
  same machine, all bind-mounted to the same real directory at cluster-creation time —
  see `docs/k8s_setup.md`). A real multi-node cluster has no shared local filesystem
  across nodes — a pod scheduled onto a different node would see an empty or missing
  directory. Production needs either a `PersistentVolumeClaim` backed by real network
  storage (EBS/GCE PD/NFS), or — better, and consistent with the same limitation already
  flagged in M5's `decisions.md` for `docker-compose.yml`'s identical hostPath-style bind
  mount — a remote MLflow tracking server with S3/GCS-backed artifact storage, so artifact
  URIs are storage keys, not host filesystem paths, and any pod on any node resolves them
  identically.
- **Namespaces.** Everything here runs in `default`. A real deployment would isolate by
  environment (`dev`/`staging`/`prod`) and/or team, with `ResourceQuota`/`NetworkPolicy`
  scoped per namespace — cheap insurance this demo skips because there's only one
  environment to isolate from.
- **GPU nodes.** Not needed for this model — a calibrated XGBoost classifier plus SHAP
  and MAPIE conformal wrapping are all CPU-bound, and the resource requests/limits in
  `k8s/deployment.yaml` reflect that. This would matter the moment a future milestone (in
  this project or elsewhere in the portfolio) added a deep-learning model — a GPU node
  pool with proper taints/tolerations and `nvidia.com/gpu` resource requests, not
  something this project's current model needs.
- **Multi-region / multi-cluster.** Out of scope here entirely, but worth naming as the
  next order of complexity beyond this demo: cross-region failover, active-active or
  active-passive serving, and a real strategy for keeping the MLflow model registry (and
  its "Production" stage) consistent across clusters — none of which a single local k3d
  cluster needs to solve.

## What I'd build next

Concrete items from the error analysis, ranked by what they'd actually fix:

1. **Recency-weighted sponsor history** (Theme 1 fix) — a lookback window that weights a
   sponsor's recent trials more than trials from a decade ago would better reflect a large
   sponsor's *current* risk profile than a flat lifetime average.
2. **Site-execution-risk signal** (Theme 2 fix) — `num_sites` alone can't distinguish a
   well-funded single-site trial from a fragile one; a real execution-risk feature (site
   experience, funding source) would.
3. **Drug-mechanism/efficacy features** (Theme 3 — the genuine ceiling raiser) — nearly half
   the worst misses have no standout feature in any group this project builds; closing that
   gap needs genuinely new information (safety signal, protocol-amendment history), not more
   tuning of the existing feature set.
4. **Remote MLflow artifact store** (infrastructure, flagged in M5's `decisions.md`) — the
   current local-file-backed `mlruns/` bakes absolute host paths into run metadata and only
   works because `docker-compose.yml` bind-mounts it at an identical path — a real deployment
   needs a tracking server with S3/GCS-backed artifacts instead.

## Known limitations

Full list with context: `docs/error_analysis.md`'s "Known Limitations" section. In brief:

- `num_arms` isn't available in the PharmaPulse mart; excluded entirely.
- `therapeutic_area` is NULL for every row in `dim_condition` — `condition_name` is used as a
  coarser proxy throughout.
- Sponsor-history features use each prior trial's `start_date`, not its outcome-resolution
  date, so a still-running prior trial counts as "not terminated" even though its true
  outcome is unknown at that point — an inherited limitation from the spec's own example
  logic, not a new bug.
- The 5:1 FN:FP cost ratio is a domain judgment made for this project, not empirically
  derived from real clinical-operations cost data — a real deployment would need that ratio
  validated by clinical-operations/portfolio-management stakeholders before trusting the
  0.22 threshold operationally.
