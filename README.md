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

**Updated in M9 — read this before the table.** `enrollment_count` was found to be target
leakage (`WITHDRAWN` is definitionally zero-enrollment; `P(label=1 | enrollment==0) = 1.000`
on TEST). It was removed, the champion was retrained, and every number below moved. The
leaked numbers are kept in the table for comparison, not as the honest result — see
["The enrollment leakage" below](#the-enrollment-leakage--interviewers-will-read-this-first)
for the full investigation, including the fix that *looked* right and turned out to leak too.

| Metric | Pre-M9 (leaked) | M9 (honest) |
|---|---|---|
| PR-AUC, temporal test (XGBoost, uncalibrated) | 0.8878 | **0.6484** |
| PR-AUC, temporal test (after isotonic calibration) | 0.8775 | 0.6309 |
| ROC-AUC, temporal test (calibrated production model) | 0.9178 | **0.7862** |
| ECE, after isotonic calibration (TEST) | 0.0238 (raw: 0.1897) | **0.0319** (raw: 0.2734) |
| Conformal coverage (target 90%) | 94.6% | **93.5%** |
| Cost-optimal threshold (5:1 FN:FP cost matrix) | 0.22 (selected on TEST — see M9-4) | **0.14** (selected on CALIB, 95% bootstrap CI [0.10, 0.20]) |
| Majority-class baseline PR-AUC (temporal test) | 0.3167 | 0.3167 |

Calibration and calibrated-model PR-AUC differ slightly within each column because isotonic
regression is a monotonic but non-linear remapping of scores, which can shuffle tie-breaks
near the decision boundary — the calibrated M9 number (0.6309) is what the Production model
and this README's other calibration/conformal/threshold numbers all use. Production model:
MLflow registry version **16**.

## The enrollment leakage ← interviewers will read this first

`enrollment_count` — the model's #1 feature pre-M9, contributing 4.4× the SHAP magnitude of
whatever ranked second — was target leakage. `WITHDRAWN` means "stopped before enrolling the
first participant," so actual enrollment is 0 *by definition of the label class*:
`P(label=1 | enrollment_count == 0) = 1.000` on TEST (n=5,700). A one-line
`if enrollment == 0` rule alone scores 0.632 PR-AUC — within 0.017 of the entire pipeline
minus enrollment. Removing it dropped TEST PR-AUC from 0.888 to 0.648, meaning **~40% of the
model's entire lift over the 0.317 majority baseline came from a column reading off the
label.**

**The leakage-detection framework built in M1/M2 (temporal-vs-random split, controlled
fixed-window ablation) tested for this and returned a genuine true negative** — there is no
temporal row-placement leakage anywhere in this feature set (see "Temporal vs. random split"
below, unchanged and still valid). It structurally could not catch this leak, because every
check in it asks "which rows went where," and this leak lives in "is this value knowable at
`start_date`?" — a different question the framework was never built to ask.

**The fix that looked obviously right also leaked, and only an ablation caught it.**
ClinicalTrials.gov's `enrollmentInfo.type` field (`ACTUAL`/`ESTIMATED`) distinguishes planned
enrollment from post-hoc actual enrollment. Keeping only `ESTIMATED`-typed values (nulling
the rest) is the textbook fix, and it scored 0.682 PR-AUC — 0.034 better than dropping
enrollment outright. That +0.034 looked like free signal. Decomposing it wasn't free:

| Variant | TEST PR-AUC | Lift over dropping enrollment entirely |
|---|---|---|
| value + missingness indicator | 0.6821 | +0.0337 |
| **indicator only — zero enrollment magnitude** | **0.6772** | **+0.0288 (85% of the total)** |
| value only, no indicator | 0.6656 | +0.0172 |
| enrollment dropped entirely | 0.6484 | — |

A bare bit carrying **no enrollment magnitude at all** supplied 85% of the "fix's" advantage.
The reason: under this fix, the missingness indicator no longer means "enrollment unknown" —
it means `enrollment_type != 'ESTIMATED'`, and every trial's record reads `ESTIMATED` at
registration, flipping to `ACTUAL` only once the trial reports. The "fix" swapped one
definitional leak for a subtler record-maintenance one. Enrollment was dropped entirely
instead. Full investigation, including why a registration-time-snapshot API (not currently
available from this warehouse) is the only thing that would legitimately bring it back:
`decisions.md` M9-1.

## Temporal vs. random split ← a different leakage question, still worth reading

Distinct from the enrollment leakage above: this experiment tests *row-placement* leakage
(does a random train/test split let future information leak backward?), not feature
semantics. **Correction:** this notebook's LogReg pipeline also had `log_enrollment_count`
in its own hardcoded feature list (separate from `train_pipeline.py`'s), so it was NOT
unaffected by the M9 fix as first assumed here — it was re-executed on the no-enrollment
feature set along with everything else, and every number below moved. **The qualitative
finding is unchanged**, which is the part that actually matters: dropping enrollment
changes what the row-placement test measures the model *with*, not what row-placement
leakage does to it.

Trained the identical `LogisticRegression` pipeline (same features, same hyperparameters)
under two split strategies:

- **Temporal split** (train < 2020, calib 2020–2022, test ≥ 2022): PR-AUC **0.5676**
- **Random split** (60/20/20 shuffle, same proportions): PR-AUC **0.3687**

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
random-split leakage were real, the leaky model should win handily. Re-run on the M9
no-enrollment feature set with the same previously-logged best hyperparameters (no fresh
Optuna sweep):

| Model | Honest PR-AUC | Leaky PR-AUC | Delta |
|---|---|---|---|
| LogReg | 0.4568 | 0.4558 | −0.0009 |
| XGBoost | 0.5338 | 0.5472 | +0.0134 |
| LightGBM | 0.5583 | 0.5672 | +0.0089 |

**Honest note on this table:** pre-M9 the tree-model deltas were ≤0.002 (noise-level on a
~0.80 PR-AUC base). Post-M9 they're 0.009–0.013 — small in absolute terms, but a larger
share of a smaller base (~1.6–2.5% relative, vs ~0.15% before). That's not strong evidence
of a temporal leak reappearing — LogReg, the model class this notebook's narrative
otherwise centers, still shows a near-zero and *negative* delta — but a weaker model
naturally has a wider noise band around a small effect, and 0.013 is close enough to worth
having in the record rather than silently kept at the old, now-false "≤0.002" figure.

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
random-split PR-AUC as "0.891" — that figure doesn't match either the pre-M9 or the current
M9 `notebooks/02_leakage_demo.ipynb` executed output. Flagged here rather than silently
reconciled; the notebook's directly-executed number (0.3687, post-M9) is the one used
throughout this README.

## Try it

```bash
docker compose up
curl -X POST localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
        "phase": "PHASE3",
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

**M9 note:** the request body's `log_enrollment_count` field is now accepted but ignored
(`enrollment_count` was dropped as target leakage — see above); it is omitted from the
example on purpose. It is kept as an optional no-op in the schema, not deleted, so it doesn't
break RegIntel's wrapper until both sides coordinate a contract update.

## Model decisions worth reading

Full log: [`decisions.md`](decisions.md). Four entries an interviewer would find most
interesting:

- **`enrollment_count` was target leakage; the obvious fix leaked too (M9-1).** See "The
  enrollment leakage" above — the headline story in this repo.
- **Cost-optimal threshold = 0.14, selected on CALIB and reported on TEST (M9-4)** — the
  previous 0.22 was selected *and* reported on TEST, a one-parameter fit on the evaluation
  split. CALIB and TEST disagreed by more than the ~0.05 stability margin, so a bootstrap CI
  (95%: [0.10, 0.20]) ships alongside the point estimate rather than instead of it.
- **Conformal coverage 93.5% vs a 90% target** — MAPIE's LAC (least-ambiguous-set) conformal
  score is finite-sample conservative at this calibration-set size; a comfortable pass, not a
  razor-thin one, with per-request interval widths that still vary meaningfully by predicted
  probability (`docs/conformal_width_vs_proba.png`).
- **`FrozenEstimator`/MAPIE API changes** — this project's pinned `scikit-learn==1.9.0` and
  `mapie==1.4.1` both shipped breaking API changes since most tutorials were written
  (`cv="prefit"` → `sklearn.frozen.FrozenEstimator`; `MapieClassifier(method="score")` →
  `SplitConformalClassifier(conformity_score="lac")`) — resolved by inspecting the installed
  package directly rather than downgrading to match stale documentation.

## Error analysis

**Rewritten completely in M9** — the previous version analyzed a model dominated by a leaked
feature; every trial, theme, and number below changed. Full essay:
[`docs/error_analysis.md`](docs/error_analysis.md). With `enrollment_count` gone,
`sponsor_prior_termination_rate` is now the #1 global SHAP feature (was #2, behind
enrollment, at 4.4× less magnitude than the top feature — now the gap to #2 is 1.2×). Three
failure themes across the 20 worst false negatives:

1. **A clean sponsor record dominates everything else** (14/20) — the direct, predictable
   consequence of removing enrollment: the model is now substantially a sponsor-reputation
   model, and a 0% historical termination rate says nothing about *this* trial's drug,
   protocol, or funding decision. Not a bug — the honest ceiling of what removing the leak
   leaves behind.
2. **`has_results = True` at mega-sponsors overrides a mediocre sponsor record** (4/20) — the
   only worst-20 rows whose sponsor rate isn't low; `has_results` (a reporting-behavior proxy
   for "large, organized sponsor," verified non-leaky in M1) pushes hard enough toward
   completion to override it.
3. **Recency** (2/20) — `start_year` reflects data-recency mechanics (recent trials haven't
   had time to reach a terminal status) as much as any real secular trend.

## Calibration

Reliability curve (before vs after, isotonic vs Platt): `notebooks/03_calibration.ipynb`.
Numbers below are for the M9 no-enrollment champion (registry v16).

- ECE before calibration (raw XGBoost, TEST): **0.2734**
- ECE after isotonic calibration (TEST): **0.0319**
- ECE after Platt scaling (TEST, for comparison): 0.0331 (isotonic still wins, by a
  narrower margin than pre-M9's 0.0238 vs 0.0405 — the weaker honest model leaves isotonic
  less room to separate from Platt)

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

**Latest run (M9, no-enrollment feature set, 36 features):** 10/36 features drifted
(`drift_share=0.278`), below the `feature_drift_threshold=0.5` dataset-level alert — **not
flagged**. (Pre-M9: 31/38, `drift_share=0.816` — flagged. Two fewer total features and a
smaller drifted count both follow mechanically from dropping `enrollment_count` and
`enrollment_missing`; not independently re-derived here, just noted.) The top-5 most-drifted
features are `start_year`, `condition_rarity`, `sponsor_prior_trial_count`,
`eligibility_criteria_length`, and `exclusion_keyword_count` — every one is either a temporal
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

Concrete items from the M9 error analysis, ranked by what they'd actually fix:

1. **Point-in-time sponsor-history fix (M9 P1-11)** — sponsor history currently uses each
   prior trial's *final* label rather than its status as of this trial's `start_date`. This
   was a known inherited limitation since M1; it's materially more important now that
   `sponsor_prior_termination_rate` is the model's #1 feature (Theme 1, 14/20 worst misses)
   rather than its #2 behind a leaked column.
2. **Registration-time enrollment, if it ever becomes available** — the only thing that would
   legitimately bring enrollment back as a feature is a versioned-record/history API exposing
   the *planned* figure as of registration. Not available from the current warehouse; see "The
   enrollment leakage" above.
3. **Site-execution-risk signal** (Theme 2, `has_results`-at-mega-sponsors) — a real
   execution-risk feature (site experience, funding source) would help distinguish "reports
   results because well-resourced" from "reports results because low-risk."
4. **Drug-mechanism/efficacy features** — closing the honest ceiling on Theme 1 needs
   genuinely new information (safety signal, protocol-amendment history), not more tuning of
   the existing feature set.
4. **Remote MLflow artifact store** (infrastructure, flagged in M5's `decisions.md`) — the
   current local-file-backed `mlruns/` bakes absolute host paths into run metadata and only
   works because `docker-compose.yml` bind-mounts it at an identical path — a real deployment
   needs a tracking server with S3/GCS-backed artifacts instead.

## Known limitations

Full list with context: `docs/error_analysis.md`'s "Known Limitations" section. In brief:

- **`enrollment_count` was target leakage and is gone (M9)** — see "The enrollment
  leakage" above. The headline honest-model number (0.648 PR-AUC) reflects its absence.
- `num_arms` isn't available in the PharmaPulse mart; excluded entirely.
- `therapeutic_area` is NULL for every row in `dim_condition` — `condition_name` is used as a
  coarser proxy throughout.
- Sponsor-history features use each prior trial's `start_date`, not its outcome-resolution
  date, so a still-running prior trial counts as "not terminated" even though its true
  outcome is unknown at that point — an inherited limitation from the spec's own example
  logic, not a new bug, but now materially more important since `sponsor_prior_termination_rate`
  is the model's #1 feature rather than its #2 behind a leaked column.
- **The 0.14 operating threshold ships with a bootstrap interval, not just a point estimate
  (M9)** — CALIB and TEST cost-optimal selection disagreed by 0.08, above the ~0.05 stability
  margin; the 95% CI is [0.10, 0.20], and honest CALIB selection costs 3.7% regret vs. the
  (unattainable) TEST-optimal choice.
- The 5:1 FN:FP cost ratio is a domain judgment made for this project, not empirically
  derived from real clinical-operations cost data — a real deployment would need that ratio
  validated by clinical-operations/portfolio-management stakeholders before trusting the
  threshold operationally, and the M9 instability finding means the ratio and the threshold
  need to be revisited together.
