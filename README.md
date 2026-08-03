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
log: [`decisions.md`](decisions.md) (summary + index; full detail in
[`docs/decision-log/`](docs/decision-log/), one file per milestone); spec with per-milestone Definition of Done:
[`ai_portfolio/02_TRIALOUTCOME_SPEC.md`](ai_portfolio/02_TRIALOUTCOME_SPEC.md).

## Headline numbers

**Updated in M9 — read this before the table.** `enrollment_count` was found to be target
leakage (`WITHDRAWN` is definitionally zero-enrollment; `P(label=1 | enrollment==0) = 1.000`
on TEST). It was removed, the champion was retrained, and every number below moved (M9-1).
A second fix (M9-11) later corrected a point-in-time bug in `sponsor_prior_termination_rate`
(a historical trial was counted as a termination based on its *current-day* status rather
than its status as of the querying trial's own `start_date`) — every number below reflects
**both** fixes now, not just the first. The leaked numbers are kept in the table for
comparison, not as the honest result — see
["The enrollment leakage" below](#the-enrollment-leakage--interviewers-will-read-this-first)
for the full investigation, including the fix that *looked* right and turned out to leak too,
and [`docs/decision-log/M9-review-fixes.md`](docs/decision-log/M9-review-fixes.md)'s M9-11
entry for the sponsor-history fix.

| Metric | Pre-M9 (leaked) | M9-1 (enrollment fixed) | M9-11 (sponsor-history also fixed) |
|---|---|---|---|
| PR-AUC, temporal test (XGBoost, uncalibrated) | 0.8878 | 0.6484 | **0.6193** |
| PR-AUC, temporal test (after isotonic calibration) | 0.8775 | 0.6309 | **0.5975** |
| ROC-AUC, temporal test (calibrated production model) | 0.9178 | 0.7862 | **0.7662** |
| ECE, after isotonic calibration (TEST) | 0.0238 (raw: 0.1897) | 0.0319 (raw: 0.2734) | **0.0310** (raw: 0.2942) |
| Conformal coverage (target 90%) | 94.6% | 93.5% | **93.1%** |
| Cost-optimal threshold (5:1 FN:FP cost matrix) | 0.22 (selected on TEST — see M9-4) | 0.14 (selected on CALIB, 95% bootstrap CI [0.10, 0.20], gap 0.08 "unstable") | **0.16** (selected on CALIB, 95% bootstrap CI [0.13, 0.21], gap 0.05 "stable") |
| Majority-class baseline PR-AUC (temporal test) | 0.3167 | 0.3167 | **0.3146** |

Calibration and calibrated-model PR-AUC differ slightly within each column because isotonic
regression is a monotonic but non-linear remapping of scores, which can shuffle tie-breaks
near the decision boundary — the calibrated M9-11 number (0.5975) is what the Production
model and this README's other calibration/conformal/threshold numbers all use. Production
model: MLflow registry version **45** (the fix plan anticipated "v17"; the real number is
higher due to accumulated dev-session registrations during this fix — reported as measured,
not forced to match the plan's guess).

## The enrollment leakage ← interviewers will read this first

`enrollment_count` — the model's #1 feature pre-M9, contributing 4.4× the SHAP magnitude of
whatever ranked second — was target leakage. `WITHDRAWN` means "stopped before enrolling the
first participant," so actual enrollment is 0 *by definition of the label class*:
`P(label=1 | enrollment_count == 0) = 1.000` on TEST. A one-line
`if enrollment == 0` rule alone scores 0.632 PR-AUC — within 0.017 of the entire pipeline
minus enrollment. Removing it dropped TEST PR-AUC from 0.888 to 0.648, meaning **~40% of the
model's entire lift over the 0.317 majority baseline came from a column reading off the
label.** (These are this specific ablation's own numbers, measured once at the time of the
M9-1 investigation on that investigation's dataset snapshot — the model's *current* honest
PR-AUC is 0.619, per the headline table above, after M9-11's separate sponsor-history fix
moved the baseline a second time. Both fixes are independent and additive; this section is
about the enrollment mechanism specifically, not the current headline number.)

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
[`docs/decision-log/M9-review-fixes.md`](docs/decision-log/M9-review-fixes.md), M9-1.

## Temporal vs. random split ← a different leakage question, still worth reading

Distinct from the enrollment leakage above: this experiment tests *row-placement* leakage
(does a random train/test split let future information leak backward?), not feature
semantics. **Correction:** this notebook's LogReg pipeline also had `log_enrollment_count`
in its own hardcoded feature list (separate from `train_pipeline.py`'s), so it was NOT
unaffected by the M9 fix as first assumed here — it was re-executed on the no-enrollment
feature set along with everything else, and every number below moved. Re-executed again
after M9-11's sponsor-history point-in-time fix (a second, unrelated feature-value change),
and every number below moved again. **The qualitative finding is unchanged across both
fixes**, which is the part that actually matters: neither fix changes what the
row-placement test measures the model *with*, only what row-placement leakage does to it.

Trained the identical `LogisticRegression` pipeline (same features, same hyperparameters)
under two split strategies:

- **Temporal split** (train < 2020, calib 2020–2022, test ≥ 2022): PR-AUC **0.5117**
- **Random split** (60/20/20 shuffle, same proportions): PR-AUC **0.3189**

The temporal split scores *higher*, not lower — the opposite of the naive "random split
leaks future sponsor history and inflates performance" story this experiment was designed
to test for. Two confounds explain the gap, and neither is leakage:

1. **Base-rate confound.** The temporal test set is entirely 2022+, where termination rates
   have risen to ~31.5% (see the Drift Monitoring section below); the random test set mirrors
   the whole-dataset average (~20%). PR-AUC is mechanically higher against a higher-prevalence
   test set for an equally-good model.
2. **History-richness confound.** The random test set spans the entire 1990–2026 window,
   including early trials with thin point-in-time sponsor/condition history; the temporal
   test set is entirely 2022+, where 30+ years of history has already accumulated, making
   those predictions systematically easier regardless of split methodology.

A **controlled ablation** isolates the actual leakage mechanism directly: fix one test
window (2020–2022), and compare a same-size "honest" training set (only rows strictly
before the window) against a "leaky" one (allowed to include rows *after* the window). If
random-split leakage were real, the leaky model should win handily.

| Model | Honest PR-AUC | Leaky PR-AUC | Delta |
|---|---|---|---|
| LogReg (M9-11, re-executed in `notebooks/02_leakage_demo.ipynb`) | 0.4158 | 0.4125 | −0.0033 |
| XGBoost (M9-1 era — not re-run under M9-11) | 0.5338 | 0.5472 | +0.0134 |
| LightGBM (M9-1 era — not re-run under M9-11) | 0.5583 | 0.5672 | +0.0089 |

**XGBoost/LightGBM rows are not yet refreshed for M9-11, flagged rather than silently left
looking current:** those two rows come from `domains/pharma/train_pipeline.py`'s
`run_controlled_ablation`, which only runs inside that file's `main()` — a full 4-family
Optuna sweep (35 trials each), not a single retrain. Re-running it under M9-11 would mean a
fresh hyperparameter search, which this fix's own stated discipline (single retrain from
already-logged hyperparameters, isolating the feature-value change rather than confounding
it with new hyperparameters — see M9-1/M9-11 in `decisions.md`) deliberately avoids. The
LogReg row above *is* refreshed, since `notebooks/02_leakage_demo.ipynb` re-executes it
directly with fixed hyperparameters. Re-running the tree-model ablation under M9-11 is
tracked as a follow-up, not silently skipped.

**Honest note on this table (pre-M9-11 context, LogReg row only):** pre-M9 the tree-model
deltas were ≤0.002 (noise-level on a ~0.80 PR-AUC base). Post-M9-1 they were 0.009–0.013 —
small in absolute terms, but a larger share of a smaller base (~1.6–2.5% relative, vs ~0.15%
before). That's not strong evidence of a temporal leak reappearing — LogReg, the model class
this notebook's narrative otherwise centers, still shows a near-zero and *negative* delta
under both M9-1 and M9-11 — but a weaker model naturally has a wider noise band around a
small effect.

**Why it matters:** this is a property of *this specific feature set* — every feature
(sponsor history, condition rarity) is computed via a point-in-time self-join
(`hist.start_date < t.start_date`) in `domains/pharma/dataset_builder.py`, so no row's
features ever encode information from after its own start date. That's the leakage guard
working as intended, not proof that random splits are safe in general — a future feature
computed carelessly (e.g. a static lifetime aggregate instead of a point-in-time join) would
still leak hard under a random split, and this same controlled-ablation methodology is
exactly how you'd catch it. The temporal split stays mandatory regardless of this negative
result. Full writeup: `notebooks/02_leakage_demo.ipynb`,
[`docs/decision-log/M1-dataset-builder.md`](docs/decision-log/M1-dataset-builder.md) and
[`docs/decision-log/M2-baseline-optuna.md`](docs/decision-log/M2-baseline-optuna.md).

**Documentation note:** `ai_portfolio/02_TRIALOUTCOME_SPEC.md`'s M1 milestone row states the
random-split PR-AUC as "0.891" — that figure doesn't match the pre-M9, M9-1, or current M9-11
`notebooks/02_leakage_demo.ipynb` executed output. Flagged here rather than silently
reconciled; the notebook's directly-executed number (0.3189, post-M9-11) is the one used
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
`{proba, uncertainty_band, coverage_guarantee, threshold_decision, top_shap, plain_english_summary, feature_pipeline_version}`
— this exact shape is what RegIntel's `trial_risk` tool wrapper (Project 4) is built against.

**M9-9 note:** `conformal_interval` was renamed to `uncertainty_band`, and `coverage_guarantee`
(`{type: "label_set", target: 0.90, empirical: 0.931, note: "..."}`) was added — the field's
90%-target coverage guarantee is on label-set membership (was the true class inside MAPIE's
predicted `{0}`/`{1}`/`{0,1}` set), not on the `[low, high]` band itself, and the old name
implied the latter. Locked-contract change, coordinated with RegIntel's spec in the same
session — see `docs/decision-log/M9-review-fixes.md` M9-9.

**M9 note:** the request body's `log_enrollment_count` field is now accepted but ignored
(`enrollment_count` was dropped as target leakage — see above); it is omitted from the
example on purpose. It is kept as an optional no-op in the schema, not deleted, so it doesn't
break RegIntel's wrapper until both sides coordinate a contract update.

## Model decisions worth reading

Full log: [`decisions.md`](decisions.md) (summary; full detail in
[`docs/decision-log/`](docs/decision-log/)). Four entries an interviewer would find most
interesting:

- **`enrollment_count` was target leakage; the obvious fix leaked too (M9-1).** See "The
  enrollment leakage" above — the headline story in this repo. →
  [`M9-review-fixes.md`](docs/decision-log/M9-review-fixes.md)
- **Cost-optimal threshold = 0.16, selected on CALIB and reported on TEST (M9-4, refit under
  M9-11)** — the previous 0.22 was selected *and* reported on TEST, a one-parameter fit on
  the evaluation split. M9-4's CALIB refit (0.14) disagreed with TEST by more than the
  ~0.05 stability margin (gap 0.08, "unstable"); M9-11's sponsor-history fix moved the
  threshold again to 0.16, narrowing the gap to exactly 0.05 ("stable" by the same rule). A
  bootstrap CI (95%: [0.13, 0.21]) ships alongside the point estimate rather than instead of
  it, and the TEST-selected value (0.21) now falls inside it. →
  [`M9-review-fixes.md`](docs/decision-log/M9-review-fixes.md)
- **Conformal coverage 93.1% vs a 90% target** — MAPIE's LAC (least-ambiguous-set) conformal
  score is finite-sample conservative at this calibration-set size; a comfortable pass, not a
  razor-thin one, with per-request interval widths that still vary meaningfully by predicted
  probability (`docs/conformal_width_vs_proba.png`). →
  [`M5-conformal-api-docker.md`](docs/decision-log/M5-conformal-api-docker.md)
- **`FrozenEstimator`/MAPIE API changes** — this project's pinned `scikit-learn==1.9.0` and
  `mapie==1.4.1` both shipped breaking API changes since most tutorials were written
  (`cv="prefit"` → `sklearn.frozen.FrozenEstimator`; `MapieClassifier(method="score")` →
  `SplitConformalClassifier(conformity_score="lac")`) — resolved by inspecting the installed
  package directly rather than downgrading to match stale documentation. →
  [`M5-conformal-api-docker.md`](docs/decision-log/M5-conformal-api-docker.md)

## Error analysis

**Rewritten completely in M9, again for M9-11** — every trial, theme, and number below changed
twice. Full essay: [`docs/error_analysis.md`](docs/error_analysis.md). With `enrollment_count`
gone, `start_year` is now the #1 global SHAP feature (mean|SHAP|=0.334), narrowly edging out
`sponsor_prior_termination_rate` (0.328, a 1.02× gap) — under M9-1, before the point-in-time
fix, `sponsor_prior_termination_rate` had been the clear #1 (1.2× the #2 feature). M9-11's fix
changed the feature's *values* (removed a hindsight leak) and that was enough to flip the top-2
ordering, not just move the numbers. Four failure themes across the 20 worst false negatives
(87 total false negatives at the operating threshold, up from 77 pre-M9-11):

1. **Recency — the model reads "too soon to know" as "will complete"** (11/20 worst; 48/87
   overall, now the largest theme) — `start_year` conflates a genuine secular rise in
   termination rates (17.8% pre-2020 → 31.5% post-2022) with a label-maturity artifact (a
   trial's outcome is only labeled once it reaches a terminal status, so recent trials are
   systematically under-represented among known terminations regardless of true risk). The
   least tractable theme — a structural property of the label, not a missing feature.
2. **Long eligibility criteria read as design rigor, sometimes wrongly** (3/20; 14/87) — a
   near-median `eligibility_criteria_length` still gets pushed toward "will complete," with
   nothing to counterbalance it when other signals are weak.
3. **A clean sponsor record still dominates a meaningful minority** (3/20; 12/87) — the
   residual version of M9-1's dominant theme: a low point-in-time termination rate says
   nothing about *this* trial's drug, protocol, or funding decision. M9-11 fixed the feature's
   values, not this structural ceiling.
4. **`has_results = True` at mega-sponsors overrides a mediocre sponsor record** (3/20; 10/87) —
   `has_results` (a reporting-behavior proxy for "large, organized sponsor," verified non-leaky
   in M1) pushes hard enough toward completion to override a middling sponsor signal.

## Calibration

Reliability curve (before vs after, isotonic vs Platt): `notebooks/03_calibration.ipynb`.
Numbers below are for the M9-11 champion (registry v45, sponsor-history point-in-time fix
applied on top of M9-1's no-enrollment fix).

- ECE before calibration (raw XGBoost, TEST): **0.2942**
- ECE after isotonic calibration (TEST): **0.0310**
- ECE after Platt scaling (TEST, for comparison): 0.0331 (M9-1-era measurement — isotonic
  still wins by a narrower margin than pre-M9's 0.0238 vs 0.0405; not independently
  recomputed against M9-11's retrained probabilities, since `notebooks/03_calibration.ipynb`
  only prints Platt's CALIB-split ECE, not a TEST-split one, so there is nothing to re-read
  off its M9-11 re-execution for this specific number — flagged rather than left implying
  it was re-verified)

Isotonic beat Platt because it makes no parametric shape assumption — it can correct
whatever miscalibration pattern the raw XGBoost scores actually have, while Platt is
constrained to a single-parameter sigmoid family. The fitted calibrator was evaluated on the
held-out TEST split (never used to fit or select it) specifically so a near-zero CALIB-split
ECE couldn't be mistaken for a calibrator that memorized its own fitting data.

## Drift monitoring

`domains/pharma/monitoring/drift_job.py` runs Evidently's `DataDriftPreset` comparing the
Production model's training population (`ml.training_dataset WHERE split='train'`,
n=66,129) against a current batch (`WHERE split='test'`, n=5,789), writes an HTML report to
`reports/`, and logs the verdict to `ml.drift_log`.

**Honesty note:** in production, "current batch" would be last week's newly-registered
trials scored by the live API. No live scoring traffic exists yet, so the held-out TEST
split is used as a stand-in for "a batch the model hasn't seen" — this validates the
monitoring *mechanism*, not a real production drift claim.

**Latest run (M9-11, no-enrollment + sponsor-history-point-in-time-fixed feature set, 36
features):** 10/36 features drifted (`drift_share=0.278`), below the
`feature_drift_threshold=0.5` dataset-level alert — **not flagged**. (Pre-M9: 31/38,
`drift_share=0.816` — flagged. Two fewer total features and a smaller drifted count both
follow mechanically from dropping `enrollment_count` and `enrollment_missing`; not
independently re-derived here, just noted.) The top-5 most-drifted features are `start_year`,
`condition_rarity`, `sponsor_prior_trial_count`, `sponsor_prior_termination_rate`, and
`eligibility_criteria_length` — every one is either a temporal feature (drifted by
construction: train is entirely pre-2020, test is entirely 2022+), a cumulative
point-in-time count/rate that mechanically shifts over calendar time, or a real secular trend
in trial design over 30+ years. `sponsor_prior_termination_rate` is new to this top-5 as of
M9-11 (drift score 0.126 → 0.416): the point-in-time fix means the feature now honestly drops
for cohorts whose prior trials hadn't yet resolved by query time, rather than quietly using
hindsight — exactly the kind of era-to-era shift a corrected point-in-time feature *should*
show, not a regression. None of this is surprising; it's exactly what you'd expect comparing
a pre-2020 population to a 2022+ batch. Full writeup with per-feature scores:
`notebooks/06_drift_report.ipynb`.

**Label-drift stretch check (PSI on the rolling termination base rate): attempted, not
completed meaningfully — two distinct structural reasons, not a coding failure.**

1. The literal construction (bins = individual `start_year` values, TRAIN years =
   "expected," TEST years = "actual") is degenerate for *any* dataset built with this
   project's temporal split: TRAIN spans 1990–2019, TEST spans 2022–2026, and the split is
   defined by `start_date` — so TRAIN and TEST share **zero** `start_year` bins by
   construction, not because of anything specific to this data.
2. A reinterpretation (bin the yearly termination-*rate value* into deciles instead of
   calendar years, so bins can overlap) is technically computable but wildly unstable: PSI
   ranges from **0.67** (3 bins) to **11.29** (10 bins) depending purely on bin-count choice,
   because there are only 30 TRAIN "observations" and 5 TEST "observations" (one rate per
   `start_year`) to bin — nowhere near enough for PSI's binning approach to be a stable
   statistic at this granularity.

**What IS real** (directly from the data, not from PSI): TRAIN termination rate **17.8%**
vs TEST termination rate **31.5%** — a genuine ~14-point absolute increase, already
documented in [`docs/decision-log/M1-dataset-builder.md`](docs/decision-log/M1-dataset-builder.md)'s
finding and already accounted for by evaluating
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
  flagged in M5's [`docs/decision-log/M5-conformal-api-docker.md`](docs/decision-log/M5-conformal-api-docker.md) for `docker-compose.yml`'s identical hostPath-style bind
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

1. **~~Point-in-time sponsor-history fix~~ — done (M9-11).** Sponsor history previously used
   each prior trial's *current-day* status rather than its status as of this trial's own
   `start_date` — a known inherited limitation since M1, made materially more important once
   `sponsor_prior_termination_rate` became a top-2 feature. Fixed: a historical trial now
   only counts toward the rate once it has an ACTUAL (not ESTIMATED) resolution date before
   the querying trial's `start_date`. Full fix, validation, and re-registration:
   [`docs/decision-log/M9-review-fixes.md`](docs/decision-log/M9-review-fixes.md), M9-11.
2. **Registration-time enrollment, if it ever becomes available** — the only thing that would
   legitimately bring enrollment back as a feature is a versioned-record/history API exposing
   the *planned* figure as of registration. Not available from the current warehouse; see "The
   enrollment leakage" above.
3. **Site-execution-risk signal** (Theme 4, `has_results`-at-mega-sponsors) — a real
   execution-risk feature (site experience, funding source) would help distinguish "reports
   results because well-resourced" from "reports results because low-risk."
4. **A finer-grained recency/label-maturity treatment** (Theme 1, now the largest failure
   theme post-M9-11) — `start_year` currently conflates a real secular trend with the fact
   that recent trials haven't had time to reach a terminal status; a feature that isolates
   trial *age at scoring time* from calendar year might separate the two effects.
5. **Drug-mechanism/efficacy features** — closing the honest ceiling on Theme 3 (clean sponsor
   record) needs genuinely new information (safety signal, protocol-amendment history), not
   more tuning of the existing feature set.
6. **Remote MLflow artifact store** (infrastructure, flagged in M5's [`docs/decision-log/M5-conformal-api-docker.md`](docs/decision-log/M5-conformal-api-docker.md)) — the
   current local-file-backed `mlruns/` bakes absolute host paths into run metadata and only
   works because `docker-compose.yml` bind-mounts it at an identical path — a real deployment
   needs a tracking server with S3/GCS-backed artifacts instead.

## Known limitations

Full list with context: `docs/error_analysis.md`'s "Known Limitations" section. In brief:

- **`enrollment_count` was target leakage and is gone (M9-1)** — see "The enrollment
  leakage" above. The current honest-model number (0.6193 PR-AUC, further moved by M9-11's
  separate sponsor-history fix) reflects its absence.
- `num_arms` isn't available in the PharmaPulse mart; excluded entirely.
- `therapeutic_area` is NULL for every row in `dim_condition` — `condition_name` is used as a
  coarser proxy throughout.
- **Sponsor-history hindsight leak fixed (M9-11), one residual gap remains.** Prior trials
  were being counted as terminations based on their *current-day* status rather than their
  status as of the querying trial's own `start_date` — fixed by requiring an ACTUAL (not
  ESTIMATED) resolution date before `start_date`. The residual gap the fix does *not*
  eliminate: a prior trial that is genuinely still running as of the querying trial's
  `start_date` (not yet resolved either way) is treated identically to one that will go on
  to complete successfully — its true future outcome is unknowable at that point by
  construction, which is correct point-in-time behavior, but it does mean the feature can't
  distinguish "sponsor with a clean track record" from "sponsor whose track record is mostly
  still pending," an inherited ambiguity from the spec's own example logic.
- **The 0.16 operating threshold ships with a bootstrap interval, not just a point estimate
  (M9-4, refit under M9-11)** — CALIB and TEST cost-optimal selection now agree within 0.05
  (down from M9-4's unstable 0.08 gap); the 95% CI is [0.13, 0.21], and honest CALIB
  selection costs 2.8% regret vs. the (unattainable) TEST-optimal choice (down from 3.7%).
- The 5:1 FN:FP cost ratio is a domain judgment made for this project, not empirically
  derived from real clinical-operations cost data — a real deployment would need that ratio
  validated by clinical-operations/portfolio-management stakeholders before trusting the
  threshold operationally.
