# TrialOutcome — Decision Log (summary)

This file is a ~150-line index. Full detail — every decision in What/Why/Failure-mode/
Scaling/Interview-question format — lives one file per milestone in
[`docs/decision-log/`](docs/decision-log/). This split happened in M9 (review-driven fix
P1-10): the single-file log had grown to 1,801 lines / 140KB, past the point anyone but
its author would read it end-to-end. Nothing was cut in the split — every entry that lived
in the old `decisions.md` is still there, verbatim, in its milestone's file; see that fix's
own entry in [`docs/decision-log/M9-review-fixes.md`](docs/decision-log/M9-review-fixes.md)
for exactly how the split was verified content-complete.

## The decisions that most shaped the system

Eight decisions, in the order they'd come up in an interview walkthrough of this repo.

1. **Enrollment leakage — the headline M9 finding.** `enrollment_count` was target leakage
   (`WITHDRAWN` is definitionally zero-enrollment; `P(label=1 | enrollment==0) = 1.000` on
   TEST). The "obvious" fix (keep only `ESTIMATED`-tagged enrollment) was built, measured,
   and *rejected on evidence* — its missingness indicator alone supplied 85% of its
   apparent lift, itself a post-hoc signal. The feature was dropped entirely, the champion
   retrained, every downstream number republished. → [`M9-review-fixes.md`](docs/decision-log/M9-review-fixes.md), first entry
2. **The leakage-detection framework found nothing here — and that's not the same as no
   leakage.** M1/M2's temporal-vs-random-split ablation methodology returned a genuine true
   negative for row-placement leakage; it structurally could not detect the enrollment
   semantic leak (#1), because it tests *which split a row lands in*, not *whether a
   feature's value was actually knowable at that row's start_date*. → [`M1-dataset-builder.md`](docs/decision-log/M1-dataset-builder.md)
3. **Isotonic calibration over Platt scaling.** ECE before calibration 0.19 (TEST); isotonic
   0.024, Platt 0.041. Isotonic wins because it makes no parametric shape assumption — fit
   on CALIB, *reported* on TEST specifically so a near-perfect CALIB-only ECE couldn't be
   mistaken for evidence of anything. → [`M3-calibration-threshold.md`](docs/decision-log/M3-calibration-threshold.md)
4. **Cost-optimal threshold selected on CALIB, reported on TEST (M9-4, superseded again by
   M9-11).** Selecting one of 99 thresholds by minimizing a TEST-computed objective is a
   one-parameter *fit*, not reporting — M3's original 0.22 broke that rule. M9-4's CALIB
   refit gave 0.14 with an unstable 0.08 CALIB/TEST gap (bootstrap CI `[0.10, 0.20]`); M9-11's
   sponsor-history point-in-time fix changed the underlying probabilities enough to move the
   threshold again — CALIB now selects 0.16, the gap narrowed to a *stable* 0.05, and the
   bootstrap CI `[0.13, 0.21]` now contains the TEST-selected value. → [`M9-review-fixes.md`](docs/decision-log/M9-review-fixes.md)
5. **Drift-direction bugfix.** Evidently's p-value-based and distance-based drift methods
   flag "drifted" in *opposite* score directions — caught by `test_drift_base.py`'s
   regression test before M6 closed, not after. → [`M6-drift-ci-readme.md`](docs/decision-log/M6-drift-ci-readme.md)
6. **Retrain-trigger fires, but never auto-promotes.** `retrain_trigger.py` detects
   drift/version mismatch and *stages* a candidate; only a human-run
   `make rollback VERSION=N` ever calls `transition_model_version_stage` toward Production —
   enforced by a CI grep check that exactly one call site in the whole codebase does that. → [`M7-retrain-rollback.md`](docs/decision-log/M7-retrain-rollback.md)
7. **K8s probe-demo substitution.** The plan's literal "rename `mlruns/`" liveness/readiness
   demo doesn't actually distinguish the two probes on this deployment; a `kubectl patch` on
   the live readinessProbe path does. Verified directly rather than shipping a demo that
   doesn't demonstrate what it claims to. → [`M8-kubernetes.md`](docs/decision-log/M8-kubernetes.md)
8. **A green CI job asserting nothing (M9-2).** `tests/conftest.py`'s dev/CI state-detection
   fixture had a scoping bug: `has_raw_cache` was only assigned inside an `except` branch,
   so on a dev machine (registry populated) the three M7 integration tests raised
   `UnboundLocalError` instead of running, while CI silently skipped them — `retrain-trigger-test`
   was green either way, having asserted nothing. → [`M9-review-fixes.md`](docs/decision-log/M9-review-fixes.md)

## Full per-milestone decision log

| Milestone | What it covers | Full detail |
|---|---|---|
| M1 | Dataset builder, point-in-time self-joins, temporal split, the leakage-framework negative result | [`M1-dataset-builder.md`](docs/decision-log/M1-dataset-builder.md) |
| M2 | mlflow/pandas version resolution, Optuna sweep (35 runs), tree-model leakage ablation confirms M1 | [`M2-baseline-optuna.md`](docs/decision-log/M2-baseline-optuna.md) |
| M3 | Isotonic vs Platt calibration, cost-optimal threshold (superseded by M9-4) | [`M3-calibration-threshold.md`](docs/decision-log/M3-calibration-threshold.md) |
| M4 | SHAP aggregation for one-hot features, multicollinearity check, 3 failure themes, `has_results` leakage clearance | [`M4-shap-error-analysis.md`](docs/decision-log/M4-shap-error-analysis.md) |
| M5 | `FrozenEstimator`/MAPIE `SplitConformalClassifier` API migrations, Docker/API contract fixes | [`M5-conformal-api-docker.md`](docs/decision-log/M5-conformal-api-docker.md) |
| M6 | Evidently drift monitoring, drift-direction bugfix, PSI label-drift honest non-completion, CI | [`M6-drift-ci-readme.md`](docs/decision-log/M6-drift-ci-readme.md) |
| M7 | Retrain trigger, rollback-doubles-as-forward-promotion, no-auto-promote CI guard | [`M7-retrain-rollback.md`](docs/decision-log/M7-retrain-rollback.md) |
| M8 | Scoped K8s deploy, probe-demo substitution, Secret handling, HPA honest-negative-result | [`M8-kubernetes.md`](docs/decision-log/M8-kubernetes.md) |
| M9 | Review-driven fixes: enrollment leakage, CI-asserts-nothing bug, plain-English sign bug, threshold refit, vocabulary pass, split-function tests, decision-log split, sponsor-history point-in-time fix, CI quality gates, and remaining P1/P2 items | [`M9-review-fixes.md`](docs/decision-log/M9-review-fixes.md) |

## Where to find specific things

- **"Why did X leak / not leak?"** → M1 (framework), M9 (the enrollment case it missed).
- **"How is the model calibrated and thresholded?"** → M3 for the mechanism, M9-4 for the
  corrected selection-split methodology.
- **"What happens on drift / retrain?"** → M6 (detection), M7 (trigger + rollback).
- **"What's still open?"** → `M9-review-fixes.md`'s own P1/P2 tracking, and
  `ai_portfolio/02_TRIALOUTCOME_SPEC.md` Section 12 (kept in lockstep with this file after
  every fix, per this project's standing rule).
