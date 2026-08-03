[← back to decisions.md summary](../../decisions.md)

---

## M7: Retraining trigger + rollback (2026-08-02)

### Decision: the M7 requirements' literal `mlflow models transition-stage` CLI command does not exist -- `rollback_production()` (wrapping `MlflowClient.transition_model_version_stage` directly) plus `make rollback VERSION=N` is the real one-command procedure, reused for both rollback and manual forward promotion
- **What:** Verified directly (`mlflow models --help` against the installed mlflow==2.22.5):
  the `mlflow models` CLI only has `build-docker`/`generate-dockerfile`/`predict`/
  `prepare-env`/`serve`/`update-pip-requirements` -- there is no `transition-stage`
  subcommand anywhere in the mlflow CLI (registry stage transitions are Python-client/
  REST-API-only, never exposed as a CLI verb, in any mlflow version this project could
  find). `domains/pharma/monitoring/rollback.py`'s `rollback_production(target_version)`
  wraps `MlflowClient.transition_model_version_stage(..., archive_existing_versions=True)`
  directly instead (the exact call the requirements' own function body specified), exposed as
  `make rollback VERSION=N`. Because this function is deliberately agnostic about
  `target_version`'s stage beforehand, `domains/pharma/monitoring/retrain_trigger.py`'s
  "Staged for review" message also points a human at `make rollback VERSION=N` for manual
  forward promotion -- one real, tested command instead of a fictional one, for both
  directions.
- **Why (vs. alternatives):** Could have shipped the requirements' literal snippet anyway (it
  would look right in a README but fail the moment anyone actually ran it) or invented a
  second, separate "promote" command. Reusing `rollback_production()` for both keeps
  exactly one function in the whole codebase that ever calls
  `transition_model_version_stage(..., stage="Production")` via an automated code path --
  which is also what M7's acceptance criteria grep check (see below) is actually protecting.
- **Failure mode:** If a future mlflow release *does* add a registry-transition CLI verb,
  this project's Makefile target would still work identically (it's calling the Python
  client underneath, not shelling out to `mlflow models transition-stage`) -- no migration
  needed.
- **Scaling story (10x/100x):** Unaffected by data or model-registry size -- this is a
  one-row API call regardless of how many versions exist.
- **Interview question this maps to:** "What do you do when a spec's literal command
  doesn't actually exist in the tool you're using?" -- verify against the real CLI/API
  before shipping it, then find (or build) the smallest real mechanism that satisfies the
  same intent, and say so explicitly rather than ship an untested snippet.

### Decision: the requirements' suggested synthetic perturbation ("multiply `log_enrollment_count` by 3") cannot cross the real drift-share threshold -- perturbing the one-hot `condition_*` columns instead
- **What:** Empirically verified against the real `ml.training_dataset` (before writing
  `tests/test_retrain_trigger.py`) that shifting `log_enrollment_count` alone -- even by a
  large additive shift, not just x3 -- never changes the drift verdict at all (stuck at
  11/38 = 28.9%, identical to the unperturbed baseline). Shifting *every* `NUMERIC_FEATURES`
  column only reaches 34.2%, still short of the 0.5 threshold. The reason: 22 of the 38
  compared columns are the one-hot `condition_*` indicators (top-20 conditions + other +
  unknown), and Evidently's per-column drift test is highly sensitive to even small
  proportion shifts in a near-constant boolean column. The test instead flips a small,
  seeded fraction (10%) of each `condition_*` column's boolean values for the `test`-split
  rows (verified: reliably crosses 0.5, ~82% in practice, comfortable margin above the exact
  boundary -- an earlier 5% flip rate landed exactly at 0.5, too tight for a robust test).
  `log_enrollment_count x3` is kept in the perturbation for continuity with the requirements'
  intent but is verified non-load-bearing on its own.
- **Why (vs. alternatives):** Could have lowered `feature_drift_threshold` just for the
  test, but that would test a different (weaker) threshold than the one actually configured
  in `config.yaml`'s `drift` section -- the whole point is demonstrating a breach of the
  *real* production gate. Could have corrupted a much larger share of all columns (verified:
  shuffling categoricals + flipping most condition columns reaches ~92-100%), but that's an
  unrealistically total corruption rather than a plausible drift scenario.
- **Failure mode:** If `config.yaml`'s condition one-hot `top_n` or the real dataset's
  positive rate changes materially, this specific flip-rate margin (10%, ~82% share) could
  drift closer to the 0.5 boundary again -- re-verify empirically (the same one-off check
  used here, not guesswork) if `test_retrain_trigger.py` ever starts flaking.
- **Scaling story (10x/100x):** At 10x/100x more trials, per-column drift tests become more
  statistically powerful (larger samples make smaller true shifts detectable), so the same
  flip-rate would very likely cross threshold with an even wider margin -- if anything, this
  perturbation gets more robust at scale, not less.
- **Interview question this maps to:** "How do you validate that a 'make it drift' test
  fixture actually drifts?" -- don't trust a suggested perturbation by inspection; score it
  against the real detector and the real configured threshold before writing the test
  around it.

### Decision: `register_model.py`'s existing direct `stage="Production"` call (M5) is treated as the same class of "manual, human-invoked" exception as `rollback.py` and `make rollback`, for M7's auto-promotion grep check
- **What:** M7's acceptance criteria says "no code path calls `transition_model_version_stage` with
  `stage=\"Production\"` except rollback.py and the manual CLI command." `domains/pharma/
  register_model.py` (M5, pre-existing) also calls this directly, registering a freshly
  refit champion straight to Production -- but it is itself a manual, human-run script
  (`make register-model`), never invoked by any automated trigger, exactly like `make
  rollback` is carved out as. `domains/pharma/monitoring/retrain_trigger.py` (the new M7
  automated-trigger code path this acceptance criteria item is actually protecting) never calls
  `transition_model_version_stage` with `stage="Production"` -- verified by grep (it only
  transitions to `"Staging"`). Confirmed via `grep -rn 'stage="Production"' domains/ core/`:
  exactly two call sites, `register_model.py` (M5, manual) and `rollback.py` (M7, manual).
- **Why (vs. alternatives):** Could have refactored `register_model.py` to route through
  `rollback_production()` too, for a literal single-call-site match to the acceptance criteria's wording.
  Not done -- `register_model.py` is an already-reviewed, already-working M5 script outside
  this milestone's actual scope, and rewriting it isn't implementation-forced; flagging the
  interpretation explicitly (per this project's standing rule) is the right level of
  intervention, not a silent pass and not an unrequested refactor.
- **Failure mode:** If a future milestone adds a second automated trigger path, that new
  code must be grepped against this same standard (never `stage="Production"` directly) --
  this decision only certifies M7's `retrain_trigger.py`, not the whole codebase in
  perpetuity.
- **Scaling story (10x/100x):** N/A -- a code-review/interpretation decision, not a
  performance-sensitive one.
- **Interview question this maps to:** "Your automated pipeline's safety check says 'no
  code path does X except these two exceptions' -- but you find a third path that does X.
  What do you do?" -- distinguish automated from human-invoked call sites before treating a
  grep hit as a violation, and say so explicitly rather than silently exempt it or silently
  over-refactor working code.

### Decision: the new `retrain-trigger-test` CI job is wired up per the requirements, but its three tests are expected to report "skipped" (not "passed") in CI -- verified working for real in this dev environment instead
- **What:** `tests/test_retrain_trigger.py`, `tests/test_version_mismatch.py`, and
  `tests/test_rollback.py` all exercise the real M7 mechanism: a genuine XGBoost retrain via
  `PharmaDatasetBuilder.fetch_raw()` (needs either a cached `data/raw_trials_cache.parquet`
  or a live `marts` schema -- both gitignored/absent on a fresh checkout) and comparison/
  rollback against a real registered MLflow Production model (`mlruns/` is gitignored, so a
  fresh CI checkout starts with zero registered versions). CI's ephemeral Postgres service
  only ever gets the `ml` schema via `make db-init`, never `marts`. Building a synthetic
  `marts`-schema fixture (and a way to bootstrap a fixture Production registration without
  tripping `register_model.py`'s real-data-calibrated `EXPECTED_PR_AUC_TEMPORAL`
  reproducibility assertion) was prototyped, then explicitly deferred as scope beyond M7's
  actual ask, after checking with the user rather than silently building it out. Each test
  requests `tests/conftest.py`'s `real_dev_state` fixture, which skips with a clear, visible
  reason when these prerequisites are absent -- the CI job is expected to report "3 skipped,"
  and all three are verified "3 passed" against this project's real dev Postgres + real
  registered Production model (v2, run `13ca29a52a5b456b93846514f8020dfa`) in this session,
  including a real drift breach (`drift_share=0.816`), a real Staging registration, a real
  version-mismatch flag, and a real bad-promotion-then-rollback cycle -- with `ml.retrain_log`
  left populated with those real rows afterward, and all synthetic perturbations/model
  versions cleaned up (verified: post-test baseline drift check matches the pre-test
  baseline exactly, 11/38 = 28.9%).
- **Why (vs. alternatives):** Same pattern M6 already established for `test_api_contract.py`
  (excluded from `test-unit` for needing a live Docker container CI doesn't stand up) --
  extended here to a "skip with a visible reason" rather than a silent `--ignore`, so the
  CI job's log output itself documents why, every run, instead of only in this file.
- **Failure mode:** If a future contributor adds real `marts` seeding to CI without updating
  `real_dev_state`'s check (a registered Production model + a raw-features cache), these
  tests would start silently skipping even once CI *could* run them for real -- the skip
  reason should be re-verified whenever CI's Postgres setup changes.
- **Scaling story (10x/100x):** N/A -- a CI-scope decision, not a data or model one.
- **Interview question this maps to:** "How do you handle a CI job whose tests need
  production-like state your CI environment doesn't have?" -- make the gap visible (a
  skip with a reason, in every CI run) rather than silently excluding the tests or
  papering over it with an unrequested fixture-engineering effort, and verify the real
  mechanism works by running it against real state at least once.

## M7 Definition of Done -- status

- [x] Synthetic drift breach on the real `ml.training_dataset` demonstrably triggers a
      training run and Staging (never Production) registration --
      `test_synthetic_drift_breach_triggers_staging_registration`, verified `drift_share=
      0.816` (well above the 0.5 gate), new MLflow version registered to stage "Staging",
      confirmed absent from `stages=["Production"]`.
- [x] Version-mismatch warning correctly fires when a simulated retrain's
      `feature_pipeline_version` differs from current Production's --
      `test_mismatched_feature_pipeline_version_is_flagged_and_logged`, asserts
      `version_mismatch=True`, a printed WARNING, and a logged `ml.retrain_log` row with
      `promoted=False`.
- [x] A simulated bad promotion (`DummyClassifier`, PR-AUC far below the real champion's
      ~0.888) is demonstrably rolled back via one-command `make rollback VERSION=N` --
      `test_bad_promotion_is_rolled_back_to_known_good_version`, verified Production held
      the bad version, then held the known-good version again after rollback.
- [x] `ml.retrain_log` populated with real rows from the test runs -- 3 real rows verified
      in the dev database after this milestone's test runs (one per test file).
- [x] All new tests pass in this dev environment: `test_retrain_trigger.py`,
      `test_version_mismatch.py`, `test_rollback.py` (3 passed); full suite still green
      (15 passed, excluding the container-dependent `test_api_contract.py`, same exclusion
      pattern as `test-unit`'s CI job).
- [x] `.github/workflows/ci.yml`'s third job (`retrain-trigger-test`) added, depends on
      `drift-check` -- expected/documented to report "3 skipped" in CI itself (see Decision
      above for why, and why that's the honest outcome rather than a false "3 passed").
- [x] Auto-promotion never happens anywhere in this codebase outside the two
      manual/human-invoked exceptions -- verified by
      `grep -rn 'stage="Production"' domains/ core/`: exactly `register_model.py` (M5,
      `make register-model`) and `rollback.py` (M7, `make rollback`); `retrain_trigger.py`
      only ever transitions to `"Staging"` (see Decision above on why `register_model.py`
      counts as the same class of exception as the acceptance criteria's literal "rollback.py and the
      manual CLI command" wording).
- [x] `02_TRIALOUTCOME_SPEC.md` M7 row updated to ✅.
- [x] `decisions.md` updated with M7 entries (this section).

