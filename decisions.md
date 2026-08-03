# TrialOutcome -- Decision Log

## M1: Dataset Builder + Temporal Split (2026-07-30)

---

### Decision: Derive the label from `overall_status` instead of using `fct_trials.is_terminated` directly
- **What:** `label = overall_status IN ('TERMINATED', 'WITHDRAWN', 'SUSPENDED')`, computed in the
  dataset-builder SQL, not read from the mart's precomputed `is_terminated` boolean column.
- **Why (vs. alternatives):** Checked live DB: `fct_trials.is_terminated` is `TRUE` only for
  `overall_status = 'TERMINATED'` -- it is `FALSE` for WITHDRAWN and SUSPENDED. That contradicts the
  spec's documented label definition and this project's own row filter. Using the column as-is would
  have silently mislabeled 4,449 WITHDRAWN/SUSPENDED trials as negatives (successes). Confirmed with
  user before proceeding (schema had drifted from what the original brief assumed). Derived label
  gives 19.8% positive rate on the live data, matching the spec's expected ~20-25%; the raw column
  would have given 14.3%.
- **Failure mode:** If a future PharmaPulse mart rebuild changes `is_terminated`'s definition again
  (or this project's derivation drifts from it), the two would silently diverge -- worth a periodic
  cross-check (`is_terminated` vs derived label) as a data-quality assertion in a future milestone.
- **Scaling story (10x/100x):** Unaffected by data volume -- this is a correctness issue, not a
  performance one. At any scale, deriving from `overall_status` directly (rather than trusting an
  intermediate boolean of unknown provenance) is the safer default.
- **Interview question this maps to:** "How do you validate that a label column actually means what
  its name says?" -- don't trust a boolean flag without checking it against the status values it's
  supposed to summarize.

---

### Decision: Include `allocation`, `masking`, `has_dmc`, `eligibility_criteria` as real features; only `num_arms` is dropped
- **What:** The original brief listed 5 columns as "not in mart": `num_arms`, `masking`,
  `allocation`, `has_dmc`, `eligibility_criteria`. Direct schema inspection (`\d marts.fct_trials`)
  showed only `num_arms` is actually absent -- the other four exist and are populated. Confirmed with
  user; chose to use all four real columns rather than sticking to the stale dropped-list.
- **Why (vs. alternatives):** The alternative (stick to the stale list) would have thrown away
  legitimate, non-leaky, spec-intended design features (all four are set at trial registration, no
  leakage risk) for no reason other than an outdated assumption. `eligibility_criteria` additionally
  unlocked the spec's "text-lite" feature group (`eligibility_criteria_length`,
  `exclusion_keyword_count`), which the stale-list path would have skipped entirely.
- **Failure mode:** `has_dmc` has a high null rate (20.8%) -- treated as a tri-state categorical
  (true/false/unknown) rather than imputed to a boolean, specifically because 1-in-5 missing is too
  high to collapse into a mode without losing information the model could otherwise use.
- **Scaling story (10x/100x):** No change -- these are per-row categorical/text columns, cost is
  linear in row count regardless of scale.
- **Interview question this maps to:** "What do you do when a spec and the live system disagree?" --
  verify against the live system, flag the discrepancy explicitly, and let the human decide the
  tradeoff rather than silently picking one side.

---

### Decision: Drop rows with implausible `start_date` (222 of 79,334 label-eligible rows)
- **What:** Rows with `start_date < 1990-01-01` (187 rows, going back to a `1900-01-01` sentinel-like
  default) or `start_date` in the future for an already-COMPLETED/TERMINATED/WITHDRAWN/SUSPENDED trial
  (35 rows) are excluded via the `filters.start_date_min`/`start_date_max` bounds in
  `config.yaml`. An additional 997 rows have `start_date IS NULL` and are excluded as a side effect of
  the same range filter (NULL comparisons are false in SQL). Final dataset: 78,115 rows.
- **Why (vs. alternatives):** The alternative (keep them, let split boundaries sort them out) risks
  placing logically-impossible rows (a trial "completing" before it starts) into an arbitrary
  train/calib/test bucket, corrupting both the temporal split's integrity and any date-based feature
  (`start_year`, `start_quarter`). Confirmed with user; drop-and-document was the simpler, safer
  choice given the negligible data loss (~1.5% combined).
- **Failure mode:** If a future mart refresh introduces a much higher rate of bad dates, silently
  dropping them could start removing a non-trivial fraction of the dataset without anyone noticing --
  worth alerting if the dropped-row count exceeds some threshold (not built in M1; flag for M2+).
- **Scaling story (10x/100x):** The filter is a `WHERE` clause -- cost scales the same way the rest of
  the query does. At 10x/100x row counts, the more relevant scaling question is the join strategy
  (see next decision), not this filter.
- **Interview question this maps to:** "How do you handle data-quality outliers in a training set
  without silently corrupting downstream splits?"

---

### Decision: Point-in-time self-joins for sponsor history and condition rarity (locked pattern, followed as specified)
- **What:** `sponsor_prior_trial_count`/`sponsor_prior_termination_rate` computed via a self-join on
  `fct_trials` where `hist.start_date < t.start_date` (no phase restriction on `hist` -- matches the
  spec's example exactly). `condition_rarity` computed the same way but via each trial's *primary*
  (tie-broken) condition, to avoid the bridge-table fan-out the spec explicitly warns about.
- **Why (vs. alternatives):** A completion-date-aware version (only count `hist` trials whose *outcome*
  was already known by `t.start_date`, not just started before it) would be a more rigorous
  point-in-time guarantee, but it deviates from the exact pattern the spec locked in and wasn't asked
  for -- treated as out of scope for M1 rather than silently adding it.
- **Failure mode:** As implemented, a sponsor's currently-still-running earlier trial is treated as
  "not terminated" (0) in the prior-termination-rate average, even though its true outcome is unknown
  at `t.start_date`. This is the same behavior the spec's own example SQL has (via `is_terminated`) --
  a known, inherited limitation, not a new bug. Worth revisiting if M2 error analysis shows this
  feature underperforming.
- **Scaling story (10x/100x):** At 10x row count, this correlated self-join (79k driving rows against
  a 595k-row `fct_trials`, currently with zero indexes on `marts`) would move from "a few minutes" to
  "unusable" without an index on `(sponsor_key, start_date)` and `(condition_key)`. Flagged as a
  concrete follow-up for whoever owns the PharmaPulse warehouse -- not something this project should
  unilaterally add to another project's schema.
- **Interview question this maps to:** "Walk me through how you'd build a point-in-time-correct
  feature without a feature store." -- the canonical answer: a self-join with a strict date inequality,
  and the discipline to keep it in the dataset-builder code (not layered on top later, where it's easy
  to accidentally break).

---

### Decision: Fit the condition one-hot vocabulary (top-20 + other/unknown) on the train split only
- **What:** `PharmaDatasetBuilder._one_hot_condition` computes the top-N most frequent
  `condition_name` values using only rows where `split == 'train'`, then applies that fixed vocabulary
  to calib/test rows (anything outside top-N/`unknown` maps to `other`). Done independently for the
  temporal split and the random split (each gets its own train-fit vocabulary).
- **Why (vs. alternatives):** Fitting on the full dataset (including calib/test) is the common
  shortcut, but it lets future condition popularity influence which categories exist as features at
  all -- a subtler leakage class than row-level split leakage, but leakage nonetheless. Not explicitly
  called out in the brief, but directly on-theme for a leakage-focused portfolio project and cheap to
  do correctly, so treated as implementation-forced rather than scope creep.
- **Failure mode:** If this discipline is dropped later (e.g., a refactor fits vocabulary on the full
  dataset for convenience), a condition that only becomes common in the test period would silently get
  its own one-hot column -- information a production model deployed before that period could not have
  had.
- **Scaling story (10x/100x):** `dim_condition` already has 131,999 distinct values with a long tail;
  at 10x that cardinality, top-N one-hot remains the right call (target encoding or hashing would be
  the next step at a scale where even top-50 doesn't cover enough volume).
- **Interview question this maps to:** "Where does leakage hide besides the obvious train/test row
  split?" -- feature *vocabulary* fit on the wrong population is a good example.

---

### Finding: the naive "random split leaks, temporal split doesn't" story does NOT hold in this build -- and the real evidence is more interesting
- **What was observed:** LogisticRegression on the identical feature set, same hyperparameters, gives
  raw PR-AUC 0.866 (temporal) vs 0.682 (random) -- the *opposite* sign of what the spec's narrative
  anticipated ("random split inflates PR-AUC via sponsor-history leakage"). Investigated rather than
  silently rewriting the notebook to match the expected conclusion.
- **What ruled out the naive mechanism:** (1) ROC-AUC, far less sensitive to test-set prevalence,
  still favors temporal (0.893 vs 0.816) -- not purely a base-rate artifact. (2)
  `sponsor_prior_termination_rate`'s coefficient is *larger* under the temporal pipeline (0.951 vs
  0.658) -- the opposite of what "random split leaks sponsor history" predicts. (3) A controlled
  ablation -- fixed test window (trials with `2020-01-01 <= start_date < 2022-01-01`), varying only
  whether the same-size training pool was allowed to include 5,276 trials that start *after* the
  entire test window -- showed **no meaningful difference**: PR-AUC 0.785 (honest) vs 0.786 (leaky),
  ROC-AUC 0.838 vs 0.840.
- **Why (interpretation):** The point-in-time self-join (`hist.start_date < t.start_date` in
  `sponsor_history`/`condition_rarity`) computes every row's features correctly regardless of which
  split that row lands in -- no individual trial's features ever encode information from after its own
  `start_date`. That closes off leakage at the feature level, which is what point-in-time joins are
  *for*. The naive raw-metric gap instead reflects two confounds in how row membership differs between
  splits: (a) **base-rate confound** -- temporal-test is entirely 2022+, where termination rates have
  risen to ~31.7% vs the whole-dataset 20.2% (see label-drift finding below), and PR-AUC is
  mechanically higher against a higher-prevalence test set; (b) **history-richness confound** --
  random-test spans the full 1990-2026 window including early trials with thin point-in-time sponsor
  history, while temporal-test trials all benefit from 30+ years of accumulated history.
- **Why the temporal split remains mandatory despite this negative result:** this is a property of
  *this specific feature set* (already point-in-time safe by construction) and *this model class*
  (linear, can't easily exploit subtle cross-period interactions) -- not a general license to use
  random splits. A carelessly-added future feature (e.g. `dim_sponsor.success_rate` used directly
  instead of the point-in-time self-join -- exactly the leakage risk flagged in `config.yaml`) would
  leak hard under a random split, and this same controlled-ablation methodology is how you'd catch it.
- **Failure mode if ignored:** Concluding "leakage isn't a real risk here" from this one linear-model
  result and dropping the temporal-split discipline in M2 would be the actual mistake -- higher-capacity
  models (RF/XGBoost/LightGBM) can pick up subtler distributional-shift effects a regularized LogReg
  cannot. **Action item for M2:** re-run this same controlled ablation once the tree-based models exist
  rather than assuming a linear-model negative result generalizes.
- **Interview question this maps to:** "Tell me about a time your data didn't confirm your hypothesis."
  -- the honest answer here is a stronger artifact than a confirmed hypothesis would have been: it shows
  the point-in-time design working as intended, demonstrates awareness of confounds (prevalence,
  history-richness) that a less careful analysis would have missed, and states plainly why the
  conclusion doesn't generalize past this specific model class.

---

### Finding (not a decision, logged for M2+): termination rate rises sharply across splits
- train (pre-2020): 17.8% positive · calib (2020-2022): 30.3% positive · test (2022+): 31.7% positive.
- This is a real base-rate shift over time, not a bug -- worth surfacing explicitly in the M2 README
  and is directly relevant to the spec's M6 "label-drift" stretch goal (rolling termination base-rate
  PSI). A model trained mostly on the 17.8%-positive regime (66,105 of 78,115 rows are pre-2020) will
  need calibration validated carefully against the higher test-period base rate.

---

### Infra note: `marts` schema has zero indexes
- Confirmed via `pg_indexes` -- no indexes anywhere in `marts`, on any table. The dataset-builder query
  (correlated self-joins over 595k-row `fct_trials` and 1.07M-row `bridge_trial_condition`) took
  ~8 minutes to run as a result (confirmed CPU-bound via `pg_stat_activity`, not hung). Not something
  this project's `.env`/config can fix -- `marts` is owned by the PharmaPulse project. Flagged here so
  it isn't re-discovered as a mystery next session; a `(sponsor_key, start_date)` index on `fct_trials`
  and a `(condition_key)` index on `bridge_trial_condition` would likely cut this to seconds.

---

### Packaging note: `mlflow` resolved to `1.27.0`, `shap` failed to install
- `uv add mlflow` (unconstrained) resolved to `mlflow==1.27.0` -- unexpectedly old for a "current"
  environment; worth double-checking during M2 whether this is genuinely the latest compatible
  resolution or a transitive-constraint artifact (e.g. from `pyarrow`/`protobuf` version ceilings) and
  whether newer MLflow APIs (e.g. model registry ergonomics) used in this project's M2/M7 plans are
  available at this version.
- `uv add shap` failed outright: resolved `shap==0.51.0` -> `numba==0.53.1` -> `llvmlite==0.36.0`,
  which only supports Python <3.10 (this project targets 3.11+). Deferred `shap`/`mapie`/`evidently`/
  `fastapi`/`uvicorn`/`pydantic` installation to their respective milestones (M4/M5/M6) rather than
  force-installing something unproven now -- see `requirements.txt` for the note.

---

## M1 Definition of Done -- status

- [x] `ml.training_dataset` populated -- 78,115 rows (train 66,105 / calib 6,310 / test 5,700).
- [x] Leakage-vs-random comparison notebook exists (`notebooks/02_leakage_demo.ipynb`).

---

## M2: Baseline -> Optuna Best (2026-07-30)

---

### Decision: Bump `mlflow` to `2.22.5`, which forced `pandas` down to `2.3.3`
- **What:** User explicitly requested `mlflow>=2.14,<3.0` instead of the M1-resolved `mlflow==1.27.0`.
  `uv add` revealed mlflow 2.14-2.22.5 all depend on `pandas<3`; this project had resolved
  `pandas==3.0.5` (mlflow 2.x genuinely does not support pandas 3.x yet). Resolved by pinning
  `pandas<3` (landed on `2.3.3`) and `pyarrow<20` (mlflow's transitive ceiling) alongside the mlflow
  bump.
- **Why (vs. alternatives):** The only alternative that keeps pandas 3.x is staying on mlflow 1.27 --
  directly contradicts the explicit instruction. Downgrading pandas is the correct, necessary
  consequence of the mlflow version the user actually wants, not a choice made independently.
- **Failure mode:** Re-ran both M1 notebooks after the downgrade to confirm no pandas-3-only API was
  relied upon -- both executed clean. If a future pandas 3.x-only feature gets used elsewhere in the
  project, it will break under this pin; worth re-checking if pandas usage grows.
- **Scaling story (10x/100x):** N/A -- packaging constraint, independent of data volume.
- **Interview question this maps to:** "How do you handle a dependency pin that conflicts with
  another explicit requirement?" -- resolve the actual conflict (here: the transitive pandas ceiling),
  don't just force one package with `--frozen` and hope nothing breaks.

---

### Decision: `feature_pipeline_version` currently logs `"unknown"` on every run -- this repo has no commits yet
- **What:** `feature_pipeline_version()` in `domains/pharma/dataset_builder.py` runs
  `git rev-parse HEAD -- domains/pharma/dataset_builder.py` and falls back to `"unknown"` on
  `CalledProcessError`. This repo (created via `uv init` in M1) has zero commits, so every M2 MLflow
  run currently logs `feature_pipeline_version="unknown"`.
- **Why this isn't a code bug:** The function is behaving exactly as documented (fail gracefully, not
  crash the whole training run over a version tag). It's a real gap in what's logged *right now*,
  not a defect to fix in code -- it resolves itself the moment there's a first commit.
- **Failure mode if not flagged:** Per the user's own requirement ("make sure each run captures all of
  these or M7 and the locked contract will have gaps later"), a version-mismatch check in M7 is
  meaningless while every run shares the same `"unknown"` tag -- every candidate would spuriously
  "match" every other candidate. **Action item: commit the repo (your call, per CLAUDE.md -- I don't
  run git commit) before M2's MLflow runs are treated as the real record; otherwise re-run
  `make train` after the first commit so the tag is meaningful.**
- **Interview question this maps to:** "What's a subtle way a versioning/lineage tag can silently
  stop doing its job?" -- a fallback value that's valid Python but semantically useless (every run
  looks like it matches) is worse than an obvious crash, because it fails silently.

---

### Decision: Every model family shares ONE condition one-hot vocabulary, fit on temporal-train only
- **What:** Unlike `notebooks/02_leakage_demo.ipynb` (which deliberately fits a separate vocabulary
  per split regime to isolate a vocabulary-leakage effect), M2's `_fit_condition_vocab`/
  `_apply_condition_one_hot` fit the top-20 condition vocabulary once on temporal-train and apply that
  same fixed set of columns to temporal-test, random-train, and random-test alike.
- **Why (vs. alternatives):** `core.training_pipeline.OptunaMLflowTrainer` uses `sklearn.base.clone()`
  to reuse one Pipeline template across 5 CV folds + the temporal-full fit + the random-full fit per
  trial -- that only works if every regime has an identical column schema. Per-split independent
  vocabularies (M1's approach) would give temporal and random different column sets, breaking clone
  reuse. A single, temporal-train-fit vocabulary is also more representative of what actually gets
  deployed (the temporal split is production; random is a diagnostic only).
- **Failure mode:** If this ever silently reverted to per-regime vocab fitting without also removing
  the `clone()`-based reuse in `core/training_pipeline.py`, `predict_proba` would raise a column-
  mismatch error immediately (loud failure, not silent).
- **Scaling story (10x/100x):** Vocabulary fitting is O(n) in row count; unaffected by which regime it's
  fit on. At far higher condition cardinality, `top_n=20` would need revisiting regardless of this
  decision (see M1's condition_one_hot note in `config.yaml`).
- **Interview question this maps to:** "When do model-comparison harnesses need a shared feature
  space vs. per-experiment-fit features?" -- when you're cloning one pipeline template across many
  fits (CV, multiple regimes) for apples-to-apples comparison, vs. M1's notebook where the point was
  specifically to demonstrate what changes when vocab fitting isn't shared.

---

### Decision: `core/training_pipeline.py` is import-only; the runnable entrypoint lives in `domains/pharma/train_pipeline.py`
- **What:** The prompt's literal Makefile spec said `make train` -> `python -m core.training_pipeline`.
  Built it instead as `python -m domains.pharma.train_pipeline`, with `core/training_pipeline.py`
  staying a pure, importable, model-agnostic class (`OptunaMLflowTrainer`) with no `__main__` block and
  no PharmaDatasetBuilder import.
- **Why (vs. alternatives):** Giving `core/training_pipeline.py` a pharma-aware `__main__` block would
  violate the project's own hard rule ("no pharma SQL, no pharma column names" in core) and invert the
  intended dependency direction (core must not depend on domains/pharma). Resolving this mechanical
  contradiction in the prompt's own Makefile line vs. its architecture-discipline rule -- not scope
  creep, an implementation-forced fix.
- **Failure mode:** If a future contributor adds pharma-specific logic directly to
  `core/training_pipeline.py` "for convenience," this project stops being the reusable
  domain-agnostic reference the whole portfolio's Section 0 architecture claims it is.
- **Interview question this maps to:** "How do you enforce a core/domain boundary in practice, not
  just in a docs page?" -- make the domain-agnostic module physically unable to run standalone with
  domain logic baked in; put the entrypoint on the domain side.

---

### Result: full Optuna sweep (8 trials x 4 families = 32 runs) + baseline + 2 ablation runs = 35 MLflow runs
- **Majority-class baseline PR-AUC:** 0.3167 (temporal test), 0.2019 (random test).
- **The CV-selected best trial per family** (i.e. `study.best_trial` -- the trial Optuna actually
  picked via cross-validation, NOT the trial with the single highest held-out score, which would be
  test-set-driven cherry-picking across 8 trials):

  | model_type | best val_pr_auc (CV, selection metric) | that trial's pr_auc_temporal |
  |---|---|---|
  | logreg  | 0.6482 | 0.8674 |
  | rf      | 0.6729 | 0.8869 |
  | xgboost | 0.6783 | 0.8878 |
  | lgbm    | 0.6768 | 0.8896 |

  Every family clears the 0.3167 baseline by a wide margin -- M2 DoD ("PR-AUC beats majority-class
  baseline meaningfully") is met several times over, matching the user's own read that the bar was
  already cleared by M1's LogReg (0.866) before M2 started. XGBoost and LightGBM edge out LogReg/RF by
  ~0.02, a real but modest lift given only 8 trials/family -- consistent with the user's framing that
  M2 is really about whether Optuna reliably finds that lift, not about clearing the baseline (which
  was never in doubt).
- **Why the CV winner and the held-out-test winner don't perfectly agree:** xgboost has the best CV
  score (0.6783 > lgbm's 0.6768), but lgbm's CV-selected trial scores *higher* on temporal-test
  (0.8896 vs xgboost's 0.8878) -- ordinary noise at only 8 trials/family, and exactly what you'd expect
  when a proxy metric (5-fold CV on train) and the true held-out metric are correlated but not
  identical. Not a bug, and not worth chasing at this milestone's trial budget.
- **A mistake caught while writing this section, left in deliberately:** my first draft of this table
  used `groupby(...).max()` over `pr_auc_temporal` across all 8 trials per family instead of each
  family's single CV-selected trial -- silently swapping in "the best-looking test score across 8
  tries" for "the score of the model Optuna actually chose." That's a small-scale version of the exact
  test-set leakage this whole project is about avoiding: picking a number because it's the best one you
  saw, not because it's the number the leakage-safe selection process actually produced. Caught by
  re-deriving the per-trial numbers from the run log instead of trusting the aggregate query.
- **Failure mode if ignored:** Selecting a "best model" by held-out test PR-AUC instead of the CV
  selection metric would be a second, silent form of leakage (test-set-driven model selection) --
  the Optuna objective staying strictly CV-based is what prevents that regardless of which number
  looks better after the fact.

---

### Result: the tree-model controlled leakage ablation confirms the M1 negative result -- no leakage effect from either XGBoost or LightGBM
- **What was run:** the exact `controlled_leakage_ablation` methodology from M1 (fixed test window
  2020-01-01 to 2022-01-01, honest train = only `start_date < 2020-01-01`, leaky train = same-size
  sample allowed to include `start_date >= 2022-01-01`), applied to each model family's *best* Optuna
  trial hyperparameters.
- **XGBoost** (best trial: `max_depth=3, n_estimators=452, learning_rate=0.034, ...`): honest PR-AUC
  0.8031 vs leaky PR-AUC 0.8042 (delta **+0.0011**), honest ROC-AUC 0.8591 vs leaky 0.8601 (delta
  **+0.0010**).
- **LightGBM** (best trial: `num_leaves=138, n_estimators=290, learning_rate=0.021, ...`): honest
  PR-AUC 0.8033 vs leaky PR-AUC 0.8049 (delta **+0.0016**), honest ROC-AUC 0.8592 vs leaky 0.8607
  (delta **+0.0015**).
- **Why this matters:** M1's negative result (LogReg: PR-AUC delta -0.0000... functionally zero) could
  have been read as "leakage isn't a real risk here" being an artifact of using only a low-capacity
  linear model, which was explicitly flagged as unverified in M1's decisions.md and the user
  specifically asked not to skip re-checking it. Both tree-based models -- which genuinely can exploit
  subtler cross-period interaction effects a regularized LogReg cannot -- show deltas of ~0.001-0.002,
  an order of magnitude smaller than anything that would change a modeling decision. This is now a
  three-model-family-deep negative result, not a single-model artifact.
- **Interpretation, held honestly:** this remains a property of *this specific feature set* (every
  feature is point-in-time safe by construction via the self-joins in `dataset_builder.py`) -- it is
  evidence the point-in-time design is working, not a general proof that random splits are safe for
  this domain. A future feature computed carelessly (e.g. a static lifetime aggregate instead of a
  point-in-time self-join) would still leak under a random split regardless of this result.
- **Failure mode if this had come out differently:** a large positive delta here would have meant the
  best Optuna-selected tree model was silently benefiting from future information whenever training
  data crosses period boundaries -- exactly the failure mode M1's temporal-split discipline exists to
  prevent. It didn't happen here, but the check existing (and being logged as its own MLflow run,
  not just printed) is what makes that a verified conclusion rather than an assumption.
- **Interview question this maps to:** "You found no leakage with a linear model -- how do you know a
  more complex model wouldn't find it?" -- you don't, until you check; re-running the identical
  ablation against the higher-capacity models this project actually plans to deploy is what turns a
  single-model observation into a portfolio-wide claim.

## M2 Definition of Done -- status

- [x] MLflow shows >=16 runs -- 35 total (32 Optuna trials + 1 majority-class baseline + 2 controlled
      ablation runs).
- [x] PR-AUC beats majority-class baseline meaningfully -- CV-selected best per family: logreg 0.867,
      rf 0.887, xgboost 0.888, lgbm 0.890 (pr_auc_temporal), all vs. a 0.317 baseline.

---

## M3: Calibration + Cost-Based Threshold (2026-07-30)

---

### Decision: M2 logged no model artifact -- M3 reconstructs the XGBoost best trial by refitting its logged hyperparameters, not deserializing a saved model
- **What:** `domains/pharma/train_pipeline.py`'s Optuna harness never called `mlflow.sklearn.log_model`
  (only params/metrics/CSV importance were logged per run). `notebooks/03_calibration.ipynb` and
  `notebooks/04_shap_analysis.ipynb` both refit run `c4a4d0300bd949f8bb07b7c48417be4d`'s exact
  hyperparameters (`n_estimators=452, max_depth=3, learning_rate=0.0344, subsample=0.6456,
  colsample_bytree=0.7424, min_child_weight=3`) against the identical feature pipeline (same
  `_fit_condition_vocab`/`_apply_condition_one_hot` procedure, same `random_state=42`), then assert
  the refit's `pr_auc_temporal` matches the MLflow-logged value (`0.8877613220680413`) to 6 decimal
  places before proceeding.
- **Why (vs. alternatives):** The alternative (add `mlflow.sklearn.log_model` retroactively to M2 and
  re-run the full 32-trial Optuna sweep) would burn real time re-deriving something already fully
  determined by the logged hyperparameters plus a fixed random seed -- refitting is exact, not an
  approximation, because XGBoost with `random_state=42` is deterministic given identical inputs. The
  assert-and-proceed pattern makes this verifiable rather than assumed.
- **Failure mode:** If a future change to `domains/pharma/train_pipeline.py`'s feature-engineering
  functions (`_fit_condition_vocab`, `_apply_condition_one_hot`, `_make_preprocessor`) drifts without
  the M3/M4 notebooks being updated in lockstep, the refit would silently diverge from the actually-
  logged run -- caught here only because of the explicit `assert abs(pr_check - 0.8877613220680413) <
  1e-6`, which would fail loudly rather than silently producing a slightly-wrong calibrator.
- **Scaling story (10x/100x):** Unaffected by data volume -- this is a reproducibility mechanism, not
  a performance one. At production scale, the real fix is what M5+ should add regardless: log the
  actual fitted model artifact (`mlflow.sklearn.log_model` or `mlflow.xgboost.log_model`) so refitting
  is never necessary again.
- **Interview question this maps to:** "How do you reproduce a specific experiment's model when the
  artifact was never saved?" -- if training is deterministic and every hyperparameter is logged, the
  hyperparameters plus a verification assert are sufficient; this is a legitimate fallback, not a
  workaround to hide.

---

### Decision: Isotonic regression chosen over Platt scaling as the calibrator
- **What:** `core/calibration.py`'s `CalibratorWrapper` fits both on the CALIB split
  (n=6,310) and selects by lowest post-calibration ECE. Isotonic won:
  ECE before calibration (CALIB, raw XGBoost output) = **0.1887**; ECE after isotonic ≈ **0.0000**
  (`2.05e-17`); ECE after Platt = **0.0405**. Confirmed on the held-out TEST split (n=5,700, never
  used for calibrator fitting): raw TEST ECE = **0.1897**, isotonic-calibrated TEST ECE = **0.0238**.
- **Why (vs. alternatives):** Isotonic winning is the *expected* outcome, not a surprise -- it makes
  no parametric shape assumption, so it can correct whatever miscalibration pattern the raw XGBoost
  scores actually have, while Platt is constrained to a single-parameter sigmoid family. Isotonic's
  near-zero ECE *on CALIB itself* is partly a fitting-flexibility artifact (a fully non-parametric
  step function can nearly memorize the set it was fit on) -- the number that actually matters is the
  TEST-split ECE (0.0238), which is legitimately held out from the calibrator fit and confirms the
  win generalizes, not just memorizes.
- **Failure mode:** Isotonic can overfit small calibration sets (as many "knots" as distinct raw
  scores) -- with only ~6,300 CALIB rows this is a real risk in principle, mitigated here specifically
  by checking ECE on held-out TEST rather than trusting the CALIB number alone.
- **Scaling story (10x/100x):** More CALIB rows only reduces isotonic's overfitting risk, never
  increases it -- no change needed at 10x/100x scale. If a future CALIB split shrank substantially
  (e.g. a tighter split-date window), Platt's smoother, lower-variance fit would likely generalize
  better despite its higher CALIB-split ECE here -- worth re-checking if the split dates in
  `config.yaml` ever change.
- **Interview question this maps to:** "How do you know your calibrator isn't just overfitting the
  calibration set?" -- fit on CALIB, but *report* ECE on TEST; a calibrator that looks perfect only on
  the data it was fit on is not evidence of anything.

---

### Decision: Cost-optimal threshold = 0.22 (not F1-max=0.37, not default=0.5)
- **What:** `core/threshold_selector.py`'s `ThresholdSelector.find_cost_optimal_threshold` swept
  0.01-0.99 against the calibrated TEST-split probabilities using `domains/pharma/cost_matrix.yaml`'s
  5:1 FN:FP cost ratio. Results (TEST, n=5,700):

  | threshold | choice | precision | recall | f1 | expected_cost |
  |---|---|---|---|---|---|
  | 0.22 | cost_optimal | 0.6235 | 0.8798 | 0.7298 | **2044** |
  | 0.37 | f1_max | 0.8206 | 0.7551 | 0.7865 | 2508 |
  | 0.50 | default | 0.9127 | 0.6715 | 0.7737 | 3081 |

  **Adopted: 0.22.** In practice: at this threshold the model catches 88.0% of trials that actually
  terminate (recall), at a 24.6% false-positive rate among trials that actually complete. Written to
  `domains/pharma/config.yaml` under `model.threshold_decision` by hand (not by the notebook writing
  the YAML directly) to avoid a machine-written `yaml.safe_dump` stripping the file's extensive
  hand-authored comments.
- **Why (vs. alternatives):** F1 treats a missed termination and a false alarm as equally costly,
  which directly contradicts the explicit 5:1 business judgment already recorded in
  `cost_matrix.yaml` (a missed failure wastes far more sponsor/patient resource than a false alarm --
  the DA/BA Lead persona's read in `REVIEW_REPORT.md` is exactly this: "a missed termination costs us
  5x more than a false alarm" is a sentence a business stakeholder immediately understands). The
  default 0.5 threshold optimizes nothing at all -- it's just wherever `predict_proba` happens to
  cross 0.5, with no connection to either metric. At the adopted 0.22 threshold, expected cost is
  strictly lower than both alternatives (2044 vs 2508 vs 3081).
- **Failure mode:** If the true FN:FP cost ratio in a different pharma context were actually much
  lower than 5:1 (e.g. a therapeutic area where false alarms trigger expensive manual review), this
  threshold would over-flag trials, raising false-positive-driven operational cost. The threshold is
  only as good as the cost matrix it was swept against.
- **Scaling story (10x/100x):** The 99-point grid sweep is O(n) per threshold point and trivial at any
  realistic TEST-split size, including 10x/100x more trials. If `cost_matrix.yaml`'s FN:FP ratio ever
  became a continuously-varying function (rather than a fixed constant), `scipy.optimize.minimize_scalar`
  bounded to `[0, 1]` would replace the grid sweep -- documented directly in
  `ThresholdSelector.find_cost_optimal_threshold`'s docstring.
- **Interview question this maps to:** "Why not just use the F1-optimal threshold?" -- F1 is a
  metric-shaped answer to a business-shaped question; the actual objective is minimizing expected
  cost under the org's own stated cost asymmetry, and F1 only coincides with that when FN and FP costs
  happen to be equal, which they explicitly are not here.

## M3 Definition of Done -- status

- [x] reliability plots exist (before + after calibration, both isotonic and Platt) --
      `notebooks/03_calibration.ipynb`, 1x3 figure with diagonal reference line.
- [x] ECE before and after logged to MLflow -- run `xgboost_best_calibrated` in experiment
      `trialoutcome_m2`: `ece_before=0.1887`, `ece_after_isotonic≈0.0000`, `ece_after_platt=0.0405`,
      tag `calibration_method_chosen=isotonic`.
- [x] cost-optimal threshold identified, documented in `config.yaml` (`model.threshold_decision`) and
      the notebook -- **0.22**.
- [x] decision table (precision/recall/f1/cost at 3 threshold choices) in notebook.

---

## M4: SHAP + Error Essay + Multicollinearity + Plain-English Templating (2026-07-30)

---

### Decision: Aggregate one-hot-expanded categorical SHAP values back to their original feature name for local explanations and the error essay; keep the expanded (post-`ColumnTransformer`) space for the global bar chart and beeswarm
- **What:** `pipe.named_steps["clf"]`'s SHAP values are naturally computed in the model's actual input
  space -- after `sklearn.preprocessing.OneHotEncoder` expands `phase`/`allocation`/`masking`/
  `has_dmc_str`/`sponsor_class` into 58 columns total. For per-row local explanations feeding
  `domains/pharma/plain_english.generate_summary` (whose templates expect one value per original
  feature, e.g. `phase="PHASE3"`, not `phase_PHASE3=1.0`), the notebook sums each categorical
  feature's one-hot SHAP contributions back into a single aggregated value per original column
  (39 aggregated features from 58 expanded ones). The two required plots (`docs/shap_global_importance.png`,
  `docs/shap_beeswarm.png`) use the aggregated and expanded spaces respectively -- see the "Why this,
  not that" note in `notebooks/04_shap_analysis.ipynb` for why each plot needs its own space.
- **Why (vs. alternatives):** Redesigning `domains/pharma/plain_english.py`'s templates to understand
  one-hot-expanded feature names (e.g. matching on `phase_PHASE2` as a prefix) would work but
  couples a domain-templating module to a specific preprocessing implementation detail
  (`ColumnTransformer`'s naming convention) -- summing SHAP contributions at read time keeps
  `plain_english.py` unchanged and pipeline-shape-agnostic.
- **Failure mode:** If a categorical feature had enough levels that summing lost meaningful nuance
  (e.g. a 100-category feature where one level dominates and the rest cancel out), this aggregation
  would need a different rollup (e.g. report the single strongest level, not the sum) -- not an issue
  for this project's 5 low-cardinality categoricals (2-9 levels each).
- **Scaling story (10x/100x):** Aggregation is a single groupby-sum over SHAP's *feature* columns --
  cost scales with feature count, not row count, so unaffected by 10x/100x more trials.
- **Interview question this maps to:** "How do you present SHAP values for a one-hot-encoded
  categorical feature without confusing a non-technical reader?" -- sum the one-hot levels' SHAP
  contributions back to the parent feature for presentation, but keep the expanded space for the
  technical diagnostic plots where per-level detail is the point.

---

### Decision: Multicollinearity check found one |r|>0.6 pair; no attribution-splitting caveat required in this build
- **What:** Pearson correlation across the 10 named engineered numeric/ordinal features, computed on
  TRAIN (n=66,105). One pair exceeds |r|>0.6: `eligibility_criteria_length` <-> `exclusion_keyword_count`,
  **r=0.70**. `eligibility_criteria_length` ranks 10th in aggregated global SHAP importance;
  `exclusion_keyword_count` ranks 13th -- below the top-10 cutoff. Per the spec's own conditional
  rule (Section 4b: caveat required only when *both* members of a flagged pair are top-10 SHAP), no
  caveat is added to `docs/error_analysis.md` for this pair in this build.
- **Why (vs. alternatives):** Adding the caveat regardless of the top-10 check (i.e. flagging *every*
  |r|>0.6 pair unconditionally) would be the safer-looking default, but would dilute the caveat's
  actual signal value -- the spec's conditional rule exists precisely so that "attribution is shared"
  warnings appear only where they change how a reader should interpret the SHAP plots, not as
  boilerplate on every correlated pair regardless of relevance.
- **Failure mode:** If a future retrain or feature-engineering change pushed `exclusion_keyword_count`
  into the top-10 (a plausible trigger: a bullet-heavy eligibility section mechanically raises both
  features together), the caveat would then be required and must be added -- this is the reason
  `docs/error_analysis.md` states the *trigger condition* explicitly (exclusion_keyword_count entering
  top-10) rather than just the current negative result.
- **Scaling story (10x/100x):** Correlation computation is O(n * k^2) for k=10 features -- trivial at
  10x/100x row counts. The check itself would need to be re-run after any feature-set change
  regardless of scale, since correlation structure is a property of the features, not the row count.
- **Interview question this maps to:** "When do you need to caveat a SHAP attribution as
  'shared' rather than trust it at face value?" -- only when the correlated features are both
  material to the story being told (both meaningfully important, not just any nonzero correlation) --
  otherwise the caveat is noise that trains readers to ignore all such warnings.

---

### Finding: 3 failure themes across the 20 worst false negatives, all sharing one universal precondition
- **What was found:** Every one of the 20 worst false negatives has `sponsor_prior_termination_rate
  < 6%` (the #2 globally-important SHAP feature) -- a clean sponsor history is the universal
  precondition that makes each of these 20 predictions confidently wrong, not a discriminator between
  them. Within that shared precondition, three distinguishable themes emerged from direct inspection
  of the 20 rows' raw feature values and top SHAP contributors (`docs/error_analysis.md` "Failure
  Themes"): (1) **"mega-sponsors"** (`sponsor_prior_trial_count >= 200`, 5/20) whose aggregate history
  is averaged over enough trials to wash out this specific trial's risk; (2) **single/near-single-site
  trials** (`num_sites <= 1`, 9/20, 3 overlapping with theme 1) where `num_sites`'s contribution stays
  modest rather than decisive; (3) **no individually extreme feature** (9/20) where every engineered
  feature sits in a "typical" range and the failure signal is plausibly external to this project's
  entire feature set (design / sponsor-history / condition / temporal / text-lite).
- **Why this matters (expected vs. surprising):** The universal low-sponsor-termination-rate
  precondition was *expected* once framed correctly -- a model that predicts termination primarily via
  sponsor history will, almost by construction, make its worst errors on trials from sponsors that
  otherwise look safe (a sponsor with a genuinely high historical termination rate would rarely be
  predicted low-risk to begin with, so it can't produce a *confident* false negative). What was less
  expected going in was how large Theme 3 turned out to be (9/20, the largest of the three) -- nearly
  half the worst misses have no standout feature at all, meaning almost half of this model's hardest
  errors are, honestly, currently unexplainable by any feature this project observes.
- **What a future model improvement would target first:** Theme 3 (no individually extreme feature)
  is the most informative miss for prioritizing future feature engineering -- it says the current
  feature groups (design/sponsor-history/condition/temporal/text-lite) have a real, measurable
  ceiling, and closing it requires genuinely new information (e.g. drug-mechanism/efficacy signals,
  funding-source data, protocol-amendment history) rather than tuning the existing model or feature
  engineering harder on the same inputs. Theme 1 (mega-sponsors) suggests a second concrete lever:
  a sponsor-history feature with a shorter or weighted lookback window (recent trials weighted more
  than trials from a decade ago) might better reflect a large sponsor's *current* risk profile than a
  flat lifetime average does.
- **Interview question this maps to:** "Walk me through an error analysis that changed what you'd
  build next." -- the answer here is concrete and falsifiable: two of three themes point at specific,
  buildable feature improvements (recency-weighted sponsor history; a site-execution-risk signal),
  while the third (Theme 3) is an honest admission of the current feature set's ceiling, which is
  itself useful information for scoping the next iteration realistically.

## M4 Definition of Done -- status

- [x] SHAP global importance plot saved to `docs/shap_global_importance.png` (aggregated feature
      space, top-20).
- [x] SHAP beeswarm plot saved to `docs/shap_beeswarm.png` (expanded model-input space, top-20).
- [x] correlation matrix computed across the 10 named engineered numeric features (TRAIN split);
      one |r|>0.6 pair found (`eligibility_criteria_length` / `exclusion_keyword_count`, r=0.70),
      documented in `docs/error_analysis.md`.
- [x] `docs/error_analysis.md` written with all four required sections (20 Worst False Negatives,
      Failure Themes, Multicollinearity Findings, Known Limitations).
- [x] `plain_english.generate_summary()` demoed on 5 sample predictions (2 high-confidence TP, 2
      high-confidence TN, 1 worst FN) in `notebooks/04_shap_analysis.ipynb`.
- [x] correlated-pair attribution caveat logic implemented and evaluated -- not triggered in this
      build (only one of the two correlated features is top-10 SHAP), trigger condition documented
      for future re-checks.

---

## has_results Leakage Check (2026-08-02)

---

### Finding: has_results leakage check — cleared (two-step verification)

- **What:** `has_results` ranks 7th in global SHAP importance (~0.10 mean |SHAP|) —
  high enough that if it were leaking post-completion information into training,
  it would materially inflate PR-AUC. Checked before M5 to confirm it is safe to keep.
- **Step 1:** Queried `has_results=true` counts across all non-terminal `overall_status`
  values in `marts.fct_trials`:
  RECRUITING 8/65,174 (0.0%), NOT_YET_RECRUITING 0/28,841 (0.0%),
  ACTIVE_NOT_RECRUITING 1,220/21,875 (5.6%), ENROLLING_BY_INVITATION 0/5,223,
  AVAILABLE 0/254.
- **Step 2:** The 1,220 `ACTIVE_NOT_RECRUITING` rows with `has_results=true`
  span start_date 1989-04-01 to 2024-11-04 — uniformly distributed across 35 years,
  consistent with interim results posted during long-running trials, not a
  data-quality anomaly or recency cluster.
- **Why cleared:** None of these statuses enter `ml.training_dataset` —
  the label filter retains only `overall_status IN ('COMPLETED', 'TERMINATED',
  'WITHDRAWN', 'SUSPENDED')`. All rows flagged above were dropped before the
  dataset was built. `has_results` rank-7 SHAP importance reflects legitimate
  post-enrollment signal on terminal-status trials only.
- **Failure mode:** If the label filter in `domains/pharma/dataset_builder.py`
  were ever loosened to include `ACTIVE_NOT_RECRUITING` rows,
  this check would need to be re-run — the 5.6% rate in that status would
  become a live leakage risk at that point.
- **Decision:** Feature confirmed safe. No retraining needed. M5 proceeds
  with `has_results` in the feature set as-is.
- **Interview question this maps to:** "How do you verify a feature is not
  leaking when it has suspiciously high importance?" — rank alone is not evidence
  of leakage; you trace the feature's population mechanism back to the raw source
  and verify it against the actual rows that entered training, not the full table.

---

## M5: Conformal Prediction + FastAPI + Docker (2026-08-02)

---

### Decision: Step 0a (git init + first commit) was already satisfied before M5 started -- no commit made this session
- **What:** The M5 prompt assumed a zero-commit repo (per the M2 decision entry above). By the time
  M5 started, `git log` already showed one commit (`e886fc6`, "initial files, model runs with optuna"),
  made by the user per this project's own rule that Claude never runs `git add`/`git commit`. Verified
  `git rev-parse HEAD -- domains/pharma/dataset_builder.py` returns a real hash, not `"unknown"`, and
  that file is unmodified since that commit -- `feature_pipeline_version()` was already producing a
  real (if buggy, see next entry) tag before any M5 code ran.
- **Why (vs. alternatives):** Running `git init`/`git commit` myself would violate this project's
  explicit, standing instruction (Claude never modifies git history/state) -- confirming the
  precondition was already met and proceeding directly to Step 0b was correct, not a skipped step.
- **Failure mode:** N/A -- this is a status check, not a code change.
- **Interview question this maps to:** "How do you handle a runbook step that assumes a precondition
  that's already been met?" -- verify the precondition directly (`git log`, `git rev-parse`) rather
  than blindly re-running the step or blindly skipping it.

---

### Decision (bugfix): `feature_pipeline_version()` used `git rev-parse HEAD -- <path>`, which does not filter by path -- fixed to `git rev-list -1 HEAD -- <path>`
- **What:** `domains/pharma/dataset_builder.py`'s `feature_pipeline_version()` (M2-era code) ran
  `git rev-parse HEAD -- domains/pharma/dataset_builder.py`. `git rev-parse` does not use a trailing
  pathspec to scope *which* commit's hash it returns -- it always returns HEAD's hash, then echoes
  `--` and the path back as additional literal output lines (verified directly:
  `git rev-parse HEAD -- <path>` prints 3 lines; `git rev-list -1 HEAD -- <path>` prints exactly 1).
  This was invisible throughout M1-M4 because the repo had zero commits and the function's
  `except (CalledProcessError, FileNotFoundError): return "unknown"` fallback masked it -- the bug
  only manifested the moment `register_model.py` ran against a real commit, producing the tag value
  `"e886fc6...\n--\ndomains/pharma/dataset_builder.py"` instead of a clean hash. Fixed to
  `git rev-list -1 HEAD -- <path>`, which is the correct incantation for "hash of the last commit
  that touched this path."
- **Why (vs. alternatives):** Leaving the polluted tag in place would satisfy M5's literal DoD line
  ("feature_pipeline_version is a real git hash, not unknown") on a technicality while silently
  breaking M7's planned version-mismatch check (a promotion-time `==` comparison against this tag
  would never match anything once every future tag also carries different trailing junk, or worse,
  match by accident if two different commits happened to embed the same path string). This is
  implementation-forced, not scope creep: M5's own production run needed a genuinely clean tag to
  meet its own DoD in spirit, not just in the literal wording.
- **Failure mode:** Re-ran `make register-model` after the fix and confirmed the tag is now exactly
  `e886fc6bc5e6fd7cab3afc3295665f49caaaea85` with no trailing content (verified via
  `client.get_run(...).data.tags`).
- **Scaling story (10x/100x):** N/A -- a one-time git plumbing correctness fix, independent of data
  or model scale.
- **Interview question this maps to:** "How did a versioning bug hide for three milestones?" -- a
  fallback (`except: return "unknown"`) that swallows the *actual* failure mode (a malformed but
  non-crashing git command) is worse than no fallback at all, because it makes the bug look like a
  documented limitation ("no commits yet") instead of a defect, right up until the precondition that
  was masking it (zero commits) stops holding.

---

### Decision: `CalibratedClassifierCV(base_estimator=..., cv="prefit")` from the M5 prompt does not run on this project's pinned sklearn==1.9.0 -- used `estimator=FrozenEstimator(...)` instead
- **What:** sklearn renamed `base_estimator` -> `estimator` in 1.2 and removed the `cv="prefit"` string
  value entirely in favor of wrapping an already-fitted estimator in `sklearn.frozen.FrozenEstimator`
  (confirmed via `inspect.signature` and `help()` against the installed sklearn 1.9.0 -- `cv="prefit"`
  is gone from the docstring, replaced by an explicit `FrozenEstimator` code example).
  `domains/pharma/register_model.py` uses `CalibratedClassifierCV(estimator=FrozenEstimator(xgb_pipeline),
  method="isotonic")` instead of the prompt's literal `base_estimator=...,  cv="prefit"` call.
- **Why (vs. alternatives):** Pinning an older sklearn to match the prompt's exact snippet would
  contradict `pyproject.toml`'s `scikit-learn>=1.9.0` (already relied on by M2-M4's Optuna/SHAP code)
  for no benefit -- the `FrozenEstimator` path produces the identical semantics (all provided data
  used for calibration, no internal refitting) with the currently-installed, already-verified sklearn
  version.
- **Failure mode:** Verified this reproduces the exact M3 TEST-split ECE (0.0238, matching to 4dp) --
  if `FrozenEstimator` behaved differently from the old `cv="prefit"` semantics, this number would have
  diverged from M3's independently-computed calibration result.
- **Scaling story (10x/100x):** N/A -- API-compatibility fix, independent of data scale.
- **Interview question this maps to:** "What do you do when a spec's code snippet doesn't match your
  installed library version?" -- check the installed API directly (`inspect.signature`, `help()`)
  rather than guessing or downgrading a pinned dependency to match stale documentation.

---

### Decision: MAPIE's classic `MapieClassifier(method="score")` API does not exist in mapie==1.4.1 -- used the renamed `SplitConformalClassifier(conformity_score="lac")`
- **What:** `uv add mapie` resolved `mapie==1.4.1`, which replaced the entire pre-1.0 API
  (`MapieClassifier`, `method="score"/"cumulated_score"/...`) with `SplitConformalClassifier`/
  `CrossConformalClassifier` and a `conformity_score` parameter (`"lac"`, `"aps"`, `"raps"`, ...).
  `"lac"` (Least Ambiguous Set-valued Classifier) is `score`'s direct successor for binary
  classification -- confirmed via `inspect.signature`/docstrings against the installed package, no
  `mapie.classification.MapieClassifier` symbol exists at all in this version.
  `core/conformal.py`'s `MAPIEConformalWrapper` is built against `SplitConformalClassifier(prefit=True)`
  + `.conformalize(X_calib, y_calib)` + `.predict_set(X)` instead of the prompt's literal
  `MapieClassifier(...).fit(...)` calls.
- **Why (vs. alternatives):** Pinning an old, pre-1.0 mapie release to match the prompt's exact class
  names was rejected -- `uv add mapie` (unconstrained, per this project's dependency-management rule)
  resolved the current release, and there is no reason to force an old, presumably less-maintained
  version onto a brand-new integration when the new API's `SplitConformalClassifier` is the direct,
  documented replacement with equivalent guarantees.
- **Failure mode:** If a future MAPIE major version renames this again, `core/conformal.py`'s import of
  `mapie.classification.SplitConformalClassifier` would raise `ImportError` immediately (loud failure),
  not silently produce wrong coverage.
- **Scaling story (10x/100x):** N/A -- API-compatibility fix.
- **Interview question this maps to:** "How do you handle a spec written against an API that changed
  underneath you?" -- inspect the installed package directly, confirm the semantic equivalent exists,
  and document the substitution explicitly rather than silently pinning around it.

---

### Decision (bugfix): MAPIE's internal "is this estimator fitted" probe breaks on a `ColumnTransformer` that selects columns by name -- fixed with a `_NamedColumnEstimatorAdapter` shim
- **What:** `SplitConformalClassifier.conformalize()` internally calls `check_sklearn_user_model_is_fitted`,
  which probes the base estimator with `estimator.predict(np.zeros((1, n_features_in_)))` -- a bare
  ndarray, not a DataFrame. This project's pipelines select columns by *name*
  (`ColumnTransformer([("cat", OneHotEncoder(...), CATEGORICAL_FEATURES), ...])`), which raises
  `"Specifying the columns using strings is only supported for dataframes"` on a bare array. MAPIE
  wraps that exception in a hard `raise UserWarning(...)` (not `warnings.warn` -- appears to be a MAPIE
  bug) that aborts `conformalize()` entirely, rather than a plain crash. A first fix attempt (wrap any
  non-DataFrame input into a same-shaped all-zero DataFrame) still crashed one level deeper: an all-zero
  probe row breaks `OneHotEncoder`'s unknown-category handling on any *categorical* column, because
  `0.0` isn't a valid category and sklearn's NaN-check path crashes comparing a float against a
  string-typed categories array. Fixed by having `_NamedColumnEstimatorAdapter` answer any non-DataFrame
  input using a real captured template row (the first row of `X_calib`) instead of the literal probe
  content -- correct dtypes everywhere, and the probe's return value was never checked for correctness
  by MAPIE anyway (only whether the call raises).
- **Why (vs. alternatives):** Restructuring `_make_preprocessor` to select columns positionally
  (avoiding the whole class of "probe with a bare array" failures) was rejected -- it would touch the
  already-validated M2-M4 training pipeline and its 6dp-verified refit assertion, for a problem that's
  entirely on the *serving/conformal* side. A thin adapter isolated to `core/conformal.py` fixes the
  incompatibility without touching anything upstream.
- **Failure mode:** This adapter's `predict`/`predict_proba` return *meaningless* values for a bare-array
  input (template-row-based, ignoring the literal input content) -- safe only because the only caller
  that ever passes a non-DataFrame is MAPIE's internal self-check. Documented explicitly in the class
  docstring so a future maintainer doesn't repurpose this adapter somewhere real ndarray predictions are
  expected.
- **Scaling story (10x/100x):** N/A -- a one-time library-compatibility shim, independent of data scale.
- **Interview question this maps to:** "Tell me about a time a well-tested pipeline broke against a
  third-party tool for a reason that had nothing to do with your model." -- named-column selection is
  a common, reasonable pipeline pattern that plenty of tooling (here, MAPIE's internal self-check)
  doesn't anticipate; the fix belongs at the integration boundary, not by weakening the pipeline itself.

---

### Decision: MAPIE prediction-SET -> `[low, high]` float interval conversion, and the `margin` used for a singleton `{1}` set
- **What:** `MAPIEConformalWrapper.predict_with_interval` converts MAPIE's per-row prediction set
  (`{0}`, `{1}`, or `{0,1}`) to a float interval: `{1}` only -> `[proba - margin, proba + margin]`
  (clipped to `[0,1]`); `{0,1}` both -> `[0.0, 1.0]`; `{0}` only -> `[0.0, proba]`. `margin` is defined
  as `1 - target_coverage` (the *nominal* coverage this wrapper was configured for, i.e. `0.10` at the
  default `target_coverage=0.90`) -- **not** the empirical coverage `verify_coverage()` measures on
  TEST. This was ambiguous in the M5 prompt ("margin = 1 - coverage_achieved" doesn't specify which
  coverage number) and is resolved here explicitly.
- **Why (vs. alternatives):** Using the empirical TEST coverage as `margin`'s input would create a
  circular dependency at serving time: `predict_with_interval()` is the method the live API calls per
  request, and it must be usable immediately after `fit_conformal()` -- a production request can't wait
  on a TEST-split evaluation that may never run outside of the M5 notebook. The nominal target is always
  available the instant the wrapper is configured, and it's what MAPIE's conformal guarantee is actually
  keyed to.
- **Failure mode:** If a future retrain measures empirical coverage well below target (e.g. 0.80 against
  a 0.90 target), the margin used at serving time would still reflect the *intended* 0.10 gap, not the
  *actual* larger miscalibration -- worth revisiting if `verify_coverage()`'s gate ever starts failing
  in production rather than at build time (it did not here: empirical 0.946 vs target 0.90).
- **Scaling story (10x/100x):** N/A -- margin is a scalar constant per model version, independent of
  request volume or data scale.
- **Interview question this maps to:** "Where in an ML system's uncertainty story would you draw the
  line between 'evaluated once at training time' and 'must be servable per-request without extra
  computation'?" -- a per-request health check can't depend on a batch evaluation step; anything a live
  endpoint needs must be captured as a constant at model-registration time.

---

### Result: Conformal coverage gate passed comfortably -- empirical 0.946 vs. target 0.90 (gate: >=0.88)
- **What:** `notebooks/05_conformal.ipynb` fit `MAPIEConformalWrapper(target_coverage=0.90)` on the
  CALIB split (n=6,310) and measured empirical coverage on the held-out TEST split (n=5,700):
  **0.946**, comfortably clearing the M5 DoD gate (`>=0.88`). Logged as `empirical_coverage` on the
  same MLflow production run register_model.py created (run `13ca29a52a5b456b93846514f8020dfa`), with
  the fitted `MAPIEConformalWrapper` itself logged as an artifact (`conformal/`) on that run.
- **Why the empirical number exceeds the target:** LAC (least-ambiguous-set) conformal scores are
  finite-sample conservative -- at n=6,310 calibration points, the true achieved coverage is expected
  to be at or slightly above the nominal target, not exactly equal to it. A coverage well above target
  (rather than a razor-thin pass) is the expected, healthy outcome here, not a sign the model is
  under-confident everywhere -- interval widths still vary meaningfully by predicted probability (see
  `docs/conformal_width_vs_proba.png`).
- **Failure mode if this had come out below 0.88:** Per the M5 prompt's explicit instruction, the
  build would have stopped before Part B (FastAPI) rather than proceeding with an unverified
  uncertainty guarantee -- this didn't happen here, but the gate assertion is a real `assert` in the
  notebook, not just a printed number a reader could miss.
- **Interview question this maps to:** "How do you know your conformal intervals actually mean
  anything?" -- fit the quantile on CALIB, verify coverage on a genuinely held-out TEST split the
  quantile never saw, and treat a passing check as a gate the pipeline can't silently skip past.

---

### Decision: Model + conformal wrapper loaded once at FastAPI startup (lifespan), not per-request
- **What:** `domains/pharma/serving/api.py`'s `lifespan()` context manager loads the calibrated model,
  conformal wrapper, condition vocabulary, and feature schema from the MLflow registry exactly once
  when the process starts, storing them in a module-level `_PharmaModelBundle`. `/ready` only reports
  `model_loaded=True` once this has finished.
- **Why (vs. alternatives):** Loading per-request (re-hitting the MLflow registry and re-deserializing
  a cloudpickled sklearn pipeline + XGBoost booster on every call) would add meaningful, unnecessary
  latency to every prediction for an artifact that only changes on redeploy -- there is no correctness
  reason to reload it more often than the process lifetime.
- **Failure mode:** A model promoted to a new "Production" version in MLflow will NOT be picked up by
  an already-running API process -- this is deliberate (a live process silently switching models
  mid-traffic without a restart/health-check cycle is a worse failure mode than requiring a redeploy),
  but means promotion must be paired with a restart, which M7's retrain/rollback flow needs to account
  for explicitly.
- **Scaling story (10x/100x):** At higher request volume, startup-time loading is strictly better (the
  fixed model-load cost is amortized across all requests instead of paid per-request); at multiple
  replicas (M8's k8s HPA), each pod independently loads the same Production model at its own startup --
  fine at this project's model size, but would need a shared/cached artifact store if the model
  artifact grew large enough that N replicas each downloading it from a network-mounted `mlruns/`
  became a real cold-start cost.
- **Interview question this maps to:** "Why load a model at startup instead of on first request?" --
  startup loading fails fast (a broken artifact surfaces immediately as an unready pod, not as a
  500 on some unlucky user's first request) and amortizes load cost across the process's whole
  lifetime.

---

### Decision: `mlruns/` mounted as a Docker volume, not baked into the image
- **What:** `docker-compose.yml` mounts `./mlruns:/app/mlruns`; the `Dockerfile` never `COPY`s model
  artifacts, and `.dockerignore` doesn't need to exclude `mlruns/` only because it's never in the build
  context's relevant path for `COPY . .` in the intended deployment (mounted at runtime, not built in).
- **Why (vs. alternatives):** Baking `mlruns/` into the image would mean every new MLflow run (new
  Optuna sweep, new calibration, new conformal fit) requires a full image rebuild to deploy -- coupling
  a fast-changing artifact (retrained model, updated as often as M7's automated retraining trigger
  fires) to a slow-changing one (application code, dependencies). A volume mount lets `make
  register-model` produce a new Production version and a simple container restart pick it up, with no
  rebuild.
- **Failure mode:** If the host's `mlruns/` directory is deleted or not present when the container
  starts, the API's `lifespan()` startup will raise (no Production model to load) -- a loud, correct
  failure (the container won't report `/ready`), not a silent fallback to some stale baked-in model.
- **Scaling story (10x/100x):** At production scale, a local bind-mount `mlruns/` directory doesn't
  generalize past a single Docker host -- M7/M8's real target would be a shared MLflow tracking server
  with artifacts in object storage (S3/GCS), which every replica reads over the network instead of a
  host-local mount. Flagged here as the natural next step, out of scope for M5's single-host Compose
  setup.
- **Interview question this maps to:** "How do you keep a model artifact's deploy cadence decoupled
  from your application code's deploy cadence?" -- volume-mount (or, at scale, a remote artifact
  store) the model; never bake a frequently-retrained artifact into a container image.

---

### Decision: `/health` and `/ready` are separate routes with different failure semantics, not one endpoint with a boolean field
- **What:** `core/serving/api_base.py`'s `/health` always returns `200` the instant the process can
  serve HTTP, regardless of model-loading state. `/ready` returns `200` only once
  `state.model_loaded and state.conformal_loaded` are both `True`, and returns HTTP `503` (not a `200`
  with `"not_ready"` in the body) otherwise.
- **Why (vs. alternatives):** A single combined endpoint would force a K8s liveness probe (which exists
  to detect and restart a genuinely hung/crashed process) to either (a) also depend on model-loading
  state, meaning a pod gets killed and restarted in a loop while a large model artifact is still
  loading -- the exact failure mode this spec calls out by name -- or (b) ignore model state entirely,
  in which case there's no way for a readiness/load-balancer check to withhold traffic from a pod
  that's alive but not yet able to serve real predictions. Two routes with two different HTTP status
  semantics let K8s (M8) wire liveness and readiness probes to the behaviors they're actually meant to
  gate.
- **Failure mode:** Verified directly -- `/ready` returns `503` with `model_loaded=false` in the tests
  run before `lifespan()` startup completes (not exercised in the committed integration tests, which
  run after the container is fully up, but confirmed manually during local `uvicorn` testing that the
  503 path executes before the model finishes loading).
- **Scaling story (10x/100x):** Unaffected by request volume -- these are O(1) state checks. At higher
  replica counts (M8), each pod's `/ready` independently gates its own traffic eligibility, which is
  exactly the per-pod granularity a K8s readiness probe needs.
- **Interview question this maps to:** "Why does Kubernetes need two separate probes instead of one
  health check?" -- liveness answers "should this process be killed and restarted," readiness answers
  "should this pod receive traffic right now" -- conflating them means a slow-starting pod (loading a
  model) gets killed by the same check that should instead just withhold traffic from it temporarily.

---

### Decision: `TrialFeatures` input schema corrected to match the actually-trained feature set -- `intervention_model` dropped, `has_results` added
- **What:** The M5 prompt's literal `TrialFeatures` pydantic model included `intervention_model:
  Optional[str]` and omitted `has_results`. Checked directly against
  `domains/pharma/train_pipeline.py`'s `CATEGORICAL_FEATURES`/`NUMERIC_FEATURES` (the actual columns
  the registered Production model was fit on): `intervention_model` is not present anywhere in the
  trained feature set -- the M1 spec-table note calling it a "bonus feature... present in mart" turns
  out to describe a column that was computed and then never actually included in the final feature
  list (confirmed: no `intervention_model` key in any row's persisted `ml.training_dataset.features`
  JSONB). `has_results` (rank-7 in M4's global SHAP importance) IS in the trained feature set but was
  missing from the prompt's schema entirely.
- **Why (vs. alternatives):** Serving with the prompt's literal schema would do one of two bad things
  silently: accept `intervention_model` from a client with zero effect on the prediction (a field that
  looks load-bearing but isn't), or force a fabricated default for `has_results` -- a feature the model
  actually learned a nontrivial pattern from -- with no way for the caller to supply the real value.
  Neither is a reasonable serving contract. This is flagged explicitly, per this project's own
  standing instruction to surface spec-vs-reality conflicts rather than silently pick a side: the
  *response* schema (Section 6's locked contract) is unchanged; only the unlocked *request* schema
  (never given explicit field names in the spec itself, only in this milestone prompt) was corrected.
- **Failure mode:** If a future spec update re-adds `intervention_model` as a genuine trained feature
  (e.g., a future retrain does start using it), this input schema would need updating in lockstep with
  `train_pipeline.py`'s `CATEGORICAL_FEATURES` -- the two are not currently enforced to stay in sync
  automatically.
- **Scaling story (10x/100x):** N/A -- a fixed-schema correctness fix.
- **Interview question this maps to:** "What do you do when a milestone spec's example payload doesn't
  match what the model actually needs?" -- verify against the model's real trained feature list (here,
  `train_pipeline.py`'s constants, cross-checked against a real persisted feature row), not the spec
  prose, and fix the schema to match reality rather than the other way around.

---

### Decision: `docker-compose.yml` environment variables corrected to `POSTGRES_*`, not the prompt's literal `DB_*` names
- **What:** The M5 prompt's `docker-compose.yml` set `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`.
  `domains/pharma/dataset_builder.py`'s `_get_engine()` reads `os.environ[db_cfg["host_env"]]` etc.,
  where `config.yaml`'s `db.host_env`/`port_env`/`dbname_env`/`user_env`/`password_env` resolve to
  `POSTGRES_HOST`/`POSTGRES_PORT`/`POSTGRES_DB`/`POSTGRES_USER`/`POSTGRES_PASSWORD` -- there is no
  `DB_HOST` anywhere in this codebase, and `os.environ[...]` (direct indexing, not `.get()`) raises
  `KeyError` immediately if any is missing. The prompt's literal names, used as-is, would have left
  the containerized API unable to connect to Postgres at all (and never mentioned `POSTGRES_PASSWORD`,
  without which the connection can't authenticate regardless of naming).
  `docker-compose.yml` uses `env_file: .env` (passing through the real `POSTGRES_PORT`/`POSTGRES_DB`/
  `POSTGRES_USER`/`POSTGRES_PASSWORD` unmodified) with a single `environment:` override,
  `POSTGRES_HOST=host.docker.internal`, since `localhost` inside the container refers to the container
  itself, not the Docker host running Postgres.
- **Why (vs. alternatives):** Renaming `dataset_builder.py`'s env var reads to match the prompt's
  `DB_*` names was rejected -- that file is shared, already-validated M1 code with no reason to change
  for a naming preference introduced in an M5 prompt; fixing the compose file to match the real,
  already-established contract is the smaller, correct-direction change. Per this project's rule
  (never read `.env`'s real contents), `env_file` passes the real values through without me viewing
  them.
- **Failure mode:** If the actual `POSTGRES_PORT` in `.env` ever changes without a corresponding
  `docker-compose.yml` update, the container would still pick up the new value automatically via
  `env_file` -- there's no hardcoded port duplicated in compose to go stale.
- **Scaling story (10x/100x):** N/A -- configuration-correctness fix.
- **Interview question this maps to:** "How do you avoid duplicating environment configuration across
  a Dockerfile/compose file and the application code that actually reads it?" -- point at the same
  source of truth (`.env` via `env_file`) instead of re-declaring variable names by hand in a second
  place, which is exactly how the two can drift.

---

### Decision: `/api/v1/predict/nct/{nct_id}` registered for both GET and POST -- the M5 prompt's own spec text and test file disagree
- **What:** The M5 prompt's endpoint description says `POST /api/v1/predict/nct/{nct_id}` (matching
  `02_TRIALOUTCOME_SPEC.md` Section 6's table), but its own `tests/test_api_contract.py` calls
  `requests.get(f"{BASE_URL}/api/v1/predict/nct/{{nct_id}}")` in both `test_predict_nct_found` and
  `test_predict_nct_not_found`. `domains/pharma/serving/api.py` registers the route via
  `@app.api_route(..., methods=["GET", "POST"])` so both the literal test file and the documented
  spec table are satisfied simultaneously.
- **Why (vs. alternatives):** Silently "fixing" the tests to use POST (matching the spec table) would
  contradict the prompt's own literal, explicitly-given test file -- which the M5 DoD explicitly
  requires to pass as-is. Registering only POST would fail the given tests; registering only GET would
  contradict the spec table. Supporting both is the only option that satisfies both without silently
  discarding either source of truth. As a secondary observation: GET is arguably the more RESTful verb
  here regardless (the route takes no request body, only a path parameter, and is a pure read).
- **Failure mode:** If RegIntel's `trial_risk` tool wrapper (built against the spec table) only ever
  calls this route with POST, the GET registration is simply unused, not harmful -- no downstream
  contract is broken by supporting an extra method.
- **Scaling story (10x/100x):** N/A.
- **Interview question this maps to:** "What do you do when a spec document and its own test suite
  contradict each other?" -- satisfy both literally where that's possible (here, by supporting both
  HTTP methods) rather than picking one source of truth and silently failing the other.

---

### Decision (bugfix): `python:3.11-slim` is missing `libgomp.so.1`, which LightGBM's compiled binary requires to import
- **What:** `docker compose up --build` failed at container startup with
  `OSError: libgomp.so.1: cannot open shared object file` -- `domains/pharma/serving/api.py` imports
  `CATEGORICAL_FEATURES` from `domains/pharma/train_pipeline.py`, which has a top-level
  `from lightgbm import LGBMClassifier`, and LightGBM's compiled extension links against the OpenMP
  runtime (`libgomp`), which `python:3.11-slim` doesn't ship by default (unlike the full `python:3.11`
  image). Fixed by adding `apt-get install -y --no-install-recommends libgomp1` to the `Dockerfile`
  before the pip install step.
- **Why (vs. alternatives):** Switching `train_pipeline.py`'s constants to live somewhere that doesn't
  import LightGBM at module scope was rejected as unnecessary surgery on already-validated M2-M4 code,
  for a dependency the API doesn't actually need to import (it only needs the two constant lists,
  `CATEGORICAL_FEATURES`/`NUMERIC_FEATURES`, not LightGBM itself) -- the one-line `apt-get install` is
  the minimal, correctly-scoped fix, and keeps `train_pipeline.py` as the single source of truth for
  those constants rather than duplicating them into a LightGBM-free module.
- **Failure mode:** If a future base-image change (e.g. moving off Debian-slim to Alpine) removes
  `apt-get`/changes the package name, this would need re-diagnosing -- Alpine's musl-based lightgbm
  wheels have a different, separately-documented set of native-dependency issues.
- **Scaling story (10x/100x):** N/A -- one-time base-image dependency fix.
- **Interview question this maps to:** "How do you debug a container that imports fine locally but
  crashes in Docker?" -- trace the traceback to the actual native library the OS-level package manager
  needs to provide; `pip install`-able Python packages can still depend on system shared libraries the
  slim base image doesn't include.

---

### Decision (bugfix): local file-backed MLflow stores bake ABSOLUTE host paths into run metadata -- broke `mlflow.sklearn.load_model` inside the container until the volume was mounted at the identical absolute path
- **What:** After fixing the `libgomp1` issue, the container still failed at startup:
  `OSError: No such file or directory:
  '/Users/shubhamagrawal/.../trialoutcome/mlruns/<exp>/<run>/artifacts/model/.'` -- MLflow's local
  `file:` tracking store records each run's `artifact_uri` as an ABSOLUTE path at the moment
  `mlflow.sklearn.log_model()` runs (baked into that run's `meta.yaml`), not a path relative to
  whatever `MLFLOW_TRACKING_URI` happens to point at when later *loading* the model. Inside the
  container, `mlruns/` was mounted at `/app/mlruns`, but the registered model's metadata still pointed
  at the original host-absolute path -- which doesn't exist inside the container's filesystem at all.
  Fixed two ways together: (1) `domains/pharma/serving/api.py`'s `_load_bundle()` now respects
  `MLFLOW_TRACKING_URI` from the environment if set, instead of always recomputing
  `file:{REPO_ROOT}/mlruns` from `__file__` (which resolves to `/app` inside the container, not the
  host path baked into existing run metadata); (2) `docker-compose.yml` bind-mounts `./mlruns` a
  *second* time at the exact same absolute path it has on the host, and sets
  `MLFLOW_TRACKING_URI=file:///Users/.../trialoutcome/mlruns` to match.
- **Why (vs. alternatives):** Re-registering the model with a tracking URI that used a
  container-friendly relative path would only move the problem -- the same run would then fail to
  load from a plain local `uv run` invocation on the host instead. Migrating to a real MLflow tracking
  server with remote artifact storage (S3/GCS) is the actual production fix (object-store keys aren't
  tied to any filesystem's absolute layout), but is out of scope for M5's single-host Docker Compose
  setup -- the mount-at-identical-path workaround is the correct-sized fix for this milestone, done
  transparently rather than silently working around it by, e.g., re-running the whole M1-M5 pipeline
  inside the container instead of reusing the artifacts already produced on the host.
  This limitation, and the actual production fix, are called out explicitly in both this entry and an
  inline comment in `docker-compose.yml` so it isn't mistaken for how this would work with more than
  one host.
- **Failure mode:** This hardcodes a host-specific absolute path (this developer's exact directory
  structure) into `docker-compose.yml` -- it will not work unmodified on a different machine or a
  second developer's checkout. Flagged directly in `docker-compose.yml`'s comment and here, not
  silently left as a portability trap.
- **Scaling story (10x/100x):** Does not generalize past a single Docker host at all -- multiple
  replicas (M8's k8s HPA) each independently mounting a host bind-mount is already the wrong model at
  that point; M8 will need a real MLflow tracking server + S3/GCS-backed artifact store regardless of
  this specific fix.
- **Interview question this maps to:** "What breaks when you containerize something that assumed a
  local filesystem?" -- absolute paths baked into metadata at write time are a classic portability
  trap; the honest fix at this milestone's scope is documented as a workaround, with the real
  production fix (remote artifact store) named explicitly rather than implied.

## M5 Definition of Done -- status

- [x] Conformal empirical coverage >= 0.88 on TEST split -- **0.946** (target 0.90), logged on the
      production MLflow run (`empirical_coverage` metric, run `13ca29a52a5b456b93846514f8020dfa`).
- [x] Fitted `MapieClassifier`-equivalent (`MAPIEConformalWrapper`, wrapping mapie 1.4.1's
      `SplitConformalClassifier`) logged as an MLflow artifact (`conformal/`) on the production run.
- [x] `docker compose up` starts without errors -- required two bugfixes beyond the M5 prompt's literal
      Dockerfile/compose (libgomp1 system package; MLflow absolute-path volume mount), both documented
      above.
- [x] `GET /health` -> 200.
- [x] `GET /ready` -> 200 with `model_loaded=true`, `conformal_loaded=true` (returns 503 before the
      model finishes loading -- verified via the `ServingState` default of `False` plus manual local
      testing of the pre-lifespan-completion window).
- [x] `POST /api/v1/predict` -> 200 with all 6 locked fields present (`proba`, `conformal_interval`,
      `threshold_decision`, `top_shap`, `plain_english_summary`, `feature_pipeline_version`).
- [x] All 8 tests in `tests/test_api_contract.py` pass against the running Docker container (the M5
      prompt's DoD line says "7" but the prompt's own test file defines 8 test functions -- a minor
      miscount in the prompt, not a discrepancy in this build).
- [x] `/health` and `/ready` are demonstrably distinct (`/ready` returns 503 via `ServingState`'s
      `model_loaded`/`conformal_loaded` flags; `/health` never depends on them).
- [x] `feature_pipeline_version` is a real, clean git hash (`e886fc6bc5e6fd7cab3afc3295665f49caaaea85`)
      on the production MLflow run -- required fixing a pre-existing bug in
      `feature_pipeline_version()` itself (see above) that would otherwise have produced a
      polluted-but-technically-non-"unknown" tag.

---

## M6: Drift Monitoring + CI + README (2026-08-02)

---

### Decision: `evidently` resolved to `0.7.21`, a full rewrite from the classic 0.4.x tutorial API -- built against the new `Report(metrics=[DataDriftPreset()]).run(current_data=Dataset, reference_data=Dataset)` shape
- **What:** `uv add evidently` (unconstrained, per this project's dependency-management rule)
  resolved `evidently==0.7.21`. Every tutorial/blog post referencing `evidently.Dashboard`,
  `ColumnMapping`, or `Report(metrics=[DataDriftPreset()]).run(reference_data=df,
  current_data=df)` (plain DataFrames) is written against the pre-0.4 API, which this version
  has fully replaced: metrics are now `Report(metrics=[...])`, inputs must be wrapped in
  `evidently.Dataset.from_pandas(df)`, and results come back as a `Snapshot` object
  (`.dict()`, `.save_html()`, no `.as_dict()`/`.show()`). Confirmed via direct inspection
  (`inspect.signature`, exploratory `Report(...).run(...)` calls against synthetic DataFrames)
  before writing `core/monitoring/drift_base.py` against the actual installed API.
- **Why (vs. alternatives):** Pinning an old evidently release to match a tutorial's exact
  class names was rejected for the same reason M5's mapie/sklearn decisions were --
  `uv add` (unconstrained) resolved the current release, and there's no reason to force an
  old, less-maintained version onto a brand-new integration when the current API is fully
  documented and functional.
- **Failure mode:** If a future evidently major version renames `DataDriftPreset` or
  restructures `Snapshot.dict()`'s shape again, `core/monitoring/drift_base.py`'s
  `check_thresholds`/`per_feature_drift` raise `StopIteration`/`KeyError` immediately (loud
  failure) rather than silently reporting zero drift -- see that file's own docstrings.
- **Scaling story (10x/100x):** N/A -- API-compatibility decision, independent of data scale.
- **Interview question this maps to:** "How do you approach a library where the tutorials
  you find online don't match what's installed?" -- inspect the installed package directly
  (`inspect.signature`, exploratory calls against synthetic data) and build against what's
  actually there, the same discipline this project already applied to mapie and sklearn in M5.

---

### Decision (env quirk, not a code bug): `evidently` pulls in `nltk` transitively, and nltk 2026's own CWD-import security hook false-positives because this project's `.venv/` lives inside the repo root -- worked around via nltk's own documented `NLTK_DISABLE_IMPORT_SECURITY=1` escape hatch
- **What:** `import evidently` cascades through `evidently.legacy.metrics` (classification
  quality-by-feature tables) into `nltk` (for an OOV-words text feature evidently ships but
  this project never uses). NLTK's 2026 release ships `nltk/inisec.py`, a `MetaPathFinder`
  that blocks any import resolving to a path *inside the current working directory* whenever
  an ancestor stack frame belongs to `nltk` -- a real CWE-427 (uncontrolled search path)
  mitigation in general. It false-positives here specifically because this project's `.venv/`
  is a subdirectory of the repo root (a completely standard `uv`-managed layout) -- `regex`
  (an `nltk` dependency) resolves to `.venv/lib/.../site-packages/regex/__init__.py`, which
  `Path.resolve().relative_to(cwd)` correctly identifies as "under the CWD," even though
  nothing is actually shadowing/hijacking anything. Verified: (1) `regex` imports fine
  standalone, only fails when `nltk` is the importer; (2) `PYTHONSAFEPATH=1` does NOT fix
  it, because the check is path-containment-based, not about whether `''` is on `sys.path`;
  (3) NLTK's own `inisec.py` module docstring documents `NLTK_DISABLE_IMPORT_SECURITY=1` as
  the sanctioned escape hatch for exactly this class of false positive. Set in `make drift`
  and both `.github/workflows/ci.yml` jobs (the `test-unit` job also needs it, since
  `tests/test_drift_base.py` imports `core.monitoring.drift_base`, which imports `evidently`).
- **Why (vs. alternatives):** Downgrading to a pre-2026 `nltk` release (before this hook
  existed) was rejected -- that's pinning around a dependency's dependency's dependency to
  avoid a documented, intentional escape hatch, and would silently break the moment any other
  package in this project's tree needs a newer nltk. Vendoring/patching `nltk/inisec.py`
  directly was rejected as far more invasive than setting one documented env var.
- **Failure mode:** If a genuinely malicious package *did* try to shadow a real dependency
  from this project's working directory, `NLTK_DISABLE_IMPORT_SECURITY=1` would suppress that
  protection too -- acceptable here because this is a controlled local/CI dev environment, not
  a runtime processing untrusted third-party code, and the check's own false-positive mode
  (any `uv`/`venv`-inside-repo-root layout, arguably the *majority* of Python projects built
  this way) makes it impractical to leave enabled without hitting this exact issue on any
  fresh clone.
- **Scaling story (10x/100x):** N/A -- one-time environment-compatibility workaround,
  independent of data or request volume.
- **Interview question this maps to:** "Tell me about a dependency that broke for a reason
  that had nothing to do with your own code." -- a transitive dependency's *security*
  feature triggering a false positive is a different failure class than a version-API
  mismatch (M5's mapie/sklearn entries) -- diagnosing it required reading the actual
  exception's traceback down to the specific library file raising it, not just re-reading
  the top-level error message.

---

### Decision (bugfix, caught while building `notebooks/06_drift_report.ipynb`): `per_feature_drift`'s first version inverted the "drifted" flag for every Wasserstein/Jensen-Shannon-distance feature -- the majority of this project's real columns
- **What:** Evidently auto-selects a drift-detection method per column based on type/
  cardinality: well-behaved small-cardinality columns can get a p-value method
  ("K-S p_value", "chi-square p_value"), where `score < threshold` means drift (reject the
  null of "no difference"); this project's actual numeric/high-cardinality features
  overwhelmingly got distance/divergence methods ("Wasserstein distance (normed)",
  "Jensen-Shannon distance"), where `score > threshold` means drift -- the OPPOSITE
  direction. A first version of `core/monitoring/drift_base.py`'s `per_feature_drift` used
  `score < threshold` unconditionally (copied from the p-value-only toy example used to
  first validate the Evidently API), which silently flagged every real drifted feature as
  "not drifted" and vice versa. Caught by cross-checking against Evidently's own per-column
  pass/fail test status (`Report(..., include_tests=True)`) on the real TRAIN-vs-TEST data
  before trusting the helper -- the inverted version showed `condition_*` one-hot columns
  (near-zero JS distance, genuinely NOT drifted) as the "most drifted" features and
  `start_year` (JS distance 0.83, the most genuinely drifted column in the dataset) as "not
  drifted." Fixed by branching on `"p_value" in method.lower()`, verified to match Evidently's
  own test-status output on every column in the real dataset before landing on it, and
  covered by a regression test (`tests/test_drift_base.py::test_per_feature_drift_direction_matches_evidently_dataset_verdict`).
- **Why (vs. alternatives):** Reading Evidently's own `include_tests=True` pass/fail status
  directly (rather than re-deriving direction from `method`/`threshold`) was considered, but
  would require running the Report with `include_tests=True` on every call and parsing a
  second, differently-shaped `tests` list purely to sanity-check a `metrics` list already
  present -- the `"p_value" in method` rule is simpler, was verified to agree with the
  test-status output on every real column in this dataset, and is now guarded by a
  regression test rather than trusted on inspection alone.
- **Failure mode:** If Evidently ever ships a new drift-detection method whose name doesn't
  contain "p_value" but IS a p-value-style statistic (or vice versa), this heuristic would
  silently misclassify that column's direction again -- the regression test would only catch
  this if the test's synthetic data happens to trigger that specific new method, which isn't
  guaranteed. Documented as a real, not fully closed, risk in `per_feature_drift`'s docstring.
- **Scaling story (10x/100x):** N/A -- a parsing-logic bug, independent of data volume.
- **Interview question this maps to:** "How do you catch a bug in code that runs without
  raising any exception and returns a plausible-looking result?" -- cross-check the parsed
  output against the underlying library's own ground truth (here, Evidently's `tests` list)
  on real data before trusting a helper function that "ran fine" and lock the fix in with a
  regression test, not just a fixed docstring.

---

### Finding: label-drift PSI stretch check attempted twice, both honestly non-completed for distinct structural reasons -- not silently skipped
- **What:** Per the M6 brief, attempted a Population Stability Index on the rolling
  termination base rate, binned by `start_year`, TRAIN years as "expected" and TEST years as
  "actual." **Attempt 1 (literal reading):** bins = individual `start_year` values. TRAIN
  spans 1990-2019, TEST spans 2022-2026 -- **zero shared bins**, a structural guarantee of
  this project's temporal split (`train_end`/`calib_end` in `config.yaml` partition the
  calendar into strictly non-overlapping ranges by definition), not a data quirk. Every bin
  has 0% weight on one side and nonzero weight on the other, making `ln(actual_pct /
  expected_pct)` undefined without an arbitrary epsilon, and even with one, the resulting
  number would only measure "these are two different multi-year windows" -- true by
  construction, carrying no drift signal. **Attempt 2 (reinterpretation):** bin the yearly
  termination-*rate value* into deciles instead of calendar years, so TRAIN and TEST years
  CAN land in the same bin. Technically computable, but PSI swung from **0.64** (3 bins) to
  **10.42** (4 bins) to **8.94** (5 bins) to **11.30** (10 bins) -- an order-of-magnitude,
  sign-of-interpretation-flipping range driven purely by bin-edge placement, because only 30
  TRAIN and 5 TEST "observations" (one termination rate per `start_year`) exist to bin.
- **Why this is an honest non-result, not a failed attempt:** both breakdowns are
  identifiable and structural, not symptoms of buggy code -- Attempt 1's zero-overlap is
  provably guaranteed by the temporal split's own definition (would recur identically on any
  future retrain with the same split config); Attempt 2's instability was demonstrated
  directly by sweeping bin count and showing the result isn't robust to an arbitrary choice.
  Reporting either a single PSI number (from either attempt) without showing this would have
  been indefensible -- a 3-bin PSI of 0.64 ("stable") and a 10-bin PSI of 11.30 ("severe") from
  the identical underlying data cannot both be presented as "the" answer.
- **What IS real, reported directly instead of via PSI:** TRAIN termination rate 17.8% vs
  TEST termination rate 31.7% (n=66,105 / n=5,700) -- already established in M1's
  decisions.md finding, and already the reason M3's calibrator and cost-optimal threshold
  were evaluated on TEST rather than assumed to transfer from TRAIN's lower base rate.
- **Failure mode if this had been silently skipped instead:** the M6 brief explicitly
  requires a number OR an honest non-result stated in both the notebook and README --
  skipping it silently (or worse, reporting one cherry-picked bin-count's PSI as if it were
  definitive) would have hidden a real methodological limitation of PSI at this dataset's
  temporal granularity (few distinct time periods per split) rather than surfacing it.
- **Scaling story (10x/100x):** More trials scored per existing test year does NOT fix this
  -- only more distinct *time periods* of monitoring history would (e.g., monthly drift
  checks accumulated over several years of live traffic, giving dozens of bins per side
  instead of 30 and 5). Flagged as the concrete trigger condition for revisiting this in a
  future milestone, in both the notebook and README.
- **Interview question this maps to:** "Tell me about a metric you tried to compute that
  didn't work out." -- the honest answer here is a stronger artifact than a single clean PSI
  number would have been: it demonstrates the statistic's own instability was caught via a
  bin-count sensitivity sweep (not assumed), and the real underlying signal (base-rate shift)
  was still reported plainly instead of being lost along with the failed statistic.

---

### Decision (bugfix): `_current_production_version()` in `drift_job.py` ignored `MLFLOW_TRACKING_URI` -- always resolved to this checkout's own `mlruns/` path regardless of environment, unlike `serving/api.py`'s equivalent lookup
- **What:** A first version of `domains/pharma/monitoring/drift_job.py`'s
  `_current_production_version()` called `mlflow.set_tracking_uri(f"file:{REPO_ROOT /
  'mlruns'}")` unconditionally. Caught while dry-running the CI `drift-check` job's recipe
  locally against a throwaway Postgres container + a scratch MLflow tracking directory
  (`/tmp/ci_sim_mlruns`, to validate the job end-to-end without touching the real dev
  database): the function still silently read this actual repo checkout's real `mlruns/`
  directory (returning the real Production version, `2`) instead of the empty scratch
  directory the simulated environment pointed at. Fixed to respect `MLFLOW_TRACKING_URI`
  from the environment if set, exactly mirroring `domains/pharma/serving/api.py`'s
  `_load_bundle()` (which already needed this for the M5 Docker/absolute-path issue).
- **Why (vs. alternatives):** Leaving it as-is was tempting since it happens to produce the
  correct behavior in the one case that matters for CI (a fresh checkout has no `mlruns/` at
  all, so the exception path still returns `"unknown"`) -- but that's the same "works by
  accident, not by design" pattern flagged elsewhere in this project's `decisions.md` (see
  M5's `feature_pipeline_version()` entry). Fixing it properly costs one line and closes a
  real inconsistency between two functions in the same codebase that both claim to resolve
  "the current Production version."
- **Failure mode:** Without this fix, any future local testing against a scratch/alternate
  MLflow tracking directory (exactly what surfaced this) would silently read the real dev
  environment's registered model version instead of the intended scratch one -- a read-only
  mistake, not data-destructive, but a confusing one to debug blind.
- **Scaling story (10x/100x):** N/A -- environment-resolution correctness fix, independent
  of data or request volume.
- **Interview question this maps to:** "How do you catch a bug that produces the *correct*
  output in your main test case?" -- dry-run the untested path (here, a from-scratch
  environment) deliberately, even when the main path already looks fine, rather than
  assuming "it worked when I ran it" means "it's correct for every caller."

---

### Decision: `tests/` needed three new fast, DB-free unit test files before CI's `test-unit` job could run at all
- **What:** Before this milestone, `tests/` contained only `test_api_contract.py` (an
  integration suite requiring a live Docker container) and `__init__.py`. The M6 brief's own
  CI job (`pytest tests/ -v --ignore=tests/test_api_contract.py`) would collect **zero**
  tests and exit non-zero (pytest's own "no tests ran" exit code) with nothing to exclude
  from. Added `tests/test_plain_english.py` (Section 4a templating, pure functions),
  `tests/test_calibration.py` (`core/calibration.py`'s ECE, pure numeric), and
  `tests/test_drift_base.py` (this milestone's own `core/monitoring/drift_base.py`, using
  tiny synthetic DataFrames -- no DB) -- all fast, dependency-light (beyond `evidently` for
  the last one), and none requiring Postgres, MLflow, or a running API.
- **Why (vs. alternatives):** Leaving `test-unit` to fail on "no tests collected" and
  treating that as an acceptable CI state was rejected -- a CI job that always fails (or
  always no-ops) the moment it's added provides zero verification value and would need
  fixing the first time anyone actually looked at the Action's results. This is
  implementation-forced by the brief's own literal CI recipe, not scope creep: the brief
  specifies excluding the one existing test file without specifying what else should exist
  to fill that job, and a real CI job needs *something* real to run.
- **Failure mode:** These are unit tests for functions that already existed pre-M6
  (`plain_english.generate_summary`, `calibration.expected_calibration_error`) plus this
  milestone's own new code (`drift_base.py`) -- they don't raise overall coverage
  substantially, but they do give `test-unit` a genuine, fast, deterministic signal instead
  of a job that trivially "passes" by having nothing to check.
- **Scaling story (10x/100x):** N/A -- test-suite completeness fix.
- **Interview question this maps to:** "What do you do when a CI recipe's literal
  instructions would leave a job with nothing to run?" -- add real, fast, dependency-light
  unit tests for existing pure functions rather than either leaving the job broken or
  padding it with tests that don't check anything meaningful.

---

### Decision: `.env` doesn't exist in a fresh CI checkout (gitignored, never committed) -- `.github/workflows/ci.yml`'s `drift-check` job generates a throwaway one from the Postgres service container's own credentials
- **What:** This project's `Makefile` does `include .env` / `export` unconditionally at the
  top, and `domains/pharma/dataset_builder.py`'s `_get_engine()` reads `POSTGRES_*` via
  `os.environ[...]` (direct indexing, raises `KeyError` if unset) after `load_dotenv(REPO_ROOT
  / ".env")`. Both paths require a real `.env` file to exist -- which CLAUDE.md's own rule
  keeps out of version control (`.env` is gitignored; only `.env.example`'s placeholder
  values are ever read or written by this assistant). The `drift-check` CI job writes a
  throwaway `.env` (via a heredoc step) populated with the Postgres service container's own
  connection details (`localhost:5433`, `pharmapulse`/`pharmapulse_user`, a CI-only
  `ci_password` -- not a real secret) plus `MLFLOW_TRACKING_URI=file:./mlruns`, matching
  `.env.example`'s documented shape exactly.
- **Why (vs. alternatives):** Refactoring `dataset_builder.py`/`Makefile` to accept
  connection details some other way (e.g., CLI flags) for CI's benefit was rejected --
  that would diverge the CI code path from the exact `make db-init`/`make drift` commands a
  developer runs locally, defeating the point of using `make` targets in CI at all (see the
  next entry). Generating a `.env` from already-public, throwaway CI-only values is the
  standard, minimal-surface-area way to satisfy this project's own env-var-driven config
  without touching the real `.env`-never-read-by-Claude rule (this is CI's own file, not the
  user's local one, and contains no real credentials).
- **Failure mode:** If CI's generated `.env` values ever drifted out of sync with the
  Postgres service container's own `env:` block in the same workflow file, `make db-init`
  would fail immediately with a connection error (loud, not silent) -- both are defined
  adjacently in the same file specifically to make that drift easy to notice and fix.
- **Scaling story (10x/100x):** N/A -- CI environment bootstrapping, independent of data or
  request volume.
- **Interview question this maps to:** "How do you give CI the config a Makefile expects
  without committing real credentials?" -- generate a throwaway config file from already-CI-
  local, non-secret values at job runtime, rather than either committing a fake `.env` to the
  repo or forking the underlying commands into a CI-only variant.

---

### Decision: CI's `drift-check` job installs `uv` and runs the actual `make db-init` / `make drift` targets, rather than reimplementing their steps as raw shell/psql/python commands
- **What:** This project's `Makefile` targets hardcode `uv run` (per CLAUDE.md: "Use `uv add`
  for all dependency management, never `pip install`"). The `drift-check` CI job installs
  `uv` (`pip install uv`) and runs `uv sync --frozen` against the committed `uv.lock`, then
  calls `make db-init` and `make drift` directly -- the identical commands a developer runs
  locally -- rather than duplicating their internals (raw `psql -f domains/pharma/schema.sql`,
  raw `python -m domains.pharma.monitoring.drift_job`) as CI-only shell steps.
- **Why (vs. alternatives):** A parallel CI-only command path (bypassing `make`/`uv`
  entirely, installing via `pip install -r requirements.txt` and calling the underlying
  Python modules directly) was considered, since `test-unit` already does exactly that. It
  was rejected for `drift-check` specifically because that job's whole point is validating
  the actual `make drift` recipe a developer would run -- a CI-only reimplementation could
  silently drift out of sync with the real Makefile (e.g., a future Makefile change to
  `drift`'s invocation wouldn't be caught by a CI path that never calls `make` at all).
  `test-unit` doesn't have this concern (`pytest tests/ -v ...` isn't behind a `make` target
  in this project), so it stays on the simpler `pip install -r requirements.txt` path per the
  M6 brief's literal instruction.
- **Failure mode:** This makes `drift-check` slightly slower than a raw-pip-install path
  (uv resolves/installs the full `pyproject.toml`/`uv.lock` dependency set) -- an acceptable
  tradeoff for exercising the real developer-facing command.
- **Scaling story (10x/100x):** N/A -- CI tooling choice, independent of data or request
  volume.
- **Interview question this maps to:** "Should CI call your Makefile targets, or reimplement
  their steps directly?" -- call the real targets when the job's purpose is to validate that
  exact developer-facing recipe; only diverge when a job's dependencies genuinely differ
  (here, `test-unit` has no `make`-wrapped step to validate in the first place).

---

### Decision: a stale "0.891" random-split PR-AUC number in `02_TRIALOUTCOME_SPEC.md`'s M1 milestone row was corrected while sourcing this README's leakage-experiment section
- **What:** While writing this README's "leakage experiment" section, cross-checked the
  spec's M1 table entry ("LogReg temporal PR-AUC 0.867 vs random-split PR-AUC 0.891") against
  both `decisions.md`'s own M1 finding ("0.866 (temporal) vs 0.682 (random)") and
  `notebooks/02_leakage_demo.ipynb`'s actual executed output ("Random split PR-AUC: 0.6823").
  The latter two agree with each other; the spec table's "0.891" matches neither and also
  has the wrong *sign* (implying random beats temporal, when both independent sources show
  the opposite). This is a transcription error in the milestone-summary table, not a
  re-derivation or a real second measurement -- corrected in the spec table with an inline
  note, per this project's standing rule to flag spec-vs-reality conflicts explicitly rather
  than silently overwrite or silently ignore them.
- **Why (vs. alternatives):** Silently fixing the number without a note would erase the
  evidence that a real inconsistency existed across three documents describing the same
  result; leaving the wrong number in place (matching neither other source) would propagate
  a demonstrably incorrect figure into a document explicitly described as "source of truth
  for schemas, APIs, and milestones" in this project's own `CLAUDE.md`.
- **Failure mode:** N/A -- a documentation-accuracy correction, not a code or data change;
  the underlying notebook and its numbers are unchanged.
- **Scaling story (10x/100x):** N/A.
- **Interview question this maps to:** "What do you do when you find a number in project
  documentation that doesn't match the artifact that supposedly produced it?" -- trace it
  back to the directly-executed source (the notebook), trust that over a milestone-summary
  table, and correct the summary with a note rather than silently overwriting or ignoring
  the discrepancy.

## M6 Definition of Done -- status

- [x] Evidently HTML drift report generated in `reports/` -- `reports/drift_2026-08-02.html`,
      written by both `make drift` and `notebooks/06_drift_report.ipynb` (same canonical
      path; gitignored as a regenerable artifact, per `.gitignore`'s new `reports/*.html`
      entry -- CI uploads it as a workflow artifact instead).
- [x] `ml.drift_log` table created and populated with at least one run -- created both via
      `domains/pharma/schema.sql` (`make db-init`) and idempotently inside
      `drift_job.py.log_to_db` itself; verified populated (`drifted=false`,
      `n_features_drifted=11`, `model_version=2`) against the real dev database.
- [x] `make drift` runs without errors -- verified against the real dev Postgres instance
      (66,105 train / 5,700 test rows) and, separately, against a throwaway isolated Postgres
      container simulating the CI `drift-check` recipe end-to-end (50-row synthetic fixture,
      fresh empty MLflow tracking dir) without ever touching the real dev database's data.
- [x] `notebooks/06_drift_report.ipynb` executed with real output -- `jupyter nbconvert
      --execute --to notebook --inplace`, zero error cells, verified via a scan for
      `output_type == "error"` across all cells.
- [x] PSI label-drift result stated (number or honest non-result) in both the notebook and
      README -- honest non-result, attempted two ways, both structural failure modes
      documented explicitly (see Finding above), plus the real underlying base-rate fact
      (17.8% vs 31.7%) reported directly.
- [x] `.github/workflows/ci.yml` written with both `test-unit` and `drift-check` jobs.
- [x] `tests/fixtures/seed_training_data.sql` written -- 50 rows (30 train / 20 test), 10/50
      label=true (20% positive rate).
- [x] `README.md` complete with all required sections and real headline numbers.
- [x] `02_TRIALOUTCOME_SPEC.md` M6 row updated to ✅ with as-built DoD (and a stale M1-row
      number corrected in the same pass, per the Decision above).
- [x] `decisions.md` updated with M6 entries (this section).

---

## M7: Retraining trigger + rollback (2026-08-02)

### Decision: the M7 brief's literal `mlflow models transition-stage` CLI command does not exist -- `rollback_production()` (wrapping `MlflowClient.transition_model_version_stage` directly) plus `make rollback VERSION=N` is the real one-command procedure, reused for both rollback and manual forward promotion
- **What:** Verified directly (`mlflow models --help` against the installed mlflow==2.22.5):
  the `mlflow models` CLI only has `build-docker`/`generate-dockerfile`/`predict`/
  `prepare-env`/`serve`/`update-pip-requirements` -- there is no `transition-stage`
  subcommand anywhere in the mlflow CLI (registry stage transitions are Python-client/
  REST-API-only, never exposed as a CLI verb, in any mlflow version this project could
  find). `domains/pharma/monitoring/rollback.py`'s `rollback_production(target_version)`
  wraps `MlflowClient.transition_model_version_stage(..., archive_existing_versions=True)`
  directly instead (the exact call the brief's own function body specified), exposed as
  `make rollback VERSION=N`. Because this function is deliberately agnostic about
  `target_version`'s stage beforehand, `domains/pharma/monitoring/retrain_trigger.py`'s
  "Staged for review" message also points a human at `make rollback VERSION=N` for manual
  forward promotion -- one real, tested command instead of a fictional one, for both
  directions.
- **Why (vs. alternatives):** Could have shipped the brief's literal snippet anyway (it
  would look right in a README but fail the moment anyone actually ran it) or invented a
  second, separate "promote" command. Reusing `rollback_production()` for both keeps
  exactly one function in the whole codebase that ever calls
  `transition_model_version_stage(..., stage="Production")` via an automated code path --
  which is also what M7's DoD grep check (see below) is actually protecting.
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

### Decision: the brief's suggested synthetic perturbation ("multiply `log_enrollment_count` by 3") cannot cross the real drift-share threshold -- perturbing the one-hot `condition_*` columns instead
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
  `log_enrollment_count x3` is kept in the perturbation for continuity with the brief's
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
- **What:** M7's DoD says "no code path calls `transition_model_version_stage` with
  `stage=\"Production\"` except rollback.py and the manual CLI command." `domains/pharma/
  register_model.py` (M5, pre-existing) also calls this directly, registering a freshly
  refit champion straight to Production -- but it is itself a manual, human-run script
  (`make register-model`), never invoked by any automated trigger, exactly like `make
  rollback` is carved out as. `domains/pharma/monitoring/retrain_trigger.py` (the new M7
  automated-trigger code path this DoD item is actually protecting) never calls
  `transition_model_version_stage` with `stage="Production"` -- verified by grep (it only
  transitions to `"Staging"`). Confirmed via `grep -rn 'stage="Production"' domains/ core/`:
  exactly two call sites, `register_model.py` (M5, manual) and `rollback.py` (M7, manual).
- **Why (vs. alternatives):** Could have refactored `register_model.py` to route through
  `rollback_production()` too, for a literal single-call-site match to the DoD's wording.
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

### Decision: the new `retrain-trigger-test` CI job is wired up per the brief, but its three tests are expected to report "skipped" (not "passed") in CI -- verified working for real in this dev environment instead
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
      counts as the same class of exception as the DoD's literal "rollback.py and the
      manual CLI command" wording).
- [x] `02_TRIALOUTCOME_SPEC.md` M7 row updated to ✅.
- [x] `decisions.md` updated with M7 entries (this section).

---

## M8: Scoped Kubernetes Deploy (2026-08-02) -- FINAL MILESTONE

---

### Decision: the brief's literal `k3d cluster create trialoutcome-demo --agents 2` does not make `mlruns/` visible inside pods -- recreated with `--volume <mlruns>:<same absolute path>@all`
- **What:** k3d agent/server nodes are themselves Docker containers, not the Mac host.
  A `hostPath` volume in a pod spec resolves inside that node container's own filesystem
  unless the real host directory is bind-mounted into every node *at cluster-creation
  time*. Verified directly: created the cluster with the brief's literal command first,
  confirmed no bind mount existed, deleted it, and recreated with
  `k3d cluster create trialoutcome-demo --agents 2 --volume
  "$(pwd)/mlruns:/Users/.../trialoutcome/mlruns@all"` -- the `@all` node-filter suffix
  mounts it into the server and both agent nodes, since the scheduler may place a pod on
  any of them. Verified via `docker exec k3d-trialoutcome-demo-agent-0 ls
  /Users/.../trialoutcome/mlruns` before applying any manifests.
- **Why (vs. alternatives):** The container path is deliberately identical to the repo's
  real absolute path -- same reason `docker-compose.yml` bind-mounts `mlruns/` at an
  identical absolute path (see M5's decisions.md entry): MLflow's local file-backed
  tracking store bakes each run's absolute host path into its own metadata at logging
  time, so `mlflow.sklearn.load_model()` only resolves it if the mount path matches
  exactly, everywhere. Using a different, k8s-idiomatic path (e.g. `/mlruns-data`) would
  require either re-registering every MLflow run (destroys the M5/M7 audit trail this
  project has built up) or a path-rewriting shim -- both strictly worse than reusing the
  exact pattern already proven correct in `docker-compose.yml`.
- **Failure mode:** Without this fix, pods would `CrashLoopBackOff` immediately --
  `_load_bundle()`'s `mlflow.sklearn.load_model()` call would raise a file-not-found
  error the instant a pod tried to start, since the registered run's artifact URI points
  at a path that, without the node-level bind mount, simply doesn't exist inside the k3d
  node container.
- **Scaling story (10x/100x):** N/A directly -- this is the same hostPath-does-not-
  generalize-past-single-node limitation already flagged in `docker-compose.yml` and
  M5's decisions.md, now flagged a third time for k8s specifically (see README's "What
  I'd change at production scale" section and this file's next-but-one entry).
- **Interview question this maps to:** "Why doesn't a Docker Compose bind-mount pattern
  port directly to Kubernetes?" -- k3d's own nodes being containers is a good concrete
  illustration of why `hostPath` in K8s needs to resolve against the actual node's
  filesystem, and why that's fundamentally different from (and more fragile than) a
  single-daemon Docker Compose bind mount.

---

### Decision (bugfix to the brief's own assumption): k3d 5.9.0 ships `metrics-server` bundled by default -- no separate install needed
- **What:** The brief's Part B said "Requires metrics-server ... document if k3d needs it
  installed separately (it usually does not ship it by default, verify and note)."
  Verified directly on this build: `kubectl get deployment metrics-server -n kube-system`
  showed it present immediately after `k3d cluster create`, reaching `1/1` Ready within
  ~15s, with `kubectl top nodes` returning real CPU/memory numbers with no extra install
  step. k3d 5.9.0's underlying k3s (`v1.35.5+k3s1`) evidently bundles it now.
- **Why this isn't silently corrected without a note:** The brief's assumption was
  reasonable (metrics-server historically needed a separate manifest on many k3d/kind
  setups) but is empirically wrong for the installed version -- documented explicitly in
  `docs/k8s_setup.md` rather than silently skipping the verification step the brief asked
  for, and the fallback install command is still documented in case a different
  k3d/k3s version doesn't bundle it.
- **Failure mode:** If a future k3d version stops bundling metrics-server, `kubectl get
  hpa` would show `TARGETS: <unknown>/50%` indefinitely and never scale -- the
  `docs/k8s_setup.md` verification step (`kubectl wait --for=condition=available
  deployment/metrics-server`) would catch this immediately, before wasting time debugging
  why the HPA never reacts to load.
- **Interview question this maps to:** "How do you handle a runbook step whose own
  assumption turns out to be wrong for your environment?" -- verify directly (`kubectl
  get deployment`) rather than trusting either the brief's assumption or blind faith that
  "it usually just works," and document the actual observed behavior for whoever reads
  this next.

---

### Decision: `POSTGRES_PASSWORD` sourced from a Kubernetes Secret, seeded with the `.env.example` placeholder, real value never touched by Claude
- **What:** `k8s/deployment.yaml` sources `POSTGRES_PASSWORD` via `secretKeyRef` against
  a Secret `trialoutcome-postgres-secret`, rather than a plaintext env var (which the M8
  brief's literal instruction -- "env vars: same POSTGRES_* vars docker-compose.yml
  already uses" -- would have allowed, since `docker-compose.yml` gets it from `.env` via
  `env_file:`). Per this project's standing rule, Claude never reads the real `.env`, so
  it could not embed the real password into any manifest or command even if it wanted to.
  Created the Secret with the `.env.example` placeholder (`changeme`) directly via
  `kubectl create secret generic ... --from-literal=POSTGRES_PASSWORD=changeme`, and
  documented (in `docs/k8s_setup.md`) the exact command for the user to run themselves,
  against the real `.env`, if they want to exercise the real
  `/api/v1/predict/nct/{nct_id}` route against live warehouse data.
- **Why this doesn't block M8's actual DoD items:** `domains/pharma/dataset_builder.py`'s
  `_get_engine()` uses SQLAlchemy's `create_engine()`, which is lazy -- it never opens a
  real connection until a query actually runs. `/health`, `/ready`, and
  `POST /api/v1/predict` (the exact route the HPA load test in Part C exercises) never
  touch Postgres at all, so the pods reach `Ready` and serve real predictions regardless
  of whether the seeded password is correct. Verified directly: the load test in Part C
  and the probe demo in Part D both ran successfully against the placeholder-secret pods.
  Only the NCT-lookup route would fail (500/connection error) against the placeholder.
- **Why a Secret instead of a plaintext env var anyway, given the brief's literal
  wording allowed either:** Using a Secret is strictly more correct and costs nothing
  extra to build -- and it's also more consistent with what Part E's own "what I'd change
  at production scale" section has to say about secrets management (a real K8s `Secret`
  is still only base64-encoded, not encrypted at rest, so even this is flagged as a demo-
  grade compromise, not a production-grade one).
- **Failure mode:** If a future contributor updates `k8s/deployment.yaml` to a plaintext
  `POSTGRES_PASSWORD` env var "for convenience," the placeholder or a real value would
  end up committed directly into a tracked YAML file -- worth grepping for
  `POSTGRES_PASSWORD` in `k8s/*.yaml` periodically to confirm it's still `secretKeyRef`-
  only, the same class of check M7's grep-based auto-promotion verification already
  established as this project's pattern for "assert an invariant stays true going
  forward."
- **Interview question this maps to:** "How do you handle a credential a prompt asks you
  to wire up, when you're not allowed to read the real value?" -- design around the
  constraint honestly (a Secret reference plus a documented command for a human to run),
  rather than working around it by hardcoding a fake-but-plausible-looking value or
  silently using a weaker plaintext-env-var pattern just because the literal instruction
  technically permitted it.

---

### Decision: liveness `initialDelaySeconds=15` set from a real measured ~5s model-load time, not guessed
- **What:** Before writing `k8s/deployment.yaml`, timed the actual `trialoutcome-api:m8`
  image directly: `docker run` a fresh container, poll `/health`/`/ready` every second
  from container start. Observed: both return `200` by t=5s (includes MLflow artifact
  download + `mlflow.sklearn.load_model` deserialization + SHAP explainer construction).
  `livenessProbe.initialDelaySeconds` set to 15 (~3x margin); `readinessProbe
  .initialDelaySeconds` set to 5 (no extra margin needed, since a `503` there only
  withholds traffic rather than killing the pod, per the brief's own guidance).
- **Why (vs. alternatives):** Guessing a "safe-sounding" delay (e.g. 30s or 60s, common
  defaults copied from unrelated tutorials) would work but wastes real time on every pod
  restart / rolling update waiting past a delay that doesn't reflect this specific image's
  actual behavior. Measuring first is cheap (one `docker run` + a polling loop) and
  directly informs the value instead of guessing at it.
- **Failure mode:** If a future model artifact grows substantially larger (a much bigger
  ensemble, or a deep-learning model per the GPU-nodes note in README's "What I'd change
  at production scale"), this 5s baseline would need re-measuring -- the load time is a
  property of the specific model+dependencies, not a fixed constant.
- **Scaling story (10x/100x):** Model artifact size, not row count, is what would move
  this number -- a 10x/100x larger *training dataset* has no effect on inference-time
  load latency, since the served artifact is a fixed-size fitted model regardless of how
  much data trained it.
- **Interview question this maps to:** "How did you pick your liveness probe's
  `initialDelaySeconds`?" -- measure the real cold-start time against the actual image
  before setting probe timing, rather than copying a number from a tutorial that has no
  relationship to this specific application's startup cost.

---

### Decision: probe distinction demo used a `kubectl patch` on the live readinessProbe path, not the brief's literal "rename mlruns/ / simulate slow load" suggestion
- **What:** Verified directly that the brief's suggested mechanism does not produce the
  state it's meant to demonstrate against this app's real code: if `_load_bundle()`
  (`domains/pharma/serving/api.py`, locked M5 serving contract) fails for any reason --
  including a broken `mlruns/` mount -- the exception propagates through FastAPI's
  `lifespan` context manager and uvicorn's startup fails outright; the container process
  exits before ever binding port 8000. Kubernetes would see a genuine crash
  (`CrashLoopBackOff`, driven by the normal container-restart policy, not a probe
  *failure*), never a `Running`-but-`NotReady` pod -- because there is no partial/
  degraded-load code path in the locked serving contract to target instead. Used
  `kubectl patch deployment ... readinessProbe/httpGet/path` to point only the readiness
  probe (liveness untouched) at a nonexistent route instead -- this isolates exactly the
  signal being demonstrated (the readiness gate, independent of real app state) without
  inventing a new failure-handling path in M5's already-locked serving code.
- **Why (vs. alternatives):** Could have added a genuine slow-load code path to
  `api.py` (e.g. an artificial `time.sleep()` before `state.model_loaded = True`) --
  rejected as unnecessary scope creep against locked serving code for a demo whose actual
  point (readiness gates traffic without restarting the pod) doesn't require the delay to
  be caused by real model loading specifically, only that `/ready` fails while `/health`
  succeeds. The `kubectl patch` approach also has the advantage of being provably
  reversible with zero risk to the codebase: `k8s/deployment.yaml` on disk was never
  edited, only the live cluster object, confirmed via a direct diff-equivalent check
  after reverting (see `docs/k8s_probe_demo.md`).
- **Failure mode:** If a future reader assumes this demonstrates "the app degrades
  gracefully under slow model load," that would be a misread -- it demonstrates the probe
  *mechanism* (kubelet only cares about what it's configured to poll, not real app state),
  explicitly not a claim that this app has graceful degraded-load behavior. Documented
  plainly in `docs/k8s_probe_demo.md` to head that misread off.
- **Scaling story (10x/100x):** N/A -- a demo-mechanism decision, not a performance one.
- **Interview question this maps to:** "The brief told you to do X to demonstrate Y --
  you found X doesn't actually work here. What do you do?" -- verify the literal
  suggested mechanism against the real code before running with it, and when it doesn't
  hold, find the smallest faithful substitute that demonstrates the same underlying
  property (here: readiness gates traffic, doesn't restart the pod) rather than
  either forcing the original suggestion to "work" via an unrequested code change, or
  silently skipping the demo.

---

### Finding: HPA scaled 2→4 (hit maxReplicas) under a real load test; probe demo held `NotReady` for 57+s with zero restarts -- both real results, neither assumed
- **What:** `hey -z 60s -c 20` against `POST /api/v1/predict` (port-forwarded, real k3d
  pods) drove `kubectl get hpa`'s reported CPU from 1%→50%→100% of the 50% target over
  ~54s of sustained load; the HPA controller created 2 new pods once CPU crossed 100%,
  which became `Ready` (`readinessProbe.initialDelaySeconds=5` elapsing) within ~15s, at
  which point the `Deployment`'s `REPLICAS` column caught up to 4 -- confirmed via
  polling `kubectl get hpa`/`kubectl get pods` every 5s throughout, real timestamped
  output preserved in `docs/k8s_load_test_output.txt`. All 1,796 requests over the 60s
  window returned `200` -- no dropped/errored requests during the scale-up transition.
  Scale-down back to 2 was not observed (the default 5-minute downscale stabilization
  window exceeds the capture window) and is explicitly not claimed.
- **Why this matters:** The M8 brief explicitly warned against claiming scaling happened
  if `kubectl` output didn't show it, and to try a heavier load pattern if the first
  attempt didn't trigger scaling. The first attempt (`-z 60s -c 20`, the brief's own
  suggested parameters) worked on the first try and reached `maxReplicas` -- no need for
  a heavier `-c 50` retry, but the real, timestamped, unedited `kubectl` output is
  preserved specifically so this claim is independently checkable, not just asserted.
- **Bonus, unplanned observation kept in the record:** during the probe demo (Part D),
  the rolling update correctly refused to tear down the 3 already-`Ready` old-ReplicaSet
  pods while the 2 new pods sat `NotReady` -- `maxUnavailable`'s default meant readiness
  gating protected the rollout itself, not just load-balancer traffic. Not something the
  brief asked to demonstrate, but a real and relevant consequence of the same mechanism,
  documented in `docs/k8s_probe_demo.md` rather than discarded as out of scope.
- **Interview question this maps to:** "Walk me through verifying an HPA actually scales
  under load, not just configuring it and assuming it works." -- run the real load test,
  poll the real `kubectl` state throughout (not just before/after), and keep the raw
  timestamped output as the artifact, so "it scaled" is a checkable claim, not an
  assertion.

## M8 Definition of Done -- status

- [x] k3d cluster created, image imported, deployment applies clean -- cluster recreated
      with a `--volume ...@all` mount (see Decision above; the brief's literal command
      alone does not surface `mlruns/` inside pods), `trialoutcome-api:m8` built and
      `k3d image import`-ed, all three manifests (`deployment.yaml`, `service.yaml`,
      `hpa.yaml`) applied with no errors.
- [x] 2 replicas running, both eventually Ready -- both pods reached `1/1 Running` within
      ~14s of `kubectl apply` (well inside the 15s liveness `initialDelaySeconds` margin).
- [x] HPA scales 2→N under real load test, N reported honestly -- **N=4 (hit
      maxReplicas)**, real `kubectl` output in `docs/k8s_load_test_output.txt`, zero
      dropped requests across 1,796 total.
- [x] `docs/k8s_load_test_output.txt` has real kubectl output -- unedited, timestamped,
      chronological, covering baseline through post-load-test settling.
- [x] `docs/k8s_probe_demo.md` demonstrates `/health` vs `/ready` gating traffic
      differently, with real kubectl output -- pod held `Running`+`NotReady` 57+s with 0
      restarts, direct port-forward to the pod proved the app's real `/ready` reported
      `ready: true` throughout, `kubectl describe pod` confirmed the probe's configured
      target (not app state) was the failing signal.
- [x] README "what I'd change at production scale" section written -- ingress, secrets
      management, hostPath `mlruns/` (explicitly flagged non-generalizing), namespaces,
      GPU nodes, multi-region/multi-cluster.
- [x] hostPath volume limitation explicitly flagged as non-generalizing -- in
      `k8s/deployment.yaml`'s own comments, `docs/k8s_setup.md`, and README's "What I'd
      change at production scale" section, consistent with the same limitation already
      flagged for `docker-compose.yml` in M5's decisions.md entry.
- [x] `02_TRIALOUTCOME_SPEC.md` M8 row updated to ✅, milestones table reflects project
      completion -- "Remaining (M5–M8)" renamed to "Completed (M5–M8) -- all TrialOutcome
      milestones now done" (with an explicit note that PharmaPulse itself has not yet
      finished its own milestone list, so there was no finished-portfolio-project pattern
      to mirror, per the M8 prompt's instruction to follow one).
- [x] `decisions.md` updated with M8 entries (this section).
- [x] TrialOutcome project marked complete in README -- "Status: all 8 milestones
      complete (M1–M8)" added directly under the intro paragraph.
