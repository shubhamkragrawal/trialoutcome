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
