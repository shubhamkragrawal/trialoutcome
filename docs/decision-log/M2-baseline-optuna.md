[← back to decisions.md summary](../../decisions.md)

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
- **What:** The spec's literal Makefile spec said `make train` -> `python -m core.training_pipeline`.
  Built it instead as `python -m domains.pharma.train_pipeline`, with `core/training_pipeline.py`
  staying a pure, importable, model-agnostic class (`OptunaMLflowTrainer`) with no `__main__` block and
  no PharmaDatasetBuilder import.
- **Why (vs. alternatives):** Giving `core/training_pipeline.py` a pharma-aware `__main__` block would
  violate the project's own hard rule ("no pharma SQL, no pharma column names" in core) and invert the
  intended dependency direction (core must not depend on domains/pharma). Resolving this mechanical
  contradiction in the spec's own Makefile line vs. its architecture-discipline rule -- not scope
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

  Every family clears the 0.3167 baseline by a wide margin -- M2 acceptance criteria ("PR-AUC beats majority-class
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

