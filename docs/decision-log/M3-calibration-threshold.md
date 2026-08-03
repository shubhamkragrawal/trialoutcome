[← back to decisions.md summary](../../decisions.md)

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

