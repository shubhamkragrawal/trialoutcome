[← back to decisions.md summary](../../decisions.md)

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

