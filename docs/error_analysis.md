# Error Analysis — TrialOutcome

Model: XGBoost best trial (MLflow run `c4a4d0300bd949f8bb07b7c48417be4d`), isotonic-calibrated
(`notebooks/03_calibration.ipynb`), cost-optimal threshold = **0.22** (`domains/pharma/config.yaml`
→ `model.threshold_decision`). All numbers below are from the TEST split (`start_date >= 2022-01-01`,
n=5,700), computed in `notebooks/04_shap_analysis.ipynb`.

## 20 Worst False Negatives

The 20 trials the model was most confident would *complete*, that actually terminated —
ranked by lowest calibrated probability of termination. `top_shap_feature` is the single
largest-magnitude SHAP contributor (one-hot-aggregated to the original feature name) for that
prediction.

| nct_id | proba | sponsor_prior_termination_rate | top_shap_feature | plain_english_summary |
|---|---|---|---|---|
| NCT05062889 | 0.025 | 3.7% | log_enrollment_count | This trial is flagged LOW RISK (missed) (2% estimated termination probability) primarily because: (1) enrollment is within a typical range at 477 participants, (2) the sponsor's prior termination rate is 4%, which is low relative to the training population, (3) no Data Monitoring Committee is registered. |
| NCT05124691 | 0.025 | 0.0% | log_enrollment_count | This trial is flagged LOW RISK (missed) (2% estimated termination probability) primarily because: (1) enrollment is unusually large at 1,001 participants, (2) the sponsor's prior termination rate is 0%, which is low relative to the training population, (3) a Data Monitoring Committee is registered. |
| NCT06400589 | 0.025 | 0.0% | log_enrollment_count | This trial is flagged LOW RISK (missed) (2% estimated termination probability) primarily because: (1) enrollment is within a typical range at 750 participants, (2) the sponsor's prior termination rate is 0%, which is low relative to the training population, (3) no Data Monitoring Committee is registered. |
| NCT02590523 | 0.029 | 0.0% | log_enrollment_count | This trial is flagged LOW RISK (missed) (3% estimated termination probability) primarily because: (1) enrollment is within a typical range at 500 participants, (2) the sponsor's prior termination rate is 0%, which is low relative to the training population, (3) the trial runs at 1 sites -- a small number of sites. |
| NCT06677203 | 0.029 | 0.0% | log_enrollment_count | This trial is flagged LOW RISK (missed) (3% estimated termination probability) primarily because: (1) enrollment is within a typical range at 123 participants, (2) the sponsor's prior termination rate is 0%, which is low relative to the training population, (3) no Data Monitoring Committee is registered. |
| NCT05513391 | 0.029 | 3.9% | log_enrollment_count | This trial is flagged LOW RISK (missed) (3% estimated termination probability) primarily because: (1) enrollment is within a typical range at 366 participants, (2) the sponsor's prior termination rate is 4%, which is low relative to the training population, (3) the trial runs at 31 sites -- a typical number of sites. |
| NCT06985498 | 0.029 | 1.5% | sponsor_prior_termination_rate | This trial is flagged LOW RISK (missed) (3% estimated termination probability) primarily because: (1) the sponsor's prior termination rate is 2%, which is low relative to the training population, (2) enrollment is unusually small at 40 participants, (3) allocation is NA (below average). |
| NCT05150496 | 0.029 | 5.6% | log_enrollment_count | This trial is flagged LOW RISK (missed) (3% estimated termination probability) primarily because: (1) enrollment is within a typical range at 640 participants, (2) the sponsor's prior termination rate is 6%, which is low relative to the training population, (3) no Data Monitoring Committee is registered. |
| NCT07381530 | 0.029 | 0.0% | log_enrollment_count | This trial is flagged LOW RISK (missed) (3% estimated termination probability) primarily because: (1) enrollment is within a typical range at 100 participants, (2) the sponsor's prior termination rate is 0%, which is low relative to the training population, (3) the trial runs at 3 sites -- a typical number of sites. |
| NCT07259564 | 0.043 | 3.3% | log_enrollment_count | This trial is flagged LOW RISK (missed) (4% estimated termination probability) primarily because: (1) enrollment is within a typical range at 120 participants, (2) the sponsor's prior termination rate is 3%, which is low relative to the training population, (3) the trial runs at 1 sites -- a small number of sites. |
| NCT06728657 | 0.044 | 2.6% | log_enrollment_count | This trial is flagged LOW RISK (missed) (4% estimated termination probability) primarily because: (1) enrollment is within a typical range at 68 participants, (2) the sponsor's prior termination rate is 3%, which is low relative to the training population, (3) the trial runs at 1 sites -- a small number of sites. |
| NCT06736509 | 0.053 | 3.6% | log_enrollment_count | This trial is flagged LOW RISK (missed) (5% estimated termination probability) primarily because: (1) enrollment is within a typical range at 139 participants, (2) the sponsor's prior termination rate is 4%, which is low relative to the training population, (3) the trial runs at 22 sites -- a typical number of sites. |
| NCT05330884 | 0.053 | 0.0% | log_enrollment_count | This trial is flagged LOW RISK (missed) (5% estimated termination probability) primarily because: (1) enrollment is unusually large at 9,200 participants, (2) the sponsor's prior termination rate is 0%, which is low relative to the training population, (3) the trial runs at 1 sites -- a small number of sites. |
| NCT06199076 | 0.053 | 4.9% | log_enrollment_count | This trial is flagged LOW RISK (missed) (5% estimated termination probability) primarily because: (1) enrollment is within a typical range at 206 participants, (2) the sponsor's prior termination rate is 5%, which is low relative to the training population, (3) the trial runs at 1 sites -- a small number of sites. |
| NCT05670912 | 0.053 | 4.0% | log_enrollment_count | This trial is flagged LOW RISK (missed) (5% estimated termination probability) primarily because: (1) enrollment is within a typical range at 105 participants, (2) the sponsor's prior termination rate is 4%, which is low relative to the training population, (3) the trial runs at 3 sites -- a typical number of sites. |
| NCT06199882 | 0.053 | 0.5% | sponsor_prior_termination_rate | This trial is flagged LOW RISK (missed) (5% estimated termination probability) primarily because: (1) the sponsor's prior termination rate is 0%, which is low relative to the training population, (2) enrollment is unusually small at 34 participants, (3) allocation is NA (below average). |
| NCT05756556 | 0.053 | 0.0% | log_enrollment_count | This trial is flagged LOW RISK (missed) (5% estimated termination probability) primarily because: (1) enrollment is within a typical range at 68 participants, (2) the sponsor's prior termination rate is 0%, which is low relative to the training population, (3) the trial runs at 1 sites -- a small number of sites. |
| NCT05785390 | 0.053 | 0.0% | log_enrollment_count | This trial is flagged LOW RISK (missed) (5% estimated termination probability) primarily because: (1) enrollment is within a typical range at 170 participants, (2) the sponsor's prior termination rate is 0%, which is low relative to the training population, (3) the trial runs at 4 sites -- a typical number of sites. |
| NCT05948553 | 0.053 | 0.0% | log_enrollment_count | This trial is flagged LOW RISK (missed) (5% estimated termination probability) primarily because: (1) enrollment is within a typical range at 200 participants, (2) the sponsor's prior termination rate is 0%, which is low relative to the training population, (3) the sponsor has run 1 prior trials -- a newer sponsor with a limited track record. |
| NCT06405061 | 0.053 | 1.1% | sponsor_prior_termination_rate | This trial is flagged LOW RISK (missed) (5% estimated termination probability) primarily because: (1) the sponsor's prior termination rate is 1%, which is low relative to the training population, (2) enrollment is unusually small at 40 participants, (3) allocation is NA (below average). |

## Failure Themes

**Common denominator across all 20 (context, not a discriminating theme):** every one of the 20
worst false negatives has `sponsor_prior_termination_rate < 6%` — well below what would trigger a
risk flag on that feature alone. `sponsor_prior_termination_rate` is the #2 globally-important SHAP
feature, and it is doing exactly what it was designed to do (a clean sponsor history genuinely
predicts completion, on average) — but that is precisely why it can't discriminate *within* this set
of misses. The three themes below group the 20 by what, beyond sponsor history, distinguishes them.

### Theme 1 — "Mega-sponsors" whose aggregate history hides trial-specific risk
- **Count:** 5 / 20 (`sponsor_prior_trial_count >= 200`): NCT05513391 (335), NCT06728657 (900),
  NCT06199076 (898), NCT06199882 (206), NCT06405061 (281).
- **Features driving the miss (SHAP):** `sponsor_prior_trial_count` (rank 8 globally) reinforcing an
  already-low `sponsor_prior_termination_rate`; `log_enrollment_count` remains the top single
  contributor for 3 of the 5, `sponsor_prior_termination_rate` for the other 2.
- **Why the model struggles:** a sponsor with hundreds of prior trials has a termination rate
  averaged over enough trials that a single failing trial barely moves it — the aggregate is
  statistically stable and genuinely low-risk *on average*, but that average carries no information
  about this specific trial's drug, protocol, or funding decision. Note 3 of these 5 rows
  (NCT06728657, NCT06199076, NCT06199882) also land in Theme 2 below.

### Theme 2 — Single/near-single-site trials
- **Count:** 9 / 20 (`num_sites <= 1`): NCT06400589, NCT02590523, NCT06985498, NCT07259564,
  NCT06728657, NCT05330884, NCT06199076, NCT06199882, NCT05756556.
- **Features driving the miss (SHAP):** `num_sites` (rank 3 globally) sits at its low end but the
  contribution stays modest rather than decisive — it nudges risk up without crossing into a
  HIGH RISK call, because `log_enrollment_count` and `sponsor_prior_termination_rate` for the same
  rows are both pulling the other direction.
- **Why the model struggles:** `num_sites` behaves as a roughly monotonic "more sites = more
  operational resilience" signal, but a single-site trial can be well-funded and well-run right up
  until it isn't — the feature set has no direct signal for site-level execution risk (e.g. site
  experience, funding source), so a small-site trial with an otherwise clean profile reads as
  low-risk almost by default.

### Theme 3 — No individually extreme feature; the failure signal is external to this feature set
- **Count:** 9 / 20 (rows not in Theme 1 or Theme 2): NCT05062889, NCT05124691, NCT06677203,
  NCT05150496, NCT07381530, NCT06736509, NCT05670912, NCT05785390, NCT05948553.
- **Features driving the miss (SHAP):** `log_enrollment_count` is still the top contributor for
  nearly all of these, but the underlying enrollment values are themselves unremarkable ("typical
  range" per the plain-English templates) — no feature in this group is at an extreme.
- **Why the model struggles:** these trials don't have an outlier design, sponsor, or site profile in
  the current feature groups (design / sponsor-history / condition / temporal / text-lite) — whatever
  drove termination (e.g. drug efficacy or safety signal, funding withdrawal, protocol amendment) is
  simply not observed by any feature this project builds. This theme is the honest floor of the
  current feature set's explanatory power, not a modeling bug.

## Multicollinearity Findings

Pearson correlation across the 10 named engineered numeric/ordinal features
(`log_enrollment_count`, `num_primary_outcomes`, `num_sites`, `eligibility_criteria_length`,
`exclusion_keyword_count`, `sponsor_prior_trial_count`, `sponsor_prior_termination_rate`,
`condition_rarity`, `start_year`, `start_quarter`), computed on TRAIN (n=66,105).

**Pairs with |r| > 0.6:**

| Feature A | Feature B | r | Both in top-10 SHAP? |
|---|---|---|---|
| eligibility_criteria_length | exclusion_keyword_count | 0.70 | No |

`eligibility_criteria_length` ranks 10th in aggregated global SHAP importance;
`exclusion_keyword_count` ranks 13th (below the top-10 cutoff). **Because only one member of this
pair is in the top-10, no attribution-splitting caveat is required for this build** — the SHAP
attribution to `eligibility_criteria_length` in the plots and demos above can be read on its own
without a shared-signal caveat. This condition (re-check whenever the model or feature set changes)
is the trigger for adding the caveat in a future run: if `exclusion_keyword_count` ever rises into
the top-10 (e.g. after a retrain with a different feature distribution), the two features' SHAP
values would need to be read together, not attributed independently, since a bullet-point-heavy
eligibility section mechanically produces both a longer `eligibility_criteria_length` and a higher
`exclusion_keyword_count`.

No other feature pair exceeds |r| > 0.6 in this feature set.

## Known Limitations

- `num_arms` not available in the mart (AACT source: `studies.number_of_arms`) — confirmed absent
  from `marts.fct_trials` in M1; excluded from the feature set entirely (see `config.yaml`
  `dropped_features`).
- `therapeutic_area` is NULL for all rows in `dim_condition` — `condition_name` is used as a proxy
  throughout (condition one-hot + condition-rarity), which is coarser than a true therapeutic-area
  grouping would be.
- Sponsor history (`sponsor_prior_trial_count`, `sponsor_prior_termination_rate`) uses each prior
  trial's `start_date`, not its `completion_date`, for the point-in-time cutoff — a still-running
  prior trial is counted as "not terminated" even though its true outcome is unknown at the current
  trial's start date. Known inherited limitation from the spec's own example SQL, documented in M1
  `decisions.md`, not a new bug introduced here.
- `feature_pipeline_version` is `"unknown"` on every MLflow run so far — this repo has no git commits
  yet, so the git-hash-based version tag (`domains/pharma/dataset_builder.py:feature_pipeline_version`)
  cannot resolve to anything meaningful. Version-mismatch checks (M7) will not be meaningful until
  after the first commit.
- The 5:1 FN:FP cost ratio in `domains/pharma/cost_matrix.yaml` is a domain assumption made for this
  project, not empirically derived from real clinical-operations cost data — a real deployment would
  need that ratio validated (or replaced) by input from clinical operations / portfolio-management
  stakeholders before the threshold in `config.yaml` `model.threshold_decision` is trusted
  operationally.
