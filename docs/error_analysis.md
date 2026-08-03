# Error Analysis — TrialOutcome

**Regenerated from scratch in M9.** Every number, every trial, and every theme on this page
changed when `enrollment_count` was removed as target leakage — the previous version of this
document analysed a model whose dominant feature encoded the label. See `decisions.md` M9-1.

Model: XGBoost champion (M2 hyperparameters, retrained in M9 on the no-enrollment feature set),
isotonic-calibrated on CALIB, MLflow registry version **16** (`Production`).
Cost-optimal threshold = **0.14**, selected on CALIB and reported on TEST
(`domains/pharma/config.yaml` → `model.threshold_decision`; see `decisions.md` M9-4 for why the
threshold moved from 0.22 and why it ships with a bootstrap interval).
All numbers below are from the TEST split (`start_date >= 2022-01-01`, n=5,700, 31.7% positive).

## Headline: what changed when the leak came out

| | Pre-M9 (leaked) | M9 (honest) |
|---|---|---|
| TEST PR-AUC (uncalibrated) | 0.8878 | **0.6484** |
| TEST ROC-AUC | 0.9178 | **0.7873** |
| #1 feature by mean \|SHAP\| | `log_enrollment_count` (1.6443) | **`sponsor_prior_termination_rate` (0.5000)** |
| False negatives at the operating threshold | — | 77 |

Majority-class baseline PR-AUC on this split is 0.3167. The honest model earns 0.648 — squarely
inside the ~0.70–0.80 ROC-AUC band the published literature reports for registry-feature trial
outcome prediction, which is where it should have been all along.

## Global SHAP importance (TEST, aggregated to original features)

| Rank | Feature | mean(\|SHAP\|) |
|---|---|---|
| 1 | `sponsor_prior_termination_rate` | 0.5000 |
| 2 | `start_year` | 0.4124 |
| 3 | `has_results` | 0.2528 |
| 4 | `eligibility_criteria_length` | 0.1381 |
| 5 | `num_sites` | 0.1246 |
| 6 | `sponsor_prior_trial_count` | 0.1225 |
| 7 | `has_dmc_str` | 0.1206 |
| 8 | `masking` | 0.0925 |
| 9 | `phase` | 0.0725 |
| 10 | `num_primary_outcomes` | 0.0609 |
| 11 | `condition_rarity` | 0.0326 |
| 12 | `exclusion_keyword_count` | 0.0291 |
| 13 | `sponsor_class` | 0.0241 |
| 14 | `allocation` | 0.0206 |
| 15 | `start_quarter` | 0.0131 |

The ordering is far flatter than before. Pre-M9, the top feature was 4.4× the second; now it is
1.2× the second. That flatness is itself the honest finding: **no single registry field predicts
trial termination strongly**, and the previous model's apparent confidence came from reading the
label off a column.

`start_year` at rank 2 deserves a caveat — it is partly a data-recency artifact (trials that
started recently are less likely to have reached a terminal status yet, and the label is only
defined on terminal statuses), not purely a real trend. It is retained because the temporal split
means the model never sees a future year at training time, but it should not be read as
"trials are getting riskier."

## 20 Worst False Negatives

The 20 trials the model was most confident would *complete* that actually terminated, ranked by
lowest calibrated probability. `top_shap_feature` is the single largest-magnitude SHAP contributor
(one-hot-aggregated to the original feature name) for that prediction. Summaries are produced by
the M9-corrected `plain_english.generate_summary` (see `decisions.md` M9-3 — factors are now
partitioned by the sign of their contribution).

| nct_id | proba | sponsor_prior_term_rate | sites | prior_trials | top_shap_feature |
|---|---|---|---|---|---|
| NCT05513391 | 0.036 | 3.9% | 31 | 335 | sponsor_prior_termination_rate |
| NCT05967130 | 0.036 | 0.5% | 1 | 189 | sponsor_prior_termination_rate |
| NCT06301308 | 0.036 | 0.0% | 1 | 51 | sponsor_prior_termination_rate |
| NCT05398263 | 0.061 | 8.9% | 64 | 2,808 | has_results |
| NCT06104683 | 0.061 | 0.0% | 4 | 14 | sponsor_prior_termination_rate |
| NCT06736509 | 0.061 | 3.6% | 22 | 28 | sponsor_prior_termination_rate |
| NCT06077773 | 0.067 | 0.0% | 48 | 3 | sponsor_prior_termination_rate |
| NCT06400589 | 0.067 | 0.0% | 1 | 14 | sponsor_prior_termination_rate |
| NCT02633514 | 0.082 | 2.7% | 1 | 812 | sponsor_prior_termination_rate |
| NCT05124691 | 0.082 | 0.0% | 3 | 13 | sponsor_prior_termination_rate |
| NCT05177770 | 0.082 | 0.0% | 9 | 10 | sponsor_prior_termination_rate |
| NCT05194839 | 0.082 | 0.0% | 51 | 3 | sponsor_prior_termination_rate |
| NCT05238025 | 0.082 | 2.1% | 120 | 48 | sponsor_prior_termination_rate |
| NCT05254236 | 0.082 | 0.0% | 1 | 15 | sponsor_prior_termination_rate |
| NCT05256134 | 0.082 | 11.8% | 63 | 1,848 | has_results |
| NCT05262517 | 0.082 | 17.9% | 32 | 2,746 | has_results |
| NCT05264506 | 0.082 | 5.8% | 103 | 499 | start_year |
| NCT05368558 | 0.082 | 15.1% | 52 | 628 | has_results |
| NCT05681481 | 0.082 | 8.3% | 40 | 36 | start_year |
| NCT05694611 | 0.082 | 4.7% | 1 | 253 | sponsor_prior_termination_rate |

Representative summaries (full set in the M9 regeneration artifacts):

- **NCT06301308** — "This trial is flagged LOW RISK (missed) (4% estimated termination
  probability) primarily because: (1) the sponsor's prior termination rate is 0%, which is low
  relative to the training population, (2) eligibility criteria are typical in length at 64 words,
  (3) masking is SINGLE (below average). One factor to watch: start_year is 2023 (above average)."
- **NCT05262517** — "This trial is flagged LOW RISK (missed) (8% estimated termination
  probability) primarily because: (1) has_results is True (below average), (2) the trial runs at 32
  sites -- a typical number of sites, (3) this is a Phase 3 trial. One factor to watch: the
  sponsor's prior termination rate is 18%, which is roughly typical for the training population."

Note the second one: the model's own summary now surfaces that the sponsor's 18% termination rate
argued *against* the low-risk call. Pre-M9 that factor would simply have been omitted, because the
summary only ever listed factors supporting the stated direction by magnitude, without checking
sign.

## Failure Themes

Across **all 77** false negatives at the operating threshold, the top SHAP contributor is
`sponsor_prior_termination_rate` for 55, `start_year` for 11, `has_results` for 10, and a condition
one-hot for 1. The three themes below reflect that distribution.

### Theme 1 — A clean sponsor record dominates everything else (14 / 20)

- **Members:** NCT05513391, NCT05967130, NCT06301308, NCT06104683, NCT06736509, NCT06077773,
  NCT06400589, NCT02633514, NCT05124691, NCT05177770, NCT05194839, NCT05238025, NCT05254236,
  NCT05694611.
- **Shared signature:** every one has `sponsor_prior_termination_rate <= 5%`, and for all 14 that
  feature is the single largest SHAP contributor, pushing hard toward "will complete."
- **Why the model struggles:** this is the direct, predictable consequence of removing the leaked
  feature. The model is now substantially a *sponsor-reputation* model, and sponsor reputation is
  an aggregate that says nothing about the specific trial's drug, protocol, or funding decision. A
  sponsor with a 0% historical termination rate will eventually terminate a trial, and when they
  do, this model has no mechanism to anticipate it. **This theme is not a bug to fix; it is the
  ceiling of the current feature set**, and it is the honest version of what pre-M9 Theme 3 was
  gesturing at.
- **Second-order concern:** `sponsor_prior_termination_rate` being rank 1 makes its known
  point-in-time weakness materially more important than it was pre-M9, when it ranked 2 behind a
  feature doing all the work. See Known Limitations below — the prior trials' *final* labels are
  used, not their status as of this trial's start date.

### Theme 2 — `has_results = True` at mega-sponsors overrides a mediocre sponsor record (4 / 20)

- **Members:** NCT05398263 (2,808 prior trials, 64 sites), NCT05256134 (1,848 / 63),
  NCT05262517 (2,746 / 32), NCT05368558 (628 / 52).
- **Shared signature:** these are the *only* rows in the worst-20 whose sponsor termination rate is
  **not** low (8.9%, 11.8%, 17.9%, 15.1% — the four highest in the table). For all four,
  `has_results = True` is the top contributor and pushes toward completion strongly enough to
  overwhelm the sponsor signal.
- **Why the model struggles:** `has_results` was verified non-leaky in M1 (1,228 actively-recruiting
  trials already carry `has_results = true`, so it is populated before completion for a meaningful
  share of trials). It is still a *reporting-behavior* signal, not a design signal: large,
  well-resourced sponsors post results consistently, so the feature substantially proxies "big
  organized sponsor." When such a sponsor terminates a trial anyway, the feature points confidently
  the wrong way. Worth re-verifying if it ever climbs above rank 3.

### Theme 3 — Recency (2 / 20)

- **Members:** NCT05264506 (2022), NCT05681481 (2023).
- **Shared signature:** `start_year` is the top contributor; both are large multi-site trials
  (103 and 40 sites) at established sponsors.
- **Why the model struggles:** `start_year` carries the recency artifact described above. For
  trials near the start of the TEST window it contributes a mild "recent trials complete" push that
  reflects label-availability mechanics as much as any real trend. This is the smallest theme and
  the one most likely to be an artifact rather than a finding.

## Multicollinearity Findings

Pearson correlation across the 9 engineered numeric/ordinal features (`num_primary_outcomes`,
`num_sites`, `eligibility_criteria_length`, `exclusion_keyword_count`, `sponsor_prior_trial_count`,
`sponsor_prior_termination_rate`, `condition_rarity`, `start_year`, `start_quarter`), computed on
TRAIN (n=66,105). `log_enrollment_count` is absent from this list — it was dropped in M9.

**Pairs with |r| > 0.6:**

| Feature A | Feature B | r | Both in top-10 SHAP? |
|---|---|---|---|
| `eligibility_criteria_length` | `exclusion_keyword_count` | 0.70 | No |

`eligibility_criteria_length` now ranks **4th** in aggregated global SHAP importance (it ranked
10th pre-M9 — everything moved up when enrollment left); `exclusion_keyword_count` ranks 12th,
still below the top-10 cutoff. **Because only one member of the pair is in the top-10, no
attribution-splitting caveat is required for this build.**

That said, the margin is thinner than it was. The trigger condition to add the caveat is unchanged
and now closer to firing: if `exclusion_keyword_count` rises into the top-10 on a future retrain,
the two features' SHAP values must be read together rather than attributed independently, since a
bullet-heavy eligibility section mechanically produces both a longer `eligibility_criteria_length`
and a higher `exclusion_keyword_count`.

No other feature pair exceeds |r| > 0.6.

## Known Limitations

- **`enrollment_count` was target leakage and is gone (M9).** `WITHDRAWN` is definitionally a
  zero-enrollment status, so `P(label=1 | enrollment == 0) = 1.000` on TEST. It contributed ~40%
  of the model's entire lift over baseline. Removing it cut TEST PR-AUC from 0.888 to 0.648. The
  fix that looked obvious — keeping only `ESTIMATED`-typed enrollment — was also rejected, on
  measurement: its missingness indicator becomes a proxy for post-hoc record state and supplies 85%
  of that variant's apparent gain. Full analysis in `decisions.md` M9-1.
- **Sponsor history uses each prior trial's final label, not its status as of this trial's
  `start_date`.** A prior trial that started in 2015 and terminated in 2019 counts as "terminated"
  in the history of a trial starting in 2018, when nobody knew that yet. Inherited from the spec's
  example SQL and flagged since M1 — but now materially more important, since
  `sponsor_prior_termination_rate` is the model's **#1** feature rather than its #2 behind a leaked
  column. Scheduled for a real fix in M9 P1-11.
- **The operating threshold is unstable.** CALIB selects 0.14, TEST would have selected 0.22, and a
  1,000-resample bootstrap puts the 95% CI at [0.10, 0.20]. The cost surface is flat across that
  range (3.7% regret), so the choice is defensible, but the point estimate should not be quoted
  without its interval. At 0.14 the model flags 70.5% of trials that actually complete — a triage
  filter, not an automated decision gate.
- **Isotonic calibration saturates.** Some served probabilities pin at exactly 0.000 / 1.000, which
  produces summaries claiming "100% estimated termination probability." Clipping to [0.01, 0.99] is
  queued as M9 P2-18; replacing isotonic with beta calibration or a monotone spline is the real fix
  and is deferred.
- **`num_arms` not available in the mart** (AACT source: `studies.number_of_arms`) — confirmed
  absent from `marts.fct_trials` in M1; excluded entirely (see `config.yaml` `dropped_features`).
- **`therapeutic_area` is NULL for all rows in `dim_condition`** — `condition_name` is used as a
  proxy throughout (condition one-hot + condition rarity), coarser than a true therapeutic-area
  grouping.
- **The 5:1 FN:FP cost ratio is a domain assumption**, not empirically derived from clinical-
  operations cost data. A real deployment needs that ratio validated (or replaced) by clinical
  operations / portfolio-management stakeholders before the threshold is trusted operationally —
  and the M9 instability finding means the ratio and the threshold have to be revisited together,
  since a flat cost surface is a property of *this* ratio at *this* base rate.
