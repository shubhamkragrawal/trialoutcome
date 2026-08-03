# Error Analysis — TrialOutcome

**Regenerated from scratch in M9, updated again for M9-11.** Every number, every trial, and every
theme on this page changed twice: once when `enrollment_count` was removed as target leakage
(M9-1), and again when `sponsor_prior_termination_rate`'s point-in-time bug was fixed (M9-11 — a
historical trial was being counted as a termination based on its *current-day* status rather than
its status as of the querying trial's own `start_date`). See `decisions.md` M9-1 and M9-11.

Model: XGBoost champion (M2 hyperparameters, retrained twice — M9-1 then M9-11 — on the
no-enrollment, point-in-time-corrected feature set), isotonic-calibrated on CALIB, MLflow registry
version **45** (`Production`). Cost-optimal threshold = **0.16**, selected on CALIB and reported on
TEST (`domains/pharma/config.yaml` → `model.threshold_decision`; see `decisions.md` M9-4/M9-11 for
why the threshold moved from 0.22 → 0.14 → 0.16 and why it ships with a bootstrap interval).
All numbers below are from the TEST split (`start_date >= 2022-01-01`, n=5,789, 31.5% positive).

## Headline: what changed when the leak came out

| | Pre-M9 (leaked) | M9-1 (enrollment fixed) | M9-11 (sponsor-history also fixed) |
|---|---|---|---|
| TEST PR-AUC (uncalibrated) | 0.8878 | 0.6484 | **0.6193** |
| TEST ROC-AUC | 0.9178 | 0.7873 | **0.7662** |
| #1 feature by mean \|SHAP\| | `log_enrollment_count` (1.6443) | `sponsor_prior_termination_rate` (0.5000) | **`start_year` (0.3341)** |
| False negatives at the operating threshold | — | 77 | **87** |

Majority-class baseline PR-AUC on this split is 0.3146. The honest model earns 0.619 — squarely
inside the ~0.70–0.80 ROC-AUC band the published literature reports for registry-feature trial
outcome prediction, which is where it should have been all along.

## Global SHAP importance (TEST, aggregated to original features)

| Rank | Feature | mean(\|SHAP\|) |
|---|---|---|
| 1 | `start_year` | 0.3341 |
| 2 | `sponsor_prior_termination_rate` | 0.3276 |
| 3 | `has_results` | 0.2422 |
| 4 | `eligibility_criteria_length` | 0.1417 |
| 5 | `has_dmc_str` | 0.1351 |
| 6 | `num_sites` | 0.1313 |
| 7 | `masking` | 0.1208 |
| 8 | `phase` | 0.0902 |
| 9 | `sponsor_prior_trial_count` | 0.0864 |
| 10 | `num_primary_outcomes` | 0.0674 |
| 11 | `sponsor_class` | 0.0318 |
| 12 | `condition_rarity` | 0.0255 |
| 13 | `exclusion_keyword_count` | 0.0197 |
| 14 | `allocation` | 0.0160 |
| 15 | `start_quarter` | 0.0116 |

**M9-11 changed the top-2 ordering, not just the values.** Under M9-1, `sponsor_prior_termination_rate`
was the clear #1 (0.5000, 1.2× the #2 feature). After M9-11's point-in-time fix corrected its values
(no longer using hindsight to count still-pending prior trials as resolved terminations),
`start_year` edged narrowly into #1 — the two are now essentially tied (0.3341 vs 0.3276, a 1.02×
gap). The ordering is still far flatter than pre-M9's 4.4× gap: **no single registry field
predicts trial termination strongly**, and the previous model's apparent confidence came from
reading the label off a column.

`start_year` deserves the same caveat it always has — it is partly a real secular trend
(termination base rates rose from 17.8% pre-2020 to ~31.5% in the 2022+ test period, `decisions.md`
M1) and partly a data-recency artifact (trials that started recently are less likely to have
reached a terminal status yet, and the label is only defined on terminal statuses). It is retained
because the temporal split means the model never sees a future year at training time, but it
should not be read as "trials are getting riskier" in isolation from that artifact.

## 20 Worst False Negatives

The 20 trials the model was most confident would *complete* that actually terminated, ranked by
lowest calibrated probability. `top_shap_feature` is the single largest-magnitude SHAP contributor
(one-hot-aggregated to the original feature name) for that prediction. Summaries are produced by
the M9-corrected `plain_english.generate_summary` (see `decisions.md` M9-3 — factors are now
partitioned by the sign of their contribution).

| nct_id | proba | sponsor_prior_term_rate | sites | prior_trials | top_shap_feature |
|---|---|---|---|---|---|
| NCT05398263 | 0.022 | 5.6% | 64 | 2,808 | has_results |
| NCT06128369 | 0.022 | 0.0% | 24 | 6 | start_year |
| NCT05681481 | 0.058 | 0.0% | 40 | 36 | start_year |
| NCT05694611 | 0.058 | 2.4% | 1 | 253 | eligibility_criteria_length |
| NCT05513391 | 0.058 | 3.6% | 31 | 335 | start_year |
| NCT05967130 | 0.058 | 0.5% | 1 | 189 | eligibility_criteria_length |
| NCT06301308 | 0.058 | 0.0% | 1 | 51 | sponsor_prior_termination_rate |
| NCT06077773 | 0.063 | 0.0% | 48 | 3 | start_year |
| NCT06400589 | 0.063 | 0.0% | 1 | 14 | sponsor_prior_termination_rate |
| NCT06104683 | 0.063 | 0.0% | 4 | 14 | start_year |
| NCT05368558 | 0.063 | 9.9% | 52 | 628 | sponsor_prior_termination_rate |
| NCT05194839 | 0.063 | 0.0% | 51 | 3 | start_year |
| NCT06061081 | 0.063 | 7.6% | 18 | 236 | start_year |
| NCT05288283 | 0.120 | 11.4% | 14 | 166 | start_year |
| NCT07381530 | 0.120 | 0.0% | 3 | 5 | start_year |
| NCT06011577 | 0.120 | 0.0% | 36 | 1 | start_year |
| NCT05758415 | 0.120 | 11.9% | 24 | 2,332 | has_results |
| NCT05753774 | 0.120 | 0.0% | 1 | 2 | eligibility_criteria_length |
| NCT05722522 | 0.120 | 11.9% | 15 | 2,333 | has_results |
| NCT05348681 | 0.120 | 0.0% | 12 | 4 | start_year |

Representative summaries (full set in the M9-11 regeneration artifacts):

- **NCT06301308** — "This trial is flagged LOW RISK (missed) (6% estimated termination
  probability) primarily because: (1) the sponsor's prior termination rate is 0%, which is low
  relative to the training population, (2) eligibility criteria are typical in length at 64 words,
  (3) masking is SINGLE (below average). One factor to watch: start_year is 2023 (above average)."
- **NCT05398263** — "This trial is flagged LOW RISK (missed) (2% estimated termination
  probability) primarily because: (1) has_results is True (below average), (2) the trial runs at
  64 sites — a large, highly multi-site trial, (3) condition_Asthma is True (below average). One
  factor to watch: start_year is 2022 (above average)."

Note both: the model's own summary now surfaces `start_year` as a "factor to watch" on nearly
every worst miss — the direct consequence of `start_year` overtaking `sponsor_prior_termination_rate`
as the #1 global driver under M9-11.

## Failure Themes

Across **all 87** false negatives at the operating threshold, the top SHAP contributor is
`start_year` for 48, `eligibility_criteria_length` for 14, `sponsor_prior_termination_rate` for 12,
`has_results` for 10, and a condition one-hot or `num_sites` for the remaining 3. **This is a
different distribution than M9-1's** (`sponsor_prior_termination_rate` was top contributor for
55/77 then) — M9-11's point-in-time fix didn't just change `sponsor_prior_termination_rate`'s
values, it demoted the feature's role in the model's *worst* misses specifically, even though its
overall global-SHAP rank barely moved (was #1, now a close #2). The four themes below reflect the
new distribution.

### Theme 1 — Recency: the model reads "too soon to know" as "will complete" (48 / 87, 11/20 worst)

- **Members (worst-20):** NCT06128369, NCT05681481, NCT05513391, NCT06077773, NCT06104683,
  NCT05194839, NCT06061081, NCT05288283, NCT07381530, NCT06011577, NCT05348681.
- **Shared signature:** `start_year` is the largest SHAP contributor, consistently pushing toward
  "will complete" for later-starting trials.
- **Why the model struggles:** `start_year` conflates two effects it cannot separate — a genuine
  secular rise in termination rates (17.8% pre-2020 → 31.5% in 2022+) and a data-recency artifact
  (a trial's label is only defined once it reaches a terminal status, so recently-started trials
  are systematically under-represented among *known* terminations at prediction time even though
  their true future termination risk may be just as high). The model correctly learns "recent →
  less often labeled terminated in TRAIN," but that's partly a labeling-maturity fact about the
  data, not a property of the trial itself. **This is now the largest theme, and the most
  structurally hard one to fix** — it is a property of how the label matures over time, not a
  missing feature.

### Theme 2 — Long, detailed eligibility criteria read as design rigor, sometimes wrongly (14 / 87, 3/20 worst)

- **Members (worst-20):** NCT05694611, NCT05967130, NCT05753774.
- **Shared signature:** `eligibility_criteria_length` is the top contributor — longer, more
  detailed eligibility sections are associated with predicted completion (a "design rigor"
  signal), but two of these three specific misses actually have *short* (59, "typical") criteria
  lengths, meaning the model is reading a near-average value as reassuring rather than neutral.
- **Why the model struggles:** the feature genuinely correlates with completion on average, but a
  near-median value carries a SHAP push in only one direction (toward completion) with nothing to
  counterbalance it when other signals are also weak/absent — a limitation of a monotonic
  tree-based push rather than a true design-quality assessment.

### Theme 3 — A clean sponsor record still dominates a meaningful minority (12 / 87, 3/20 worst)

- **Members (worst-20):** NCT06301308, NCT06400589, NCT05368558.
- **Shared signature:** every one has `sponsor_prior_termination_rate <= 10%`, and for these three
  it remains the single largest SHAP contributor.
- **Why the model struggles:** this is the residual version of M9-1's Theme 1 — the model is
  still, for a meaningful subset of misses, substantially a sponsor-reputation model, and a low
  historical termination rate says nothing about *this* trial's drug, protocol, or funding
  decision. M9-11's fix corrected the feature's *values* (removing the hindsight leak) but not
  this structural ceiling — a sponsor with a genuinely clean point-in-time record will still
  eventually have a trial terminate, and the model has no mechanism to anticipate which one.

### Theme 4 — `has_results = True` at mega-sponsors overrides a mediocre sponsor record (10 / 87, 3/20 worst)

- **Members (worst-20):** NCT05398263 (2,808 prior trials, 64 sites), NCT05758415 (2,332 / 24),
  NCT05722522 (2,333 / 15).
- **Shared signature:** these three all have `sponsor_prior_termination_rate` in the 5.6–11.9%
  range — not the lowest in the table — but `has_results = True` is the top contributor and pushes
  toward completion strongly enough to override the (already middling) sponsor signal.
- **Why the model struggles:** `has_results` was verified non-leaky in M1 (populated pre-completion
  for a meaningful share of trials). It is still a *reporting-behavior* signal, not a design
  signal: large, well-resourced sponsors post results consistently, so the feature substantially
  proxies "big organized sponsor." When such a sponsor terminates a trial anyway, the feature
  points confidently the wrong way.

## Multicollinearity Findings

Pearson correlation across the 9 engineered numeric/ordinal features (`num_primary_outcomes`,
`num_sites`, `eligibility_criteria_length`, `exclusion_keyword_count`, `sponsor_prior_trial_count`,
`sponsor_prior_termination_rate`, `condition_rarity`, `start_year`, `start_quarter`), computed on
TRAIN (n=66,129). `log_enrollment_count` is absent from this list — it was dropped in M9-1.

**Pairs with |r| > 0.6:**

| Feature A | Feature B | r | Both in top-10 SHAP? |
|---|---|---|---|
| `eligibility_criteria_length` | `exclusion_keyword_count` | 0.70 | No |

`eligibility_criteria_length` ranks **4th** in aggregated global SHAP importance (it ranked 10th
pre-M9 — everything moved up when enrollment left, and M9-11 didn't change this rank);
`exclusion_keyword_count` ranks 13th, still well below the top-10 cutoff. **Because only one
member of the pair is in the top-10, no attribution-splitting caveat is required for this build.**

That said, the margin is thinner than it was pre-M9. The trigger condition to add the caveat is
unchanged and still worth watching: if `exclusion_keyword_count` rises into the top-10 on a future
retrain, the two features' SHAP values must be read together rather than attributed independently,
since a bullet-heavy eligibility section mechanically produces both a longer
`eligibility_criteria_length` and a higher `exclusion_keyword_count`.

No other feature pair exceeds |r| > 0.6.

## Known Limitations

- **`enrollment_count` was target leakage and is gone (M9-1).** `WITHDRAWN` is definitionally a
  zero-enrollment status, so `P(label=1 | enrollment == 0) = 1.000` on TEST. It contributed ~40%
  of the model's entire lift over baseline (measured at the time of that investigation). Removing
  it cut TEST PR-AUC from 0.888 to 0.648. The fix that looked obvious — keeping only
  `ESTIMATED`-typed enrollment — was also rejected, on measurement: its missingness indicator
  becomes a proxy for post-hoc record state and supplies 85% of that variant's apparent gain. Full
  analysis in `decisions.md` M9-1.
- **Sponsor-history hindsight leak fixed (M9-11); one residual gap remains.** Prior trials were
  being counted as terminations based on their *current-day* status rather than their status as of
  this trial's own `start_date` — fixed by requiring an ACTUAL (not ESTIMATED) resolution date
  strictly before `start_date`. Measured effect on the 2023+ subset: average rate dropped
  0.108 → 0.067 (−38% relative), 58% of rows changed. The residual gap: a prior trial genuinely
  still running as of `start_date` is treated identically to one that will later complete
  successfully — its true future outcome is unknowable at that point by construction (correct
  point-in-time behavior), but it means the feature still can't distinguish "clean track record"
  from "track record mostly still pending." Full analysis in `decisions.md` M9-11.
- **The operating threshold moved twice, and is now more stable.** M9-4 refit CALIB to 0.14 against
  TEST's 0.22 (gap 0.08, "unstable," 95% CI [0.10, 0.20]). M9-11's sponsor-history fix moved CALIB
  to 0.16, narrowing the gap to TEST's 0.21 to exactly 0.05 ("stable" by the same rule), 95% CI
  [0.13, 0.21], with TEST's value now falling inside the CI. Regret vs. the unattainable
  TEST-optimal choice improved from 3.7% to 2.8%. At 0.16 the model flags 72.9% of trials that
  actually complete — a triage filter, not an automated decision gate.
- **Isotonic calibration saturates.** Some served probabilities pin at exactly 0.000 / 1.000, which
  produces summaries claiming "100% estimated termination probability." Clipping to [0.01, 0.99] is
  queued as M9 P2-18; replacing isotonic with beta calibration or a monotone spline is the real fix
  and is deferred.
- **`num_arms` not available in the mart** (AACT source: `studies.number_of_arms`) — confirmed
  absent from `marts.fct_trials` in M1; excluded entirely (see `config.yaml` `dropped_features`).
- **`therapeutic_area` is NULL for all rows in `dim_condition`** — `condition_name` is used as a
  proxy throughout (condition one-hot + condition rarity), coarser than a true therapeutic-area
  grouping.
- **`start_year` conflates a real secular trend with a label-maturity artifact**, and after M9-11
  it is the single largest driver of both global SHAP importance and the worst false negatives
  (Theme 1, 48/87). This is the least tractable of the known limitations — it isn't a data-quality
  bug to fix, but a structural property of predicting a label that can only be observed once a
  trial reaches a terminal status.
- **The 5:1 FN:FP cost ratio is a domain assumption**, not empirically derived from clinical-
  operations cost data. A real deployment needs that ratio validated (or replaced) by clinical
  operations / portfolio-management stakeholders before the threshold is trusted operationally.
