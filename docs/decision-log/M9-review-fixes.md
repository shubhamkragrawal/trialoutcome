[← back to decisions.md summary](../../decisions.md)

---

# M9 — Review-driven fixes

Source: `final_review.md` (senior DS/AI engineer + hiring-manager review of commit
`6d93c28`, 2026-08-02). Plan: `ai_portfolio/M9_REVIEW_FIXES_PLAN.md`. Spec addendum:
`02_TRIALOUTCOME_SPEC.md` Section 10.

## Decision (review-driven, P0): `enrollment_count` dropped as target leakage; fix path A rejected on evidence after being shown to be available

- **What:** Investigated `enrollment_count` after `final_review.md` §1.1 flagged it as
  target-leaking. Confirmed the review's evidence against the repo's own cached data
  (n=78,115) and feature pipeline: `P(label=1 | enrollment_count == 0) = 0.981` overall
  and `1.000` on TEST. The mechanism is definitional, not statistical — `WITHDRAWN`
  means "stopped before enrolling the first participant," so actual enrollment is 0 by
  definition of the label class. Ablation with the champion hyperparameters on the
  locked temporal split reproduced the review's numbers exactly: full pipeline
  **0.8878** PR-AUC → without enrollment **0.6484**, a delta of **0.2394**, i.e. ~40% of
  the total lift over the 0.3167 majority baseline came from a column encoding the
  label. Dropped `enrollment_count`, `log_enrollment_count`, and `enrollment_missing`;
  retrained the champion on the reduced feature set (37 features, down from 39);
  re-registered as Production.

- **Fix path taken: B (drop), after A was investigated, found *available*, and rejected
  on measured evidence.** This is the part worth reading. The plan's decision tree said
  to prefer path A (keep planned-only enrollment via `enrollment_type`) and to fall back
  to B only if `enrollment_type` was unavailable upstream. The diagnosis found:
    - `enrollment_type` is **not** in `marts.fct_trials` and **not** in `staging.stg_trials`
      — so by the plan's own literal trigger, path A was already blocked.
    - But it **is** recoverable from `raw.ct_studies`, whose JSONB payload carries
      `protocolSection.designModule.enrollmentInfo.type` (`ACTUAL`/`ESTIMATED`) for all
      596,690 staged studies. Path A was therefore genuinely buildable with a
      raw→staging→mart passthrough in PharmaPulse. I built it as an experiment before
      committing to it.
    - Measured on the modeling set, `enrollment_type` splits **ACTUAL 70,439 /
      ESTIMATED 3,101 / missing 4,575**. It cleanly isolates the definitional leak:
      **3,994 of 3,994** `WITHDRAWN` trials with `type='ACTUAL'` have
      `enrollment_count = 0`, while `WITHDRAWN`+`ESTIMATED` rows (n=57) have none.
    - So path A "works" — but it nulls **96%** of the column, and the resulting model
      scores 0.6821 vs path B's 0.6484. That +0.0337 looked like free signal.
    - **Decomposing where that +0.0337 comes from is what killed path A.** Under path A
      the surviving `enrollment_missing` indicator no longer means "enrollment unknown";
      it means `enrollment_type != 'ESTIMATED'`. And *that* is post-hoc: every trial's
      enrollment record reads `ESTIMATED` at registration and only flips to `ACTUAL`
      once the trial reports actual enrollment. Measured contributions over path B:

      | Path A variant | TEST PR-AUC | lift over B |
      |---|---|---|
      | value + missingness indicator | 0.6821 | +0.0337 |
      | **indicator only (no enrollment magnitude at all)** | **0.6772** | **+0.0288** |
      | value only (no indicator) | 0.6656 | +0.0172 |
      | path B (dropped entirely) | 0.6484 | — |

      A bare bit carrying zero enrollment magnitude supplies **85% of path A's entire
      advantage**. Path A does not remove the leak; it swaps an obvious definitional
      leak for a subtler record-maintenance one. Even the "value only" variant leaks,
      because 96% of rows share one imputed median and are trivially separable.
    - Rejected path A and took B. Also confirmed the raw payload is a current snapshot
      with no version history (`protocolSection` has no revision module), so genuine
      registration-time planned enrollment is **not** recoverable from this warehouse at
      all — the trigger-to-reconsider below is corrected accordingly.

- **Numbers republished:** `domains/pharma/config.yaml` (feature list, `dropped_features`,
  missingness policy, threshold block), `register_model.py` baseline constant,
  `train_pipeline.py` `NUMERIC_FEATURES`, `serving/api.py` request schema + row builder.
  README, `docs/error_analysis.md`, SHAP plots, and the M6 drift baseline follow in the
  same M9 sweep.

- **Why (vs. alternatives):**
    - *Keep and disclose as a known limitation:* rejected — a known limitation is where
      you put things you can't fix, not the thing producing your headline number.
    - *Regularize / re-weight the feature:* rejected — leakage is a semantics problem,
      not an over-fitting problem you tune away.
    - *Zero → missing patch:* rejected explicitly by the review (§1.1) — leaves graded
      post-hoc signal in the nonzero values.
    - *Path A (planned-enrollment only):* rejected **on measurement, not on
      availability** — see the decomposition above. This is the non-obvious call in M9,
      and the ablation that produced it is the reason it's defensible rather than a
      preference.

- **Failure mode:** the leakage-detection framework built in M1/M2 (temporal-vs-random
  split, controlled fixed-window ablation, honest-vs-leaky sponsor history) produced a
  genuine **true negative** — there is no temporal row-placement leakage in this dataset.
  But it structurally *could not* detect this leak, because every test in it asks "which
  rows went where," and this leak lives in "is this value knowable at `start_date`?"
  `core/dataset_builder_base.py`'s docstring already drew that distinction; nothing
  pointed it at `enrollment_count` until the review did. The second-order failure mode is
  the one path A demonstrates: once you know to look for semantic leakage, the *fix* can
  reintroduce it in a less visible form, and only an ablation tells you.

- **Scaling story (10x / 100x):** unchanged — a semantics problem, not a scale one. At
  100x the same leak produces a more confidently wrong model faster. What *does* scale is
  the check: a feature-semantics review ("what populates this column, and when?") is
  O(number of features), runs once per feature at design time, and is the only thing that
  would have caught this before training.

- **Trigger to reconsider (CORRECTED vs. the plan's template):** the plan proposed
  "if PharmaPulse ever exposes `enrollment_type`." That trigger is now known to be
  insufficient — `enrollment_type` **is** exposed in `raw`, and using it does not fix the
  problem. The correct trigger is: **if a registration-time snapshot becomes available**
  — ClinicalTrials.gov's versioned-record/history API, which returns each field as of the
  original registration date. That, and only that, makes planned enrollment a legitimate
  feature. Absent it, enrollment stays out.

- **Interview question this maps to:** *"Your leakage tests found nothing but there was
  leakage anyway. Walk me through how."* — the framework was correctly designed for the
  leakage class it tested (temporal row placement) and returned a true negative; it
  missed a different class (feature semantics); a review-driven ablation caught it. The
  follow-up worth volunteering: *"and then the obvious fix turned out to leak too, which
  I only found because I ablated the fix instead of trusting it."*

## Decision (review-driven, P0): threshold selected on CALIB and reported on TEST; instability contingency fired, bootstrap CI reported

- **What:** `ThresholdSelector.find_cost_optimal_threshold()` was being swept against
  TEST-split probabilities, with precision/recall/expected-cost then reported at the
  winning threshold on that same split. Moved selection to CALIB (n=6,310), froze the
  threshold, and report the decision table on TEST (n=5,700). Corrected the class
  docstring, which had argued in writing that "threshold selection is a downstream
  reporting step, not a fitting step" — that argument is wrong, and the docstring
  committing to it was the more damaging half of the bug.
- **Result:** CALIB selects **0.14**; TEST would have selected **0.22**. The plan set
  ~0.05 as the divergence that means "unstable, needs a bootstrap CI rather than a point
  estimate." The gap is 0.08, so that contingency fired and the bootstrap was run rather
  than noted:
    - 1,000-resample bootstrap of the CALIB selection: **95% CI [0.10, 0.20]**, median
      0.14 — an interval 0.10 wide.
    - The TEST-optimal 0.22 falls **outside** that CI.
    - The cost surface is nearly flat across the region: on TEST, cost is 3,130 at 0.14
      vs 3,019 at 0.22 — **3.7% regret** for selecting honestly. That 3.7% is precisely
      the optimism the old TEST-selected number was hiding, which is a satisfyingly small
      and concrete answer to "how much did this actually matter."
- **Operational consequence, stated plainly:** at 0.14 the model flags **70.5%** of
  trials that actually complete (recall 0.957, precision 0.386). A 5:1 FN:FP matrix
  applied to a genuinely weaker post-M9 model asks for a very aggressive operating point.
  This is a triage filter, not an automated decision gate, and the README/config say so
  rather than quoting recall alone.
- **Why (vs. alternatives):**
    - *Nested CV on the threshold:* better statistics, more machinery than a
      one-parameter fit warrants; CALIB-fit/TEST-report captures most of the
      credibility for a fraction of the work.
    - *Keep 0.22 because it's the better operating point:* rejected — it is only better
      *as measured on the split that chose it*. Choosing it knowingly is worse than
      having chosen it accidentally.
    - *Report the point estimate alone:* rejected once the CI came back 0.10 wide;
      a point estimate implies a precision this operating point does not have.
- **Failure mode:** if the cost surface had been sharply peaked instead of flat, an
  0.08 selection error would translate into a large real cost gap rather than 3.7%, and
  a point estimate would be actively dangerous. The flatness is what makes this
  recoverable — and flatness is a property of *this* cost matrix and base rate, so it
  must be re-checked whenever either changes, not assumed.
- **Scaling story (10x / 100x):** with 10x the CALIB rows the bootstrap CI narrows
  roughly as 1/√n and the point estimate becomes meaningful on its own. At 100x, the
  binding constraint stops being statistical and becomes operational — the threshold
  would be set by review capacity (how many trials a team can actually triage per week),
  with the cost sweep used to price that capacity rather than to pick the number.
- **Interview question this maps to:** *"Walk me through your threshold selection and the
  split it ran on."* — CALIB fit, TEST report, and the honest extra: the two splits
  disagreed by more than my own stability criterion allowed, so the deliverable is an
  interval and a 3.7% regret figure, not a number.

## Decision (review-driven, P0): plain-English summaries partition SHAP factors by sign; `high_risk` → `HIGH RISK` in prose only

- **What:** `generate_summary()` cited the top-3 SHAP contributors by absolute magnitude
  as "reasons for the flag" without consulting the sign of their contribution. Factors
  are now partitioned into risk-increasing and risk-decreasing, ordered by magnitude
  within each; a HIGH RISK flag cites only risk-increasing factors, a LOW RISK flag only
  risk-decreasing ones, and the strongest opposing factor is surfaced separately as
  "One factor argues against the flag" / "One factor to watch" rather than dropped.
  Separately, the human-readable string now renders `HIGH RISK`/`LOW RISK`; the
  machine-readable `threshold_decision` field is untouched (`high_risk`/`low_risk`) since
  it is part of the locked contract RegIntel builds against.
- **Scope of the bug, measured:** on the TEST split at the M9 threshold, **2,943 of 4,473
  HIGH RISK predictions (65.8%)** cited at least one risk-decreasing factor among their
  three stated reasons. This was not a rare edge case; it was the majority of served
  high-risk explanations.
- **Before / after (NCT01451853, proba 0.213, real TEST row):**
    - *Before:* "This trial is flagged high_risk (21% estimated termination probability)
      primarily because: (1) **the sponsor's prior termination rate is 0%, which is low
      relative to the training population**, (2) start_year is 2026 (above average),
      (3) has_results is False (above average)."
    - *After:* "This trial is flagged HIGH RISK (21% estimated termination probability)
      primarily because: (1) start_year is 2026 (above average), (2) has_results is False
      (above average), (3) a Data Monitoring Committee is registered. One factor argues
      against the flag: the sponsor's prior termination rate is 0%, which is low relative
      to the training population."
    - The before-text is the failure in one line: a spotless sponsor record offered as a
      reason the trial is high-risk. Its SHAP contribution was −0.682 — the single
      largest-magnitude factor, and pointing the opposite way.
- **Why (vs. alternatives):**
    - *Drop opposing factors silently:* rejected — that would overstate confidence.
      A model whose top driver argues against its own flag is telling the reader
      something real, and a borderline call should look borderline.
    - *Sort by signed contribution instead of partitioning:* rejected — it produces the
      right top-of-list most of the time but still lets a wrong-signed factor into
      position 3 whenever few factors point the flagged way, which is exactly the
      borderline case where the summary is read most carefully.
    - *Change `threshold_decision` to `HIGH RISK` too:* rejected — that is the locked
      cross-project contract; a display concern must not reach into the machine-readable
      field. The split (enum stays, prose humanizes) is the whole point.
- **Failure mode:** if a flag has NO supporting factors at all, the summary now says so
  explicitly ("none of its top factors point that way -- the flag rests on the threshold,
  not on any single driver") instead of manufacturing a justification. That string
  appearing in production is a genuine signal that the threshold and the explanation have
  come apart, and it should be alarming rather than invisible. Unit-tested.
- **Scaling story (10x / 100x):** the partition is O(k) over an already-computed top-k
  and costs nothing. At scale the real risk is template coverage, not sorting: every
  feature without an entry in `_TEMPLATES` falls back to "`feature` is `value`
  (above/below average)", which is not a real population comparison. The fallback is
  already documented as such; at 100x features it would need frozen training-population
  quantiles to stay honest.
- **Interview question this maps to:** *"Show me a plain-English summary you'd be
  comfortable putting in front of a Clinical Operations VP."* — with the before/after
  above, and the 65.8% figure, because "I found it, measured how often it fired, and
  fixed it" is a better answer than a clean example.

## Addendum to M9-1: `notebooks/02_leakage_demo.ipynb` was not, in fact, unaffected

The M9 plan and the first pass of this fix assumed `notebooks/02_leakage_demo.ipynb` (the
temporal-vs-random row-placement leakage experiment, M1) was independent of the enrollment
fix, since it predates M9 and tests a different leakage class. That assumption was wrong in
one respect: the notebook hardcodes its own `NUMERIC` feature list (separate from
`train_pipeline.py`'s), and that list included `log_enrollment_count`/`enrollment_missing`.
Once `PharmaDatasetBuilder.build_features()` stopped producing those columns (M9-1), the
notebook would `KeyError` on re-execution.

- **Fixed:** notebook 02's `NUMERIC` list, re-executed end to end. Numbers moved
  substantially (temporal PR-AUC 0.8657→0.5676, random 0.6823→0.3687) — mechanically
  expected, since enrollment was a dominant feature here too, just less overwhelmingly so
  for a linear model than for XGBoost. The **qualitative finding is unchanged**: temporal
  still scores higher than random, ROC-AUC still rules out a pure base-rate artifact, and
  the sponsor-history coefficient still points the "wrong" way for a leakage story.
- **Controlled ablation re-run** (XGBoost + LightGBM, same previously-logged best
  hyperparameters, no fresh Optuna sweep — logged as new MLflow runs under
  `controlled_ablation_{model}_best`):

  | Model | Honest PR-AUC | Leaky PR-AUC | Delta |
  |---|---|---|---|
  | LogReg | 0.4568 | 0.4558 | −0.0009 |
  | XGBoost | 0.5338 | 0.5472 | +0.0134 |
  | LightGBM | 0.5583 | 0.5672 | +0.0089 |

  Pre-M9 the tree-model deltas were ≤0.002 (noise-level on a ~0.80 PR-AUC base); post-M9
  they're 0.009–0.013, a larger share of a smaller base (~1.6–2.5% relative vs ~0.15%
  before). Recorded honestly rather than kept at the old figure — not strong evidence a
  temporal leak reappeared (LogReg's delta is still negative), but a weaker model has a
  wider noise band around a small effect, and this is close enough to the boundary that
  it's worth tracking on future retrains rather than assuming ≤0.002 is a permanent property
  of this feature set.
- **Why this matters beyond the specific fix:** it's a second, smaller instance of the same
  lesson M9-1 is about — a change made for one reason (dropping a leaked feature) had a
  side effect (breaking a notebook, and moving a second experiment's numbers) that only
  surfaced by actually trying to re-run everything downstream, not by reasoning about scope
  from the plan document alone.
- **Methodology note (added post-P0-review, per `docs/M9_P0_REVIEW.md`'s recommended
  follow-up):** "actually trying to re-run everything downstream," concretely, means
  `jupyter nbconvert --to notebook --execute` against notebooks 02–06 in place, not manual
  cell-by-cell execution or reasoning about which cells "should" be affected. That's the
  same mechanism that caught this KeyError — a full re-execute fails loudly on a stale
  hardcoded list; spot-checking individual cells would not have.

## Possible future revisit (flagged, not changed): `WITHDRAWN`/`SUSPENDED` in the positive class

The review (§1.1) and the M9 fix plan both raise this without asking for it to be changed:
"never started" (`WITHDRAWN`) and "started then stopped" (`TERMINATED`) are different
business events with different interventions, and the interaction with `enrollment_count`
was specifically bad — `WITHDRAWN` is definitionally zero-enrollment, which is exactly the
leak M9-1 removed. A `COMPLETED` vs `TERMINATED`-only label (dropping `WITHDRAWN`/
`SUSPENDED` from the positive class entirely) is a defensible alternative framing, immune to
this specific leak by construction, and was not implemented here per the plan's explicit
instruction: **label definition is a spec-level decision requiring approval, not an
implementation-forced change.** The current label (`TERMINATED`/`WITHDRAWN`/`SUSPENDED` all
positive) stays as-is. If revisited: re-derive the base rate (currently 19.9% derived vs
14.3% if `WITHDRAWN`/`SUSPENDED` were excluded), re-run the full M1–M9 pipeline, and treat it
as a new milestone, not a patch — the split dates, cost matrix, and every downstream number
in this repo are keyed to the current definition.

## Decision (review-driven, P0): `tests/conftest.py` scoping bug fixed; CI guard added, shaped as soft-fail after a real spec/reality conflict surfaced

- **What:** `_has_real_dev_state()` had two bugs stacked on each other, described in the
  updated docstring in full. (1) `_set_tracking_uri()` was imported but never called, so
  `MlflowClient()` read an ambient tracking URI instead of this repo's own `mlruns/`.
  (2) `has_raw_cache` was only assigned inside the `except MlflowException` branch — on a
  dev machine, where a Production model IS registered, the `try` succeeds and
  `has_raw_cache` is never assigned, so the final `return has_production_model and
  has_raw_cache` raised `UnboundLocalError`. That's the exact "ERROR on dev, SKIP in CI"
  split the review described: dev hits the `UnboundLocalError`, CI's fresh mlruns/ store
  raises `MlflowException` on the missing registered model, which the `except` branch
  *does* compute `has_raw_cache` for (False on a fresh checkout), so CI skips cleanly while
  dev errors. Fixed: call `_set_tracking_uri()` first; compute `has_raw_cache`
  unconditionally, outside the try/except. Verified: all three M7 tests (retrain trigger,
  version mismatch, rollback) now pass for real against this dev machine's actual
  registered model and raw cache — `3 passed`, not skipped or errored.
- **CI guard — plan conflict surfaced and resolved, not silently picked.** The plan
  specified an unconditional hard-fail if all M7 tests skip in CI. Checked against reality:
  CI's ephemeral Postgres only ever gets the `ml` schema (`make db-init`), never `marts`,
  and `mlruns/`/`data/` are both gitignored — so on CI as it exists today, all three tests
  WILL always skip, correctly, by the same deliberate scope decision M7's own
  `decisions.md` entry already documents (building a synthetic marts fixture + a way to
  bootstrap a fixture Production registration was explicitly deferred as beyond M7's ask).
  A literal hard-fail guard would make `retrain-trigger-test` permanently red starting
  immediately, for a known and already-approved limitation — a worse outcome than the
  silent-pass bug it was meant to fix. Flagged to the user rather than guessed at (per the
  standing rule against silently resolving spec/reality conflicts); user chose **soft-fail**:
  the guard step runs with `continue-on-error: true`, so an all-skipped run shows a visible
  yellow warning in the Actions UI on every run — impossible to mistake for "the M7 tests
  genuinely passed," which was the actual problem — without turning the pipeline red for a
  scope decision nobody is asking to revisit this session. If the marts fixture is ever
  built, this step passes silently like any other with zero further change needed.
- **Why (vs. alternatives):**
    - *Hard-fail exactly as specified:* rejected by the user after the conflict was
      surfaced — see above.
    - *Build the synthetic marts+registry fixture now, so the tests genuinely run in CI:*
      rejected — real scope growth (fake marts data satisfying every point-in-time join,
      plus bypassing `register_model.py`'s real-data reproducibility assertion) well beyond
      P0-2's 30-minute estimate and beyond what M7's own decisions.md already scoped out.
      Remains the "real" fix if this is ever prioritized.
    - *Silently implement the plan's literal guard without flagging the conflict:* rejected
      — CLAUDE.md's standing rule is to flag spec/reality conflicts explicitly, not
      quietly resolve them in either direction.
- **Failure mode:** the soft-fail guard depends on someone actually looking at the yellow
  warning; a hard failure is unmissable, a soft one is merely visible. That tradeoff was
  made deliberately in exchange for not permanently breaking the pipeline's headline
  pass/fail signal over a known limitation — worth revisiting if this project ever needs CI
  to be a hard release gate rather than a portfolio-demo signal.
- **Trigger to reconsider:** if the synthetic marts+registry fixture is ever built (making
  the skip genuinely avoidable), switch this step back to hard-fail — the soft-fail
  rationale evaporates the moment skipping stops being the CORRECT behavior on this runner.
- **Scaling story (10x/100x):** unaffected by data volume; this is entirely about CI
  environment provisioning, not model or dataset scale.
- **Interview question this maps to:** "How did you catch that a green CI job was actually
  testing nothing?" — the review found it; the fix makes the dev-machine path assert real
  behavior (verified: 3/3 pass) and makes the CI-skip path visible instead of invisible,
  after surfacing (not guessing past) a real conflict between the fix plan and this
  project's own already-documented scope decisions.

---

## M9-5: vocabulary find-and-replace (`prompt`/`brief`/`DoD` → `spec`/`requirements`/`acceptance criteria`)

Presentation fix, not a design decision — no full What/Why/Failure-mode format per the
plan. Repo-wide replacement of assignment-shaped vocabulary in code comments/docstrings
and this file with requirements-engineering vocabulary, per `M9_REVIEW_FIXES_PLAN.md`
§P1-5: `the M<N> prompt` → `the M<N> spec`, `the M<N> brief` → `the M<N> requirements`,
`M<N>'s DoD` → `M<N>'s acceptance criteria`, and the bare forms of each. Applied to
`domains/`, `core/`, `tests/`, and every prior M1–M8 entry in this file (`config.yaml`
comments included, since they're read alongside the code they configure). The content of
every flagged deviation is unchanged — only the noun describing what was deviated from.
`CLAUDE.md` and `/mnt/user-data/uploads/` were left untouched, as instructed (they
legitimately describe the prompt/brief-driven workflow this project was actually built
under). Verification: `grep -rn -i "the m[1-8].*brief\|the m[1-8].*prompt\|m[1-8].*dod\b"`
plus a broader case-insensitive `\b(prompt|brief|DoD)\b` scan both return zero hits across
`domains/`, `core/`, `tests/`, and this file.

---

## Decision (review-driven, P1): `temporal_split` raises on null-dated rows instead of silently routing them to TEST; `tests/test_dataset_builder_base.py` added

- **What:** `core/dataset_builder_base.py`'s `temporal_split` used
  `np.select(conditions, choices, default="test")`. Pandas comparisons against
  `NaT` are always `False`, so a null-dated row fails all three conditions and
  silently falls through to the `default` — previously `"test"`. Fixed:
  `temporal_split` now raises `ValueError` (naming the row count and the
  offending column) if `dates.isna().any()` before ever reaching `np.select`.
  Added `tests/test_dataset_builder_base.py` (5 tests, no DB/MLflow
  dependency, matching `test_calibration.py`'s fast-unit-test pattern):
  `temporal_split` respects date boundaries including both boundary edges;
  `temporal_split` raises on a `NaT` row instead of miscategorizing it;
  `random_split` is deterministic for a fixed `random_state` and differs
  across seeds; `controlled_leakage_ablation` actually enforces its
  fixed-window contract (`honest_train` strictly precedes
  `test_window_start`, `leaky_train` genuinely draws from both sides of the
  window given real post-cutoff data exists, and the two are equal-sized by
  construction).
- **Precondition checked before landing the raise:** queried the actual
  cached dataset (`data/raw_trials_cache.parquet`, n=78,115, the same cache
  `domains/pharma/dataset_builder.py.fetch_raw()` reads by default) for
  `start_date` nulls — **zero**. The raise is safe to land; the next
  `make build-dataset` will not break on today's cache. This guard exists for
  a *future* stale/partial cache, not a known problem in the current one.
- **Why (vs. alternatives):**
    - *Leave the silent `default="test"` and only add tests around the
      current (safe) behavior:* rejected — a test suite that documents a
      known-dangerous default without fixing it just formalizes the risk
      instead of removing it, and the review (§2.2/§2.10) specifically
      flagged this as a "subtle failure mode," not a hypothetical one.
    - *Route null-dated rows to a new `"invalid"` split instead of raising:*
      rejected — a fourth split value would need threading through every
      downstream consumer (`_one_hot_condition`, `write_to_db`,
      `feature_columns`) for a case that should never legitimately occur
      post-M1's `start_date_min`/`start_date_max` filtering; raising forces
      the caller to fix the input, which is the correct contract for a
      leakage-guard function.
- **Failure mode:** the guard only fires inside `temporal_split` itself — if
  a future caller pre-filters nulls with a *different* column name than what
  it later passes as `date_col`, the guard would not catch a nulls-elsewhere
  problem. Scoped to exactly what M1's own dataset-builder subclass calls
  (`date_col="start_date"`), not a general null-audit of the whole DataFrame.
- **Scaling story (10x/100x):** `dates.isna().any()` is a single vectorized
  pass over the date column — O(n), negligible next to the correlated
  self-joins that already dominate `fetch_raw()`'s runtime at any row count.
- **Interview question this maps to:** "Your leakage guard was tested. Show
  me the test that would catch someone weakening it." — before this fix, a
  stale or partially-filtered cache could silently leak null-dated rows into
  TEST with no error and no test to catch it; `test_temporal_split_raises_on_null_dates`
  is that test, and it fails loudly against the pre-fix behavior.

---

## Decision (review-driven, P1): `decisions.md` split into a ~150-line summary + `docs/decision-log/` (one file per milestone)

- **What:** The single-file `decisions.md` had grown to 1,801 lines / 140KB across M1–M8
  (2,201 lines including M9's P0 entries) — long past the point anyone reads it end-to-end.
  Split into `docs/decision-log/M1-dataset-builder.md` through
  `docs/decision-log/M9-review-fixes.md` (one file per milestone, full What/Why/Failure-
  mode/Scaling/Interview-question detail preserved verbatim) plus a new root `decisions.md`
  (75 lines): an 8-point "most load-bearing decisions" executive summary, each linking to
  its milestone's full file, and a full per-milestone table of contents.
- **How the split was done, and how completeness was verified:** milestone boundaries were
  found via `grep -n "^## M[0-9]"` against the old file, each range extracted with `sed -n
  '<start>,<end>p'`, and a `[← back to decisions.md summary]` breadcrumb prepended. Verified
  no content loss by accounting for every line: old file 2,201 lines − 2 title lines − 8×2
  lines for the `---`/blank separators between milestones (16 lines) = 2,183 lines, which
  matches the sum of content lines actually written across the 9 new files exactly. Every
  entry that existed in the old file exists, unedited, in exactly one new file.
- **Why (vs. alternatives):**
    - *Keep one file, add a table of contents at the top:* rejected — a ToC makes the first
      screen navigable but does nothing for the underlying problem (nobody reads a
      140KB file linearly, ToC or not), and this project's own README already links to
      specific entries by milestone, showing the "split by milestone" instinct was already
      the right one.
    - *Split by decision *category* (leakage, calibration, infra, ...) instead of by
      milestone:* rejected — categories would cut across a milestone's own narrative order
      (e.g. M5's FrozenEstimator fix only makes sense next to the MAPIE fix it's paired
      with), and this project's own spec ties Definition-of-Done criteria to milestones, so
      milestone-aligned files match how a reader would actually look something up ("what
      happened in M7?").
    - *Auto-generate the summary from the full files (e.g. extract each "What:" line):*
      rejected — a mechanically extracted summary would include every decision, defeating
      the point of a *curated* 8-entry "read these first" list; the summary's value is in
      the editorial choice of which 8 decisions actually shaped the system, not completeness.
- **Failure mode:** the summary can drift from the full files if a future milestone's
  headline decision isn't added to the "most load-bearing" list — nothing enforces that sync
  automatically. Mitigated the same way `decisions.md`/spec-Section-12 sync already is
  elsewhere in this project: a standing rule to update both after every milestone, not a
  technical guarantee.
- **Scaling story (10x/100x):** unaffected by row/model scale — this is a documentation-
  volume problem. At 10x more milestones (a longer-running project), the same
  one-file-per-milestone pattern scales linearly in file count, not in any single file's
  size, which is the actual property being optimized for.
- **Interview question this maps to:** "Walk me through the most important decisions in
  this project." — before this fix, the honest answer required scrolling past irrelevant
  milestones to find the interesting parts; after it, the answer is the 8-entry list at the
  top of `decisions.md`, each with one click to full detail.

---

## Decision (review-driven, P1): `sponsor_prior_termination_rate`'s point-in-time bug fixed with a data-driven hybrid, not the plan's literal Option A/B

- **What:** `domains/pharma/dataset_builder.py`'s `sponsor_history` CTE computed
  `sponsor_prior_termination_rate` by checking each historical trial's **current-day**
  `overall_status` (`TERMINATED`/`WITHDRAWN`/`SUSPENDED`), with no gate on whether that
  status had actually been reached before the querying trial's own `start_date`. A sponsor's
  prior trial that was still `RECRUITING` in, say, 2015 but terminated in 2021 was counted as
  a termination when computing a *2015* trial's feature — using knowledge that didn't exist
  at 2015's `start_date`. Same root-cause shape as M9-1's `enrollment_count` leak (a
  point-in-time feature silently using present-day information), on a different column.
  - **Diagnostic investigation first, per the plan's own instruction:** ran the plan's exact
    diagnostic SQL queries to check whether a real status-history table exists anywhere in
    PharmaPulse (mart, staging, or raw JSONB) that would let the fix reconstruct "what was
    this trial's status as of date X" directly. **None exists** — `marts.fct_trials` only
    ever stores the current/latest status, full stop.
  - **Plan's literal Option A/B did not fit reality, so a third, data-driven option was
    built instead — flagged, not silently substituted:** Option A (a real status-history
    join) is impossible without the missing table. Option B (the plan's fallback:
    `completion_date < t.start_date`, unfiltered) would have introduced a *different* bias,
    discovered the same way M9-1's `enrollment_type` distinction was discovered — checking
    `raw.ct_studies`' own JSONB payload. `marts.fct_trials.completion_date` is populated
    even for still-open trials with a future **ESTIMATED** target completion date, and is
    ESTIMATED (not ACTUAL) for **46% of WITHDRAWN** and **92% of SUSPENDED** trials —
    exactly the two statuses this feature cares about most. Using Option B literally would
    have "fixed" the hindsight leak by introducing a fabricated-date leak in its place (an
    ESTIMATED future date sliding in under `t.start_date` by coincidence, not because the
    trial had actually resolved).
  - **Fix actually shipped:** a new `hist_resolution` CTE trusts `completion_date` only when
    `raw.ct_studies.payload -> 'protocolSection' -> 'statusModule' -> 'completionDateStruct'
    ->> 'type' = 'ACTUAL'` (mirrors M9-1's `enrollmentInfo.type` pattern exactly).
    `sponsor_history`'s `AVG(CASE ...)` now requires all three: the historical trial started
    before `t.start_date` (unchanged), its resolution date is ACTUAL (not ESTIMATED), and
    that resolution date is itself `< t.start_date`. A prior trial with no ACTUAL resolution
    by then contributes 0, not a hindsight-derived 1.
  - **Validated on a subset before applying at full scale, same discipline as M9-1's
    ablation:** measured on the 2023+ subset (n=3,169) before touching the full 78k-row
    build — average `sponsor_prior_termination_rate` dropped **0.108 → 0.067** (a −38%
    relative change), and **58% of rows** had their value change at all. This confirmed the
    fix has real, material effect (not a no-op) before committing to a full rebuild + retrain.
  - **Full-scale results:** `data/raw_trials_cache.parquet` rebuilt to 78,260 rows (was
    78,115 — a `WHERE type = 'ACTUAL'` join changes which historical rows *count*, not the
    querying trial's own eligibility, so the small row-count change came from cache
    invalidation/refresh timing, not the fix itself). Splits: train=66,129 / calib=6,342 /
    test=5,789. XGBoost champion retrained (same hyperparameters as M9-1's refit — again a
    single retrain, not a fresh Optuna sweep, so this delta isolates the feature-value change):
    `pr_auc_temporal=0.619294` (uncalibrated) / `0.597534` (isotonic-calibrated),
    `roc_auc_temporal=0.766199`, `ece_test=0.031024`.
  - **Threshold recalculated, and its stability classification changed:** M9-4's CALIB-vs-TEST
    threshold-selection instability check was re-run against the new probabilities — the gap
    narrowed from M9-4's 0.08 (classified "unstable," triggering the bootstrap-CI contingency)
    to **0.05** (now "stable" by the plan's own margin). CALIB-selected threshold moved
    0.14 → **0.16**; bootstrap 95% CI (1,000 resamples of the CALIB selection) is
    **[0.13, 0.21]**, and the TEST-selected threshold (0.21) now falls *inside* that CI — a
    materially different, more reassuring picture than M9-4's TEST-selected value sitting at
    the CI's edge. Regret vs. the (unattainable) TEST-optimal choice also improved slightly,
    3.7% → **2.8%**.
  - **Registered as MLflow registry version 45, not the plan's anticipated "v17":** the
    fix-plan document speculated M9-11 would land as v17 (one increment past M9-1's v16).
    In practice, `register_model.py` was re-run three times during this fix (once to reveal
    the new PR-AUC via its own tripwire assert, safely failing pre-registration; once after
    updating `EXPECTED_PR_AUC_TEMPORAL` to register for real, creating v44; once more after
    catching that `THRESHOLD` was still 0.14 while `config.yaml` had already moved to 0.16,
    creating v45) — plus accumulated dev-session registrations from earlier in this fix
    session. v45 is the real, current Production version; v44 was archived automatically by
    `register_model.py`'s own archive-previous-Production logic. Reported the real number
    rather than forcing the plan's anticipated one, per CLAUDE.md's "flag conflicts
    explicitly" rule.
  - **Downstream propagation, all re-executed/republished rather than hand-edited:**
    `ml.training_dataset` (live Postgres) was found to have drifted 7 rows out of sync with
    the rebuilt cache mid-session (a real, not-fully-root-caused inconsistency — resolved
    pragmatically via `TRUNCATE` + full rebuild from the deterministic cache, restoring the
    exact expected split counts, rather than chasing the write path further); notebooks
    02–06 all re-executed via `nbconvert` (never hand-edited outputs); notebook 04's SHAP
    consistency check confirmed the DB-sourced JSONB-parsed feature matrix reproduces the
    retrained pipeline's `pr_auc_temporal` to 6dp; notebook 05 re-fit the conformal wrapper
    against the new Production run (`aa8ec5b5710b490d9b0963b15476bad0`, v45) and logged a
    fresh `conformal` artifact onto it, empirical coverage now **0.931** (target 0.90,
    still comfortably passing); `config.yaml`'s `model.threshold_decision` block had its
    prose comments updated for M9-11 but three of its actual numeric fields
    (`test_would_have_selected`, `regret_vs_test_optimal_pct`,
    `expected_cost_at_threshold`, `recall_at_threshold`, `fp_rate_at_threshold`,
    `precision_at_threshold`, `f1_at_threshold`) were left at stale M9-4-era values in an
    earlier pass of this fix — caught and corrected against `notebooks/03_calibration.ipynb`'s
    actual re-executed output before this fix was considered done, since this is the exact
    block `domains/pharma/serving/api.py` reads at load time, not just documentation;
    notebook 06's drift comparison picked up the resynced
    `ml.training_dataset` and surfaced a genuine, expected consequence of the fix:
    `sponsor_prior_termination_rate`'s drift score jumped from 0.126 (outside the top-5) to
    0.416 (now the 4th most-drifted feature) — exactly what a newly point-in-time-honest
    feature should show when compared across a pre-2020 vs. 2022+ population, not a
    regression.
- **Why (vs. alternatives):**
    - *Ship the plan's literal Option B (unfiltered `completion_date < t.start_date`)
      anyway, since it's simpler and the plan already named it:* rejected — the 46%/92%
      ESTIMATED-date contamination on exactly the statuses this feature targets would have
      replaced one leak with a smaller but real one, discoverable by the same JSONB-payload
      check that caught M9-1's leak. Shipping a fix that's "less leaky" but still leaky
      would not have survived a follow-up review.
    - *Build a real status-history table from `raw.ct_studies`' historical payload
      snapshots (if any exist) instead of a same-payload type filter:* not pursued — the
      diagnostic query confirmed no such historical/versioned snapshot exists in this
      dataset; only the current payload's own `type` field was available, which is what
      the shipped fix actually uses.
    - *Skip the subset validation and go straight to the full rebuild:* rejected — mirrors
      M9-1's own review-praised discipline of measuring an effect on a smaller sample before
      committing to a full retrain; the 2023+ subset check is what confirmed this was a
      real, material fix (58% of rows changed) rather than a cosmetic no-op before spending
      the time on a full rebuild + retrain + re-registration + six notebook re-executions.
- **Failure mode:** if a future ClinicalTrials.gov payload refresh ever back-fills an
  ESTIMATED-typed `completionDateStruct` to ACTUAL retroactively (a real possibility — CT.gov
  updates trial records after the fact), `hist_resolution` would silently start counting a
  trial as resolved earlier than the querying trial's `start_date` "always was," reintroducing
  a smaller version of the same hindsight leak this fix removed. Nothing in the current
  pipeline detects a `type` flip on a historical record between dataset builds.
- **Trigger to reconsider:** if that ACTUAL/ESTIMATED backfill behavior is ever observed in
  practice (e.g., a diagnostic query showing a `nct_id` whose `type` changed across two raw
  data pulls), or if `raw.ct_studies` ever gains a genuine append-only status-history table
  (making Option A finally buildable), revisit this CTE.
- **Scaling story (10x/100x):** the `hist_resolution` CTE adds one more JOIN over
  `raw.ct_studies`' JSONB payload per historical trial — same asymptotic cost class as
  M9-1's `enrollment_type` check, since both are single-column JSONB path extractions keyed
  on `nct_id`. At 10x/100x more trials, this scales the same way the rest of `sponsor_history`
  already does (an indexed join, not a per-row Python loop) — no new bottleneck introduced by
  this fix specifically.
- **Interview question this maps to:** "Tell me about a time a 'fix' you almost shipped would
  have introduced a new bug." — the answer is the Option B rejection above: the plan's own
  proposed fallback was checked against the raw payload before shipping and found to
  reintroduce leakage on the two statuses (WITHDRAWN, SUSPENDED) the feature cares about
  most, which is exactly the kind of thing a diagnostic check (not blind trust in a plan
  document) is supposed to catch.

---

## Decision (review-driven, P1): CI quality gates — ruff, mypy on `core/`, coverage floor

- **What:** Added `pyproject.toml`'s `[tool.ruff]`/`[tool.ruff.lint]`/`[tool.mypy]` sections
  and a new `lint-and-type-check` CI job (`ruff check .` → `ruff format --check .` →
  `uv run mypy core/`), gating `test-unit` on it via `needs:`. `test-unit`'s pytest
  invocation gained `--cov=core --cov=domains --cov-report=term-missing --cov-fail-under=25`.
  `ruff`, `mypy`, and `pandas-stubs` added via `uv add --dev` (never `pip install`, per
  CLAUDE.md). Fixed everything both tools flagged in the actually-checked scope (`ruff
  check .` repo-wide; `mypy` on `core/` only, per the plan) rather than suppressing:
  - **ruff, 34 initial findings → 0:** 21 auto-fixed (`ruff check --fix` + `ruff format`,
    14 files reformatted — this codebase had never been run through the formatter).
    3 fixed by hand: `core/conformal.py`'s bare `zip(proba, sets)` got `strict=True` (both
    arrays are derived from the same input rows by construction — a length mismatch would
    be a real bug worth crashing on, not silently truncating);
    `domains/pharma/register_model.py`'s `dict(...)` call rewritten as a literal;
    `tests/test_retrain_trigger.py`'s nested `if` collapsed into one `and`-chained condition.
  - **ruff config deviates from the plan's literal `ignore = ["E501"]`:** also ignores
    `N803`/`N806` (sklearn-style `X`/`X_train` naming). `core/conformal.py`, `core/explain.py`,
    and `core/training_pipeline.py` all use this convention throughout — enforcing snake_case
    here would fight the one naming pattern every reader of this codebase (or of scikit-learn
    itself) already recognizes, for zero clarity gain. Flagged explicitly per CLAUDE.md's
    "say so if a plan-vs-repo conflict needs resolving" rule, not silently changed.
  - **mypy, 45 initial findings → 0 in `core/`:** ~17 were `import-untyped` noise (sklearn/
    evidently/shap/mapie ship no stubs or `py.typed` marker in the installed versions) —
    resolved via `[[tool.mypy.overrides]]` `ignore_missing_imports` for those four modules
    specifically, not a global suppression. The remaining 28 were real gaps in this
    project's own code: missing parameter/return annotations across
    `threshold_selector.py`/`explain.py`/`conformal.py`/`calibration.py`/
    `dataset_builder_base.py` (added `ArrayLike`/`Any`/`dict[str, Any]` as appropriate);
    `min()`/`max()` called with `.get` as the sort key (returns `Optional`, doesn't satisfy
    the ordering-key protocol — switched to a lambda indexing the dict directly);
    `core/conformal.py`'s `self.mapie_` narrowed to `None` even right after assignment
    inside `fit_conformal()` (mypy doesn't flow-narrow instance attributes the way it does
    locals) — fixed by assigning to a local `mapie` variable, calling `.conformalize()` on
    that, and only then assigning the fully-constructed object to `self.mapie_`, which is
    also cleaner than the original two-step assign-then-mutate pattern;
    `core/training_pipeline.py`'s `SplitData.dates_train: pd.Series | None` used directly
    inside `run_family()` without narrowing — added an explicit `assert ... is not None`
    with a message explaining *why* it's always populated for the temporal regime this
    method is called with, rather than silencing the check;
    `core/calibration.py`'s `predict()` used `self.isotonic_`/`self.platt_` (both
    `Optional`) after only checking `self.result_ is None` — same root cause as the
    `mapie_` case, fixed with explicit asserts documenting that the three attributes are
    always set together inside `fit()`.
  - **Coverage floor set to 25%, not a "typical" 70-80%:** the plan's own P1-12 section
    doesn't specify a percentage (only its ruff/mypy snippet is concrete); measured
    coverage on the CI-runnable fast-test suite is ~31-44% depending on whether the
    DB/MLflow-dependent M7 integration tests happen to run (they do on a dev machine with
    real state, skip in CI per the M9-2 entry above). Flagged to the user rather than
    picking an arbitrary "sounds right" number for a document meant to be precise — user
    chose 25%, a non-regression tripwire below the current baseline, not a target. Several
    `core/` modules (`conformal.py`, `explain.py`, `threshold_selector.py`,
    `serving/api_base.py`) sit at 0% in the fast suite specifically, since they're
    currently exercised only through the integration tests CI skips — raising the floor to
    something a reviewer would expect (60-80%) requires writing new fast unit tests for
    those modules, out of scope for this fix's 30-minute estimate.
- **Why (vs. alternatives):**
    - *Global `ignore_missing_imports = true` for mypy, not per-module:* rejected — that
      would also silence real import errors in this project's own future modules, not just
      the four known-unstubbed third-party libraries.
    - *`# type: ignore` comments instead of fixing the underlying narrowing gaps:* rejected
      for the `mapie_`/`isotonic_`/`platt_` cases specifically — the local-variable
      reordering and explicit asserts are both a real narrowing fix AND better runtime
      documentation of an invariant the code already relied on implicitly.
- **Failure mode:** a 25% coverage floor won't catch most real regressions — it only fires
  if coverage drops by double digits, which usually means an entire test file stopped
  running (e.g. a collection error), not that one function lost its test. This is
  explicit in the floor's own justification above, not hidden as if it were a real
  quality bar.
- **Trigger to reconsider:** the coverage floor should rise the next time someone adds
  fast unit tests for `core/conformal.py`/`explain.py`/`threshold_selector.py` — bump it to
  just below whatever the new measured number is, same non-regression-tripwire logic.
- **Scaling story (10x/100x):** ruff/mypy runtime scales with file count, not data/model
  scale — irrelevant at 10x/100x row counts. The coverage floor's main scaling axis is
  *test suite* growth: as more fast unit tests land, the floor should ratchet up to track
  them, per the trigger above.
- **Interview question this maps to:** "How do you enforce code quality in CI?" — the
  answer is now visible in `ci.yml`'s `lint-and-type-check` job, and "why is your coverage
  floor only 25%?" has an honest answer instead of a made-up-sounding-authoritative one.

---

## Decision (review-driven, P1): imputation medians frozen at training time and loaded at serving, not recomputed from a single-row request

- **What:** `domains/pharma/dataset_builder.py`'s `build_features()` fills
  `num_primary_outcomes`, `num_sites`, and `sponsor_prior_termination_rate` NaNs
  with `.median()` computed over the fetched population, before the temporal
  split — this is TRAIN's own imputation, unchanged by this fix. The gap this
  fix closes is downstream, at serving: `domains/pharma/serving/api.py`'s
  `_row_from_trial_features()` had exactly one live "missing value" code path
  (`sponsor_prior_termination_rate=None` on the request), and it was hardcoded
  to `0.0` — not the training-time median — a documented limitation in the
  code's own docstring since M5 ("deferred past M5's acceptance criteria").
  Added `PharmaDatasetBuilder.compute_imputation_constants(raw)`, which
  evaluates the exact same three `.median()` expressions `build_features()`
  already uses, as a standalone, loggable dict. `register_model.py` now logs
  this dict as an `imputation_constants.json` MLflow artifact alongside the
  existing `condition_vocab.json`/`feature_schema.json` on every future
  registration run. `api.py`'s `_load_bundle()` loads it and stores it on
  `_PharmaModelBundle`; `_row_from_trial_features()` now fills the missing
  `sponsor_prior_termination_rate` from that frozen constant instead of `0.0`.
- **v45 backfilled without a retrain, per this fix's scope:** the constants
  were computed from `data/raw_trials_cache.parquet` (the same cache M9-11's
  retrain used to produce v45) and attached directly to v45's existing run
  (`aa8ec5b5710b490d9b0963b15476bad0`) via `MlflowClient().log_dict(run_id=...)`
  — no new run, no new registry version. Computed values:
  `num_primary_outcomes=1.0`, `num_sites=2.0`,
  `sponsor_prior_termination_rate≈0.01020408`.
- **Scope decision, flagged rather than silently expanded:** the fix plan's
  example JSON lists all three constants, but only `sponsor_prior_termination_rate`
  has an `Optional`/missing-value path in `TrialFeatures` today —
  `num_primary_outcomes`/`num_sites` are required (non-nullable) request
  fields, so there is no live code path that would ever consume frozen
  constants for them. All three are still computed and logged (parity with
  the plan, and so the artifact doesn't need a second revision if those
  fields are ever made optional), but only the one field with a genuine
  missing-value path was wired into the request-handling code — making the
  other two required fields optional as well was judged out of scope for a
  bug fix about *stale* imputation, not a request to widen the schema.
- **Test:** `tests/test_imputation_constants.py` (4 tests, no DB/MLflow
  dependency): `compute_imputation_constants` reproduces `.median()` exactly
  on a fixture with NaNs; a request with the field omitted uses the bundle's
  frozen constant, not a hardcoded value; the constant a fixture's
  `compute_imputation_constants` call produces is the *exact* value
  `_row_from_trial_features` falls back to when that same dict is loaded onto
  the bundle (the literal fixture-equality check the plan's DoD asked for); a
  request that *does* supply the field is unaffected.
- **Why (vs. alternatives):**
    - *Recompute a "median" from the live request at serving:* rejected —
      this is the bug being fixed; a median of one row is not a median, and
      the pre-fix `0.0` fallback was an arbitrary constant with no connection
      to the training distribution at all.
    - *Refactor build_features()'s three fillna() calls to call the new
      `compute_imputation_constants()` internally (single source of truth):*
      not done — build_features() operates on `raw.copy()` inside the same
      method scope those medians are computed from, so introducing a call to
      a separate method there would need to thread a computed-once dict
      through in a way that changes build_features()'s existing structure for
      no behavior change; duplicating the three expressions (flagged
      explicitly in `compute_imputation_constants`'s own docstring, with a
      pointer to the test that would catch drift) was judged the smaller,
      more auditable diff.
    - *Widen `TrialFeatures` to make `num_primary_outcomes`/`num_sites`
      optional too, to fully exercise all three logged constants:* rejected
      per CLAUDE.md's scope-creep rule — nothing about this bug required
      loosening those fields' required-ness; that would be a request-schema
      change made because the tooling was already open, not because the fix
      needed it.
- **Failure mode:** `compute_imputation_constants` and `build_features()`
  read the same three `.median()` expressions from two separate places in the
  same file. If a future change to `build_features()`'s imputation logic
  (e.g., switching `num_sites` to a different strategy) isn't mirrored in
  `compute_imputation_constants`, the served fallback silently stops matching
  what the model actually saw at training time, and nothing short of
  `tests/test_imputation_constants.py`'s fixture-equality test (or noticing
  the drift by inspection) would catch it — there is no automated coupling
  between the two beyond that test.
- **Scaling story (10x/100x):** the constants are three scalars computed once
  per training run — irrelevant to row/model scale. The one axis this does
  scale with is *feature count*: if more numeric features gain a
  missing-value path in `TrialFeatures` in the future, each needs its own
  entry in `compute_imputation_constants` and its own frozen-constant fallback
  in `_row_from_trial_features` — a linear, not compounding, cost per field.
- **Interview question this maps to:** "What happens if a request comes in
  with a missing feature value?" — before this fix, the honest answer was
  "an arbitrary hardcoded 0.0 with no connection to training"; after it, "the
  exact median the model was trained against, frozen as an MLflow artifact
  and loaded at serving startup, verified by a test that ties the two
  together."

---

## Decision (review-driven, P1): `ml.prediction_log` added, `/predict` writes to it as a background task, `drift_job.py` gains a real `--source=prediction_log` mode with an honestly-documented schema gap

- **What:** Drift monitoring (M6) has only ever compared TRAIN vs TEST
  (`ml.training_dataset`), a proxy for "training population vs. a batch the
  model hasn't seen," documented honestly as such since M6 because no live
  scoring traffic has ever existed. This fix makes a real production
  monitoring path possible: added `ml.prediction_log` to `schema.sql` (exact
  columns per the M9 fix plan: `request_id`, `nct_id`, `proba`,
  `threshold_decision`, `feature_pipeline_version`, `model_version`,
  `features_hash`, `conformal_low`/`conformal_high`, `top_shap_feature`,
  `latency_ms`, `created_at`, plus the two indexes the plan specified).
  `domains/pharma/serving/api.py` gained: a `request_id` middleware (uuid4
  per request, also returned as an `X-Request-ID` response header);
  `_features_hash()` (sha256 of the exact scored feature row, for dedup/audit
  without persisting the full row); `_log_prediction_background()`, wired
  into both `/api/v1/predict` and `/api/v1/predict/nct/{nct_id}` via FastAPI
  `BackgroundTasks` so the write happens strictly after the response is
  already on the wire — it cannot add latency to what a caller experiences.
  Any exception inside the write (DB down, schema not migrated, etc.) is
  caught and printed to stdout, never re-raised: the API's fail-loud contract
  (`_require_bundle()` returning 503 when the model isn't loaded) is
  deliberately NOT extended to this path — a client must always get their
  prediction even if the log DB is unreachable. `drift_job.py` gained a
  `--source={training,prediction_log}` / `--lookback=N` CLI (`argparse`),
  threaded into `PharmaDriftMonitor.__init__`/`load_current()`; `make drift`
  now accepts `SOURCE=`/`LOOKBACK=` (`make drift SOURCE=prediction_log
  LOOKBACK=14`), defaulting to the unchanged pre-existing `training` behavior
  so `retrain_trigger.py`'s and `tests/test_retrain_trigger.py`'s existing
  `PharmaDriftMonitor()` call sites (no args) are unaffected.
- **Schema gap surfaced and handled explicitly, not silently glossed over:**
  the plan's own `ml.prediction_log` schema (verbatim, landed as specified)
  logs `proba`/`threshold_decision`/`features_hash`/etc. but does **not**
  persist the full engineered feature vector a request was scored on — only
  a hash of it. This means a feature-level Evidently `DataDriftPreset`
  comparison (the kind `training` mode runs) has **zero columns in common**
  between `ml.prediction_log` and the TRAIN reference, regardless of how many
  predictions ever get logged — a real limitation of the specified schema,
  not a bug in this fix's implementation of it. `PharmaDriftMonitor.run()`
  detects the empty-or-non-comparable case explicitly (`current.empty or not
  common_cols`) and logs an honest zero-comparison verdict
  (`drifted=False, n_features_drifted=0, drift_share=0.0`) rather than either
  crashing on a degenerate Evidently call over disjoint columns or fabricating
  a "not drifted" result over data that was never actually compared. Verified
  live against the real dev Postgres (not mocked): `make drift
  SOURCE=prediction_log LOOKBACK=7` printed `(0, 13)` current rows and the
  exact honest message before any traffic existed; a single real `/predict`
  call via `fastapi.testclient.TestClient` was confirmed to write a row with
  the correct `request_id` (matching the response's `X-Request-ID` header),
  `model_version=45`, and a populated `features_hash`/`top_shap_feature`; then
  re-running the same drift check showed `(1, 13)` current rows and **still**
  `0` shared columns — confirming the gap is about column overlap, not row
  count. The test row was deleted afterward (`DELETE FROM ml.prediction_log
  WHERE request_id = ...`) so this manual verification doesn't masquerade as
  real served traffic in the table going forward, matching M7's own precedent
  of cleaning up synthetic test artifacts from live state.
- **Why (vs. alternatives):**
    - *Extend `ml.prediction_log`'s schema to also store the feature vector
      (as JSONB, mirroring `ml.training_dataset.features`), closing the gap
      immediately instead of just documenting it:* not done — the plan's DoD
      specifies this exact schema verbatim and asks for the gap to be
      documented, not silently expanded; a schema change beyond what was
      specified is exactly the kind of unauthorized scope growth CLAUDE.md's
      standing rule asks to flag before making. Documented as the real next
      step instead.
    - *Make the write synchronous (block the response until the log insert
      completes):* rejected — the plan explicitly asks for async/background,
      and a synchronous write would mean a slow or down log DB directly
      degrades `/predict`'s latency or availability, the opposite of the
      "API stays up if the log DB is down" requirement.
    - *Let a prediction_log write failure propagate as a 5xx:* rejected for
      the same reason — logging is best-effort by design; serving is not.
    - *Skip the empty-batch guard and let Evidently run on a 0-shared-column
      comparison:* not attempted directly, but reasoned through — Evidently's
      `DataDriftPreset` over an empty common-column DataFrame either raises
      or produces a metrics payload `check_thresholds()` can't parse
      (`StopIteration` on the `DriftedColumnsCount` lookup, per
      `core/monitoring/drift_base.py`'s own documented failure mode) — an
      unhelpful crash in place of an honest, informative empty-result message.
- **Failure mode:** the empty-or-non-comparable guard is keyed on `current.empty
  or not common_cols` — if `ml.prediction_log` is ever extended to log even
  one column that happens to share a name with a reference feature (unlikely
  given the current schema, but not structurally prevented), `common_cols`
  would become non-empty and Evidently would run a comparison over that one
  column alone, silently producing a `drift_share` computed over a
  near-meaningless single-column intersection rather than the honest
  zero-comparison message — worth re-checking if this table's schema ever
  changes.
- **Scaling story (10x/100x):** the background-task write is O(1) per
  request, off the response's critical path — irrelevant to request-rate
  scale in the sense that it never slows a response down, though at very high
  QPS the write itself would need to move off FastAPI's in-process
  `BackgroundTasks` (which run in the same event loop / worker process) to a
  real queue (Celery/RQ, or a managed log sink) to avoid competing with
  request-handling threads for CPU — noted as the real 10x/100x answer, not
  implemented here since current traffic is zero.
- **Interview question this maps to:** "You said drift monitoring works —
  walk me through what actually gets compared to what." Before this fix, the
  honest answer was "TRAIN vs TEST, because there's no traffic, and there's
  no mechanism to ever log real traffic even if there were." After it: "TRAIN
  vs TEST as a dev proxy, or TRAIN vs the last N days of `ml.prediction_log`
  once real traffic exists — and today, if you actually run that second mode,
  it tells you honestly that it has nothing to compare yet, which is the
  correct thing for it to say."

---

## Decision (review-driven, P1, LOCKED-CONTRACT CHANGE): `conformal_interval` renamed to `uncertainty_band`; `coverage_guarantee` field added

- **What:** `PredictionResponse.conformal_interval` (a `[low, high]` band
  derived from MAPIE's prediction *set* over labels, converted heuristically
  — see `core/conformal.py`'s `predict_with_interval` docstring) renamed to
  `uncertainty_band` in `core/serving/api_base.py`. Added a new
  `CoverageGuarantee` submodel and `coverage_guarantee` field:
  `{type: "label_set", target: <float>, empirical: <float>, note: str}`.
  `domains/pharma/serving/api.py`'s `_predict_from_row()` populates `target`
  from `b.conformal_wrapper.target_coverage` (the loaded MAPIE wrapper's own
  attribute — the actual value `predict_with_interval()`'s margin is derived
  from, not a separately-hardcoded copy) and `empirical` from the Production
  run's logged `empirical_coverage` MLflow metric (0.9310761789600968 as of
  the M9-11 retrain — loaded once in `_load_bundle()`, not recomputed or
  hardcoded, so it stays correct across future retrains without a code
  change). `tests/test_api_contract.py`'s field-name assertions updated
  (`uncertainty_band`/`coverage_guarantee` in place of `conformal_interval`,
  plus new assertions on `coverage_guarantee`'s sub-fields). `ml.prediction_log`'s
  `conformal_low`/`conformal_high` DB columns (M9-7) are UNCHANGED — those are
  internal storage column names, not part of the locked HTTP contract, so
  this rename does not touch them; `api.py`'s write path now reads from
  `response.uncertainty_band` instead.
- **Locked-contract change, coordinated with RegIntel in the same session,
  per CLAUDE.md's standing rule:** `01_REGINTEL_SPEC.md`'s Section 2 contract-
  verification note and its M5 milestone DoD row (the two places that spec
  states TrialOutcome's locked response shape verbatim) both updated to the
  new field set, each with an explicit "Updated in M9-9 — see TrialOutcome
  Section 12 for the rationale" pointer, per the fix plan's own instruction.
  `02_TRIALOUTCOME_SPEC.md` Section 6's contract table and Section 12's
  "Unresolved / explicitly deferred" and "Effect on cross-project contracts"
  subsections all updated in the same pass so no file is left describing the
  old shape as current.
- **Verified end-to-end against the real Production model before landing
  (not mocked):** `fastapi.testclient.TestClient` against the actual running
  app (model v45) returned `coverage_guarantee: {type: "label_set", target:
  0.9, empirical: 0.9310761789600968, note: "..."}` and `uncertainty_band:
  [0.0, 0.0225]` for a real scored request — confirming both the rename and
  the dynamically-loaded (not hardcoded) `target`/`empirical` values work
  together correctly.
- **Why (vs. alternatives):**
    - *Option B (rebuild as a true conformalized probability interval via
      `MapieRegressor` on `predict_proba`), per the fix plan's own more
      expensive alternative:* not pursued this session — the plan itself
      recommends Option A (rename + document) given the locked-contract
      implications, and reserves Option B as a future revisit; doing both in
      one session would conflate a naming/documentation fix with a modeling
      methodology change, two decisions that deserve to be evaluated
      independently.
    - *Make `coverage_guarantee` a plain `dict` field instead of a typed
      `CoverageGuarantee` submodel:* rejected — `SHAPContributor` already
      establishes the pattern of typed submodels for structured sub-objects
      in this response; a bare dict would be the only untyped field in an
      otherwise fully-typed contract.
    - *Hardcode `target`/`empirical` as the plan's example JSON literals
      (0.90/0.946):* rejected — 0.946 is a pre-M9 number (the leaked-feature
      model's coverage); hardcoding either value would silently go stale on
      the next retrain exactly the way M9-8's imputation-constant fix was
      about *not* doing that elsewhere in this same API. Both are read live
      from the loaded conformal wrapper and the run's own logged metric.
- **Failure mode:** if a future retrain's conformal-fitting step fails to log
  `empirical_coverage` (e.g., `register_model.py`/notebook 05's logging call
  is skipped or edited out), `_load_bundle()`'s `.get("empirical_coverage",
  float("nan"))` fallback means `coverage_guarantee.empirical` would silently
  serve `NaN` rather than raising — a client would get a response with a
  non-numeric `empirical` field instead of a loud startup failure. Mirrors
  the existing (pre-M9-9) fallback behavior for `pr_auc`/`ece` in the same
  bundle, so this isn't a new failure mode introduced by this fix, but it's
  worth naming since `coverage_guarantee` is new.
- **Scaling story (10x/100x):** irrelevant to row/model scale — this is a
  field-naming and documentation fix. The one thing that *would* need to
  change at scale is Option B's eventual adoption (a true conformalized
  probability interval): MAPIE's regression-style conformal has its own
  computational profile independent of this rename, and swapping it in would
  be a `core/conformal.py`-internal change that this rename's contract shape
  (`uncertainty_band` as a `tuple[float, float]`) already accommodates
  without a further schema change.
- **Interview question this maps to:** "What does `conformal_interval`
  guarantee?" — before this fix, the honest answer required a caveat the
  field name didn't hint at ("nothing on the probability itself, actually,
  despite the name"). After it: "`uncertainty_band`'s `coverage_guarantee`
  field says so directly — label-set membership, 90% target, 93.1% empirical
  on TEST, not a promise about the probability band."

---

# P2 — Polish

## Decision (review-driven, P2): serving/dev dependency split; plan's literal serving list corrected on two counts, not followed verbatim

- **What:** Split `pyproject.toml`'s single `dependencies` list into
  `[project.dependencies]` (serving — installed by the Docker image) and
  `[dependency-groups] dev` (training/notebooks/CI tooling — never shipped).
  Added `make export-requirements` (`uv export --no-dev --no-hashes --format
  requirements.txt`) so `requirements.txt` is a generated courtesy artifact,
  not hand-edited; nothing in the repo's own tooling reads it anymore
  (Dockerfile and every CI job use `uv sync --frozen[--no-dev]` directly
  against `uv.lock`).
- **The plan's literal serving list was wrong on two counts, corrected
  rather than followed as written:**
    - **`shap` belongs in serving, not dev.** The plan put it in the dev
      group. Every `/predict` response's `top_shap` field — part of the
      LOCKED cross-project contract — is computed via `shap.TreeExplainer`
      on the live request inside `_top_shap_contributors()`; without it the
      API cannot answer a single prediction. `pyyaml`/`python-dotenv` were
      similarly missing from the plan's literal list despite being read at
      every `_load_bundle()` call (`config.yaml`'s threshold,
      `_get_engine()`'s `load_dotenv`).
    - **`lightgbm` does NOT belong in serving, though the plan listed it.**
      Root cause: `api.py` imported `CATEGORICAL_FEATURES` from
      `train_pipeline.py` purely for `_original_feature_of()`'s SHAP
      aggregation — and `train_pipeline.py` imports `optuna` and
      `lightgbm` at module level for its 4-model-family Optuna sweep, so
      merely importing `api.py` silently pulled both training-only
      packages into the servable graph (the Dockerfile's own pre-existing
      `libgomp1` comment already flagged the LightGBM half of this as a
      known wart, without tracing it to its actual cause). Fixed the root
      cause instead of accepting the transitive cost: moved
      `CATEGORICAL_FEATURES`/`NUMERIC_FEATURES` to `dataset_builder.py`
      (already a hard serving dependency, imports neither package) and had
      `train_pipeline.py` import them back from there — one definition,
      still. Verified via `sys.modules` inspection after `import
      domains.pharma.serving.api`: neither `optuna` nor `lightgbm` loads.
      `xgboost` itself still needs `libgomp1` (OpenMP), so that apt package
      stays; the Dockerfile comment was corrected to say so rather than
      still blaming LightGBM.
- **Measured image-size delta:** `docker image inspect`/`docker images`
  reported the *new* image as larger (938MB→1.5GB via `.Size`, 3.55GB→5.12GB
  via `docker images`) — a red herring from the containerd image store's
  attestation/manifest-list accounting under BuildKit, not actual shipped
  content (confirmed both builds independently emit an attestation
  manifest). The number that reflects what's actually on disk is `du -sh /`
  inside each running container: **2.5GB → 1.8GB, a 28% reduction**
  (`/usr/local/lib/python3.11/site-packages` 2.3GB → `/app/.venv/.../site-
  packages` 1.5GB). Breakdown confirmed jupyterlab/evidently/optuna/
  matplotlib*/seaborn/nbconvert/ipykernel/plotly/statsmodels/jedi/debugpy/
  babel are gone from the image (*mlflow itself pulls matplotlib back in
  transitively — unavoidable without dropping mlflow, not a leak in this
  split). One line item dominates what's left either way and is unaffected
  by this fix: `nvidia-nccl-cu12` (401MB, an XGBoost-on-Linux wheel
  dependency for multi-GPU training, useless for CPU inference) was present
  at the same size in the BEFORE image too — not introduced by this change,
  flagged as a real future-revisit (`uv`'s `override-dependencies` could
  plausibly strip it) rather than attempted in this session.
- **Why (vs. alternatives):**
    - *Follow the plan's literal serving/dev split verbatim:* rejected —
      would have shipped an API that 500s on its first `/predict` call
      (missing shap) while still carrying lightgbm's dead weight; the whole
      point of a leanness fix is defeated by keeping an admittedly-unused
      package because a document said to.
    - *Accept the optuna/lightgbm transitive cost rather than move two
      constant lists:* rejected — the fix is two lines moved plus updated
      imports in three files, materially smaller than the packages it
      removes from the image.
- **Failure mode:** if a future PR adds a new `domains.pharma.serving.api`
  import of anything under `train_pipeline.py` (or any other dev-group-only
  module), the optuna/lightgbm exclusion silently regresses — nothing
  currently tests "the servable import graph stays free of dev-only
  packages" as an ongoing invariant; `sys.modules` inspection was a
  one-time manual check, not a CI gate.
- **Scaling story (10x/100x):** dependency count doesn't scale with
  row/model volume — this is a one-time footprint fix. The nvidia-nccl-cu12
  finding is the one line item that WOULD matter at higher request volume
  (larger image → slower cold starts/autoscaling in a K8s context), and is
  exactly the kind of thing worth revisiting if this project's Docker image
  is ever actually deployed at scale rather than run locally.
- **Interview question this maps to:** "How did you keep your serving image
  lean?" — the honest answer includes the shap/lightgbm correction as the
  interesting part: a plan document's dependency split isn't automatically
  correct, and tracing WHY a training-only package was reachable from the
  API (one transitively-imported constant) is a more defensible answer than
  reciting a package list.

## Decision (review-driven, P2): `test_api_contract.py` rewritten against TestClient with a real (tiny) fitted model, not a mock

- **What:** Renamed the Docker-dependent `test_api_contract.py` to
  `test_api_contract_e2e.py` (kept for local end-to-end verification against
  the real Production model via `make test-api`). The new
  `test_api_contract.py` runs against `fastapi.testclient.TestClient`
  directly, all 8 of the original file's tests (health, ready, predict
  schema, field types, threshold applied, nct found, nct not found, missing
  features 422 — the spec's own "7 vs 8" miscount, already noted in Section
  12's M5 row, is unchanged here), no Docker, no live Postgres, no real
  MLflow registry. Added to the `test-unit` CI job (the `--ignore` flag
  moved from `test_api_contract.py` to `test_api_contract_e2e.py`).
- **Fixture choice: a real tiny fitted pipeline, not a mock.** The fix
  plan's own text allowed either "mock the model bundle or use a tiny
  fixture model." Chose the latter: `_build_fixture_bundle()` fits a real
  `XGBClassifier` + `CalibratedClassifierCV(FrozenEstimator(...))` +
  `MAPIEConformalWrapper` + `SHAPExplainer` on ~120 rows of synthetic data
  shaped exactly like `PharmaDatasetBuilder.build_features()`'s output,
  mirroring `register_model.py`'s own procedure at toy scale. A
  `_FakeEngine`/`_FakeConnection` pair stands in for the one SQL query
  `/predict/nct/{id}` issues (and no-ops the prediction-log INSERT). Every
  route under test exercises the REAL prediction path — preprocessing,
  calibration, conformal interval math, SHAP aggregation, plain-English
  templating — not a hand-stubbed response shape that could drift from what
  the real bundle actually produces without a test noticing.
- **Why (vs. alternatives):**
    - *Mock `calibrated_model.predict_proba`/`conformal_wrapper.predict_with_interval`
      directly:* rejected — would test that the route calls a mock
      correctly, not that `_predict_from_row`'s actual math (SHAP
      aggregation, conformal interval construction, plain-English
      partitioning) produces a valid response; a refactor that silently
      broke the real pipeline could still pass.
    - *Load the real Production MLflow model in CI:* rejected — the plan
      explicitly anticipated this failing on a fresh checkout (no
      `mlruns/`), which is exactly CI's situation; a fixture model is the
      documented fallback.
- **Failure mode:** the fixture's synthetic data has no real relationship
  between features and label (label is a threshold on one column) — these
  tests validate response SHAPE and plumbing, not model quality; a
  regression in prediction *accuracy* would not be caught here (nor should
  it be — that's TEST-split PR-AUC's job, not a contract test's).
- **Scaling story (10x/100x):** the fixture model trains in ~1 second on
  every test run (10 XGBoost estimators, 80 rows) — irrelevant to
  production row/model scale, since it never touches real data.
- **Interview question this maps to:** "Your contract tests need a model —
  where do you get one in CI without the real registry?" — a tiny, genuinely
  fitted model beats a mock because it can't silently drift from what the
  real prediction path actually does.

## Decision (review-driven, P2): raw XGBoost pipeline logged as its own MLflow artifact; api.py no longer reaches through three layers of sklearn private attributes

- **What:** `register_model.py` now logs `xgb_pipeline` (the raw fitted
  pipeline, pre-`CalibratedClassifierCV`) as its own artifact
  (`mlflow.sklearn.log_model(xgb_pipeline, artifact_path="raw_pipeline")`)
  on every future registration run, alongside the existing `model`
  (calibrated) artifact. `api.py`'s `_load_bundle()` now loads it directly
  (`mlflow.sklearn.load_model(f"runs:/{mv.run_id}/raw_pipeline")`) instead
  of reaching for `calibrated_model.calibrated_classifiers_[0].estimator.estimator`
  — three layers of `CalibratedClassifierCV`/`FrozenEstimator` private
  attributes with no cross-sklearn-version compatibility guarantee.
- **Backfilled onto the existing v45 Production run, no new registry
  version:** extracted `xgb_pipeline` from the currently-loaded calibrated
  model via the old private-attribute chain ONE more time (the only
  remaining legitimate use of it in this codebase) and re-logged it onto
  v45's existing run (`aa8ec5b5710b490d9b0963b15476bad0`) via
  `mlflow.start_run(run_id=...)`. Verified end-to-end against the real
  Production model afterward: `_load_bundle()` loads successfully, and a
  live docker-compose `/predict` call (see M9-19 entry) returns a correct,
  fully-populated response including `top_shap` — confirming the new load
  path produces a working `SHAPExplainer` just like the old one did.
- **Why (vs. alternatives):**
    - *Leave api.py's private-attribute reach and only fix it on the next
      unrelated retrain:* rejected — the whole point is removing a
      sklearn-version-bump timebomb before it fires, not after.
    - *Retrain to get a clean v46 with the artifact logged the normal way:*
      rejected — no model or feature change is involved; forcing a retrain
      (and a new registry version) for a pure serving-robustness fix would
      be scope creep on a model that hasn't changed.
- **Failure mode:** any FUTURE retrain that doesn't go through
  `register_model.py`'s `main()` (a hand-rolled MLflow run, say) would need
  to remember to log `raw_pipeline` too, or `_load_bundle()`'s
  `mlflow.sklearn.load_model(f"runs:/{mv.run_id}/raw_pipeline")` raises a
  clean `MlflowException` (artifact not found) — a loud failure at startup,
  not a silent misbehavior, but still a manual step someone could forget.
- **Scaling story (10x/100x):** one extra `log_model` call per training
  run — irrelevant to row/model scale, a fixed one-time cost per retrain.
- **Interview question this maps to:** "What happens to your serving code
  when sklearn ships a minor version bump?" — before this fix, the honest
  answer was "hope `CalibratedClassifierCV`'s internal attribute names
  didn't change"; after it, "the raw pipeline is its own versioned MLflow
  artifact, loaded through the public `mlflow.sklearn.load_model` API, not
  a private attribute chain."

## Decision (review-driven, P2): `feature_pipeline_version()` widened to a content hash of all three pipeline-defining files

- **What:** Replaced `git rev-list -1 HEAD -- dataset_builder.py` (a
  single-file git hash) with a sha256 content hash of
  `dataset_builder.py` + `train_pipeline.py` + `config.yaml`'s concatenated
  bytes (`_PIPELINE_FILES`, hashed in `feature_pipeline_version()`). Fixes
  two blind spots the review flagged (§2.9): (1) a change to
  `train_pipeline.py`'s `CATEGORICAL_FEATURES`/`NUMERIC_FEATURES` or
  `config.yaml`'s missingness policy/dropped features/split dates moved
  neither the old hash nor M7's version-mismatch check, even though both
  files define real pipeline behavior; (2) `.dockerignore` excludes
  `.git/`, so `git rev-list` silently returned `"unknown"` for any run
  trained inside a container — this function can no longer return
  `"unknown"` at all (that was specifically a git-lookup-failure fallback),
  so `register_model.py`'s now-dead "commit the repo first" warning was
  removed too. Added `tests/test_feature_pipeline_version.py` (5 tests, no
  DB/MLflow dependency, `_PIPELINE_FILES` monkeypatched to `tmp_path`
  throwaway files): hash changes when any of the three files changes,
  stays stable when a fourth/unrelated file changes, deterministic for
  identical content.
- **Deliberate behavior change, named explicitly:** unlike the old git
  hash (which only moved on commit), this ALSO changes on uncommitted
  edits to any of the three files. Per the fix plan's own framing: "this
  hash covers uncommitted edits too, which git rev-list cannot — that's a
  feature, not a bug." A training run against dirty working-tree state
  should be tagged as such, not silently attributed to the last commit's
  (different) feature logic.
- **Why (vs. alternatives):**
    - *Widen the git-hash approach to cover all three files (e.g. combine
      three `git rev-list` calls) instead of switching to a content hash:*
      rejected — doesn't fix the `.dockerignore`/`.git/` blind spot, which
      is the second, independently-real bug the review flagged; a content
      hash fixes both problems with one mechanism instead of two.
- **Failure mode:** scoped to exactly these three files — a change to a
  FOURTH file that also affects feature semantics (e.g. a shared `core/`
  helper `dataset_builder_base.py` itself) would not move this hash. Named
  explicitly in the function's own docstring as the boundary of what this
  guarantees, not silently assumed to be exhaustive.
- **Scaling story (10x/100x):** `hasher.update(path.read_bytes())` over
  three files is O(file size), negligible next to the correlated self-joins
  that already dominate `fetch_raw()`'s runtime — irrelevant at any row
  count.
- **Interview question this maps to:** "Your feature-pipeline-version check
  had a blind spot — how did you find it and what did you replace it
  with?" — the review named the blind spot; the fix is a content hash over
  the actual set of files that define pipeline behavior, verified by a test
  that would catch a regression to single-file scope.

## Decision (review-driven, P2): structured JSON logging across the four batch entrypoints; a capsys-visibility bug found and fixed in the process

- **What:** Added `core/logging_utils.py` (`JsonFormatter` +
  `configure_json_logging()`, domain-agnostic per this project's core/ vs
  domains/pharma/ split) and replaced every `print()` call in
  `drift_job.py`, `register_model.py`, `retrain_trigger.py`, and
  `dataset_builder.py` with `logger.info()`/`logger.warning()` calls
  (`logger = logging.getLogger(__name__)`, one per module, per the fix
  plan). `configure_json_logging()` is called at each module's import time
  (not gated behind `if __name__ == "__main__":`) so the structured output
  shows up whether the module is run as a CLI (`make drift`,
  `make register-model`), imported by another script (`retrain_trigger.py`
  imports `register_model.py`), or called from a notebook/test — and is
  idempotent, so importing more than one of these four in the same process
  never stacks duplicate handlers.
- **`dataset_builder.py` is the one of the four that `api.py` also
  imports (for `PharmaDatasetBuilder`/`CATEGORICAL_FEATURES`) — flagged,
  not silently absorbed:** this means `configure_json_logging()` also runs
  inside the live API process at startup, not just this module's own CLI
  entrypoint. Judged a harmless, even positive, side effect: nothing
  previously attached a handler to the root logger there at all (any
  app-internal `logging.*` call was invisible beyond Python's WARNING+
  "handler of last resort"), and `configure_json_logging()`'s idempotency
  guard means multiple entrypoints importing it in one process is
  explicitly a supported case, not an accident.
- **Real bug found and fixed: `logging.StreamHandler(sys.stdout)` breaks
  pytest's `capsys`.** `tests/test_version_mismatch.py`'s
  `assert "WARNING" in captured.out` started failing the moment
  `retrain_trigger.py`'s version-mismatch warning moved from `print()` to
  `logger.warning()` — not because the message changed, but because
  `logging.StreamHandler(sys.stdout)` captures whatever `sys.stdout` object
  IS at handler-construction time (module import, long before `capsys`
  monkeypatches `sys.stdout` for an individual test), and keeps writing to
  that stale reference forever. The JSON output was genuinely being
  produced (visible in pytest's own "Captured stdout call" section) but
  invisible to `capsys.readouterr()`. Fixed with `_StdoutProxy` — a
  stream-like object whose `write()`/`flush()` resolve `sys.stdout` at
  CALL time via the module-level `sys` reference, so it observes capsys's
  per-test swap correctly. Added
  `tests/test_logging_utils.py::test_configured_logging_is_visible_to_capsys`
  as a regression test for exactly this failure mode, plus tests for
  formatter output shape and idempotency.
- **Why (vs. alternatives):**
    - *Gate `configure_json_logging()` behind each file's own
      `if __name__ == "__main__":` block, to avoid touching the API
      process's logging at all:* considered and rejected — `api.py` never
      calls `dataset_builder.py`'s `build_features()`/`write_to_db()` (the
      methods with the converted `logger.info()` calls), so the log lines
      themselves never fire during serving either way; gating would also
      mean `PharmaDatasetBuilder` called from a notebook or test gets NO
      visible output at all (a real regression from the old `print()`
      behavior), for no corresponding benefit.
    - *Use `logging.StreamHandler()`'s default stderr instead of stdout:*
      rejected — same staleness bug applies to `sys.stderr`, and stdout is
      what the pre-existing `print()`-based tests and `make drift | jq`
      shell-pipeline usage both expect.
- **`/metrics` endpoint:** added via
  `Instrumentator().instrument(app).expose(app)`
  (`prometheus-fastapi-instrumentator`, added to the serving dependency
  group). Verified with a live TestClient request (200, valid Prometheus
  text format) and against the real docker-compose container (see M9-19
  entry). Nothing scrapes it today — the same honest "the surface exists,
  traffic doesn't" framing M9-7's `ml.prediction_log` entry already uses
  for drift monitoring.
- **Failure mode:** the JSON formatter's `msg` field is built from
  `record.getMessage()`, which already interpolates `%`-style args — a
  logging call with a malformed format string (e.g. mismatched `%s` count)
  raises at log time same as it always would with stdlib logging; not a
  new failure mode introduced by the JSON wrapping.
- **Scaling story (10x/100x):** irrelevant to row/model scale — this is a
  log-format and observability-surface fix. `/metrics`' request-count/
  latency histograms are the one piece that DOES matter at real traffic
  volume — they're what a Prometheus scrape target would consume for
  alerting, which is exactly the "what would you monitor in production"
  question this exists to have a real answer to.
- **Interview question this maps to:** "How did you catch that your new
  logging setup broke a test in a way that had nothing to do with the log
  message content?" — `capsys` captured nothing, the message was still
  correct, and the only way to find it was noticing the JSON payload
  appeared in pytest's "Captured stdout call" diagnostic section but not in
  the fixture's own return value — a stream-identity bug, not a
  content bug.

## Decision (review-driven, P2): served probabilities clipped to [0.001, 0.999]; plain-English never says "100%"/"0%"

- **What:** `api.py`'s `_predict_from_row()` now clips
  `proba = float(np.clip(proba_arr[0], 0.001, 0.999))` immediately after
  reading the calibrated model's output — before the threshold decision,
  SHAP summary, or response are built from it, so every downstream
  consumer sees the clipped value, not just the JSON `proba` field.
  `plain_english.py` gained `_format_probability()`: values that would
  round to "100%"/"0%" under plain `f"{proba:.0%}"` formatting (>=0.995 /
  <0.005) render as "at least 99%" / "less than 1%" instead; every other
  value formats exactly as before. Both layers matter independently: the
  clip bounds the served NUMBER, the formatter bounds what plain English
  ever DISPLAYS even for an unclipped caller (defensive — `_format_probability`
  doesn't assume its input was already clipped).
- **Range note (flagged, not silently deviated from):** the M9 fix-plan
  document (`M9_REVIEW_FIXES_PLAN.md`'s own §P2-18 text) specifies
  `[0.01, 0.99]`; this session's actual task instructions specified
  `[0.001, 0.999]` and were followed as the authoritative, more current ask
  — the plan document was not updated to match. Recorded here so a future
  reader diffing the plan against the code doesn't mistake this for an
  unflagged deviation.
- **Test coverage:** `tests/test_plain_english.py` gained 4 tests
  (`_format_probability`'s normal/high-extreme/low-extreme behavior, plus
  `generate_summary` never containing the literal strings "100%"/"0%").
  `tests/test_api_contract.py` gained
  `test_proba_is_clipped_to_avoid_100_or_0_percent`, which monkeypatches
  the fixture bundle's `calibrated_model.predict_proba` to return genuine
  `0.0`/`1.0` and asserts `_predict_from_row()`'s response clips to
  `0.999`/`0.001` and the plain-English summary contains neither "100%"
  nor "0%".
- **Why (vs. alternatives):**
    - *Only fix the display layer (plain_english.py), leave the served
      `proba` JSON field unclipped:* rejected — a consumer reading the raw
      `proba` field directly (RegIntel's `trial_risk` tool wrapper, a BI
      dashboard) would still see literal `1.0`/`0.0`, which is exactly the
      "100%/0% certainty" credibility problem the fix targets, just moved
      one layer down.
    - *Switch the calibrator to a monotone spline / beta calibration that
      structurally can't saturate, instead of clipping:* the fix plan
      itself names this as a possible future revisit — not pursued this
      session; clipping is the 20-minute fix, a calibrator swap is a
      modeling-methodology change that deserves its own evaluation (ECE,
      TEST-split reliability curve) independent of this polish pass.
- **Failure mode:** the clip bounds are hardcoded (`0.001`/`0.999`), not
  derived from anything about this specific model's calibration curve — a
  model whose real, well-calibrated extreme probabilities are much closer
  to 0/1 than these bounds would have its output slightly compressed at the
  tails. Judged acceptable: the entire point is that "compressed at the
  tails" is exactly the honest thing to communicate, not a modeling error
  to correct for.
- **Scaling story (10x/100x):** `np.clip` on a scalar is O(1) — irrelevant
  to row/model/request scale.
- **Interview question this maps to:** "Would you ever tell a stakeholder a
  trial has a 100% termination probability?" — no, and the API can no
  longer produce that string even if the calibrator's raw output would
  have implied it.

## Decision (review-driven, P2): container hardening — non-root user, digest-pinned base image, HEALTHCHECK; verified against a real running container, not just a Dockerfile diff

- **What:** `Dockerfile` changes, landed together with the M9-13 serving/dev
  dependency split (same file, one coherent rewrite):
    - **Non-root:** `RUN adduser --disabled-password --gecos "" appuser`;
      `chown -R appuser:appuser /app` after `COPY . .`; `USER appuser`
      before `CMD`.
    - **Digest-pinned base image:** `FROM
      python@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93`
      (pulled and inspected live: `docker pull python:3.11-slim &&
      docker inspect ... --format='{{index .RepoDigests 0}}'`), not the
      floating `python:3.11-slim` tag.
    - **HEALTHCHECK:** `--interval=30s --timeout=10s --start-period=60s
      --retries=3 CMD curl -f http://localhost:8000/health || exit 1`
      (`curl` added to the `apt-get install` alongside `libgomp1`).
- **Verified end-to-end against a real running container, not just a
  Dockerfile diff:** `docker compose up --build -d` against this
  developer's real `.env`/`mlruns/` (docker-compose's existing
  bind-mount setup, unchanged by this fix) — confirmed `docker exec ...
  whoami` returns `appuser` (not root); `curl localhost:8000/health` and
  `/metrics` both return 200 on the live container; a real
  `POST /api/v1/predict` call returned a fully-correct response (proba,
  clipped uncertainty_band, coverage_guarantee, partitioned SHAP,
  plain-English summary, feature_pipeline_version — exercising M9-9, M9-15,
  M9-16, M9-17, and M9-18 together in one live request); and, after waiting
  past the 60s `start-period`, `docker inspect --format='{{.State.Health.Status}}'`
  reported `healthy`. Torn down with `docker compose down` afterward.
- **Why (vs. alternatives):**
    - *`USER 1000:1000` (the fix plan's literal snippet) instead of
      `adduser` + a named user:* used `adduser` instead — a named
      `appuser` in `/etc/passwd` is what lets `docker exec ... whoami`
      (and any tool that resolves a username, not just a bare UID) report
      something meaningful; functionally equivalent for the non-root
      requirement itself.
    - *Skip the live-container verification and trust the Dockerfile
      diff:* rejected — the whole point of a hardening change is that it
      shouldn't silently break something (non-root permissions on the
      bind-mounted `mlruns/`, the HEALTHCHECK's `curl` dependency); a
      `docker build` that merely succeeds doesn't prove the container
      actually serves a correct prediction as `appuser`.
- **Failure mode:** the non-root `appuser` only has whatever permissions
  the bind-mounted `mlruns/`/`config.yaml` volumes grant on the HOST side —
  verified read access works (the live prediction above proves it), but a
  host filesystem with more restrictive permissions than this developer's
  machine could still break model loading; not something this fix can
  guarantee across arbitrary host configurations.
- **Trigger to reconsider the digest pin:** re-pull and update it if a
  security patch to the base image is ever needed — a digest never
  auto-updates, which is the safety property being bought, but means this
  file needs a manual bump when `python:3.11-slim` itself gets patched.
- **Scaling story (10x/100x):** irrelevant to row/model/request scale —
  this is image supply-chain and process-privilege hardening, a one-time
  property of the image, not something that degrades under load.
- **Interview question this maps to:** "Walk me through your container's
  security posture." — non-root process, digest-pinned base (no
  floating-tag supply-chain surprise), and a real liveness probe that
  reflects actual server health per `core/serving/api_base.py`'s own
  `/health`-vs-`/ready` distinction — verified against a live container,
  not asserted from a Dockerfile diff alone.

## Decision (review-driven, P2): staying on MLflow's model-registry "stages" API despite the FutureWarning, with a named migration path and trigger

- **What:** This codebase uses `stage="Production"`/`"Staging"`/`"Archived"`
  throughout — `register_model.py`'s Production registration and
  previous-version archival, `retrain_trigger.py`'s Staging registration,
  `rollback.py`'s `rollback_production()`, `drift_job.py`'s/`api.py`'s
  Production-version lookups via `get_latest_versions(...,
  stages=["Production"])` — all of which emit
  `mlflow.tracking.client.MlflowClient.transition_model_version_stage`/
  `get_latest_versions` `FutureWarning`s under mlflow 2.22.5 ("Model
  registry stages will be removed in a future major release"). Decision:
  keep the stages API as-is for the remainder of this project's active
  development, not migrate to the alias-based replacement now.
- **Why this, not migrating now:**
    - `pyproject.toml`'s `mlflow>=2.14,<3.0` ceiling (pre-existing, already
      in place before M9) is the actual mitigation — stages remain
      available and functional for this entire pin range; the warning is
      forward-looking, not an imminent break. Nothing in mlflow 2.x's
      roadmap removes stages; only "a future major release" (3.x) does.
    - The migration itself is well-understood and small in scope, which is
      exactly why it doesn't need to happen preemptively: replace
      `stage="Production"` writes with
      `client.set_registered_model_alias(name, "champion", version)` and
      `stage="Production"` reads with
      `mlflow.sklearn.load_model("models:/{name}@champion")` (or
      `get_model_version_by_alias`). Every one of this codebase's stage
      call sites is a thin, mechanical wrapper around exactly these two
      operations (register/promote, and look-up-current-Production) — no
      call site does anything stage-semantics-specific (no reliance on
      "Staging" as a distinct queryable state beyond "the alias I haven't
      promoted yet," which aliases support equally well via a second alias
      like `"candidate"`). Estimated at roughly a day: rename the stage
      arguments, re-verify M7's retrain/rollback tests
      (`test_retrain_trigger.py`/`test_rollback.py`/
      `test_version_mismatch.py`) against the alias API, no model retrain
      or schema change required.
    - Migrating now, before it's forced, would be effort spent on a
      library-API-currency concern with zero user-facing or model-quality
      benefit — directly the kind of proactive scope CLAUDE.md's "is this
      implementation-forced, or is this scope creep?" standing question is
      meant to catch. This one isn't implementation-forced: mlflow 2.22.5
      works today, and will keep working for the `<3.0` pin's entire
      range.
- **Trigger to reconsider:** migrate when mlflow 3.x is actually released
  AND this project needs to upgrade past it (e.g. a security fix, a new
  MAPIE/sklearn compatibility requirement that only ships against mlflow
  3.x). Until then, the `<3.0` ceiling is sufficient, and is documented as
  such directly in `pyproject.toml`'s dependency comment (not just here).
- **Failure mode:** if the `<3.0` ceiling is ever accidentally loosened
  (e.g. a careless `uv add mlflow --upgrade` past the pin) without this
  decision being revisited first, every stage-based call site above breaks
  simultaneously at the next `register-model`/`check-drift-retrain`/
  `rollback` run — a single dependency-version bump, not a code change,
  would be the actual trigger for an otherwise-silent breakage. The pin is
  the guardrail against exactly that.
- **Scaling story (10x/100x):** irrelevant to row/model/request scale — a
  library-API-lifecycle decision, not a performance or architecture one.
- **Interview question this maps to:** "Your code uses a deprecated MLflow
  API — why haven't you fixed it?" — because it isn't broken yet, the
  `<3.0` pin is the documented reason it won't break under this project,
  and the migration path (aliases, ~1 day, every call site already
  identified) is ready to execute the moment mlflow 3.x is actually forced
  on this project rather than being speculative work against a warning.

## M9-21: doc drift cleanup sweep

Presentation/documentation-hygiene items, per the fix plan's six-item list. Most
are one-liners; two turned out to need more than a text replacement, noted below.

- **`docs/error_analysis.md`:** checked for the stale "feature_pipeline_version=
  unknown, no git commits yet" line the fix plan named — already gone (M9-1's
  entry already notes this essay was "rewritten from scratch in M9"). No change
  needed; confirmed rather than assumed clean.
- **`notebooks/04_shap_analysis.ipynb`:** the markdown cell pointing to
  `logs/query_log.md` "for the persisted copy" was fixed — `logs/` is gitignored
  and never exists on a fresh checkout, so that pointer was a dead link for
  anyone but the original author. Dropped the pointer, kept the actual query-log
  entry (already reproduced inline in the same cell) as the sole copy.
- **`notebooks/01_dataset_audit.ipynb` funnel — root cause was NOT the row
  counts, it was a missing `ORDER BY`.** The fix plan described this as "fix the
  non-monotonic funnel (`plausible_start_date` count above
  `phase_2_3_and_label_status` count)" — investigated before treating it as a
  simple stale-output re-run. The four-branch `UNION ALL` funnel query had no
  `ORDER BY` at all; Postgres is free to return `UNION ALL` rows in any order,
  and it did — two different re-executions of the identical query (once during
  this fix, once again after adding a fix) returned two DIFFERENT scrambled row
  orders, while the underlying counts were themselves always correct and
  monotonically decreasing (596,690 → 114,618 → 79,478 → 78,260). Fixed the
  actual cause: added a `stage_order` integer column to each branch and
  `ORDER BY stage_order`, dropped before display. A stale re-run without this
  fix would have "passed" by accident on some future re-execution and silently
  regressed on the next — the query itself needed to change, not just its cached
  output. Re-executed the full notebook via `nbconvert --execute --inplace`
  (matching M9-1's addendum precedent: full re-execution, not manual cell
  edits) — also caught and fixed a second, smaller drift found in the same pass:
  the final markdown cell's "1,228 RECRUITING/ACTIVE_NOT_RECRUITING trials"
  no longer matched the re-executed count (1,224); updated in the same commit
  since it's adjacent content in the same notebook re-execution.
- **`config.yaml` n=79,334 vs n=78,260 — reconciled the `missingness_policy`
  block specifically, not every occurrence of "79,334" in the repo.** The
  `missingness_policy` section's `null_rate_observed` comments were explicitly
  labeled "measured against the Phase2/3 + label-eligible filtered set
  (2026-07-30, n=79,334)" — a real, current-documentation claim about the
  dataset's size that had gone stale across M9-1's enrollment_count removal and
  M9-11's sponsor-history rebuild (current cache: n=78,260, per decisions.md
  M9-11). Re-measured directly against `data/raw_trials_cache.parquet` (live
  query, not a rescaled estimate) and updated all six `null_rate_observed`
  values with real numbers. Two occurrences deliberately left untouched, not
  silently missed:
    - `config.yaml`'s own line 37 ("222 of 79,334 label-eligible rows... had
      start_date < 1990-01-01") describes a DIFFERENT population — the
      pre-start-date-filter row count at the time of the original M1
      investigation, not the current post-all-filters dataset size the
      missingness block describes. `_RAW_FEATURE_SQL`'s `WHERE start_date >=
      start_date_min` clause means those 222 rows are excluded from the cache
      at the SQL level and cannot be recounted from it — rewriting "79,334" to
      "78,260" here would misrepresent what the number means, not fix a stale
      figure.
    - `docs/decision-log/M1-dataset-builder.md` also contains "79,334" — a
      historical milestone decision-log entry, which this project's own
      convention (see every M9-* entry: corrections are ADDED, not retroactive
      edits to earlier milestones' entries) treats as a point-in-time record,
      not live documentation to keep current.
- **`.gitignore`:** `.DS_store` → `.DS_Store` (the wrong casing silently failed
  to match on this case-sensitive filesystem). Verified with `git status
  --porcelain --ignored`: the repo's actual `.DS_Store` file now shows as
  ignored (`!!`), and was never accidentally tracked in git history in the
  first place (checked via `git ls-files`).
- **`core/conformal.py`'s `verify_coverage()`:** `passed = empirical >= 0.88`
  (hardcoded) → `passed = empirical >= self.target_coverage - 0.02` (derived).
  No behavior change at this project's actual `target_coverage=0.90`
  (0.90 − 0.02 = 0.88 exactly) — the bug was latent, not yet triggered: a
  future `MAPIEConformalWrapper(target_coverage=0.95)` would have silently kept
  the old 0.88 floor, passing a run that under-covers its own 0.95 target by a
  full 5pp instead of the intended 2pp tolerance. Added an over-coverage
  warning (`empirical > target_coverage + 0.05` → new `over_covered` field in
  the returned dict, plus a printed warning) — over-coverage isn't a failure,
  but it does mean the conformal quantile is wider than it needs to be,
  trading away interval informativeness for margin nobody asked for. Added
  `tests/test_conformal.py` (4 tests, stubbed `self.mapie_.predict_set()`, no
  real MAPIE fit needed): tolerance derives correctly at a non-default target
  (would have passed incorrectly under the old hardcoded gate), the default
  target's boundary still matches the old 0.88 behavior exactly, over-coverage
  is flagged past the +5pp threshold, and not flagged within it.
- **Why (vs. alternatives), scoped to the two items with a real design
  choice:**
    - *Blindly text-replace every "79,334" in the repo with "78,260":*
      rejected — the M1-dataset-builder.md and config.yaml line-37 occurrences
      describe different things than the missingness-policy block does;
      correcting a genuinely stale documentation claim is different from
      erasing a historical record or restating a different measurement
      incorrectly.
    - *Fix notebook 01's funnel by just re-running it once and accepting
      whatever order came back:* rejected once the second re-execution proved
      the order itself was non-deterministic — a fix that "happens to look
      right" on one run is not a fix.
- **Failure mode:** none of these six items has an ongoing failure mode beyond
  what's already named per-item above (the `verify_coverage` scoping note, the
  funnel's `ORDER BY` now being load-bearing for display correctness).
- **Scaling story (10x/100x):** none of these six items are scale-sensitive —
  documentation/display/tolerance-formula fixes, not data-volume-dependent
  logic.
- **Interview question this maps to:** "How do you keep documentation from
  drifting out of sync with the code and data it describes?" — the honest
  answer for this sweep is that half these items required actually re-querying
  or re-executing something to get a true current number, not just editing
  text to look consistent; the funnel bug in particular would have kept
  silently misleading readers if "fixed" by re-running once and trusting
  the output.
