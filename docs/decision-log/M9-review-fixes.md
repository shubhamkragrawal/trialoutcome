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
