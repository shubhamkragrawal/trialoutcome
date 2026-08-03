[← back to decisions.md summary](../../decisions.md)

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
- **What:** Per the M6 requirements, attempted a Population Stability Index on the rolling
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
- **Failure mode if this had been silently skipped instead:** the M6 requirements explicitly
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
  integration suite requiring a live Docker container) and `__init__.py`. The M6 requirements' own
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
  implementation-forced by the requirements' own literal CI recipe, not scope creep: the requirements
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
  M6 requirements' literal instruction.
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
- [x] `02_TRIALOUTCOME_SPEC.md` M6 row updated to ✅ with as-built acceptance criteria (and a stale M1-row
      number corrected in the same pass, per the Decision above).
- [x] `decisions.md` updated with M6 entries (this section).

