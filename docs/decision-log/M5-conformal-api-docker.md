[← back to decisions.md summary](../../decisions.md)

---

## M5: Conformal Prediction + FastAPI + Docker (2026-08-02)

---

### Decision: Step 0a (git init + first commit) was already satisfied before M5 started -- no commit made this session
- **What:** The M5 spec assumed a zero-commit repo (per the M2 decision entry above). By the time
  M5 started, `git log` already showed one commit (`e886fc6`, "initial files, model runs with optuna"),
  made by the user per this project's own rule that Claude never runs `git add`/`git commit`. Verified
  `git rev-parse HEAD -- domains/pharma/dataset_builder.py` returns a real hash, not `"unknown"`, and
  that file is unmodified since that commit -- `feature_pipeline_version()` was already producing a
  real (if buggy, see next entry) tag before any M5 code ran.
- **Why (vs. alternatives):** Running `git init`/`git commit` myself would violate this project's
  explicit, standing instruction (Claude never modifies git history/state) -- confirming the
  precondition was already met and proceeding directly to Step 0b was correct, not a skipped step.
- **Failure mode:** N/A -- this is a status check, not a code change.
- **Interview question this maps to:** "How do you handle a runbook step that assumes a precondition
  that's already been met?" -- verify the precondition directly (`git log`, `git rev-parse`) rather
  than blindly re-running the step or blindly skipping it.

---

### Decision (bugfix): `feature_pipeline_version()` used `git rev-parse HEAD -- <path>`, which does not filter by path -- fixed to `git rev-list -1 HEAD -- <path>`
- **What:** `domains/pharma/dataset_builder.py`'s `feature_pipeline_version()` (M2-era code) ran
  `git rev-parse HEAD -- domains/pharma/dataset_builder.py`. `git rev-parse` does not use a trailing
  pathspec to scope *which* commit's hash it returns -- it always returns HEAD's hash, then echoes
  `--` and the path back as additional literal output lines (verified directly:
  `git rev-parse HEAD -- <path>` prints 3 lines; `git rev-list -1 HEAD -- <path>` prints exactly 1).
  This was invisible throughout M1-M4 because the repo had zero commits and the function's
  `except (CalledProcessError, FileNotFoundError): return "unknown"` fallback masked it -- the bug
  only manifested the moment `register_model.py` ran against a real commit, producing the tag value
  `"e886fc6...\n--\ndomains/pharma/dataset_builder.py"` instead of a clean hash. Fixed to
  `git rev-list -1 HEAD -- <path>`, which is the correct incantation for "hash of the last commit
  that touched this path."
- **Why (vs. alternatives):** Leaving the polluted tag in place would satisfy M5's literal acceptance criteria line
  ("feature_pipeline_version is a real git hash, not unknown") on a technicality while silently
  breaking M7's planned version-mismatch check (a promotion-time `==` comparison against this tag
  would never match anything once every future tag also carries different trailing junk, or worse,
  match by accident if two different commits happened to embed the same path string). This is
  implementation-forced, not scope creep: M5's own production run needed a genuinely clean tag to
  meet its own acceptance criteria in spirit, not just in the literal wording.
- **Failure mode:** Re-ran `make register-model` after the fix and confirmed the tag is now exactly
  `e886fc6bc5e6fd7cab3afc3295665f49caaaea85` with no trailing content (verified via
  `client.get_run(...).data.tags`).
- **Scaling story (10x/100x):** N/A -- a one-time git plumbing correctness fix, independent of data
  or model scale.
- **Interview question this maps to:** "How did a versioning bug hide for three milestones?" -- a
  fallback (`except: return "unknown"`) that swallows the *actual* failure mode (a malformed but
  non-crashing git command) is worse than no fallback at all, because it makes the bug look like a
  documented limitation ("no commits yet") instead of a defect, right up until the precondition that
  was masking it (zero commits) stops holding.

---

### Decision: `CalibratedClassifierCV(base_estimator=..., cv="prefit")` from the M5 spec does not run on this project's pinned sklearn==1.9.0 -- used `estimator=FrozenEstimator(...)` instead
- **What:** sklearn renamed `base_estimator` -> `estimator` in 1.2 and removed the `cv="prefit"` string
  value entirely in favor of wrapping an already-fitted estimator in `sklearn.frozen.FrozenEstimator`
  (confirmed via `inspect.signature` and `help()` against the installed sklearn 1.9.0 -- `cv="prefit"`
  is gone from the docstring, replaced by an explicit `FrozenEstimator` code example).
  `domains/pharma/register_model.py` uses `CalibratedClassifierCV(estimator=FrozenEstimator(xgb_pipeline),
  method="isotonic")` instead of the spec's literal `base_estimator=...,  cv="prefit"` call.
- **Why (vs. alternatives):** Pinning an older sklearn to match the spec's exact snippet would
  contradict `pyproject.toml`'s `scikit-learn>=1.9.0` (already relied on by M2-M4's Optuna/SHAP code)
  for no benefit -- the `FrozenEstimator` path produces the identical semantics (all provided data
  used for calibration, no internal refitting) with the currently-installed, already-verified sklearn
  version.
- **Failure mode:** Verified this reproduces the exact M3 TEST-split ECE (0.0238, matching to 4dp) --
  if `FrozenEstimator` behaved differently from the old `cv="prefit"` semantics, this number would have
  diverged from M3's independently-computed calibration result.
- **Scaling story (10x/100x):** N/A -- API-compatibility fix, independent of data scale.
- **Interview question this maps to:** "What do you do when a spec's code snippet doesn't match your
  installed library version?" -- check the installed API directly (`inspect.signature`, `help()`)
  rather than guessing or downgrading a pinned dependency to match stale documentation.

---

### Decision: MAPIE's classic `MapieClassifier(method="score")` API does not exist in mapie==1.4.1 -- used the renamed `SplitConformalClassifier(conformity_score="lac")`
- **What:** `uv add mapie` resolved `mapie==1.4.1`, which replaced the entire pre-1.0 API
  (`MapieClassifier`, `method="score"/"cumulated_score"/...`) with `SplitConformalClassifier`/
  `CrossConformalClassifier` and a `conformity_score` parameter (`"lac"`, `"aps"`, `"raps"`, ...).
  `"lac"` (Least Ambiguous Set-valued Classifier) is `score`'s direct successor for binary
  classification -- confirmed via `inspect.signature`/docstrings against the installed package, no
  `mapie.classification.MapieClassifier` symbol exists at all in this version.
  `core/conformal.py`'s `MAPIEConformalWrapper` is built against `SplitConformalClassifier(prefit=True)`
  + `.conformalize(X_calib, y_calib)` + `.predict_set(X)` instead of the spec's literal
  `MapieClassifier(...).fit(...)` calls.
- **Why (vs. alternatives):** Pinning an old, pre-1.0 mapie release to match the spec's exact class
  names was rejected -- `uv add mapie` (unconstrained, per this project's dependency-management rule)
  resolved the current release, and there is no reason to force an old, presumably less-maintained
  version onto a brand-new integration when the new API's `SplitConformalClassifier` is the direct,
  documented replacement with equivalent guarantees.
- **Failure mode:** If a future MAPIE major version renames this again, `core/conformal.py`'s import of
  `mapie.classification.SplitConformalClassifier` would raise `ImportError` immediately (loud failure),
  not silently produce wrong coverage.
- **Scaling story (10x/100x):** N/A -- API-compatibility fix.
- **Interview question this maps to:** "How do you handle a spec written against an API that changed
  underneath you?" -- inspect the installed package directly, confirm the semantic equivalent exists,
  and document the substitution explicitly rather than silently pinning around it.

---

### Decision (bugfix): MAPIE's internal "is this estimator fitted" probe breaks on a `ColumnTransformer` that selects columns by name -- fixed with a `_NamedColumnEstimatorAdapter` shim
- **What:** `SplitConformalClassifier.conformalize()` internally calls `check_sklearn_user_model_is_fitted`,
  which probes the base estimator with `estimator.predict(np.zeros((1, n_features_in_)))` -- a bare
  ndarray, not a DataFrame. This project's pipelines select columns by *name*
  (`ColumnTransformer([("cat", OneHotEncoder(...), CATEGORICAL_FEATURES), ...])`), which raises
  `"Specifying the columns using strings is only supported for dataframes"` on a bare array. MAPIE
  wraps that exception in a hard `raise UserWarning(...)` (not `warnings.warn` -- appears to be a MAPIE
  bug) that aborts `conformalize()` entirely, rather than a plain crash. A first fix attempt (wrap any
  non-DataFrame input into a same-shaped all-zero DataFrame) still crashed one level deeper: an all-zero
  probe row breaks `OneHotEncoder`'s unknown-category handling on any *categorical* column, because
  `0.0` isn't a valid category and sklearn's NaN-check path crashes comparing a float against a
  string-typed categories array. Fixed by having `_NamedColumnEstimatorAdapter` answer any non-DataFrame
  input using a real captured template row (the first row of `X_calib`) instead of the literal probe
  content -- correct dtypes everywhere, and the probe's return value was never checked for correctness
  by MAPIE anyway (only whether the call raises).
- **Why (vs. alternatives):** Restructuring `_make_preprocessor` to select columns positionally
  (avoiding the whole class of "probe with a bare array" failures) was rejected -- it would touch the
  already-validated M2-M4 training pipeline and its 6dp-verified refit assertion, for a problem that's
  entirely on the *serving/conformal* side. A thin adapter isolated to `core/conformal.py` fixes the
  incompatibility without touching anything upstream.
- **Failure mode:** This adapter's `predict`/`predict_proba` return *meaningless* values for a bare-array
  input (template-row-based, ignoring the literal input content) -- safe only because the only caller
  that ever passes a non-DataFrame is MAPIE's internal self-check. Documented explicitly in the class
  docstring so a future maintainer doesn't repurpose this adapter somewhere real ndarray predictions are
  expected.
- **Scaling story (10x/100x):** N/A -- a one-time library-compatibility shim, independent of data scale.
- **Interview question this maps to:** "Tell me about a time a well-tested pipeline broke against a
  third-party tool for a reason that had nothing to do with your model." -- named-column selection is
  a common, reasonable pipeline pattern that plenty of tooling (here, MAPIE's internal self-check)
  doesn't anticipate; the fix belongs at the integration boundary, not by weakening the pipeline itself.

---

### Decision: MAPIE prediction-SET -> `[low, high]` float interval conversion, and the `margin` used for a singleton `{1}` set
- **What:** `MAPIEConformalWrapper.predict_with_interval` converts MAPIE's per-row prediction set
  (`{0}`, `{1}`, or `{0,1}`) to a float interval: `{1}` only -> `[proba - margin, proba + margin]`
  (clipped to `[0,1]`); `{0,1}` both -> `[0.0, 1.0]`; `{0}` only -> `[0.0, proba]`. `margin` is defined
  as `1 - target_coverage` (the *nominal* coverage this wrapper was configured for, i.e. `0.10` at the
  default `target_coverage=0.90`) -- **not** the empirical coverage `verify_coverage()` measures on
  TEST. This was ambiguous in the M5 spec ("margin = 1 - coverage_achieved" doesn't specify which
  coverage number) and is resolved here explicitly.
- **Why (vs. alternatives):** Using the empirical TEST coverage as `margin`'s input would create a
  circular dependency at serving time: `predict_with_interval()` is the method the live API calls per
  request, and it must be usable immediately after `fit_conformal()` -- a production request can't wait
  on a TEST-split evaluation that may never run outside of the M5 notebook. The nominal target is always
  available the instant the wrapper is configured, and it's what MAPIE's conformal guarantee is actually
  keyed to.
- **Failure mode:** If a future retrain measures empirical coverage well below target (e.g. 0.80 against
  a 0.90 target), the margin used at serving time would still reflect the *intended* 0.10 gap, not the
  *actual* larger miscalibration -- worth revisiting if `verify_coverage()`'s gate ever starts failing
  in production rather than at build time (it did not here: empirical 0.946 vs target 0.90).
- **Scaling story (10x/100x):** N/A -- margin is a scalar constant per model version, independent of
  request volume or data scale.
- **Interview question this maps to:** "Where in an ML system's uncertainty story would you draw the
  line between 'evaluated once at training time' and 'must be servable per-request without extra
  computation'?" -- a per-request health check can't depend on a batch evaluation step; anything a live
  endpoint needs must be captured as a constant at model-registration time.

---

### Result: Conformal coverage gate passed comfortably -- empirical 0.946 vs. target 0.90 (gate: >=0.88)
- **What:** `notebooks/05_conformal.ipynb` fit `MAPIEConformalWrapper(target_coverage=0.90)` on the
  CALIB split (n=6,310) and measured empirical coverage on the held-out TEST split (n=5,700):
  **0.946**, comfortably clearing the M5 acceptance criteria gate (`>=0.88`). Logged as `empirical_coverage` on the
  same MLflow production run register_model.py created (run `13ca29a52a5b456b93846514f8020dfa`), with
  the fitted `MAPIEConformalWrapper` itself logged as an artifact (`conformal/`) on that run.
- **Why the empirical number exceeds the target:** LAC (least-ambiguous-set) conformal scores are
  finite-sample conservative -- at n=6,310 calibration points, the true achieved coverage is expected
  to be at or slightly above the nominal target, not exactly equal to it. A coverage well above target
  (rather than a razor-thin pass) is the expected, healthy outcome here, not a sign the model is
  under-confident everywhere -- interval widths still vary meaningfully by predicted probability (see
  `docs/conformal_width_vs_proba.png`).
- **Failure mode if this had come out below 0.88:** Per the M5 spec's explicit instruction, the
  build would have stopped before Part B (FastAPI) rather than proceeding with an unverified
  uncertainty guarantee -- this didn't happen here, but the gate assertion is a real `assert` in the
  notebook, not just a printed number a reader could miss.
- **Interview question this maps to:** "How do you know your conformal intervals actually mean
  anything?" -- fit the quantile on CALIB, verify coverage on a genuinely held-out TEST split the
  quantile never saw, and treat a passing check as a gate the pipeline can't silently skip past.

---

### Decision: Model + conformal wrapper loaded once at FastAPI startup (lifespan), not per-request
- **What:** `domains/pharma/serving/api.py`'s `lifespan()` context manager loads the calibrated model,
  conformal wrapper, condition vocabulary, and feature schema from the MLflow registry exactly once
  when the process starts, storing them in a module-level `_PharmaModelBundle`. `/ready` only reports
  `model_loaded=True` once this has finished.
- **Why (vs. alternatives):** Loading per-request (re-hitting the MLflow registry and re-deserializing
  a cloudpickled sklearn pipeline + XGBoost booster on every call) would add meaningful, unnecessary
  latency to every prediction for an artifact that only changes on redeploy -- there is no correctness
  reason to reload it more often than the process lifetime.
- **Failure mode:** A model promoted to a new "Production" version in MLflow will NOT be picked up by
  an already-running API process -- this is deliberate (a live process silently switching models
  mid-traffic without a restart/health-check cycle is a worse failure mode than requiring a redeploy),
  but means promotion must be paired with a restart, which M7's retrain/rollback flow needs to account
  for explicitly.
- **Scaling story (10x/100x):** At higher request volume, startup-time loading is strictly better (the
  fixed model-load cost is amortized across all requests instead of paid per-request); at multiple
  replicas (M8's k8s HPA), each pod independently loads the same Production model at its own startup --
  fine at this project's model size, but would need a shared/cached artifact store if the model
  artifact grew large enough that N replicas each downloading it from a network-mounted `mlruns/`
  became a real cold-start cost.
- **Interview question this maps to:** "Why load a model at startup instead of on first request?" --
  startup loading fails fast (a broken artifact surfaces immediately as an unready pod, not as a
  500 on some unlucky user's first request) and amortizes load cost across the process's whole
  lifetime.

---

### Decision: `mlruns/` mounted as a Docker volume, not baked into the image
- **What:** `docker-compose.yml` mounts `./mlruns:/app/mlruns`; the `Dockerfile` never `COPY`s model
  artifacts, and `.dockerignore` doesn't need to exclude `mlruns/` only because it's never in the build
  context's relevant path for `COPY . .` in the intended deployment (mounted at runtime, not built in).
- **Why (vs. alternatives):** Baking `mlruns/` into the image would mean every new MLflow run (new
  Optuna sweep, new calibration, new conformal fit) requires a full image rebuild to deploy -- coupling
  a fast-changing artifact (retrained model, updated as often as M7's automated retraining trigger
  fires) to a slow-changing one (application code, dependencies). A volume mount lets `make
  register-model` produce a new Production version and a simple container restart pick it up, with no
  rebuild.
- **Failure mode:** If the host's `mlruns/` directory is deleted or not present when the container
  starts, the API's `lifespan()` startup will raise (no Production model to load) -- a loud, correct
  failure (the container won't report `/ready`), not a silent fallback to some stale baked-in model.
- **Scaling story (10x/100x):** At production scale, a local bind-mount `mlruns/` directory doesn't
  generalize past a single Docker host -- M7/M8's real target would be a shared MLflow tracking server
  with artifacts in object storage (S3/GCS), which every replica reads over the network instead of a
  host-local mount. Flagged here as the natural next step, out of scope for M5's single-host Compose
  setup.
- **Interview question this maps to:** "How do you keep a model artifact's deploy cadence decoupled
  from your application code's deploy cadence?" -- volume-mount (or, at scale, a remote artifact
  store) the model; never bake a frequently-retrained artifact into a container image.

---

### Decision: `/health` and `/ready` are separate routes with different failure semantics, not one endpoint with a boolean field
- **What:** `core/serving/api_base.py`'s `/health` always returns `200` the instant the process can
  serve HTTP, regardless of model-loading state. `/ready` returns `200` only once
  `state.model_loaded and state.conformal_loaded` are both `True`, and returns HTTP `503` (not a `200`
  with `"not_ready"` in the body) otherwise.
- **Why (vs. alternatives):** A single combined endpoint would force a K8s liveness probe (which exists
  to detect and restart a genuinely hung/crashed process) to either (a) also depend on model-loading
  state, meaning a pod gets killed and restarted in a loop while a large model artifact is still
  loading -- the exact failure mode this spec calls out by name -- or (b) ignore model state entirely,
  in which case there's no way for a readiness/load-balancer check to withhold traffic from a pod
  that's alive but not yet able to serve real predictions. Two routes with two different HTTP status
  semantics let K8s (M8) wire liveness and readiness probes to the behaviors they're actually meant to
  gate.
- **Failure mode:** Verified directly -- `/ready` returns `503` with `model_loaded=false` in the tests
  run before `lifespan()` startup completes (not exercised in the committed integration tests, which
  run after the container is fully up, but confirmed manually during local `uvicorn` testing that the
  503 path executes before the model finishes loading).
- **Scaling story (10x/100x):** Unaffected by request volume -- these are O(1) state checks. At higher
  replica counts (M8), each pod's `/ready` independently gates its own traffic eligibility, which is
  exactly the per-pod granularity a K8s readiness probe needs.
- **Interview question this maps to:** "Why does Kubernetes need two separate probes instead of one
  health check?" -- liveness answers "should this process be killed and restarted," readiness answers
  "should this pod receive traffic right now" -- conflating them means a slow-starting pod (loading a
  model) gets killed by the same check that should instead just withhold traffic from it temporarily.

---

### Decision: `TrialFeatures` input schema corrected to match the actually-trained feature set -- `intervention_model` dropped, `has_results` added
- **What:** The M5 spec's literal `TrialFeatures` pydantic model included `intervention_model:
  Optional[str]` and omitted `has_results`. Checked directly against
  `domains/pharma/train_pipeline.py`'s `CATEGORICAL_FEATURES`/`NUMERIC_FEATURES` (the actual columns
  the registered Production model was fit on): `intervention_model` is not present anywhere in the
  trained feature set -- the M1 spec-table note calling it a "bonus feature... present in mart" turns
  out to describe a column that was computed and then never actually included in the final feature
  list (confirmed: no `intervention_model` key in any row's persisted `ml.training_dataset.features`
  JSONB). `has_results` (rank-7 in M4's global SHAP importance) IS in the trained feature set but was
  missing from the spec's schema entirely.
- **Why (vs. alternatives):** Serving with the spec's literal schema would do one of two bad things
  silently: accept `intervention_model` from a client with zero effect on the prediction (a field that
  looks load-bearing but isn't), or force a fabricated default for `has_results` -- a feature the model
  actually learned a nontrivial pattern from -- with no way for the caller to supply the real value.
  Neither is a reasonable serving contract. This is flagged explicitly, per this project's own
  standing instruction to surface spec-vs-reality conflicts rather than silently pick a side: the
  *response* schema (Section 6's locked contract) is unchanged; only the unlocked *request* schema
  (never given explicit field names in the spec itself, only in this milestone spec) was corrected.
- **Failure mode:** If a future spec update re-adds `intervention_model` as a genuine trained feature
  (e.g., a future retrain does start using it), this input schema would need updating in lockstep with
  `train_pipeline.py`'s `CATEGORICAL_FEATURES` -- the two are not currently enforced to stay in sync
  automatically.
- **Scaling story (10x/100x):** N/A -- a fixed-schema correctness fix.
- **Interview question this maps to:** "What do you do when a milestone spec's example payload doesn't
  match what the model actually needs?" -- verify against the model's real trained feature list (here,
  `train_pipeline.py`'s constants, cross-checked against a real persisted feature row), not the spec
  prose, and fix the schema to match reality rather than the other way around.

---

### Decision: `docker-compose.yml` environment variables corrected to `POSTGRES_*`, not the spec's literal `DB_*` names
- **What:** The M5 spec's `docker-compose.yml` set `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`.
  `domains/pharma/dataset_builder.py`'s `_get_engine()` reads `os.environ[db_cfg["host_env"]]` etc.,
  where `config.yaml`'s `db.host_env`/`port_env`/`dbname_env`/`user_env`/`password_env` resolve to
  `POSTGRES_HOST`/`POSTGRES_PORT`/`POSTGRES_DB`/`POSTGRES_USER`/`POSTGRES_PASSWORD` -- there is no
  `DB_HOST` anywhere in this codebase, and `os.environ[...]` (direct indexing, not `.get()`) raises
  `KeyError` immediately if any is missing. The spec's literal names, used as-is, would have left
  the containerized API unable to connect to Postgres at all (and never mentioned `POSTGRES_PASSWORD`,
  without which the connection can't authenticate regardless of naming).
  `docker-compose.yml` uses `env_file: .env` (passing through the real `POSTGRES_PORT`/`POSTGRES_DB`/
  `POSTGRES_USER`/`POSTGRES_PASSWORD` unmodified) with a single `environment:` override,
  `POSTGRES_HOST=host.docker.internal`, since `localhost` inside the container refers to the container
  itself, not the Docker host running Postgres.
- **Why (vs. alternatives):** Renaming `dataset_builder.py`'s env var reads to match the spec's
  `DB_*` names was rejected -- that file is shared, already-validated M1 code with no reason to change
  for a naming preference introduced in an M5 spec; fixing the compose file to match the real,
  already-established contract is the smaller, correct-direction change. Per this project's rule
  (never read `.env`'s real contents), `env_file` passes the real values through without me viewing
  them.
- **Failure mode:** If the actual `POSTGRES_PORT` in `.env` ever changes without a corresponding
  `docker-compose.yml` update, the container would still pick up the new value automatically via
  `env_file` -- there's no hardcoded port duplicated in compose to go stale.
- **Scaling story (10x/100x):** N/A -- configuration-correctness fix.
- **Interview question this maps to:** "How do you avoid duplicating environment configuration across
  a Dockerfile/compose file and the application code that actually reads it?" -- point at the same
  source of truth (`.env` via `env_file`) instead of re-declaring variable names by hand in a second
  place, which is exactly how the two can drift.

---

### Decision: `/api/v1/predict/nct/{nct_id}` registered for both GET and POST -- the M5 spec's own text and test file disagree
- **What:** The M5 spec's endpoint description says `POST /api/v1/predict/nct/{nct_id}` (matching
  `02_TRIALOUTCOME_SPEC.md` Section 6's table), but its own `tests/test_api_contract.py` calls
  `requests.get(f"{BASE_URL}/api/v1/predict/nct/{{nct_id}}")` in both `test_predict_nct_found` and
  `test_predict_nct_not_found`. `domains/pharma/serving/api.py` registers the route via
  `@app.api_route(..., methods=["GET", "POST"])` so both the literal test file and the documented
  spec table are satisfied simultaneously.
- **Why (vs. alternatives):** Silently "fixing" the tests to use POST (matching the spec table) would
  contradict the spec's own literal, explicitly-given test file -- which the M5 acceptance criteria explicitly
  requires to pass as-is. Registering only POST would fail the given tests; registering only GET would
  contradict the spec table. Supporting both is the only option that satisfies both without silently
  discarding either source of truth. As a secondary observation: GET is arguably the more RESTful verb
  here regardless (the route takes no request body, only a path parameter, and is a pure read).
- **Failure mode:** If RegIntel's `trial_risk` tool wrapper (built against the spec table) only ever
  calls this route with POST, the GET registration is simply unused, not harmful -- no downstream
  contract is broken by supporting an extra method.
- **Scaling story (10x/100x):** N/A.
- **Interview question this maps to:** "What do you do when a spec document and its own test suite
  contradict each other?" -- satisfy both literally where that's possible (here, by supporting both
  HTTP methods) rather than picking one source of truth and silently failing the other.

---

### Decision (bugfix): `python:3.11-slim` is missing `libgomp.so.1`, which LightGBM's compiled binary requires to import
- **What:** `docker compose up --build` failed at container startup with
  `OSError: libgomp.so.1: cannot open shared object file` -- `domains/pharma/serving/api.py` imports
  `CATEGORICAL_FEATURES` from `domains/pharma/train_pipeline.py`, which has a top-level
  `from lightgbm import LGBMClassifier`, and LightGBM's compiled extension links against the OpenMP
  runtime (`libgomp`), which `python:3.11-slim` doesn't ship by default (unlike the full `python:3.11`
  image). Fixed by adding `apt-get install -y --no-install-recommends libgomp1` to the `Dockerfile`
  before the pip install step.
- **Why (vs. alternatives):** Switching `train_pipeline.py`'s constants to live somewhere that doesn't
  import LightGBM at module scope was rejected as unnecessary surgery on already-validated M2-M4 code,
  for a dependency the API doesn't actually need to import (it only needs the two constant lists,
  `CATEGORICAL_FEATURES`/`NUMERIC_FEATURES`, not LightGBM itself) -- the one-line `apt-get install` is
  the minimal, correctly-scoped fix, and keeps `train_pipeline.py` as the single source of truth for
  those constants rather than duplicating them into a LightGBM-free module.
- **Failure mode:** If a future base-image change (e.g. moving off Debian-slim to Alpine) removes
  `apt-get`/changes the package name, this would need re-diagnosing -- Alpine's musl-based lightgbm
  wheels have a different, separately-documented set of native-dependency issues.
- **Scaling story (10x/100x):** N/A -- one-time base-image dependency fix.
- **Interview question this maps to:** "How do you debug a container that imports fine locally but
  crashes in Docker?" -- trace the traceback to the actual native library the OS-level package manager
  needs to provide; `pip install`-able Python packages can still depend on system shared libraries the
  slim base image doesn't include.

---

### Decision (bugfix): local file-backed MLflow stores bake ABSOLUTE host paths into run metadata -- broke `mlflow.sklearn.load_model` inside the container until the volume was mounted at the identical absolute path
- **What:** After fixing the `libgomp1` issue, the container still failed at startup:
  `OSError: No such file or directory:
  '/Users/shubhamagrawal/.../trialoutcome/mlruns/<exp>/<run>/artifacts/model/.'` -- MLflow's local
  `file:` tracking store records each run's `artifact_uri` as an ABSOLUTE path at the moment
  `mlflow.sklearn.log_model()` runs (baked into that run's `meta.yaml`), not a path relative to
  whatever `MLFLOW_TRACKING_URI` happens to point at when later *loading* the model. Inside the
  container, `mlruns/` was mounted at `/app/mlruns`, but the registered model's metadata still pointed
  at the original host-absolute path -- which doesn't exist inside the container's filesystem at all.
  Fixed two ways together: (1) `domains/pharma/serving/api.py`'s `_load_bundle()` now respects
  `MLFLOW_TRACKING_URI` from the environment if set, instead of always recomputing
  `file:{REPO_ROOT}/mlruns` from `__file__` (which resolves to `/app` inside the container, not the
  host path baked into existing run metadata); (2) `docker-compose.yml` bind-mounts `./mlruns` a
  *second* time at the exact same absolute path it has on the host, and sets
  `MLFLOW_TRACKING_URI=file:///Users/.../trialoutcome/mlruns` to match.
- **Why (vs. alternatives):** Re-registering the model with a tracking URI that used a
  container-friendly relative path would only move the problem -- the same run would then fail to
  load from a plain local `uv run` invocation on the host instead. Migrating to a real MLflow tracking
  server with remote artifact storage (S3/GCS) is the actual production fix (object-store keys aren't
  tied to any filesystem's absolute layout), but is out of scope for M5's single-host Docker Compose
  setup -- the mount-at-identical-path workaround is the correct-sized fix for this milestone, done
  transparently rather than silently working around it by, e.g., re-running the whole M1-M5 pipeline
  inside the container instead of reusing the artifacts already produced on the host.
  This limitation, and the actual production fix, are called out explicitly in both this entry and an
  inline comment in `docker-compose.yml` so it isn't mistaken for how this would work with more than
  one host.
- **Failure mode:** This hardcodes a host-specific absolute path (this developer's exact directory
  structure) into `docker-compose.yml` -- it will not work unmodified on a different machine or a
  second developer's checkout. Flagged directly in `docker-compose.yml`'s comment and here, not
  silently left as a portability trap.
- **Scaling story (10x/100x):** Does not generalize past a single Docker host at all -- multiple
  replicas (M8's k8s HPA) each independently mounting a host bind-mount is already the wrong model at
  that point; M8 will need a real MLflow tracking server + S3/GCS-backed artifact store regardless of
  this specific fix.
- **Interview question this maps to:** "What breaks when you containerize something that assumed a
  local filesystem?" -- absolute paths baked into metadata at write time are a classic portability
  trap; the honest fix at this milestone's scope is documented as a workaround, with the real
  production fix (remote artifact store) named explicitly rather than implied.

## M5 Definition of Done -- status

- [x] Conformal empirical coverage >= 0.88 on TEST split -- **0.946** (target 0.90), logged on the
      production MLflow run (`empirical_coverage` metric, run `13ca29a52a5b456b93846514f8020dfa`).
- [x] Fitted `MapieClassifier`-equivalent (`MAPIEConformalWrapper`, wrapping mapie 1.4.1's
      `SplitConformalClassifier`) logged as an MLflow artifact (`conformal/`) on the production run.
- [x] `docker compose up` starts without errors -- required two bugfixes beyond the M5 spec's literal
      Dockerfile/compose (libgomp1 system package; MLflow absolute-path volume mount), both documented
      above.
- [x] `GET /health` -> 200.
- [x] `GET /ready` -> 200 with `model_loaded=true`, `conformal_loaded=true` (returns 503 before the
      model finishes loading -- verified via the `ServingState` default of `False` plus manual local
      testing of the pre-lifespan-completion window).
- [x] `POST /api/v1/predict` -> 200 with all 6 locked fields present (`proba`, `conformal_interval`,
      `threshold_decision`, `top_shap`, `plain_english_summary`, `feature_pipeline_version`).
- [x] All 8 tests in `tests/test_api_contract.py` pass against the running Docker container (the M5
      spec's acceptance criteria line says "7" but the spec's own test file defines 8 test functions -- a minor
      miscount in the spec, not a discrepancy in this build).
- [x] `/health` and `/ready` are demonstrably distinct (`/ready` returns 503 via `ServingState`'s
      `model_loaded`/`conformal_loaded` flags; `/health` never depends on them).
- [x] `feature_pipeline_version` is a real, clean git hash (`e886fc6bc5e6fd7cab3afc3295665f49caaaea85`)
      on the production MLflow run -- required fixing a pre-existing bug in
      `feature_pipeline_version()` itself (see above) that would otherwise have produced a
      polluted-but-technically-non-"unknown" tag.

