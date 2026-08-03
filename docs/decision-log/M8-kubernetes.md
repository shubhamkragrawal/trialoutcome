[← back to decisions.md summary](../../decisions.md)

---

## M8: Scoped Kubernetes Deploy (2026-08-02) -- FINAL MILESTONE

---

### Decision: the requirements' literal `k3d cluster create trialoutcome-demo --agents 2` does not make `mlruns/` visible inside pods -- recreated with `--volume <mlruns>:<same absolute path>@all`
- **What:** k3d agent/server nodes are themselves Docker containers, not the Mac host.
  A `hostPath` volume in a pod spec resolves inside that node container's own filesystem
  unless the real host directory is bind-mounted into every node *at cluster-creation
  time*. Verified directly: created the cluster with the requirements' literal command first,
  confirmed no bind mount existed, deleted it, and recreated with
  `k3d cluster create trialoutcome-demo --agents 2 --volume
  "$(pwd)/mlruns:/Users/.../trialoutcome/mlruns@all"` -- the `@all` node-filter suffix
  mounts it into the server and both agent nodes, since the scheduler may place a pod on
  any of them. Verified via `docker exec k3d-trialoutcome-demo-agent-0 ls
  /Users/.../trialoutcome/mlruns` before applying any manifests.
- **Why (vs. alternatives):** The container path is deliberately identical to the repo's
  real absolute path -- same reason `docker-compose.yml` bind-mounts `mlruns/` at an
  identical absolute path (see M5's decisions.md entry): MLflow's local file-backed
  tracking store bakes each run's absolute host path into its own metadata at logging
  time, so `mlflow.sklearn.load_model()` only resolves it if the mount path matches
  exactly, everywhere. Using a different, k8s-idiomatic path (e.g. `/mlruns-data`) would
  require either re-registering every MLflow run (destroys the M5/M7 audit trail this
  project has built up) or a path-rewriting shim -- both strictly worse than reusing the
  exact pattern already proven correct in `docker-compose.yml`.
- **Failure mode:** Without this fix, pods would `CrashLoopBackOff` immediately --
  `_load_bundle()`'s `mlflow.sklearn.load_model()` call would raise a file-not-found
  error the instant a pod tried to start, since the registered run's artifact URI points
  at a path that, without the node-level bind mount, simply doesn't exist inside the k3d
  node container.
- **Scaling story (10x/100x):** N/A directly -- this is the same hostPath-does-not-
  generalize-past-single-node limitation already flagged in `docker-compose.yml` and
  M5's decisions.md, now flagged a third time for k8s specifically (see README's "What
  I'd change at production scale" section and this file's next-but-one entry).
- **Interview question this maps to:** "Why doesn't a Docker Compose bind-mount pattern
  port directly to Kubernetes?" -- k3d's own nodes being containers is a good concrete
  illustration of why `hostPath` in K8s needs to resolve against the actual node's
  filesystem, and why that's fundamentally different from (and more fragile than) a
  single-daemon Docker Compose bind mount.

---

### Decision (bugfix to the requirements' own assumption): k3d 5.9.0 ships `metrics-server` bundled by default -- no separate install needed
- **What:** The requirements' Part B said "Requires metrics-server ... document if k3d needs it
  installed separately (it usually does not ship it by default, verify and note)."
  Verified directly on this build: `kubectl get deployment metrics-server -n kube-system`
  showed it present immediately after `k3d cluster create`, reaching `1/1` Ready within
  ~15s, with `kubectl top nodes` returning real CPU/memory numbers with no extra install
  step. k3d 5.9.0's underlying k3s (`v1.35.5+k3s1`) evidently bundles it now.
- **Why this isn't silently corrected without a note:** The requirements' assumption was
  reasonable (metrics-server historically needed a separate manifest on many k3d/kind
  setups) but is empirically wrong for the installed version -- documented explicitly in
  `docs/k8s_setup.md` rather than silently skipping the verification step the requirements asked
  for, and the fallback install command is still documented in case a different
  k3d/k3s version doesn't bundle it.
- **Failure mode:** If a future k3d version stops bundling metrics-server, `kubectl get
  hpa` would show `TARGETS: <unknown>/50%` indefinitely and never scale -- the
  `docs/k8s_setup.md` verification step (`kubectl wait --for=condition=available
  deployment/metrics-server`) would catch this immediately, before wasting time debugging
  why the HPA never reacts to load.
- **Interview question this maps to:** "How do you handle a runbook step whose own
  assumption turns out to be wrong for your environment?" -- verify directly (`kubectl
  get deployment`) rather than trusting either the requirements' assumption or blind faith that
  "it usually just works," and document the actual observed behavior for whoever reads
  this next.

---

### Decision: `POSTGRES_PASSWORD` sourced from a Kubernetes Secret, seeded with the `.env.example` placeholder, real value never touched by Claude
- **What:** `k8s/deployment.yaml` sources `POSTGRES_PASSWORD` via `secretKeyRef` against
  a Secret `trialoutcome-postgres-secret`, rather than a plaintext env var (which the M8
  requirements' literal instruction -- "env vars: same POSTGRES_* vars docker-compose.yml
  already uses" -- would have allowed, since `docker-compose.yml` gets it from `.env` via
  `env_file:`). Per this project's standing rule, Claude never reads the real `.env`, so
  it could not embed the real password into any manifest or command even if it wanted to.
  Created the Secret with the `.env.example` placeholder (`changeme`) directly via
  `kubectl create secret generic ... --from-literal=POSTGRES_PASSWORD=changeme`, and
  documented (in `docs/k8s_setup.md`) the exact command for the user to run themselves,
  against the real `.env`, if they want to exercise the real
  `/api/v1/predict/nct/{nct_id}` route against live warehouse data.
- **Why this doesn't block M8's actual acceptance criteria items:** `domains/pharma/dataset_builder.py`'s
  `_get_engine()` uses SQLAlchemy's `create_engine()`, which is lazy -- it never opens a
  real connection until a query actually runs. `/health`, `/ready`, and
  `POST /api/v1/predict` (the exact route the HPA load test in Part C exercises) never
  touch Postgres at all, so the pods reach `Ready` and serve real predictions regardless
  of whether the seeded password is correct. Verified directly: the load test in Part C
  and the probe demo in Part D both ran successfully against the placeholder-secret pods.
  Only the NCT-lookup route would fail (500/connection error) against the placeholder.
- **Why a Secret instead of a plaintext env var anyway, given the requirements' literal
  wording allowed either:** Using a Secret is strictly more correct and costs nothing
  extra to build -- and it's also more consistent with what Part E's own "what I'd change
  at production scale" section has to say about secrets management (a real K8s `Secret`
  is still only base64-encoded, not encrypted at rest, so even this is flagged as a demo-
  grade compromise, not a production-grade one).
- **Failure mode:** If a future contributor updates `k8s/deployment.yaml` to a plaintext
  `POSTGRES_PASSWORD` env var "for convenience," the placeholder or a real value would
  end up committed directly into a tracked YAML file -- worth grepping for
  `POSTGRES_PASSWORD` in `k8s/*.yaml` periodically to confirm it's still `secretKeyRef`-
  only, the same class of check M7's grep-based auto-promotion verification already
  established as this project's pattern for "assert an invariant stays true going
  forward."
- **Interview question this maps to:** "How do you handle a credential a spec asks you
  to wire up, when you're not allowed to read the real value?" -- design around the
  constraint honestly (a Secret reference plus a documented command for a human to run),
  rather than working around it by hardcoding a fake-but-plausible-looking value or
  silently using a weaker plaintext-env-var pattern just because the literal instruction
  technically permitted it.

---

### Decision: liveness `initialDelaySeconds=15` set from a real measured ~5s model-load time, not guessed
- **What:** Before writing `k8s/deployment.yaml`, timed the actual `trialoutcome-api:m8`
  image directly: `docker run` a fresh container, poll `/health`/`/ready` every second
  from container start. Observed: both return `200` by t=5s (includes MLflow artifact
  download + `mlflow.sklearn.load_model` deserialization + SHAP explainer construction).
  `livenessProbe.initialDelaySeconds` set to 15 (~3x margin); `readinessProbe
  .initialDelaySeconds` set to 5 (no extra margin needed, since a `503` there only
  withholds traffic rather than killing the pod, per the requirements' own guidance).
- **Why (vs. alternatives):** Guessing a "safe-sounding" delay (e.g. 30s or 60s, common
  defaults copied from unrelated tutorials) would work but wastes real time on every pod
  restart / rolling update waiting past a delay that doesn't reflect this specific image's
  actual behavior. Measuring first is cheap (one `docker run` + a polling loop) and
  directly informs the value instead of guessing at it.
- **Failure mode:** If a future model artifact grows substantially larger (a much bigger
  ensemble, or a deep-learning model per the GPU-nodes note in README's "What I'd change
  at production scale"), this 5s baseline would need re-measuring -- the load time is a
  property of the specific model+dependencies, not a fixed constant.
- **Scaling story (10x/100x):** Model artifact size, not row count, is what would move
  this number -- a 10x/100x larger *training dataset* has no effect on inference-time
  load latency, since the served artifact is a fixed-size fitted model regardless of how
  much data trained it.
- **Interview question this maps to:** "How did you pick your liveness probe's
  `initialDelaySeconds`?" -- measure the real cold-start time against the actual image
  before setting probe timing, rather than copying a number from a tutorial that has no
  relationship to this specific application's startup cost.

---

### Decision: probe distinction demo used a `kubectl patch` on the live readinessProbe path, not the requirements' literal "rename mlruns/ / simulate slow load" suggestion
- **What:** Verified directly that the requirements' suggested mechanism does not produce the
  state it's meant to demonstrate against this app's real code: if `_load_bundle()`
  (`domains/pharma/serving/api.py`, locked M5 serving contract) fails for any reason --
  including a broken `mlruns/` mount -- the exception propagates through FastAPI's
  `lifespan` context manager and uvicorn's startup fails outright; the container process
  exits before ever binding port 8000. Kubernetes would see a genuine crash
  (`CrashLoopBackOff`, driven by the normal container-restart policy, not a probe
  *failure*), never a `Running`-but-`NotReady` pod -- because there is no partial/
  degraded-load code path in the locked serving contract to target instead. Used
  `kubectl patch deployment ... readinessProbe/httpGet/path` to point only the readiness
  probe (liveness untouched) at a nonexistent route instead -- this isolates exactly the
  signal being demonstrated (the readiness gate, independent of real app state) without
  inventing a new failure-handling path in M5's already-locked serving code.
- **Why (vs. alternatives):** Could have added a genuine slow-load code path to
  `api.py` (e.g. an artificial `time.sleep()` before `state.model_loaded = True`) --
  rejected as unnecessary scope creep against locked serving code for a demo whose actual
  point (readiness gates traffic without restarting the pod) doesn't require the delay to
  be caused by real model loading specifically, only that `/ready` fails while `/health`
  succeeds. The `kubectl patch` approach also has the advantage of being provably
  reversible with zero risk to the codebase: `k8s/deployment.yaml` on disk was never
  edited, only the live cluster object, confirmed via a direct diff-equivalent check
  after reverting (see `docs/k8s_probe_demo.md`).
- **Failure mode:** If a future reader assumes this demonstrates "the app degrades
  gracefully under slow model load," that would be a misread -- it demonstrates the probe
  *mechanism* (kubelet only cares about what it's configured to poll, not real app state),
  explicitly not a claim that this app has graceful degraded-load behavior. Documented
  plainly in `docs/k8s_probe_demo.md` to head that misread off.
- **Scaling story (10x/100x):** N/A -- a demo-mechanism decision, not a performance one.
- **Interview question this maps to:** "The requirements told you to do X to demonstrate Y --
  you found X doesn't actually work here. What do you do?" -- verify the literal
  suggested mechanism against the real code before running with it, and when it doesn't
  hold, find the smallest faithful substitute that demonstrates the same underlying
  property (here: readiness gates traffic, doesn't restart the pod) rather than
  either forcing the original suggestion to "work" via an unrequested code change, or
  silently skipping the demo.

---

### Finding: HPA scaled 2→4 (hit maxReplicas) under a real load test; probe demo held `NotReady` for 57+s with zero restarts -- both real results, neither assumed
- **What:** `hey -z 60s -c 20` against `POST /api/v1/predict` (port-forwarded, real k3d
  pods) drove `kubectl get hpa`'s reported CPU from 1%→50%→100% of the 50% target over
  ~54s of sustained load; the HPA controller created 2 new pods once CPU crossed 100%,
  which became `Ready` (`readinessProbe.initialDelaySeconds=5` elapsing) within ~15s, at
  which point the `Deployment`'s `REPLICAS` column caught up to 4 -- confirmed via
  polling `kubectl get hpa`/`kubectl get pods` every 5s throughout, real timestamped
  output preserved in `docs/k8s_load_test_output.txt`. All 1,796 requests over the 60s
  window returned `200` -- no dropped/errored requests during the scale-up transition.
  Scale-down back to 2 was not observed (the default 5-minute downscale stabilization
  window exceeds the capture window) and is explicitly not claimed.
- **Why this matters:** The M8 requirements explicitly warned against claiming scaling happened
  if `kubectl` output didn't show it, and to try a heavier load pattern if the first
  attempt didn't trigger scaling. The first attempt (`-z 60s -c 20`, the requirements' own
  suggested parameters) worked on the first try and reached `maxReplicas` -- no need for
  a heavier `-c 50` retry, but the real, timestamped, unedited `kubectl` output is
  preserved specifically so this claim is independently checkable, not just asserted.
- **Bonus, unplanned observation kept in the record:** during the probe demo (Part D),
  the rolling update correctly refused to tear down the 3 already-`Ready` old-ReplicaSet
  pods while the 2 new pods sat `NotReady` -- `maxUnavailable`'s default meant readiness
  gating protected the rollout itself, not just load-balancer traffic. Not something the
  requirements asked to demonstrate, but a real and relevant consequence of the same mechanism,
  documented in `docs/k8s_probe_demo.md` rather than discarded as out of scope.
- **Interview question this maps to:** "Walk me through verifying an HPA actually scales
  under load, not just configuring it and assuming it works." -- run the real load test,
  poll the real `kubectl` state throughout (not just before/after), and keep the raw
  timestamped output as the artifact, so "it scaled" is a checkable claim, not an
  assertion.

## M8 Definition of Done -- status

- [x] k3d cluster created, image imported, deployment applies clean -- cluster recreated
      with a `--volume ...@all` mount (see Decision above; the requirements' literal command
      alone does not surface `mlruns/` inside pods), `trialoutcome-api:m8` built and
      `k3d image import`-ed, all three manifests (`deployment.yaml`, `service.yaml`,
      `hpa.yaml`) applied with no errors.
- [x] 2 replicas running, both eventually Ready -- both pods reached `1/1 Running` within
      ~14s of `kubectl apply` (well inside the 15s liveness `initialDelaySeconds` margin).
- [x] HPA scales 2→N under real load test, N reported honestly -- **N=4 (hit
      maxReplicas)**, real `kubectl` output in `docs/k8s_load_test_output.txt`, zero
      dropped requests across 1,796 total.
- [x] `docs/k8s_load_test_output.txt` has real kubectl output -- unedited, timestamped,
      chronological, covering baseline through post-load-test settling.
- [x] `docs/k8s_probe_demo.md` demonstrates `/health` vs `/ready` gating traffic
      differently, with real kubectl output -- pod held `Running`+`NotReady` 57+s with 0
      restarts, direct port-forward to the pod proved the app's real `/ready` reported
      `ready: true` throughout, `kubectl describe pod` confirmed the probe's configured
      target (not app state) was the failing signal.
- [x] README "what I'd change at production scale" section written -- ingress, secrets
      management, hostPath `mlruns/` (explicitly flagged non-generalizing), namespaces,
      GPU nodes, multi-region/multi-cluster.
- [x] hostPath volume limitation explicitly flagged as non-generalizing -- in
      `k8s/deployment.yaml`'s own comments, `docs/k8s_setup.md`, and README's "What I'd
      change at production scale" section, consistent with the same limitation already
      flagged for `docker-compose.yml` in M5's decisions.md entry.
- [x] `02_TRIALOUTCOME_SPEC.md` M8 row updated to ✅, milestones table reflects project
      completion -- "Remaining (M5–M8)" renamed to "Completed (M5–M8) -- all TrialOutcome
      milestones now done" (with an explicit note that PharmaPulse itself has not yet
      finished its own milestone list, so there was no finished-portfolio-project pattern
      to mirror, per the M8 spec's instruction to follow one).
- [x] `decisions.md` updated with M8 entries (this section).
- [x] TrialOutcome project marked complete in README -- "Status: all 8 milestones
      complete (M1–M8)" added directly under the intro paragraph.

