# TrialOutcome M8 — Part D: Probe distinction demo

Goal: demonstrate that `/health` (liveness) and `/ready` (readiness) gate
traffic differently — a pod can be `Running` (liveness passing) while
`NotReady` (readiness failing) indefinitely, without being restarted.

## Adaptation from the milestone brief, flagged explicitly

The brief suggested "temporarily rename the `mlruns/` volume mount or
simulate a slow model load" to produce this state. That mechanism does not
actually work against this app's real lifespan implementation
(`domains/pharma/serving/api.py`'s `_load_bundle()`, locked M5 serving
code): if the MLflow registry lookup fails (e.g. because the model
artifacts aren't where expected), `_load_bundle()` raises, FastAPI's
`lifespan` context manager propagates that exception, and uvicorn's startup
fails outright — the container process exits non-zero before it ever binds
port 8000. Kubernetes then sees a genuine crash (`CrashLoopBackOff`,
restarted by the **liveness** mechanism's normal container-restart policy,
not a liveness *probe* failure), not a `Running`-but-`NotReady` pod. There's
no partial/degraded-load code path in the locked serving contract to
target instead.

**What was actually done instead:** the live Deployment's `readinessProbe`
path was temporarily patched (via `kubectl patch`, not a file edit — see
below) to a nonexistent route, while `livenessProbe` was left untouched
pointing at `/health`. This isolates exactly the signal being demonstrated
— the readiness gate, independent of the app's real internal state — without
inventing a new failure-handling path in code the M5 contract already locked
down. **`k8s/deployment.yaml` itself was never modified** — this was a
purely runtime `kubectl patch` against the live cluster object, reverted
afterward (confirmed: `git diff`-equivalent — the file on disk still reads
`path: /ready`, see the diff check at the bottom of this doc).

## Commands run

```bash
# 1. Baseline -- all pods Running and Ready
kubectl get pods -l app=trialoutcome-api -o wide

# 2. Patch ONLY readinessProbe's path (liveness untouched)
kubectl patch deployment trialoutcome-api --type=json \
  -p='[{"op":"replace","path":"/spec/template/spec/containers/0/readinessProbe/httpGet/path","value":"/definitely-not-a-real-route"}]'

# 3. Watch new pods come up
kubectl get pods -l app=trialoutcome-api

# 4. Revert
kubectl patch deployment trialoutcome-api --type=json \
  -p='[{"op":"replace","path":"/spec/template/spec/containers/0/readinessProbe/httpGet/path","value":"/ready"}]'
kubectl rollout status deployment/trialoutcome-api
```

## Real kubectl output

**Before (baseline — all 4 pods Running + Ready, RESTARTS 0):**

```
NAME                                READY   STATUS    RESTARTS   AGE     IP          NODE
trialoutcome-api-7f7786b4c8-27qcg   1/1     Running   0          5m56s   10.42.0.5   k3d-trialoutcome-demo-agent-0
trialoutcome-api-7f7786b4c8-675cn   1/1     Running   0          5m56s   10.42.1.5   k3d-trialoutcome-demo-agent-1
trialoutcome-api-7f7786b4c8-c7qmm   1/1     Running   0          3m10s   10.42.0.6   k3d-trialoutcome-demo-agent-0
trialoutcome-api-7f7786b4c8-gdnbd   1/1     Running   0          3m10s   10.42.2.5   k3d-trialoutcome-demo-server-0
```

**After patch — new pods Running but 0/1 (NotReady), old pods left alone by
the rolling update because the new pods never satisfy the readiness gate:**

```
NAME                                READY   STATUS    RESTARTS   AGE     IP
trialoutcome-api-7895c4c7df-kt5bh   0/1     Running   0          8s      10.42.2.6
trialoutcome-api-7895c4c7df-pnj44   0/1     Running   0          8s      10.42.1.6
trialoutcome-api-7f7786b4c8-27qcg   1/1     Running   0          6m4s    10.42.0.5
trialoutcome-api-7f7786b4c8-675cn   1/1     Running   0          6m4s    10.42.1.5
trialoutcome-api-7f7786b4c8-gdnbd   1/1     Running   0          3m18s   10.42.2.5
```

**Sustained over ~57s of polling every 6s — still `Running`, still `0/1`,
`RESTARTS` stays at `0` the entire time** (full poll log, timestamps real):

```
=== 15:40:44 ===  kt5bh 0/1 Running 0  14s   pnj44 0/1 Running 0  14s
=== 15:40:50 ===  kt5bh 0/1 Running 0  20s   pnj44 0/1 Running 0  20s
=== 15:40:56 ===  kt5bh 0/1 Running 0  26s   pnj44 0/1 Running 0  26s
=== 15:41:02 ===  kt5bh 0/1 Running 0  33s   pnj44 0/1 Running 0  33s
=== 15:41:09 ===  kt5bh 0/1 Running 0  39s   pnj44 0/1 Running 0  39s
=== 15:41:15 ===  kt5bh 0/1 Running 0  45s   pnj44 0/1 Running 0  45s
=== 15:41:21 ===  kt5bh 0/1 Running 0  51s   pnj44 0/1 Running 0  51s
=== 15:41:27 ===  kt5bh 0/1 Running 0  57s   pnj44 0/1 Running 0  57s
```

**Direct proof the app itself is healthy** — port-forwarded straight to the
`NotReady` pod (`trialoutcome-api-7895c4c7df-kt5bh`), bypassing the Service
entirely:

```
--- /health on the NotReady pod ---
{"status":"ok","timestamp":"2026-08-02T19:41:54.662119+00:00"}
HTTP 200
--- /ready on the NotReady pod (the app's real /ready route) ---
{"status":"ready","model_loaded":true,"conformal_loaded":true}
HTTP 200
```

**`kubectl describe pod` confirms the mechanism precisely** — the kubelet's
configured readiness probe target is the bogus path, and it's *that* which
is failing (404), not the application:

```
    Readiness:  http-get http://:8000/definitely-not-a-real-route delay=5s timeout=1s period=5s #success=1 #failure=6
  Warning  Unhealthy  0s (x17 over 76s)  kubelet  Readiness probe failed: HTTP probe failed with statuscode: 404
```

This is the cleanest possible isolation of the mechanism: the app's real
`/ready` endpoint was reporting `ready` the entire time — kubelet simply
wasn't asking it. `NotReady` here is 100% a property of what the probe
polls, not of application state, which is exactly the liveness/readiness
distinction the M8 milestone asks to demonstrate: readiness gates traffic
independent of whether the process is alive and serving.

**After revert — rollout completes, all pods back to Running + Ready:**

```
Waiting for deployment "trialoutcome-api" rollout to finish: 3 out of 4 new replicas have been updated...
Waiting for deployment "trialoutcome-api" rollout to finish: 3 of 4 updated replicas are available...
deployment "trialoutcome-api" successfully rolled out

NAME                                READY   STATUS    RESTARTS   AGE
trialoutcome-api-7f7786b4c8-27qcg   1/1     Running   0          8m4s
trialoutcome-api-7f7786b4c8-675cn   1/1     Running   0          8m4s
trialoutcome-api-7f7786b4c8-6bcfp   1/1     Running   0          9s
trialoutcome-api-7f7786b4c8-gdnbd   1/1     Running   0          5m18s
```

**Diff check confirming `k8s/deployment.yaml` on disk was never touched**
(the patch only ever modified the live cluster object):

```
$ grep -A2 readinessProbe k8s/deployment.yaml
          readinessProbe:
            # Shorter initial delay is fine here -- a 503 from /ready just
            # withholds traffic (doesn't kill the pod), unlike liveness.
            httpGet:
              path: /ready
```

## Bonus, unplanned observation worth keeping

The rolling update never touched the 3 already-`Ready` old-ReplicaSet pods
while the new pods sat `NotReady` — `maxUnavailable`'s default (25%) meant
Kubernetes correctly refused to tear down working capacity in favor of pods
that couldn't prove they were ready to serve. That's readiness gating
protecting a rollout, not just a load balancer — a second, real consequence
of the liveness/readiness distinction beyond the one this demo specifically
set out to show.
