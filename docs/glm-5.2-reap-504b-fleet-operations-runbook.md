# GLM-5.2-REAP-504B Fleet Operations Runbook

How to **operate** the live GLM-5.2-REAP-504B serving fleet day to day:
check health, recover from Spot preemption, run the continual-learning burn,
and reason about capacity. This is the operator's companion to the build doc
`glm-5.2-reap-504b-g4-runbook.md` (which covers reproducing a single host from a
clean G4 image, hardware profile, vLLM launch flags, and the proxy/HTTPS setup).
Read that one before provisioning a *new* host; read this one to keep the
existing fleet healthy.

## What the fleet is

- A self-hosted fleet of GCP **G4** VMs (`g4-standard-192` / `g4-standard-384`)
  in project `openagentsgemini`, each running **vLLM** serving the
  `glm-5.2-reap-504b-g4` model on `http://localhost:8000` behind a private
  proxy + (optional) public HTTPS origin.
- It backs two workloads:
  1. The **public GLM lane** on `openagents.com` — GLM-only, fail-closed,
     reached with the `glm-saturation` / `internal_stress` demand routing. It
     never spills to a paid provider when GLM is unavailable.
  2. The **continual-learning burn** — `khala-glm-continual-learning-burn.mjs`
     drives many parallel GLM calls over a corpus and writes candidate /
     dataset artifacts.

## Topology: durable vs Spot

The fleet is intentionally split:

- **Durable replicas** live in `us-central1-b` (named `...-g4-4g-b-...` and
  `...-g4-8g-b-...`). They are **not** Spot and survive capacity churn. Treat
  them as the always-on floor.
- **Spot replicas** are spread across regions (`us-west1-a`, `us-east1-b/d`,
  `us-east5-a/b/c`, `us-south1-b`, `us-central1-f`, ...) named with a `-spot-`
  segment. They are cheap added capacity that **will get preempted**. A fleet
  showing "1/10 ready" almost always means the durable floor is up and the Spot
  replicas were preempted — that is normal Spot behavior, not an outage.

## 1. Health check

List running GLM VMs:

```sh
gcloud compute instances list \
  --filter="name~glm52-reap AND status=RUNNING" \
  --format="table(name,zone,status,machineType.basename())"
```

Per-VM serving check (vLLM `/v1/models` should return `glm-5.2-reap-504b-g4`):

```sh
gcloud compute instances list --filter="name~glm52-reap AND status=RUNNING" \
  --format="value(name,zone)" > /tmp/glm_running.txt

while read -r name zone; do
  [ -z "$name" ] && continue
  ok=$(gcloud compute ssh "$name" --zone "$zone" \
        --command 'curl -s -m 5 http://localhost:8000/v1/models | grep -o glm-5.2-reap-504b-g4 | head -1' \
        --ssh-flag="-o ConnectTimeout=12" </dev/null 2>/dev/null)
  [ "$ok" = "glm-5.2-reap-504b-g4" ] && echo "ok   $name" || echo "down $name"
done < /tmp/glm_running.txt
```

> **Gotcha — `</dev/null` is required.** `gcloud compute ssh` reads from stdin,
> so without `</dev/null` it swallows the rest of the `while read` loop and only
> the first VM is ever checked. The loop above redirects it explicitly.

A VM that is `RUNNING` but not yet serving is usually **still loading the
checkpoint** — vLLM mounts the ~1.5T model disk and loads the 504B weights,
which takes several minutes after a (re)start. "Booting/loading" right after a
restart is expected; recheck in a few minutes before treating it as broken.

## 2. Recovery from Spot preemption

When the fleet is degraded because Spot replicas were preempted:

```sh
# Identify preempted Spot replicas
gcloud compute instances list \
  --filter="name~glm52-reap AND status=TERMINATED" \
  --format="value(name,zone)"

# Restart them (cheap: starts the existing VM + its durable disk; no new
# provisioning). Spot may DECLINE if there is no capacity in the region — that
# is an external cloud-capacity risk and costs nothing when it fails.
gcloud compute instances list --filter="name~glm52-reap AND status=TERMINATED" \
  --format="value(name,zone)" | while read -r name zone; do
    [ -z "$name" ] && continue
    echo "starting $name ($zone)"
    gcloud compute instances start "$name" --zone "$zone"
  done
```

Then, per the `glm-5.2-reap-504b-g4-runbook.md` **Recovery** section, for each
restarted VM:

1. Verify the durable model disk is attached.
2. Raw vLLM `ACTION=status`; if `/v1/models` is not ready, `ACTION=start`.
3. Poll raw vLLM until `modelsEndpoint=ready` (allow minutes for checkpoint load).
4. Proxy `ACTION=install-systemd` or `ACTION=restart-systemd`, then
   `ACTION=status` and `ACTION=smoke`.
5. Re-admit healthy replicas with
   `scripts/admit-glm-52-reap-504b-fleet-gce.sh`.
6. Confirm the public GLM lane serves; commit/comment only public-safe summary
   artifacts.

Do **not** force a Spot stop/start just to refresh a doc. Prefer restarting
existing VMs over provisioning new G4 hosts — new provisioning is expensive and
slow; restarting an existing Spot VM only pays when capacity is actually
granted.

Durability helpers live alongside this fleet: the keep-warm units and the
durable canary watchdog (`install-glm-52-reap-504b-durable-canary-gce.sh`) help
preempted replicas recover faster. Keep the keep-warm timer **disabled** during
decision-grade Terminal-Bench runs (warm probes contend with the singleflight
model lane) and re-enable it for normal steady-state operation.

## 3. The continual-learning burn

The burn is the steady-state GLM workload. It runs **many parallel GLM calls**
and is the primary way to keep the fleet productive.

```sh
# From the openagents repo (clean worktree at origin/main):
CL_CONCURRENCY=12 CL_MAX_TOKENS=512 \
  bun scripts/khala-glm-continual-learning-burn.mjs        # continuous loop
# add --once or --cycles N for bounded runs
```

Key facts:

- It routes through `https://openagents.com/api/v1/chat/completions` with
  `model: openagents/khala` on the **GLM-only** `glm-saturation` /
  `internal_stress` route — **no paid fallback**. If GLM is down the calls fail
  closed; they never spill to a paid provider.
- Burn keys come from `~/work/.secrets/khala-heartbeat.env`
  (`KHALA_HEARTBEAT_KEYS`). Never print them.
- `CL_CONCURRENCY` is the number of parallel in-flight runners (default 4).
- Output: `~/work/.khala-continual-learning/corpus-dataset-<date>.jsonl` (and
  `remediation-candidates-*.jsonl`, `receipt-*.json`). **A growing
  `corpus-dataset-<today>.jsonl` is the liveness/productivity signal.**
- Setting `OPENAGENTS_ADMIN_API_TOKEN` enables the trace lane; without it the
  trace lane is skipped and the burn still runs.

**Scale concurrency to serving capacity.** Effective burn throughput is bounded
by how many VMs are actually serving, not by `CL_CONCURRENCY` alone — the GLM
lane has singleflight/backpressure, so over-concurrency just queues. Start the
burn against the durable floor and raise `CL_CONCURRENCY` toward 16–24 as the
Spot replicas finish loading and admit.

If the burn process dies, relaunch it; it is a plain loop with no durable
supervisor by default.

## 4. Capacity model (why GLM workers are not account-bound)

This is the operationally important distinction:

- The **codex/claude coding lane** (`codex_agent_task` / `claude_agent_task`
  Khala workers) is **account-rate-limited**: each linked ChatGPT/Anthropic
  account sustains ~**2 concurrent** workers; a 3rd concurrent call per account
  429s. Total concurrency there scales only with the number of *distinct*
  connected accounts (`khala fleet connect`).
- The **GLM fleet workers** (the burn, and GLM-lane traffic generally) are
  **GPU-bound, not account-bound**. Concurrency scales with the number of
  serving G4 replicas, independent of any per-account API rate limit. This is
  the lever for running large parallel workloads without adding accounts.

So: to add coding-agent throughput, connect more accounts; to add GLM
throughput, recover/add serving replicas and raise burn concurrency.

## 5. Receipt discipline

Public-safe receipts may contain run IDs, hashes, token counts, public
hardware/runtime metadata, aggregate timings, and sanitized failure classes.
They must **not** contain secrets, raw prompts, raw responses, hidden reasoning,
burn keys, or admin tokens. The burn's corpus/candidate artifacts are local;
publish only digests and aggregate stats.

## Related docs

- `glm-5.2-reap-504b-g4-runbook.md` — build/reproduce a single G4 host, vLLM
  launch flags, proxy + public HTTPS, the canonical Recovery checklist.
- `glm-5.2-sglang-preflight-runbook.md` — SGLang preflight path.
- `gpt-oss-20b-khala-live-roadmap.md` — the sibling self-hosted GPT-OSS lane.
