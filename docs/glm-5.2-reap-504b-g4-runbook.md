# GLM-5.2 504B REAP G4 runbook

Status: `private_canary`

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/92

Receipt:
[`docs/evidence/2026-06-24-glm-52-reap-504b-integration-receipt.json`](evidence/2026-06-24-glm-52-reap-504b-integration-receipt.json)

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../profiles/glm-5.2-reap-504b-b12x-g4.json)

Public-safety boundary: this runbook contains public model metadata, pinned
runtime settings, public-safe run IDs, hashes, counts, and aggregate metrics
only. It contains no bearer token, model-provider credentials, raw prompts, raw
responses, private source, hidden reasoning traces, weights, checkpoints,
compiled engines, profiler dumps, raw benchmark folders, or large logs.

## What This Lane Is

This is the Hydralisk private canary lane for serving
`0xSero/GLM-5.2-504B` REAP/NVFP4 on OpenAgents Google Cloud infrastructure.
The current evidence proves a 4-GPU runtime envelope in two admitted shapes:
four selected RTX PRO 6000 GPUs inside an 8-GPU G4 fallback host, and a second
standalone 4 x G4 Spot replica. It does not yet prove on-demand or
reservation-backed `g4-standard-192` capacity is currently obtainable.

Final status for this packet:

- `private_canary`: yes
- `servable`: internally, behind private proxy and operator controls
- `eval_passed`: preliminary Terminal-Bench pilot only, not a final leaderboard
  claim
- `blocked`: public production promise, billing, customer routing, and
  on-demand or reservation-backed 4x G4 durability

## Upstream Model Facts

Source links:

- Model card: https://huggingface.co/0xSero/GLM-5.2-504B
- Report: https://huggingface.co/0xSero/GLM-5.2-504B/blob/main/REPORT.md
- SM120 serving reference: https://github.com/0xSero/glm-5.2-sm120

Pinned model:

- Repository: `0xSero/GLM-5.2-504B`
- Revision: `cb6b1e0451b9d560cda864f84187869c9a679712`
- License: MIT
- Architecture: `GlmMoeDsaForCausalLM`
- Model type: `glm_moe_dsa`
- Context window: 1,048,576 tokens
- Layers: 78
- Dense layers: 3
- MoE layers: 75
- MTP layers: 1
- Routed experts per MoE layer: 168
- Experts per token: 8
- Shared experts: 1
- Quantization: ModelOpt NVFP4 / `modelopt_fp4`
- REAP recovery: Router-KD
- Safetensor shards expected: 63
- Safetensors index metadata total: 318,247,808,128 bytes
- Local safetensor shard bytes in staged copy: 308,829,060,264 bytes

The 0xSero report identifies loop behavior as the main observed regression
after REAP pruning and Router-KD recovery. Hydralisk therefore defaults the
private proxy to:

```json
{
  "min_p": 0.05,
  "repetition_penalty": 1.05,
  "max_tokens": 1024,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

Use `repetition_penalty=1.10` only as an explicit loop-abatement bump for
workloads that show repetition. Reasoning experiments must opt into thinking
deliberately and must not commit hidden reasoning traces.

## Hardware Profile

Primary intended shape:

- Provider: GCE
- Project: `openagentsgemini`
- Zones: `us-central1-b`, `us-central1-f`
- Machine: `g4-standard-192`
- Accelerator: 4 x `nvidia-rtx-pro-6000`

First admitted fallback:

- Instance: `hydralisk-glm52-reap-504b-g4-8g-b-20260624214500`
- Zone: `us-central1-b`
- Machine: `g4-standard-384`
- Accelerator: 8 x `nvidia-rtx-pro-6000`
- Active GPUs for this lane: `0,1,2,3`
- Public HTTPS origin: reserved static address, value not tracked
- Network IP: private VPC address, value not tracked
- Provisioning: Spot
- Termination action: Stop
- Boot disk: 1500 GB Hyperdisk Balanced
- Boot disk auto-delete: false

Second admitted standalone canary:

- Instance: `hydralisk-glm52-reap-504b-g4-4g-b-20260625154532`
- Zone: `us-central1-b`
- Machine: `g4-standard-192`
- Accelerator: 4 x `nvidia-rtx-pro-6000`
- Public HTTPS origin: reserved static address, value not tracked
- Network IP: private VPC address, value not tracked
- Provisioning: Spot
- Termination action: Stop
- Max run duration: 604800 seconds
- Boot disk auto-delete: false
- Model staging: read-only mount of a cloned model disk from the first canary

Second endpoint evidence:
[`docs/evidence/2026-06-25-glm-52-reap-504b-second-endpoint.md`](evidence/2026-06-25-glm-52-reap-504b-second-endpoint.md)

Admission evidence:
[`docs/evidence/2026-06-24-glm-52-reap-504b-g4-admission.md`](evidence/2026-06-24-glm-52-reap-504b-g4-admission.md)

## Reproduce From A Clean G4 Host

1. Admit a G4 host.

```bash
ACTION=start RUN_ID=<run-id> \
  scripts/probe-glm-52-reap-504b-g4-gce.sh
```

Prefer the 4x target when capacity is available. Use the 8x fallback only when
the 4x `g4-standard-192` shape is capacity-blocked, and keep the claim boundary
explicit.

The 2026-06-25 second-endpoint attempt found on-demand 4x and 8x G4 capacity
exhausted in the tested zones, then admitted a 4x Spot host. Treat that as
useful burst capacity, not durable production capacity.

2. Stage the pinned checkpoint onto durable disk.

```bash
ACTION=start RUN_ID=<run-id> \
  scripts/stage-glm-52-reap-504b-gce.sh
```

The staging script downloads:

```bash
hf download 0xSero/GLM-5.2-504B \
  --revision cb6b1e0451b9d560cda864f84187869c9a679712 \
  --local-dir /opt/hydralisk/models/glm-5.2-504b
```

Use out-of-band Hugging Face credentials only if the environment requires
them. Do not commit those credentials or raw transfer logs.

For same-zone replica bring-up, an operator may clone an existing Hydralisk
model disk instead of downloading the checkpoint again. Mount the clone
read-only on the new VM and symlink `/opt/hydralisk/models/glm-5.2-504b` to the
cloned model directory. This is a staging shortcut only; each replica still
needs an independent proxy token, public origin, watchdog identity, and
public-safe smoke evidence.

3. Verify staging.

```bash
ACTION=status RUN_ID=<run-id> \
  scripts/stage-glm-52-reap-504b-gce.sh
```

Expected public-safe manifest facts:

- Complete: true
- Local safetensor shard count: 63
- Missing shard count: 0
- Unexpected shard count: 0
- Weight-map entries: 154,433

4. Launch raw vLLM.

```bash
ACTION=start RUN_ID=<run-id> \
  DOCKER_RESTART_POLICY=unless-stopped \
  MAX_MODEL_LEN=250000 \
  MAX_NUM_SEQS=2 \
  MAX_NUM_BATCHED_TOKENS=4096 \
  GPU_DEVICES=0,1,2,3 \
  TP_SIZE=4 \
  DCP_SIZE=4 \
  scripts/launch-glm-52-reap-504b-b12x-gce.sh
```

The launcher runs the pinned image:

```text
voipmonitor/vllm@sha256:ce23a9b075bd7138ce3b12ee29609b98606e5050e2def4a29bbb917ad96e5997
```

Core vLLM arguments:

```text
vllm serve /model
  --served-model-name glm-5.2-reap-504b-g4
  --host 127.0.0.1
  --port 8000
  --trust-remote-code
  --tensor-parallel-size 4
  --decode-context-parallel-size 4
  --quantization modelopt_fp4
  --kv-cache-dtype fp8
  --attention-backend B12X_MLA_SPARSE
  --moe-backend b12x
  --tool-call-parser glm47
  --reasoning-parser glm45
  --max-model-len 250000
  --max-num-seqs 2
  --max-num-batched-tokens 4096
  --gpu-memory-utilization 0.95
```

The launcher also sets the GLM DSA indexer override:

```text
FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
```

5. Poll raw readiness.

```bash
ACTION=status RUN_ID=<run-id> \
  scripts/launch-glm-52-reap-504b-b12x-gce.sh
```

Expected:

- `containerStatus=running`
- `restartPolicy=unless-stopped`
- `modelsEndpoint=ready`
- GPU memory near 93.4 GiB used on GPUs 0-3

6. Install the private proxy.

```bash
ACTION=install-systemd RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-private-proxy-gce.sh
```

The proxy should bind to a private listener on the host, requires a bearer
token for `/v1/models` and generation routes, and exposes public-safe `/health`,
`/hydralisk/v1/capabilities`, `/hydralisk/v1/metrics`, and receipt lookup.
When the public HTTPS front runs on the same VM and forwards to the private
proxy, bind the proxy to the VM's private interface rather than loopback. Keep
the private address out of tracked docs.

The capabilities and metrics endpoints also expose the public-safe replica
routing shape used by Khala pool selection. Set `REPLICA_REF` and
`REPLICA_PROFILE_REF` when installing the proxy; the GLM installer defaults the
primary host to `glm52-reap-primary-g4-tp4` and
`glm-reap-504b-g4-tp4-mtp2-rp105`. The dynamic metrics include inflight count,
singleflight/backpressure, last 429/busy accounting, host-local keep-warm
summary, provisioning class, `maxRunDuration` presence, watchdog status, and
reserved/draining flags. They intentionally do not expose endpoint URLs, public
or private IPs, bearer tokens, raw prompts, raw responses, weights, or raw
logs. See:
[`docs/evidence/2026-06-25-glm-52-reap-504b-replica-routing-metadata.md`](evidence/2026-06-25-glm-52-reap-504b-replica-routing-metadata.md)

7. Check and smoke the private proxy.

```bash
ACTION=status RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-private-proxy-gce.sh

ACTION=smoke RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-private-proxy-gce.sh
```

8. If Khala needs to reach the lane from Cloudflare Workers, expose only the
bearer-gated proxy through HTTPS.

```bash
ACTION=setup RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-public-https-gce.sh

ACTION=smoke RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-public-https-gce.sh
```

The public HTTPS script reserves or reuses a regional static address, attaches
it to the GLM host, adds a tag-targeted `80/443` firewall rule, installs Caddy,
and fronts only `/health`, `/v1/*`, and `/hydralisk/*`. Raw vLLM remains
host-local. The actual origin URL and bearer token belong only in secret
stores; tracked docs and issue comments should use the shape
`https://<operator-secret-hostname>`. The generated Caddyfile uses an ordered
`route` block so the allowlisted reverse proxy is evaluated before the
fail-closed `404` fallback.

Public HTTPS evidence:
[`docs/evidence/2026-06-25-glm-52-reap-504b-public-https-origin.md`](evidence/2026-06-25-glm-52-reap-504b-public-https-origin.md)

9. Install the durable canary watchdog and keep-warm path when the lane should
stay up for Khala.

```bash
ACTION=setup RUN_ID=<run-id> \
  scripts/install-glm-52-reap-504b-durable-canary-gce.sh

ACTION=smoke RUN_ID=<run-id> \
  scripts/install-glm-52-reap-504b-durable-canary-gce.sh
```

This keeps the current Spot host labeled honestly as a canary, but removes the
manual restart gap: the VM has an external Cloud Scheduler -> Cloud Run
watchdog that starts it after STOP, and the host has enabled Docker, proxy, and
Caddy. Keep-warm units are installed, but the timer should stay disabled during
decision-grade Terminal-Bench runs because warm probes contend with the
singleflight model lane. After the benchmark owner clears the run, enable warm
probes explicitly:

```bash
ENABLE_KEEPWARM_TIMER=1 ACTION=setup-keepwarm RUN_ID=<run-id> \
  scripts/install-glm-52-reap-504b-durable-canary-gce.sh

ALLOW_MODEL_KEEPWARM_SMOKE=1 ACTION=smoke RUN_ID=<run-id> \
  scripts/install-glm-52-reap-504b-durable-canary-gce.sh
```

If Spot capacity is unavailable, the watchdog retries on schedule; it cannot
guarantee zonal stock.

For more than one GLM replica, set distinct global watchdog names with the
script environment variables (`WATCHDOG_SERVICE_ACCOUNT_NAME`,
`WATCHDOG_ROLE_ID`, `WATCHDOG_RUN_JOB`, and `WATCHDOG_SCHEDULER_JOB`). The
keep-warm systemd units are host-local, so their default unit names are safe on
separate VMs; use a distinct `KEEPWARM_LOG_DIR` when you need operator log paths
to make the replica identity obvious. Reusing the default watchdog names would
collapse multiple replicas into one global control-plane target.

Durable canary evidence:
[`docs/evidence/2026-06-25-glm-52-reap-504b-durable-canary.md`](evidence/2026-06-25-glm-52-reap-504b-durable-canary.md)

Second standalone endpoint evidence:
[`docs/evidence/2026-06-25-glm-52-reap-504b-second-endpoint.md`](evidence/2026-06-25-glm-52-reap-504b-second-endpoint.md)

10. Run Terminal-Bench only through the private proxy and public-safe summary
reducer.

```bash
export HYDRALISK_TB_BASE_URL=http://127.0.0.1:8080
export HYDRALISK_TB_MODEL=glm-5.2-reap-504b-g4
export HYDRALISK_BEARER_TOKEN=<remote proxy token, never committed>

scripts/run-glm-52-reap-terminal-bench-preflight.sh
```

Use IAP/SSH tunneling for local operator access. Keep `n-concurrent=1` until a
separate Hydralisk concurrency receipt admits more.

Run Harbor with raw job output outside the repository, then reduce the job
directory with Hydralisk's public-safe reducer. The reducer reads only
job/trial `result.json` and `config.json` files; it does not read agent
trajectories, terminal panes, recordings, raw logs, verifier artifacts, or
environment files.

```bash
harbor run \
  --dataset terminal-bench@2.0 \
  --agent terminus-2 \
  --model openai/glm-5.2-reap-504b-g4 \
  --n-concurrent 1 \
  --jobs-dir /tmp/hydralisk-terminal-bench \
  --job-name glm52-reap-mtp2-full-$(date -u +%Y%m%d%H%M%S) \
  --env-file <local env file with private proxy URL + token, never committed> \
  --yes \
  --quiet

uv run hydralisk-terminal-bench-summary \
  --harbor-job-dir /tmp/hydralisk-terminal-bench/<job-name> \
  --output-dir docs/evidence \
  --json-name <public-safe-receipt>.json \
  --markdown-name <public-safe-report>.md \
  --runner-version 0.15.0 \
  --omit-min-p \
  --claim-status decision_grade_candidate
```

## Evidence Summary

Load smoke:

- Run ID: `20260624224500`
- Result: pass
- Visible completion characters: 18
- Prompt tokens: 22
- Completion tokens: 9
- Total tokens: 31
- First visible token latency: 0.2376 seconds
- Total request time: 0.4610 seconds

Tuned serving envelope:

- Max context: 250,000 tokens
- Max sequences: 2
- Max batched tokens: 4096
- Full-context concurrency admitted: 1 request
- Short concurrent utility requests tested: 2
- MTP: opt-in only, not default

Private proxy:

- Run ID: `20260624235325`
- Systemd unit: `hydralisk-glm52-reap-private-proxy.service`
- Health/models/metrics: ready
- Smoke status: pass
- Smoke run ref: `hydralisk-run-91b8c576959c4e6d8633626c7adf9d36`
- Metrics after smoke: 1 request, 1 response, 0 errors

Public HTTPS origin:

- Status: pass
- Public origin shape: `https://<operator-secret-hostname>`
- Front: Caddy `v2.11.4`
- Caddy upstream: HTTP/1.1 to the bearer-gated Hydralisk proxy
- Health/models: ready
- Authenticated completion HTTP status: 200
- Completion wall time: 0.988 seconds
- Total tokens: 13
- Endpoint value and bearer token: not tracked

Durable canary:

- Status: Spot auto-restart canary
- Max run duration in current instance scheduling: not present
- Boot disk auto-delete: false
- Raw vLLM restart policy: `unless-stopped`
- Docker, Caddy, and proxy: enabled
- Keep-warm timer: installed; intentionally disabled during decision-grade
  Terminal-Bench runs, then enabled every 4 minutes after the run
- External watchdog: Cloud Scheduler every 5 minutes triggering a Cloud Run job
  that conditionally starts the VM only when it is not running
- Evidence:
  [`docs/evidence/2026-06-25-glm-52-reap-504b-durable-canary.md`](evidence/2026-06-25-glm-52-reap-504b-durable-canary.md)

Second standalone endpoint:

- Status: pass for a second Spot 4 x G4 endpoint
- On-demand admission: capacity-exhausted in tested 4x and 8x zones
- Spot admission: `g4-standard-192`, 4 x RTX PRO 6000
- Model staging: cloned model disk mounted read-only
- Public HTTPS origin: pass, bearer-gated, value not tracked
- Durable canary: distinct global watchdog installed; host-local keep-warm
  timer enabled on the second VM
- Warm smoke: HTTP 200, 0.428 seconds wall, 25 total tokens
- Single-request streaming median: 0.281 seconds TTFT, 46.7 completion tok/s
  including TTFT, 49.4 completion tok/s excluding TTFT on the 160-token case
- Same-endpoint concurrency: one request passed, one was rejected with 429,
  matching the singleflight policy
- Evidence:
  [`docs/evidence/2026-06-25-glm-52-reap-504b-second-endpoint.md`](evidence/2026-06-25-glm-52-reap-504b-second-endpoint.md)

Terminal-Bench 2.0 pilot:

- Total tasks: 89
- Solved: 60
- Failing so far: 25
- Environment-broken: 2
- Not started: 2
- Solved / attempted: 69.0%
- Solved / properly attempted: 70.6%
- Status: preliminary pilot, not final leaderboard claim

## What Is Not Promised

- Public production SLA
- Ungated public endpoint
- Billing, credits, customer routing, settlement, or payout
- OpenAgents product-surface claim outside Hydralisk evidence
- Repeatable standalone 4x G4 capacity availability
- On-demand 4x G4 capacity availability
- Two concurrent full-250K requests
- Terminal-Bench leaderboard finality
- Customer or third-party traffic

## Recovery

After a Spot stop, planned VM stop, or host refresh:

1. Start the GCE instance and verify the durable 1.5T disk is attached.
2. Run raw vLLM `ACTION=status`.
3. If `/v1/models` is not ready, run raw vLLM `ACTION=start`.
4. Poll raw vLLM until `modelsEndpoint=ready`.
5. Run proxy `ACTION=install-systemd` or `ACTION=restart-systemd`.
6. Run proxy `ACTION=status`.
7. Run proxy `ACTION=smoke`.
8. If using the public HTTPS origin, run the public HTTPS smoke and verify the
   redacted summary before arming product secrets.
9. Commit or comment only the public-safe summary artifacts.

Do not force a full Spot stop/start solely for a doc refresh. Spot capacity
re-admission is an external cloud-capacity risk, not a model-serving property.

## Receipt Discipline

Public-safe receipts may contain run IDs, hashes, token counts, public
hardware/runtime metadata, aggregate timings, and sanitized failure classes.
They must not contain secrets, raw prompts, raw responses, hidden reasoning,
private source, weights, compiled engines, raw benchmark logs, or profiler
dumps.
