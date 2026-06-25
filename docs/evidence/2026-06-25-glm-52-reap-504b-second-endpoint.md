# GLM-5.2 504B REAP second endpoint

Date: 2026-06-25

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/95

Runbook:
[`docs/glm-5.2-reap-504b-g4-runbook.md`](../glm-5.2-reap-504b-g4-runbook.md)

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Public-safety boundary: this note contains service shape, public-safe hashes,
token counts, aggregate timings, and cost estimates only. It contains no public
origin hostname, public IP address, private VPC address, bearer token, raw
prompt, raw response, hidden reasoning trace, provider credential, model
weight, checkpoint, compiled engine, profiler dump, or raw log.

## Result

PASS for a second independent GLM-5.2 REAP endpoint. This endpoint was brought
up without restarting, warming, or sending model traffic through the first
endpoint that is reserved for the live Harbor Terminal-Bench run.

This is the current capacity boundary:

- The second endpoint is available for inference serving from the Hydralisk
  side.
- It is a Spot 4 x G4 canary, not on-demand or reservation-backed capacity.
- It is warmed by its host-local keep-warm timer and watched by its own Cloud
  Scheduler watchdog resources.
- It remains singleflight. Same-endpoint concurrency is intentionally rejected
  with 429 so one request keeps the full fast lane.
- Multiple-developer reliability requires routing across independent replicas,
  not pushing concurrent requests into one replica.

## Admission

On-demand capacity was tested first because it is the cleaner durability class
for Khala serving. Google Cloud returned zonal stockout for the tested G4
shapes:

| Provisioning | Shape | Zones tested | Result |
| --- | --- | --- | --- |
| On-demand | `g4-standard-192` | `us-central1-b`, `us-central1-f` | capacity exhausted |
| On-demand | `g4-standard-384` | `us-central1-b`, `us-central1-f` | capacity exhausted |
| Spot | `g4-standard-192` | `us-central1-b` | admitted |

Admitted endpoint:

- Instance: `hydralisk-glm52-reap-504b-g4-4g-b-20260625154532`
- Zone: `us-central1-b`
- Machine: `g4-standard-192`
- Accelerator: 4 x `nvidia-rtx-pro-6000`
- Provisioning: Spot
- Termination action: Stop
- Max run duration: 604800 seconds
- Automatic restart: false
- Boot disk auto-delete: false

## Model Staging

The pinned model was staged by cloning the existing same-zone Hydralisk model
disk instead of re-downloading from Hugging Face. The clone was attached to the
new VM and mounted read-only, then `/opt/hydralisk/models/glm-5.2-504b` was
symlinked to the cloned model directory.

Public-safe verification:

- Safetensor shard count: 63
- Model directory size: 288 GB
- Safetensors index: present
- Raw weights: not committed

## vLLM Readiness

The endpoint uses the same MTP-2/no-`min_p` serving profile as the faster
Khala canary lane:

- Tensor parallel size: 4
- Decode-context parallel size: 4
- Max model length: 250,000 tokens
- Max sequences: 2
- Max batched tokens: 4096
- MTP speculative tokens: 2
- KV cache dtype: FP8
- Quantization: ModelOpt FP4
- Attention backend: B12X MLA sparse
- MoE backend: B12X

Startup observations:

- Weight load time: 498.22 seconds
- Model loading time per worker: about 517.5 seconds
- Engine init and warmup time: 138.91 seconds
- Compilation time inside warmup: 44.88 seconds
- GPU KV cache size: 377,942 tokens
- Max concurrency at 250K context: 1.51x
- GPU memory after readiness: about 93,341 MiB used per GPU, with about
  3,908 MiB free per GPU

## Proxy And Public HTTPS

The private Hydralisk proxy is bearer-gated and independently tokened from the
first lane. It was rebound to the VM private interface so the local Caddy front
can reach it while raw vLLM remains host-local.

Private proxy smoke:

- Status: pass
- HTTP status: 200
- Run ref: `hydralisk-run-eb09f7d154794727837ffec982b98ac1`
- Wall time: 15.250 seconds
- Prompt tokens: 22
- Completion tokens: 9
- Total tokens: 31
- Singleflight: true
- Max inflight requests: 1

The first public HTTPS setup attempt exposed a reusable script bug: Caddy's
fallback `respond 404` could win before the reverse proxy. The script now
wraps the reverse proxy and fallback response in an ordered `route` block.

Public HTTPS smoke after the route fix:

- Status: pass
- Public origin shape: `https://<operator-secret-hostname>`
- Origin host SHA-256:
  `43300002df018dc379363b6e53bc3f5d0c94bf7d6d555c6ee376ff1b8ea1527d`
- Health/models: ready
- Authenticated completion HTTP status: 200
- Wall time: 0.924 seconds
- Prompt tokens: 11
- Completion tokens: 2
- Total tokens: 13

## Keep-Warm And Watchdog

The second endpoint uses distinct global watchdog names so it does not collide
with the first canary. The keep-warm service and timer are host-local, so they
use the default unit names on the second VM while writing to a second-endpoint
log directory:

- Cloud Run job: `hydralisk-glm52-reap-watchdog-second`
- Cloud Scheduler job: `hydralisk-glm52-reap-watchdog-second-5m`
- Watchdog service account:
  `hydralisk-glm52-reap-wd2@openagentsgemini.iam.gserviceaccount.com`
- Watchdog custom role: `hydraliskGlm52Watchdog2`
- Keep-warm timer: `hydralisk-glm52-reap-keepwarm.timer`
- Keep-warm log directory:
  `/var/log/hydralisk/glm52-reap-keepwarm-second`

Latest explicit durable smoke:

- Cloud Run execution: `hydralisk-glm52-reap-watchdog-second-db9nb`
- Keep-warm timestamp: `2026-06-25T16:12:17Z`
- HTTP status: 200
- Wall time: 0.428 seconds
- Prompt tokens: 17
- Completion tokens: 8
- Total tokens: 25

Keep-warm is enabled on this second endpoint because it is not participating
in the live Harbor Terminal-Bench run. Keep-warm remains disabled on the first
endpoint until that benchmark owner clears the lane.

Fresh control-plane check after publishing this evidence:

- Instance status: `RUNNING`
- Docker: enabled and active
- Caddy: enabled and active
- Private proxy: enabled and active
- Keep-warm timer: enabled, next run scheduled every 4 minutes
- Latest keep-warm timestamp: `2026-06-25T16:20:30Z`
- Latest keep-warm HTTP status: 200

## Streaming Benchmark

Benchmark window: `2026-06-25T16:14:23Z`

Path: private Hydralisk proxy on the second VM, bearer token read locally on
the host and not printed.

Single-request median:

- TTFT: 0.281 seconds
- Wall time: 3.577 seconds
- Completion tokens: 160
- Completion tok/s including TTFT: 46.71
- Completion tok/s excluding TTFT: 49.37

Measured cases:

| Case | TTFT | Wall | Completion tokens | Completion tok/s incl. TTFT | Completion tok/s excl. TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| warmup-64 | 0.288s | 1.589s | 64 | 40.27 | 49.16 |
| single-160-1 | 0.282s | 3.478s | 160 | 46.00 | 50.06 |
| single-160-2 | 0.266s | 3.145s | 160 | 50.87 | 55.56 |
| single-160-3 | 0.281s | 3.676s | 160 | 43.53 | 47.13 |
| single-512-1 | 0.281s | 10.799s | 512 | 47.41 | 48.68 |

Same-endpoint concurrency probe:

- Attempted concurrent requests: 2
- Passed: 1
- Rejected: 1
- Rejected status: 429
- Rejected wall time: 0.017 seconds

Plain-language read: one 4 x G4 GLM replica is one fast interactive slot at
the current 250K profile. It should not be treated as a many-user concurrent
server. To serve multiple developers at speed, Khala should route across an
endpoint pool and pick an idle warmed replica. If all replicas are busy, Khala
should queue, fail over, or return honest backpressure instead of silently
stacking requests on one GPU process.

## Cost Reference

Monthly estimate uses the same 730-hour pricing table from the canary status
evidence:

| Shape | GPUs | Spot / month | On-demand / month | DWS flex / month |
| --- | ---: | ---: | ---: | ---: |
| `g4-standard-192` | 4 | $2,696.21 | $13,139.80 | $6,570.00 |
| `g4-standard-384` | 8 | $5,392.42 | $26,279.59 | $13,140.00 |

This endpoint costs like the 4-GPU row while it is admitted as Spot. A
reliable Khala pool made of two on-demand 4-GPU replicas would be roughly
`2 x $13,139.80/month` before storage, egress, and control-plane overhead.
Two Spot replicas are far cheaper, but Spot does not guarantee re-admission
after preemption or planned stops.

## Claim Boundary

Admitted:

- Second independent 4 x G4 GLM-5.2 REAP endpoint.
- Authenticated public HTTPS origin shape for Worker integration.
- Warmed private-proxy throughput around 47 completion tok/s including TTFT.
- Distinct global watchdog resources plus host-local keep-warm on the second VM.
- Public-safe evidence for a cloned-disk staging path.

Not admitted:

- Public product SLA.
- On-demand or reservation-backed G4 availability.
- More than one in-flight generation per endpoint.
- Worker-side Khala arming or deployment.
- Customer traffic, billing, settlement, or public product promise.
