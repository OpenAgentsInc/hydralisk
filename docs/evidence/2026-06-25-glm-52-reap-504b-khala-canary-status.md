# GLM-5.2 504B REAP Khala canary status

Date: 2026-06-25

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Runbook:
[`docs/glm-5.2-reap-504b-g4-runbook.md`](../glm-5.2-reap-504b-g4-runbook.md)

Public-safety boundary: this note contains live service status, public-safe
hashes, token counts, aggregate timings, throughput, and private-network
endpoint metadata only. It contains no bearer token, model-provider
credentials, raw prompts, raw responses, private source, hidden reasoning
traces, weights, checkpoints, compiled engines, profiler dumps, or raw logs.

## Why GCE Did Not Allocate The 4x Shape

The failed primary admission attempts for standalone 4x G4 were cloud capacity
failures, not script or quota failures. GCE returned
`ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` for `g4-standard-192` with
4 x `nvidia-rtx-pro-6000` in both `us-central1-b` and `us-central1-f`, across
Spot and Standard attempts.

Hydralisk therefore used the explicit fallback:

- Instance: `hydralisk-glm52-reap-504b-g4-8g-b-20260624214500`
- Zone: `us-central1-b`
- Machine: `g4-standard-384`
- GPUs: 8 x RTX PRO 6000 physically present
- Active GPUs for this canary: `0,1,2,3`

The 4-GPU runtime claim remains a runtime-envelope claim on four selected GPUs
inside the 8x fallback host. It is not a fresh standalone 4x capacity claim.

Live quota/capacity context checked on 2026-06-25:

- `us-central1` CPU quota had headroom.
- The project can see `nvidia-rtx-pro-6000` accelerator types in multiple
  zones and regions.
- The blocker evidenced for the primary shape remains zonal resource-pool
  stockout.

## Current Live Status

Raw vLLM status at `2026-06-25T02:07:24Z`:

- Container: running
- Docker restart policy: `unless-stopped`
- `/v1/models`: ready
- Raw bind: `127.0.0.1:8000`
- Model dir: `/opt/hydralisk/models/glm-5.2-504b`
- HF cache dir: `/var/lib/hydralisk/huggingface`
- Root disk: `1.5T`, `338G` used, `1.1T` available

GPU memory:

| GPU | Used MiB | Free MiB |
| --- | ---: | ---: |
| 0 | 93425 | 3826 |
| 1 | 93423 | 3828 |
| 2 | 93425 | 3826 |
| 3 | 93425 | 3826 |
| 4 | 0 | 97250 |
| 5 | 0 | 97250 |
| 6 | 0 | 97250 |
| 7 | 0 | 97250 |

Private proxy status at `2026-06-25T02:07:27Z`:

- Systemd unit: `hydralisk-glm52-reap-private-proxy.service`
- State: active
- Bind: `10.128.0.38:8080`
- Public bind: false
- Health: ready
- Authenticated models endpoint: ready
- Metrics endpoint: ready
- Inflight limit: 1
- Current inflight: 0

The VM has no external IP. The default VPC internal firewall allows traffic
from `10.128.0.0/9`, so Khala-side services on the same private GCE network can
reach the proxy at:

```text
http://10.128.0.38:8080
```

The bearer token remains only on the VM in:

```text
/var/lib/hydralisk/glm52-reap-private-proxy/bearer-token
```

## Streaming Performance Probe

Probe time: `2026-06-25T02:04:20Z`

Request shape:

- Endpoint: private proxy
- Model alias: `openagents/glm-5.2-reap-504b`
- Streaming: true
- Requested max tokens: 160
- Temperature: 0.2
- Prompt SHA-256:
  `87e9119d6797d0a41165a0dd26f0fcd99219d796ebb7120711db7bb35288148d`
- Visible completion SHA-256:
  `66b47fefb6b5fb26a6a61687b0a77cc17c4dc762c29dd2a2c59858117a6e3df5`

Result:

- HTTP status: 200
- Proxy run ref: `hydralisk-run-0be4f71361ff43f99b2f56f64b8d208f`
- Prompt tokens: 42
- Completion tokens: 156
- Total tokens: 198
- Visible completion characters: 587
- SSE data events: 158
- Wall time: 4.975 s
- First byte: 0.618 s
- TTFT: 0.618 s
- Decode window after TTFT: 4.357 s
- Completion tokens/sec excluding TTFT: 35.80
- Completion tokens/sec including TTFT: 31.36
- Visible chars/sec including TTFT: 117.98

## Keep-Warm

Installed live on 2026-06-25:

- Timer: `hydralisk-glm52-reap-keepwarm.timer`
- Service: `hydralisk-glm52-reap-keepwarm.service`
- Cadence: every 4 minutes
- Base URL: `http://10.128.0.38:8080`
- Log directory: `/var/log/hydralisk/glm52-reap-keepwarm`
- Latest public-safe output:
  `/var/log/hydralisk/glm52-reap-keepwarm/latest-public.json`

First keep-warm run:

- Checked at: `2026-06-25T02:07:02Z`
- HTTP status: 200
- Proxy run ref: `hydralisk-run-82aa640b51a740f7b3b0b608829380eb`
- Prompt tokens: 21
- Completion tokens: 10
- Total tokens: 31
- Wall time: 0.907 s
- First byte: 0.653 s
- TTFT: 0.653 s
- Decode window after TTFT: 0.254 s
- Completion tokens/sec excluding TTFT: 39.39
- Completion tokens/sec including TTFT: 11.03

The warm probe is intentionally small and singleflight-safe. If the proxy is
busy, a warm request may fail or be skipped without widening public exposure.

## Claim Boundary

This is still a private Khala canary, not a public production route.

Admitted:

- Internal GCE private-network access at `10.128.0.38:8080`.
- Bearer-authenticated OpenAI-compatible proxy.
- Warm-resident 4-GPU GLM-5.2 REAP service.
- Observed streaming TTFT and throughput from a public-safe synthetic probe.

Not admitted:

- Public internet endpoint.
- Public SLA.
- Customer routing or billing.
- Standalone 4x G4 capacity availability.
- Multi-tenant concurrency beyond the current singleflight proxy limit.
