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

Speed update: after the benchmark matrix below, the private canary was promoted
to the MTP-2/no-`min_p` speed profile. See:
[`2026-06-25-glm-52-reap-504b-mtp2-speed-gate.md`](2026-06-25-glm-52-reap-504b-mtp2-speed-gate.md)

Post-speed-gate raw vLLM status at `2026-06-25T03:31:56Z`:

- Container: running
- Docker restart policy: `unless-stopped`
- `/v1/models`: ready
- Raw bind: `127.0.0.1:8000`
- MTP: enabled
- Speculative tokens: 2
- Default `min_p`: omitted for MTP compatibility
- Model dir: `/opt/hydralisk/models/glm-5.2-504b`
- HF cache dir: `/var/lib/hydralisk/huggingface`
- Root disk: `1.5T`, `338G` used, `1.1T` available

GPU memory:

| GPU | Used MiB | Free MiB |
| --- | ---: | ---: |
| 0 | 93475 | 3776 |
| 1 | 93475 | 3776 |
| 2 | 93475 | 3776 |
| 3 | 93471 | 3780 |
| 4 | 0 | 97250 |
| 5 | 0 | 97250 |
| 6 | 0 | 97250 |
| 7 | 0 | 97250 |

Private proxy status at `2026-06-25T03:31:56Z`:

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

## Controlled Benchmark Run

Benchmark window: `2026-06-25T02:28:57Z` to `2026-06-25T02:48:20Z`.

The keep-warm timer was stopped during the controlled benchmark window and
restored afterward. All benchmark requests ran on the VM against raw localhost
vLLM endpoints, so IAP/SSH control-plane latency is not included in TTFT or
tokens/sec. The copied benchmark artifacts are public-safe summaries only and
were left outside the tracked repo under `.hydralisk/`; this note records the
human-readable receipt.

Public-safety boundary for the benchmark rows: prompt text and model text are
not recorded. The source artifacts contain prompt hashes, visible completion
hashes, token counts, aggregate timings, endpoint labels, and GPU memory
summaries only.

Profiles tested:

| Profile | GPUs | Runtime shape | Raw endpoint(s) | Result |
| --- | ---: | --- | --- | --- |
| `4x-tp-current` | 4 | One tensor-parallel vLLM process, TP=4, DCP=4 | `127.0.0.1:8000` | Passed |
| `8x-tp-single` | 8 | One tensor-parallel vLLM process, TP=8, DCP=8 | `127.0.0.1:8000` | Passed |
| `dual-4x-replicas` | 8 | Two independent TP=4/DCP=4 vLLM processes | `127.0.0.1:8000`, `127.0.0.1:8001` | Passed |

Launch/readiness observations:

| Event | Started | Ready | Approx. ready delay | Notes |
| --- | --- | --- | ---: | --- |
| 8x TP relaunch | `2026-06-25T02:31:31Z` | `2026-06-25T02:34:51Z` | 3m 20s | Replaced the live 4x process during the test |
| 4x primary restore | `2026-06-25T02:38:29Z` | `2026-06-25T02:42:13Z` | 3m 44s | Restored the routed primary process |
| 4x replica B launch | `2026-06-25T02:42:29Z` | `2026-06-25T02:46:18Z` | 3m 49s | Loaded second full resident copy on GPUs 4-7 |

The current live state after the benchmark is back to the routed 4x canary:
only GPUs 0-3 are resident, the private proxy points at port `8000`, and
replica B was stopped because it was not yet behind a production router. A
follow-up speed gate then promoted that 4x canary to MTP-2/no-`min_p`.

### Single-Request Decode

Median results from streaming requests:

| Profile | Case | Prompt tokens | Completion tokens | TTFT | Wall | Completion tok/s excl. TTFT | Completion tok/s incl. TTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4x TP | Tiny, max 32 | 21 | 32 | 0.249s | 1.115s | 36.93 | 28.70 |
| 8x TP | Tiny, max 32 | 21 | 31 | 0.258s | 1.143s | 36.04 | 27.74 |
| 4x TP | Decode, max 160 | 27 | 160 | 0.251s | 4.719s | 35.81 | 33.86 |
| 8x TP | Decode, max 160 | 27 | 160 | 0.232s | 4.811s | 34.95 | 33.26 |
| 4x TP | Decode, max 512 | 34 | 512 | 0.258s | 14.550s | 35.82 | 35.19 |
| 8x TP | Decode, max 512 | 34 | 512 | 0.409s | 15.117s | 34.81 | 33.87 |

Plain-language read: the 8-GPU tensor-parallel process did not make interactive
generation faster in this benchmark. Short TTFT was similar, and decode
throughput was slightly lower than the 4-GPU process. This is consistent with
the topology: adding more GPUs also adds more per-token synchronization.

The first tiny request after a fresh 8x or second-replica load showed a warmup
spike, so tiny medians are more useful than tiny means. The 8x tiny case had
one first-run TTFT outlier around 23.6s. The dual-replica tiny case had one
first-run TTFT outlier on each replica around 25-26s. Later tiny requests were
back around 0.24-0.26s TTFT.

### Long-Prefill Behavior

Single long-prefill runs:

| Profile | Case | Prompt tokens | Completion tokens | TTFT | Wall | Completion tok/s excl. TTFT | Completion tok/s incl. TTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4x TP | Synthetic prefill, ~9.6k prompt tokens | 9,613 | 42 | 41.683s | 42.751s | 39.35 | 0.98 |
| 8x TP | Synthetic prefill, ~9.6k prompt tokens | 9,613 | 46 | 39.351s | 40.581s | 37.40 | 1.13 |
| 4x TP | Synthetic prefill, ~38.4k prompt tokens | 38,398 | 32 | 35.017s | 35.804s | 40.65 | 0.89 |
| 8x TP | Synthetic prefill, ~38.4k prompt tokens | 38,398 | 32 | 36.776s | 37.585s | 39.57 | 0.85 |

These runs are deliberately synthetic and should be interpreted as prefill
pressure probes, not application prompts. The surprising ordering between the
9.6k and 38.4k cases suggests cache/warmup effects in the B12X/vLLM path; do
not infer a precise prefill scaling curve from one sample. The useful result is
that long prompts are dominated by TTFT/prefill time, while post-TTFT decode
stays near 37-41 completion tok/s.

### Concurrency And Capacity

Concurrent decode probe:

| Shape | Concurrent requests | Completion tokens | Wall | Aggregate completion tok/s | Per-request behavior |
| --- | ---: | ---: | ---: | ---: | --- |
| One 4x TP process | 2 to same replica | 320 | 6.414s | 49.89 | Each request slowed to about 25 tok/s including TTFT |
| One 8x TP process | 2 to same replica | 309 | 6.318s | 48.91 | Similar aggregate to 4x TP, not a clear win |
| Two 4x replicas | 1 per replica | 320 | 4.762s | 67.20 | Each request stayed near single-request speed |

This is the important capacity result. The same eight physical GPUs produce
better practical throughput as two independent 4-GPU replicas than as one
8-GPU tensor-parallel process for these interactive decode workloads. The
single 8x TP process may still be useful for future very-large-KV or batch
experiments, but it did not improve the current Khala canary path.

## Why 4 GPUs And 8 GPUs Behave Differently

This host has eight RTX PRO 6000 GPUs, but they are not one flat low-latency
GPU fabric. `nvidia-smi topo -m` showed:

- GPUs 0-3 share NUMA node 0.
- GPUs 4-7 share NUMA node 1.
- Within each side, some links are `PIX` or `NODE`.
- Between GPU 0-3 and GPU 4-7, links are `SYS`.
- There is no NVLink fabric reported.

In plain English: the current 4-GPU profile keeps one request inside one side
of the machine. The 8-GPU tensor-parallel profile splits every token step
across both sides of the machine. That can add bandwidth and memory headroom,
but it also means every generated token pays extra coordination cost across
the CPU/PCIe/NUMA boundary.

That is why "use eight GPUs" is not the same as "make this twice as fast":

- For one interactive request, more tensor-parallel ranks can increase
  synchronization overhead.
- For long prompts, the real cost is often TTFT/prefill rather than decode.
- For multiple users, independent replicas can preserve low per-request
  latency while increasing aggregate capacity.
- For giant contexts or larger batches, 8x TP may still become attractive, but
  it needs a separate workload-specific gate.

## Capacity Model

Current routed Khala canary:

- One 4-GPU resident process on GPUs 0-3.
- Proxy singleflight limit: 1.
- Honest full-context policy: one 250K-token request at a time.
- Short raw vLLM concurrency passed at two simultaneous requests, but the
  proxy intentionally serializes until a product-level routing and fairness
  policy exists.

Capacity options:

| Option | Uses provisioned GPUs | Routed today | Best use | Measured behavior | Main drawback |
| --- | ---: | --- | --- | --- | --- |
| Current 4x TP canary | 4 of 8 | Yes | Lowest-risk private Khala canary | ~35-36 decode tok/s; ~0.25s short TTFT after warmup | Pays for unused GPUs on 8x fallback host |
| One 8x TP process | 8 of 8 | Not left live | Larger KV/batch experiments | Similar or slightly lower decode tok/s than 4x; no concurrency win in this probe | Cross-NUMA synchronization and more complex tuning |
| Two 4x replicas | 8 of 8 | Not yet | Higher aggregate interactive capacity | 67.2 aggregate tok/s for one concurrent request per replica | Needs proxy/load-balancer work, health checks, and routing policy |

The clean next production move is not "flip TP to 8" by default. It is to add a
proper two-replica router for ports `8000` and `8001`, then keep both replicas
warm, expose a single private Khala endpoint, and admit two singleflight lanes
with per-lane health and backpressure.

## Cost Model

Source: Google Cloud accelerator-optimized VM pricing page, checked
2026-06-25 with Iowa (`us-central1`) selected:
<https://cloud.google.com/products/compute/pricing/accelerator-optimized>

The page row used for this host class lists:

| Shape | GPUs | vCPU | Memory | On-demand | Current spot | DWS flex-start | 1-year CUD | 3-year CUD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `g4-standard-192` | 4 | 192 | 720GiB | $17.99972/hr | $3.69344/hr | $9.00000/hr | $12.42000/hr | $7.91780/hr |
| `g4-standard-384` | 8 | 384 | 1440GiB | $35.99944/hr | $7.38688/hr | $18.00000/hr | $24.84000/hr | $15.83560/hr |

Monthly estimate uses 730 hours:

| Shape | On-demand / month | Current spot / month | DWS flex / month | 1-year CUD / month | 3-year CUD / month |
| --- | ---: | ---: | ---: | ---: | ---: |
| `g4-standard-192` | $13,139.80 | $2,696.21 | $6,570.00 | $9,066.60 | $5,779.99 |
| `g4-standard-384` | $26,279.59 | $5,392.42 | $13,140.00 | $18,133.20 | $11,559.99 |

Important caveats:

- The live VM is `g4-standard-384` Spot with a 6-hour max run duration and
  `STOP` termination action, so the pricing-page spot estimate is the closest
  public comparator, not a bill.
- These rows do not include persistent disk, snapshot, network egress, logging,
  or any product-layer cost outside the VM shape.
- Because the current canary is on an 8-GPU fallback host, using only GPUs 0-3
  still burns the 8-GPU VM while it is up.
- If standalone `g4-standard-192` capacity becomes available, a single 4x
  canary should cost roughly half as much as this fallback host.
- If the 8-GPU fallback must stay up anyway, two 4x replicas use the already
  paid-for idle half and materially improve served capacity once routed.

Plain-language cost translation:

- Current routed state: operationally behaves like a 4-GPU service, but costs
  like the 8-GPU VM because that is what GCE actually allocated.
- 8x TP: same host cost as current fallback, but no measured latency/capacity
  advantage for the benchmarked interactive workload.
- Dual 4x replicas: same host cost as current fallback, better aggregate
  interactive throughput, but requires router/proxy integration before it is a
  real Khala serving lane.

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
- Post-benchmark timer state: active
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
- Controlled 4x TP, 8x TP, and dual-4x-replica benchmark evidence on the same
  8-GPU G4 host.

Not admitted:

- Public internet endpoint.
- Public SLA.
- Customer routing or billing.
- Standalone 4x G4 capacity availability.
- Multi-tenant concurrency beyond the current singleflight proxy limit.
- 8x TP as the default serving profile.
- Dual-replica routing until the proxy/load-balancer work is implemented.
