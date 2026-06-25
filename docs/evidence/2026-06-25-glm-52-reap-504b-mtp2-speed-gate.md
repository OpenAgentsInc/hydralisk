# GLM-5.2 504B REAP MTP-2 speed gate

Date: 2026-06-25

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Launcher:
[`scripts/launch-glm-52-reap-504b-b12x-gce.sh`](../../scripts/launch-glm-52-reap-504b-b12x-gce.sh)

Private proxy helper:
[`scripts/expose-glm-52-reap-504b-private-proxy-gce.sh`](../../scripts/expose-glm-52-reap-504b-private-proxy-gce.sh)

Public-safety boundary: this note contains public-safe timing aggregates,
token counts, hashes, sanitized error classes, and live service status only. It
contains no bearer token, model-provider credentials, raw prompts, raw
responses, private source, hidden reasoning traces, weights, checkpoints,
compiled engines, profiler dumps, or raw model logs.

## Result

PASS. The private Khala GLM-5.2 504B REAP canary is now running a faster
per-request serving profile:

- GPUs: `0,1,2,3`
- Tensor parallel size: 4
- Decode-context parallel size: 4
- Max model length: 250,000 tokens
- Max sequences: 2
- Max batched tokens: 4096
- MTP: enabled
- Speculative tokens: 2
- Proxy default `min_p`: disabled / omitted
- Repetition penalty: 1.05
- Keep-warm: active

The live proxy receipt advertises:

- `profile.speculation.mode`: `mtp2`
- `requestDefaults.sampling.repetitionPenalty`: `1.05`
- `requestDefaults.sampling.maxTokens`: `1024`
- No `requestDefaults.sampling.minP`

## Why `min_p` Had To Go

The first MTP-3 benchmark reused the existing sampler defaults and failed for
longer decode requests. vLLM returned a streaming error event:

```text
The min_p and logit_bias sampling parameters are not yet supported with
speculative decoding.
```

That is consistent with the current vLLM sampling-parameter surface, where
`min_p` and `logit_bias` are explicit sampling parameters:
<https://docs.vllm.ai/en/latest/api/vllm/sampling_params/>

Hydralisk therefore changed the proxy deploy helper so `DEFAULT_MIN_P=` is an
intentional "omit min_p" value instead of being overwritten back to `0.05`.
Keep `min_p=0.05` for non-MTP fallback lanes; do not inject it into MTP
requests.

## Candidate Matrix

Baseline comparison is the controlled 4x TP non-MTP benchmark from
[`2026-06-25-glm-52-reap-504b-khala-canary-status.md`](2026-06-25-glm-52-reap-504b-khala-canary-status.md).

Median streaming results:

| Profile | Context ceiling | Spec tokens | `min_p` | Case | TTFT | Completion tok/s excl. TTFT | Completion tok/s incl. TTFT | Result |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 4x TP baseline | 250K | 0 | 0.05 | Decode 160 | 0.251s | 35.81 | 33.86 | Passed |
| 4x TP baseline | 250K | 0 | 0.05 | Decode 512 | 0.258s | 35.82 | 35.19 | Passed |
| MTP-3 | 250K | 3 | 0.05 | Decode 160/512 | n/a | n/a | n/a | Failed: vLLM rejects `min_p` with speculative decoding |
| MTP-3 | 250K | 3 | omitted | Decode 160 | 0.270s | 42.16 | 39.41 | Passed |
| MTP-3 | 250K | 3 | omitted | Decode 512 | 0.306s | 41.94 | 39.85 | Passed |
| MTP-3 | 250K | 3 | omitted | Decode 1024 | 0.471s | 42.45 | 41.45 | Passed |
| MTP-2 | 250K | 2 | omitted | Decode 160 | 0.289s | 48.67 | 44.63 | Passed |
| MTP-2 | 250K | 2 | omitted | Decode 512 | 0.287s | 48.06 | 46.74 | Passed |
| MTP-2 | 250K | 2 | omitted | Decode 1024 | 0.458s | 51.32 | 49.91 | Passed |
| MTP-2 fast-lane probe | 65K | 2 | omitted | Decode 160 | 0.250s | 48.71 | 45.26 | Passed |
| MTP-2 fast-lane probe | 65K | 2 | omitted | Decode 512 | 0.284s | 49.45 | 48.20 | Passed |
| MTP-2 fast-lane probe | 65K | 2 | omitted | Decode 1024 | 0.445s | 48.56 | 47.33 | Passed |

The 65K fast-lane probe did not justify replacing the 250K profile as the
default. It was slightly faster on the 512-token case and slower on the
1024-token case. The live canary keeps the full 250K admitted context.

## Live Proxy Speed

After switching the raw model to MTP-2 and restarting the private proxy without
default `min_p`, an authenticated streaming proxy benchmark passed:

- Run window: 2026-06-25 around `03:30Z`
- Endpoint: `http://10.128.0.38:8080`
- Auth: bearer token read locally on the VM
- Prompt/output storage: omitted; hashes only

Median proxy-path results:

| Case | Passed | Completion tokens | TTFT | Wall | Completion tok/s excl. TTFT | Completion tok/s incl. TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Proxy decode 160 | 3/3 | 148 | 0.283s | 3.418s | 48.01 | 44.25 |
| Proxy decode 512 | 2/2 | 512 | 0.280s | 10.852s | 48.48 | 47.23 |

Compared to the previous 4x TP baseline:

- Decode 160 proxy-equivalent throughput rose from about `33.86` to
  `44.25` completion tok/s including TTFT: about `+31%`.
- Decode 512 throughput rose from about `35.19` to `47.23` completion tok/s
  including TTFT: about `+34%`.
- Post-TTFT decode is now roughly `48` completion tok/s on the private proxy
  path.

## Live Status After Change

Checked at `2026-06-25T03:31:56Z`:

- Keep-warm timer: active
- Raw vLLM `/v1/models`: ready
- Private proxy `/health`: ready
- Raw bind: `127.0.0.1:8000`
- Proxy bind: `10.128.0.38:8080`
- Running container: `hydralisk-glm52-reap-504b`

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

## Decision

Admit MTP-2/no-`min_p` as the private Khala speed canary profile. Do not admit
MTP-3 as default because MTP-2 was faster across the practical decode cases.
Do not make the 65K fast-lane probe the default because it gives up context
for only marginal and mixed decode benefit.

## Claim Boundary

Admitted:

- Private 4-GPU G4 GLM-5.2 REAP speed canary.
- MTP-2 speculative decoding at the 250K admitted context envelope.
- Proxy default `min_p` omitted for MTP compatibility.
- Live private proxy throughput around 44-47 completion tok/s including TTFT
  on the tested 160/512-token streaming cases.

Not admitted:

- Public internet endpoint.
- Public SLA.
- Public customer routing or billing.
- `min_p` with speculative decoding.
- MTP-3 as default.
- 65K context fast lane as default.
- Multi-tenant concurrency beyond the current singleflight proxy limit.
