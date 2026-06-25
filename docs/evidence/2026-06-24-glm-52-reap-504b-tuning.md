# GLM-5.2 504B REAP G4 tuning evidence

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/88

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Launcher:
[`scripts/launch-glm-52-reap-504b-b12x-gce.sh`](../../scripts/launch-glm-52-reap-504b-b12x-gce.sh)

Private proxy helper:
[`scripts/expose-glm-52-reap-504b-private-proxy-gce.sh`](../../scripts/expose-glm-52-reap-504b-private-proxy-gce.sh)

Public receipt:
[`docs/evidence/2026-06-24-glm-52-reap-504b-tuning-receipt.json`](2026-06-24-glm-52-reap-504b-tuning-receipt.json)

Public-safety boundary: this packet contains run IDs, launch settings, hashes,
token counts, aggregate timings, GPU memory envelopes, and sanitized failure
classes only. It contains no bearer token, model-provider credentials, raw
prompts, raw responses, private source, hidden reasoning traces, weights,
checkpoints, compiled engines, profiler dumps, or raw model logs.

## Result

PASS. GLM-5.2 504B REAP/NVFP4 has a production-candidate 4-GPU G4 launch
envelope for the private Hydralisk lane:

- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- Tensor parallel size: 4
- Decode-context parallel size: 4
- Max model length: 250,000 tokens
- Max sequences: 2
- Max batched tokens: 4096
- GPU memory utilization: 0.95
- KV cache dtype: `fp8`
- Attention backend: `B12X_MLA_SPARSE`
- MoE backend: `b12x`
- Quantization: `modelopt_fp4`
- MTP: MTP-2 enabled by default after the 2026-06-25 speed gate

This was tested on four selected RTX PRO 6000 GPUs inside the admitted 8x
fallback host. Standalone 4x `g4-standard-192` admission is still
capacity-blocked, so public claims must keep that boundary explicit.

## Context ramp

| Run ID | Max context | Max seqs | Batch tokens | Prompt tokens | Completion tokens | Wall time | KV tokens | vLLM full-context concurrency | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `20260624232000` | 65,536 | 1 | 8192 | 60,012 | 8 | 23.467 s | 442,368 | 6.75x | passed |
| `20260624233000` | 131,072 | 1 | 16384 | 120,012 | 6 | 152.335 s | 324,352 | 2.47x | passed |
| `20260624235000` | 250,000 | 1 | 4096 | 240,012 | 8 | 281.219 s | 485,158 | 1.94x | passed |

The 250K run is the largest stable context gate tested. The model card's
architectural context window is larger, but this G4 serving profile only admits
250K until a separate long-context gate extends it.

## Batch-token gate

At 250K context, `max_num_batched_tokens=32768` failed before readiness:

- Run ID: `20260624234000`
- Sanitized failure class: `failed_kv_cache_admission`
- vLLM required 3.58 GiB of KV cache for one 250K request but reported only
  1.26 GiB available.
- vLLM estimated the maximum admissible model length at 87,808 tokens for that
  over-wide batch-token setting.

The production-candidate 250K profile therefore pins
`max_num_batched_tokens=4096`. Higher batch-token settings remain valid tuning
work only after a new memory envelope proves them.

## Concurrency gate

The recommended server process runs with `max_num_seqs=2` so it can admit short
concurrent utility traffic:

- Run ID: `20260624240000`
- Max context: 250,000 tokens
- Max sequences: 2
- Max batched tokens: 4096
- Short concurrent requests: 2
- Short concurrency result: passed
- Aggregate wall time: 25.862 s
- Per-request wall time p50: 25.861 s
- Per-request wall time p95: 25.861 s

Full-context concurrency is not admitted at 2 x 250K. vLLM reported 1.94x
maximum concurrency at 250,000 tokens/request, so the honest public ceiling is:
one full 250K request at a time, with `max_num_seqs=2` reserved for shorter
concurrent requests and scheduler headroom.

## MTP gate

Historical note: MTP loaded and completed a tiny smoke during the original
tuning pass, but it was not made the default until the 2026-06-25 MTP-2 speed
gate:

- Run ID: `20260624241000`
- Max context: 32,768 tokens
- Max sequences: 1
- Max batched tokens: 4096
- Speculative method: MTP
- Speculative tokens: 3
- Tiny smoke result: passed
- Tiny smoke wall time: 5.306 s
- vLLM warning: speculative settings capped scheduled tokens and may be
  suboptimal without further batch-token tuning.

The non-MTP path has already produced sub-second tiny smokes after warmup. Keep
MTP opt-in until a dedicated latency/quality comparison proves a win at the
target context.

## Sampler guardrail

Both loop-abatement candidates completed the public-safe synthetic guardrail
probe with `min_p=0.05`:

| Repetition penalty | Result | Completion tokens | Wall time |
| ---: | --- | ---: | ---: |
| 1.05 | passed after warmup | 22 | 0.834 s |
| 1.10 | passed | 23 | 0.880 s |

Default private serving should use `repetition_penalty=1.05` and
`min_p=0.05`. Use `repetition_penalty=1.10` as the explicit loop-abatement bump
for workloads that show repetition.

## Operator defaults

The raw launcher defaults now match the tuned speed envelope:

```bash
MAX_MODEL_LEN=250000
MAX_NUM_SEQS=2
MAX_NUM_BATCHED_TOKENS=4096
MTP=1
NUM_SPECULATIVE_TOKENS=2
```

The private proxy advertises `ADMITTED_CONTEXT_TOKENS=250000`, injects the GLM
sampler defaults, omits default `min_p` for MTP compatibility, keeps
`enable_thinking=false` unless a client opts in, and keeps
`MAX_INFLIGHT_REQUESTS=1` at the proxy layer so full-context traffic stays
single-flight until a public concurrency policy is introduced. The speed-gate
evidence is:
[`2026-06-25-glm-52-reap-504b-mtp2-speed-gate.md`](2026-06-25-glm-52-reap-504b-mtp2-speed-gate.md).

Proxy refresh smoke:

- Run ID: `20260624243000`
- Proxy run ref:
  `hydralisk-run-049109202d794cf5b633922e2ff1b7b1`
- Evidence ref advertised by `/v1/models`:
  `docs/evidence/2026-06-24-glm-52-reap-504b-tuning.md`
- Admitted context in receipt: 250,000 tokens
- Max inflight requests in receipt: 1
- HTTP status: 200
- Prompt SHA-256:
  `5f8103cbebaac77e42161be89a636352dafbb3ceda5efef0aa6484d67233dfe2`
- Visible completion SHA-256:
  `deb72954879f318cd0fcb41355e82f54fbed51947d68e71b465fd31aba03f166`
- Visible completion characters: 18
- Prompt tokens: 22
- Completion tokens: 9
- Total tokens: 31
- Wall time: 0.505 s

## Claim boundary

The honest claim after this gate is:

```text
GLM-5.2 504B REAP/NVFP4 passes a private 4-GPU G4 production-candidate serving
profile at 250K context with max_num_seqs=2, max_num_batched_tokens=4096, and
MTP-2/no-min_p speed serving, using four RTX PRO 6000 GPUs on the admitted 8x
fallback host. Standalone 4x G4 admission remains capacity-blocked, and two
concurrent full-250K requests are not admitted.
```
