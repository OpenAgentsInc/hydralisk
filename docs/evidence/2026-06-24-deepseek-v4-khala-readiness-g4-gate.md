# DeepSeek V4 Khala readiness G4 timing gate

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/62

## Summary

Issue #62 added and ran a resident-server timing gate for the DeepSeek V4 Flash
8 x G4 lane. The gate starts the server, waits for `/v1/models`, runs a warmup
request, then runs repeated streaming requests and records p50/p95 timing
metrics without committing raw prompt or response text.

The tested image was:

```text
image=hydralisk-deepseek-v4-b12x-g4-vllm-issue60-vector-v3:20260624v3vector2
digest=sha256:a43653081fe01ab53901ecdff7415d2d8c40a6ea76b5c923aaff1b0e0e661451
model=nvidia/DeepSeek-V4-Flash-NVFP4
revision=e3cd60e7de98e9867116860d522499a728de1cf9
```

The target remained:

```text
instance=hydralisk-deepseek-v4-b12x-g4-8g-b-20260624155352
project=openagentsgemini
zone=us-central1-b
machine=g4-standard-384
accelerator=nvidia-rtx-pro-6000 x8
externalIp=<none>
```

## Gate

Thresholds:

| Metric | Threshold |
| --- | ---: |
| Server start to ready | <= 180 s |
| Warmup max latency | <= 45 s |
| Streaming TTFT p95 | <= 2.0 s |
| Decode tok/s p50 | >= 8.0 |
| End-to-end tok/s p50 | >= 8.0 |
| Successful requests | 5 / 5 |

Result:

```text
khalaReadinessGatePassed=true
```

Measured summary:

| Metric | Result |
| --- | ---: |
| Server start to ready | 116 s |
| Warmup max latency | 30.157707 s |
| Successful streaming requests | 5 / 5 |
| TTFT p50 | 0.277751 s |
| TTFT p95 | 0.2892222 s |
| Decode tok/s p50 | 11.328077 |
| Decode tok/s p95 | 11.3945372 |
| End-to-end tok/s p50 | 10.616448 |
| End-to-end tok/s p95 | 10.6470746 |

Per-request stream metrics:

| Index | TTFT (s) | Decode tok/s | End-to-end tok/s | Completion tokens |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.289939 | 11.401052 | 10.634613 | 32 |
| 1 | 0.277564 | 11.279571 | 10.575205 | 32 |
| 2 | 0.286355 | 11.245239 | 10.515491 | 32 |
| 3 | 0.277751 | 11.368478 | 10.650190 | 32 |
| 4 | 0.277575 | 11.328077 | 10.616448 | 32 |

Prompt evidence:

```text
promptBytes=121
promptSha256=3b19edc4cddb35982c1558c552341c6cbbdc7e42153d9cc0f2f58ff4fe02bdc5
```

The prompt body and all model response text were intentionally excluded from
tracked evidence.

## Interpretation

This timing gate is enough to justify continuing toward a Khala integration
candidate. It is not enough, by itself, to ship the model in Khala.

What this proves:

- the v3 vector-gather fallback keeps the model resident and stable for five
  repeated warmed streaming requests;
- warmed TTFT is sub-300 ms p95 for this short prompt/32-token shape;
- warmed decode is about 11 tok/s on the 8 x RTX PRO 6000 G4 host;
- the current container can meet an explicit timing gate while keeping raw
  prompt and response text out of tracked evidence.

What remains before a Khala serving claim:

- answer quality must pass a small public-safe evaluation gate;
- a longer output shape should be measured, not just 32 tokens;
- at least one longer-context prompt should pass within the current
  `max_model_len=2048` cap, and the cap should be raised if the memory/runtime
  path supports it;
- concurrency remains unproven because this lane still uses `max_num_seqs=1`;
- `HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY=1` still bypasses the compressed sparse
  top-k indexer path.

## Local Artifacts

Generated but not tracked:

```text
.hydralisk/deepseek-v4-khala-readiness-20260624issue62gate1/readiness-public.json
.hydralisk/deepseek-v4-khala-readiness-20260624issue62gate1/readiness-report.md
/var/log/hydralisk/deepseek-khala-readiness-20260624issue62gate1
```

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
