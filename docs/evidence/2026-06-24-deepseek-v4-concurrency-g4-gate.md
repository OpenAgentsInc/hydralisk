# DeepSeek V4 concurrency G4 gate

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/65

## Summary

Issue #65 extended the resident DeepSeek V4 Khala readiness probe with measured
concurrent streaming requests. The live G4 lane admitted `max_num_seqs=2` and
completed two concurrent streamed requests, but failed the explicit concurrency
gate on TTFT and throughput.

The tested image remained:

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

The concurrency prompt was supplied from a temp file outside the repository.
Tracked evidence records only prompt hash, prompt bytes, token counts, and
timing metadata.

```text
promptBytes=173
promptSha256=2b50023e27447e6db8ed8af9d87d3bd91e05fce23847f55682fa903634323e57
streamPromptTokens=41
maxTokens=96
maxNumSeqs=2
concurrentStreamRequests=2
minPromptTokens=20
minStreamCompletionTokens=48
```

Thresholds:

| Metric | Threshold |
| --- | ---: |
| Server start to ready | <= 180 s |
| Warmup max latency | <= 45 s |
| Streaming TTFT p95 | <= 5.0 s |
| Decode tok/s p50 | >= 6.0 |
| End-to-end tok/s p50 | >= 5.0 |
| Prompt tokens | >= 20 |
| Completion tokens per measured stream | >= 48 |
| Successful measured streams | 2 / 2 |
| Concurrent measured stream requests | 2 |

Result:

```text
khalaReadinessGatePassed=false
```

Measured summary:

| Metric | Result |
| --- | ---: |
| Server start to ready | 116 s |
| Warmup max latency | 30.454614 s |
| Stream warmup requests | 1 |
| Concurrent measured stream requests | 2 |
| Successful measured streams | 2 / 2 |
| Minimum stream prompt tokens | 41 |
| Minimum stream completion tokens | 96 |
| TTFT p50 | 7.072120 s |
| TTFT p95 | 13.0776319 s |
| Decode tok/s p50 | 2.9847005 |
| Decode tok/s p95 | 3.10661135 |
| End-to-end tok/s p50 | 2.5087415 |
| End-to-end tok/s p95 | 2.81151185 |

Measured concurrent streams:

| Index | TTFT (s) | Decode tok/s | End-to-end tok/s | Prompt tokens | Completion tokens | Finish reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.399329 | 2.849244 | 2.845153 | 41 | 96 | length |
| 1 | 13.744911 | 3.120157 | 2.172330 | 41 | 96 | length |

The uncounted streaming warmup showed that the same short shape is fast when
served alone:

| Index | TTFT (s) | Decode tok/s | End-to-end tok/s | Prompt tokens | Completion tokens | Finish reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stream-warmup-0 | 0.413299 | 11.210684 | 10.801779 | 41 | 96 | length |

## Interpretation

This is a real concurrency failure, not a capacity admission failure:

- the server reached readiness with `max_num_seqs=2`;
- both concurrent requests completed with 96 output tokens;
- single-stream warmup throughput stayed around 11 tok/s;
- two concurrent measured streams dropped to roughly 3 tok/s per request;
- one concurrent request waited 13.7 s for first token.

Therefore the current DeepSeek V4 Flash G4 lane is not credible as a shared
Khala service. The honest boundary is single-flight, resident, explicitly
prewarmed canary traffic only.

The next useful step is to choose between:

- implementing an external single-flight/queueing serving envelope for a
  narrow internal Khala canary; or
- doing scheduler/kernel work to make `max_num_seqs > 1` viable before product
  integration.

The SWA-only sparse-indexer bypass also remains unresolved.

## Local Artifacts

Generated but not tracked:

```text
.hydralisk/deepseek-v4-concurrency-20260624issue65concurrency1/readiness-public.json
.hydralisk/deepseek-v4-concurrency-20260624issue65concurrency1/readiness-report.md
/var/log/hydralisk/deepseek-khala-readiness-20260624issue65concurrency1
```

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
