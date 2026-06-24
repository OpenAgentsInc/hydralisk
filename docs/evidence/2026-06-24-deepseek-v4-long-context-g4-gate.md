# DeepSeek V4 long-context G4 gate

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/64

## Summary

Issue #64 extended the resident DeepSeek V4 Khala readiness probe so longer
prompt/output runs can require minimum observed prompt tokens and streamed
completion tokens. It also added uncounted streaming warmups, because the live
G4 lane showed a large first-use streaming long-shape penalty that nonstream
warmup did not remove.

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

The long-context gate used a runtime prompt file outside the repository.
Tracked evidence records only the prompt hash, prompt bytes, token counts, and
timing metadata.

```text
promptBytes=8783
promptSha256=482c9b6fd81f55c369cde381bd6672e78edc0f1d810687de24f94f05d6e265a7
streamPromptTokens=1796
maxTokens=160
minPromptTokens=700
minStreamCompletionTokens=96
```

Thresholds:

| Metric | Threshold |
| --- | ---: |
| Server start to ready | <= 180 s |
| Warmup max latency | <= 45 s |
| Streaming TTFT p95 | <= 5.0 s |
| Decode tok/s p50 | >= 8.0 |
| End-to-end tok/s p50 | >= 6.0 |
| Prompt tokens | >= 700 |
| Completion tokens per measured stream | >= 96 |
| Successful measured streams | 2 / 2 |

## Failed Attempts

The first attempt used the existing short nonstream warmup. It proved the model
can consume the longer prompt and emit the longer output, but the first measured
stream paid a first-use streaming long-shape penalty.

| Run | Stream warmups | Gate passed | TTFT p95 | Decode tok/s p50 | E2E tok/s p50 | Min prompt tokens | Min completion tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| issue64long1 | 0 | false | 10.265401 | 8.708863 | 7.726424 | 1796 | 160 |

The second attempt used a full-length nonstream warmup. That still did not
remove the first measured streaming penalty.

| Run | Stream warmups | Gate passed | TTFT p95 | Decode tok/s p50 | E2E tok/s p50 | Min prompt tokens | Min completion tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| issue64long2 | 0 | false | 10.230644 | 8.765250 | 7.737336 | 1796 | 152 |

In both failed attempts, the second measured stream was already fast. The
blocker was the first measured streaming long-shape path.

## Passing Run

The passing run used one short nonstream warmup and one uncounted long
streaming warmup before the two measured long streams.

```text
khalaReadinessGatePassed=true
```

Measured summary:

| Metric | Result |
| --- | ---: |
| Server start to ready | 116 s |
| Warmup max latency | 27.994585 s |
| Stream warmup requests | 1 |
| Successful measured streams | 2 / 2 |
| Minimum stream prompt tokens | 1796 |
| Minimum stream completion tokens | 160 |
| TTFT p50 | 0.1698925 s |
| TTFT p95 | 0.20696485 s |
| Decode tok/s p50 | 11.1068105 |
| Decode tok/s p95 | 11.14811285 |
| End-to-end tok/s p50 | 11.04586 |
| End-to-end tok/s p95 | 11.1147208 |

Measured streams:

| Index | TTFT (s) | Decode tok/s | End-to-end tok/s | Prompt tokens | Completion tokens | Finish reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.211084 | 11.060919 | 10.969348 | 1796 | 160 | length |
| 1 | 0.128701 | 11.152702 | 11.122372 | 1796 | 160 | length |

The uncounted streaming warmup itself exposed the cost that must be paid before
serving long streamed outputs:

| Index | TTFT (s) | Decode tok/s | End-to-end tok/s | Prompt tokens | Completion tokens | Finish reason |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| stream-warmup-0 | 10.840387 | 4.453844 | 2.750532 | 1796 | 77 | stop |

## Interpretation

This clears a materially stronger resident-server gate than issues #62 and
#63: with the long streaming path prewarmed, the G4 lane served a 1,796-token
prompt and two 160-token measured streamed outputs at about 11 tok/s with
sub-250 ms TTFT.

It is still not a general Khala serving claim:

- long streamed outputs require an explicit streaming prewarm;
- the first unprewarmed long streaming request still has about 10.8 s TTFT;
- the configured context remains `max_model_len=2048`, far below the model's
  advertised million-token context;
- concurrency remains unproven because this lane still uses `max_num_seqs=1`;
- `HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY=1` still bypasses the compressed sparse
  top-k indexer path;
- the quality gate is still tiny and nonstream quality latency remains uneven.

The next useful gate is concurrency at `max_num_seqs > 1` on the same
prewarmed resident path, or a sparse-indexer correctness replacement if we want
to reduce the serving caveats before concurrency.

## Local Artifacts

Generated but not tracked:

```text
.hydralisk/deepseek-v4-long-context-20260624issue64long1/readiness-public.json
.hydralisk/deepseek-v4-long-context-20260624issue64long1/readiness-report.md
.hydralisk/deepseek-v4-long-context-20260624issue64long2/readiness-public.json
.hydralisk/deepseek-v4-long-context-20260624issue64long2/readiness-report.md
.hydralisk/deepseek-v4-long-context-20260624issue64long3/readiness-public.json
.hydralisk/deepseek-v4-long-context-20260624issue64long3/readiness-report.md
/var/log/hydralisk/deepseek-khala-readiness-20260624issue64long1
/var/log/hydralisk/deepseek-khala-readiness-20260624issue64long2
/var/log/hydralisk/deepseek-khala-readiness-20260624issue64long3
```

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
