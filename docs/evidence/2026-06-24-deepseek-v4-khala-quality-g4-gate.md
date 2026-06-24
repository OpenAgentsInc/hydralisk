# DeepSeek V4 Khala quality G4 gate

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/63

## Summary

Issue #63 extended the resident DeepSeek V4 Khala readiness probe with
runtime-supplied quality cases. The harness accepts case IDs, prompt text,
per-case max tokens, and regex or substring expectations at runtime, then
records only hashes, lengths, token counts, latency, finish reasons, pass/fail,
and matched expectation counts.

The live run used the same tested G4 image:

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

## Timing Gate

The run used one warmup request, three warmed streaming requests, and three
runtime quality cases.

```text
khalaReadinessGatePassed=true
qualityGatePassed=true
```

Measured timing summary:

| Metric | Result |
| --- | ---: |
| Server start to ready | 115 s |
| Warmup max latency | 30.326379 s |
| Successful streaming requests | 3 / 3 |
| TTFT p50 | 0.357261 s |
| TTFT p95 | 0.357405 s |
| Decode tok/s p50 | 11.970492 |
| Decode tok/s p95 | 12.111621 |
| End-to-end tok/s p50 | 9.774685 |
| End-to-end tok/s p95 | 10.0831573 |

Per-request stream metrics:

| Index | TTFT (s) | Decode tok/s | End-to-end tok/s | Completion tokens | Finish reason |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 0.291436 | 12.127302 | 9.774685 | 15 | stop |
| 1 | 0.357421 | 11.970492 | 9.745790 | 19 | stop |
| 2 | 0.357261 | 11.848431 | 10.117432 | 25 | stop |

## Quality Gate

The quality case bodies were supplied from a temp file outside the repository.
Tracked evidence intentionally excludes raw prompt text, response text, and
expectation text.

| Case ID | Passed | Prompt bytes | Prompt SHA-256 | Response bytes | Response SHA-256 | Latency (s) | Prompt tokens | Completion tokens | Finish reason | Regex matches | Substring matches |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| qa-arithmetic-small | true | 43 | `6fcde72d35581650122fc9f2e7ed786feb80e04f4b9d26c7030e81150486f9a6` | 1 | `4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a` | 10.371589 | 18 | 2 | stop | 1 / 1 | 0 / 0 |
| qa-copy-token | true | 63 | `177e5aeaff1d1a615346aee866a5fdaf2ce9a7e60e6d925ef4a5cc895dc3efb8` | 4 | `2f9acb02faa121bb2a3621951f57b4c690655337edee2e5ac350be2b3be88ea8` | 18.172683 | 16 | 3 | stop | 1 / 1 | 0 / 0 |
| qa-common-color | true | 65 | `b30a44d041f7ce6df4d7c772bbe71e62ecc57fa29213aafa36277d06d5710b0a` | 4 | `16477688c0e00699c6cfa4497a3612d7e83c532062b64b250fed8908128ed548` | 0.397918 | 18 | 2 | stop | 1 / 1 | 0 / 0 |

Prompt evidence for the repeated streaming timing prompt:

```text
promptBytes=125
promptSha256=2148b48dbb93dd81d420509b3360fb79f439df463632f2b2835848e6a998c54f
```

## Interpretation

This run clears the first public-safe quality smoke for the DeepSeek V4 Flash
G4 lane. The model answered three tiny deterministic cases correctly while the
resident timing gate continued to pass.

It is still not enough for a Khala serving claim:

- the quality suite is intentionally tiny;
- two tiny quality completions took roughly 10 s and 18 s, so nonstream
  per-request latency still needs investigation;
- no longer-output gate has passed;
- no longer-context prompt has passed beyond the short smoke shape;
- concurrency remains unproven because this lane still uses `max_num_seqs=1`;
- `HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY=1` still bypasses the compressed sparse
  top-k indexer path.

The next useful gate is a longer-output and longer-context resident run using
the same public-safe hash-only receipt shape.

## Local Artifacts

Generated but not tracked:

```text
.hydralisk/deepseek-v4-quality-20260624issue63quality1/readiness-public.json
.hydralisk/deepseek-v4-quality-20260624issue63quality1/readiness-report.md
/var/log/hydralisk/deepseek-khala-readiness-20260624issue63quality1
```

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
