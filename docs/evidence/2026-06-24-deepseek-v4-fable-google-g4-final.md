# DeepSeek-V4-Fable Google G4 final tracker

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/81

Child issues:

- https://github.com/OpenAgentsInc/hydralisk/issues/78
- https://github.com/OpenAgentsInc/hydralisk/issues/79
- https://github.com/OpenAgentsInc/hydralisk/issues/80

Status: `no_go_current_b12x_g4_runtime_envelope`

## Answer

Hydralisk cannot run `Chunjiang-Intelligence/DeepSeek-v4-Fable` on the current
Google G4 B12x runtime envelope today.

The Google infrastructure capacity is sufficient to stage the artifact:

- the existing 8 x RTX PRO 6000 G4 host had enough disk and GPU memory class;
- all 47 merged checkpoint shards and five metadata files were staged;
- the staged file total is `298428071600` bytes;
- the staged manifest SHA-256 is
  `0610e0fc3f79512a9cc11b6ce93e48e1bdf6c25e0e694d52f4046f43c06a8833`.

The blocker is runtime compatibility, not quota or storage:

```text
ValueError: moe_backend='flashinfer_b12x' is not supported for MXFP4 MoE.
```

The first private canary failed before `/v1/models`, before generation, and
before any TTFT or tokens/sec measurement. Fable therefore remains unavailable
for Khala, public aliases, and MPP.

## Child Results

| Issue | Result | Meaning |
| --- | --- | --- |
| #78 | `go_for_staging_high_runtime_risk` | Capacity, disk, network, and checkpoint naming were good enough to stage. |
| #79 | `staged_private_g4_ready_for_canary` | All 47 shards plus metadata were staged and hashed in private G4 storage. |
| #80 | `blocked_runtime_loader_backend_selection` | Current B12x G4 runtime cannot construct the Fable MoE backend. |

## Practical Path Forward

The next executable path is not more GCloud quota hunting for this specific
experiment. It is a runtime backend probe:

1. Try `moe_backend=triton` against the already staged checkpoint as a
   correctness-first private canary.
2. If `triton` constructs and loads, measure a tiny localhost-only generation
   and record TTFT/tokens/sec.
3. If `triton` cannot load or is too slow, test another vLLM-supported backend
   from the error's allowed set.
4. If none of the supported backends work on G4, build or patch B12x support
   for this Fable/DeepSeek-V4 quantization selector.

This remains a research lane. Do not route Fable through Khala, public model
aliases, MPP, or any shared serving lane until a private canary reaches
`/v1/models`, generates, and passes an explicit policy gate.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
