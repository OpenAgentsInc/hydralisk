# DeepSeek-V4-Fable merged checkpoint private canary

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/80

Depends on:

- https://github.com/OpenAgentsInc/hydralisk/issues/78
- https://github.com/OpenAgentsInc/hydralisk/issues/79

Status: `blocked_runtime_loader_backend_selection`

## Decision

- Private merged-checkpoint load succeeded: `false`
- `/v1/models` reached: `false`
- Generation attempted: `false`
- TTFT measured: `false`
- Decode tokens/sec measured: `false`
- Khala general route allowed: `false`
- Public aliases allowed: `false`
- MPP public sale allowed: `false`
- Next step: `select_supported_fable_moe_backend_or_build_mxfp4_b12x_support`

## Target

```text
project=openagentsgemini
zone=us-central1-b
instance=hydralisk-deepseek-v4-b12x-g4-8g-b-20260624155352
machine=g4-standard-384
gpuCount=8
gpuName=NVIDIA RTX PRO 6000 Blackwell Server Edition
modelPath=/opt/hydralisk/models/deepseek-v4-fable-merged
image=hydralisk-deepseek-v4-b12x-g4-vllm-issue60-vector-v3:20260624v3vector2
tokenizer=nvidia/DeepSeek-V4-Flash-NVFP4
tokenizerRevision=e3cd60e7de98e9867116860d522499a728de1cf9
```

The server was started with localhost binding only:

```text
host=127.0.0.1
port=8000
tensorParallelSize=8
moeBackend=flashinfer_b12x
linearBackend=triton
attentionBackend=FLASHINFER_MLA_SPARSE_DSV4
maxModelLen=2048
maxNumSeqs=1
maxNumBatchedTokens=512
gpuMemoryUtilization=0.95
```

No Khala route, public alias, MPP listing, or public OpenAI-compatible route
was changed.

## Result

The canary started at `2026-06-24T21:36:48Z`. The vLLM worker failed during
engine initialization at `2026-06-24T21:37:19Z`, before `/v1/models`, before
weight-load success, and before any completion request.

Public-safe root cause:

```text
ValueError: moe_backend='flashinfer_b12x' is not supported for MXFP4 MoE.
Expected one of ['deep_gemm', 'flashinfer_trtllm',
'flashinfer_trtllm_afp8', 'flashinfer_cutlass',
'flashinfer_cutlass_afp8', 'triton', 'triton_unfused', 'humming',
'marlin', 'aiter', 'aiter_mxfp4_fp8', 'aiter_mxfp4_mxfp4', 'xpu',
'cpu', 'emulation'].
```

The staged checkpoint config reports:

```json
{
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "scale_fmt": "ue8m0",
  "weight_block_size": [128, 128]
}
```

Even though the config reports FP8 metadata, the installed vLLM DeepSeek-V4
MoE quantization path selected its MXFP4 MoE backend selector and rejected the
B12x backend used by the current admitted DeepSeek-V4 G4 lane.

## Classification

Blocker class: `runtime_loader_backend_selection`.

It is not a storage blocker: issue #79 verified all 47 shards and metadata.

It is not a GPU memory blocker: the failure occurred during model construction,
and after the canary all eight GPUs reported `0 MiB` memory used.

It is not an NCCL/topology blocker: the model did not reach collective-heavy
weight load or generation.

It is not a quality/latency blocker: no completion request was attempted.

## Interpretation

The full merged checkpoint is staged, but Fable does not currently load on the
proven Hydralisk G4 `flashinfer_b12x` runtime envelope. The next useful step is
not another staging attempt. It is one of:

1. try a supported backend for this vLLM MXFP4-selected MoE path, most likely
   `triton` first as a correctness probe, then `flashinfer_trtllm` only if the
   supported path clears construction; or
2. build B12x support for this Fable/DeepSeek-V4 quantization selector if we
   want the same G4 B12x path to serve the merged checkpoint.

Until that backend-selection gate clears, Fable remains a private research
experiment and must not be routed through Khala, public aliases, or MPP.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
