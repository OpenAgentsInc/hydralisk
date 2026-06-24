# DeepSeek-V4-Flash FlashInfer DSV4 G4 live smoke

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/41

## Summary

Issue #41 is no longer blocked by gcloud auth or IAM. The canonical DSV4
wrapper ran against the existing 8 x G4 target:

```text
targetInstance=hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036
targetZone=us-central1-b
accelerator=nvidia-rtx-pro-6000
gpuCount=8
```

The run reached `/v1/models`, then the tiny public-safe completion smoke failed
inside the selected FlashInfer DSV4 sparse MLA path. This validates the stock
DSV4 fallback as a real runtime attempt on RTX PRO 6000 and shows that it is
not enough by itself for generation on SM120.

## Engine

```text
MODEL_ID=nvidia/DeepSeek-V4-Flash-NVFP4
MODEL_REVISION=e3cd60e7de98e9867116860d522499a728de1cf9
MOE_BACKEND=flashinfer_b12x
VLLM_VERSION=0.23.0
TORCH=2.11.0+cu130
CUDA=13.0
VLLM_LINEAR_BACKEND=triton
VLLM_ENFORCE_EAGER=1
VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4
HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=bf16_einsum
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=hopper
HYDRALISK_B12X_CLAMP_PATCH=1
HYDRALISK_B12X_CLAMP_LIMIT=10.0
MAX_MODEL_LEN=2048
MAX_NUM_BATCHED_TOKENS=512
GPU_MEMORY_UTILIZATION=0.95
```

Base image digest:

```text
vllm/vllm-openai@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f
```

Derived image digest:

```text
hydralisk-deepseek-v4-b12x-g4-vllm@sha256:767c0a05a0f26926d5c5e003df8305fc8b18a48f0d62e583a5266cecf4a6a35c
```

## Result

Readiness:

```text
READY=1
```

Model list returned:

```text
id=nvidia/DeepSeek-V4-Flash-NVFP4
max_model_len=2048
```

Completion receipt:

```json
{"ready":true,"completion":false,"status":"completion_failed_or_timed_out"}
```

The first public-safe root error:

```text
vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py forward_mqa
flashinfer.mla._core.trtllm_batch_decode_sparse_mla_dsv4
tvm.error.InternalError: Error in function 'TllmGenFmhaRunner' at /workspace/include/flashinfer/trtllm/fmha/fmhaRunner.cuh:37: Unsupported architecture
```

## Interpretation

This clears the local launch-shape bug and proves the existing vLLM
`FLASHINFER_MLA_SPARSE_DSV4` backend is not an SM120-safe generation path on
the current G4 stack. It avoids the earlier `flash_mla_sparse_fwd` call, but
lands in FlashInfer's TRTLLM FMHA runner, which still rejects this architecture.

The next concrete patch point is a tiny FlashInfer TRTLLM DSV4 sparse MLA FMHA
repro on RTX PRO 6000. That should isolate `TllmGenFmhaRunner` outside the full
model before any more full-model DeepSeek smoke. If the repro confirms the same
SM120 guard, Hydralisk needs either:

- an SM120-capable FlashInfer/TRTLLM DSV4 attention patch;
- a correctness-first fallback attention implementation for DeepSeek V4 on G4;
- or known-good H100/H200/B200/GB200 capacity for the provider recipe.

## Cleanup

After the smoke, the target host had no running Docker containers and all eight
GPUs reported 0 MiB memory in use.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
