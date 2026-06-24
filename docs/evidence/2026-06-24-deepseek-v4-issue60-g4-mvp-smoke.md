# DeepSeek V4 issue #60 8x G4 MVP smoke

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/60

## Summary

Issue #60 moved the DeepSeek-V4-Flash NVFP4 G4 lane from a suspected NCCL
blocker to a public-safe OpenAI-compatible completion smoke on the existing
8 x G4 target.

The target stayed the private spot VM:

```text
instance=hydralisk-deepseek-v4-b12x-g4-8g-b-20260624155352
project=openagentsgemini
zone=us-central1-b
machine=g4-standard-384
accelerator=nvidia-rtx-pro-6000 x8
gpu=NVIDIA RTX PRO 6000 Blackwell Server Edition
driver=580.159.03
cuda=13.0
externalIp=<none>
```

## What changed

The first fix was not NCCL. A minimal 8-rank Torch/NCCL all-gather fixture
passed on the same G4 host and container envelope, so the earlier full-model
failure was not a generic G4 collective failure.

The full-model path then moved through three runtime blockers:

1. Sparse MLA fallback dtype and cache layout:
   - accepted float/float8 query and KV rows by computing in fp32 internally;
   - supported both the synthetic 4D HND KV cache layout and the live 3D
     FlashInfer DSV4 MLA cache layout.
2. Sparse attention indexer top-k logits:
   - added `HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY=1` to fail open before the
     SM120-unsupported DeepGEMM paged-MQA logits kernel.
3. MLA indexer scheduler metadata:
   - under the same SWA-only flag, zeroed the preallocated scheduler metadata
     instead of calling the second SM120-unsupported DeepGEMM metadata kernel.

The final derived image was:

```text
baseImage=hydralisk-deepseek-v4-b12x-g4-vllm-issue60-indexer:20260624165303
baseImageDigest=sha256:e0f9b1bd5880797273880b541cd1bc6770aed15e7b310d8027f061ae70f44673
derivedImage=hydralisk-deepseek-v4-b12x-g4-vllm-issue60-metadata:20260624170016
derivedImageDigest=sha256:ace20aadf862812912b4b63ff9bd19f8b04d335680f2983a1200ebf6eaf47418
```

## Smoke

The final run used:

```text
MODEL_ID=nvidia/DeepSeek-V4-Flash-NVFP4
MODEL_REVISION=e3cd60e7de98e9867116860d522499a728de1cf9
MOE_BACKEND=flashinfer_b12x
VLLM_LINEAR_BACKEND=triton
VLLM_ENABLE_EXPERT_PARALLEL=0
VLLM_ENFORCE_EAGER=1
VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4
HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=bf16_einsum
HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK=1
HYDRALISK_DEEPSEEK_INDEXER_PATCH=1
HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY=1
TENSOR_PARALLEL_SIZE=8
MAX_MODEL_LEN=2048
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=512
GPU_MEMORY_UTILIZATION=0.95
```

The OpenAI-compatible server reached `/v1/models` and completed one
public-safe `/v1/chat/completions` request:

```json
{
  "id": "chatcmpl-9e0a1cdcf04d2f5f",
  "model": "nvidia/DeepSeek-V4-Flash-NVFP4",
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 12,
    "completion_tokens": 3,
    "prompt_tokens_details": null
  },
  "finish_reason": "stop"
}
```

No prompt text or response text is committed in this receipt.

## Boundary

This is a real MVP execution proof, not a production serving claim.

The current lane is intentionally correctness-first and degraded:

- `max_model_len=2048`;
- one sequence smoke only;
- `HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY=1` disables the compressed sparse top-k
  expert-cache indexer path and uses sliding-window attention routes only;
- first request pays heavy Triton/TileLang JIT latency;
- no quality eval, throughput eval, or long-context eval has passed.

The next work should either restore an SM120-supported sparse indexer/metadata
kernel path or explicitly characterize the SWA-only fallback quality and
latency before any serving promise.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
