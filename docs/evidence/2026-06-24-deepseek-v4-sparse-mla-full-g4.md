# DeepSeek V4 sparse MLA full 8x G4 smoke

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/59

## Summary

Issue #59 ran the full DeepSeek-V4-Flash NVFP4 model smoke on a fresh 8 x G4
spot VM using the derived Hydralisk provider stack with the sparse MLA fallback
enabled.

The run admitted the target hardware:

```text
instance=hydralisk-deepseek-v4-b12x-g4-8g-b-20260624155352
zone=us-central1-b
machine=g4-standard-384
accelerator=nvidia-rtx-pro-6000 x8
provisioning=SPOT
externalIp=<none>
```

The derived image built successfully from `vllm/vllm-openai:latest` and applied
the Hydralisk local site-packages patches:

```text
baseImageDigest=sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f
derivedImageDigest=sha256:52ffbbdff07a27764609b9b9a26c5b3eba0ba2ee53e91559df634e68d34fe1e8
vllm=0.23.0
torch=2.11.0+cu130
cuda=13.0
localPatches=b12x_clamp,sparse_mla_fallback
```

The import probe confirmed all eight GPUs and the patched sparse MLA branch:

```text
deviceCount=8
deviceName=NVIDIA RTX PRO 6000 Blackwell Server Edition
capability=12.0
deepGemmImport=true
sparseMlaFallbackPatched=true
sparseMlaFallbackEnv=1
```

## Full-Model Smoke

The full-model command used:

```text
MODEL_ID=nvidia/DeepSeek-V4-Flash-NVFP4
MODEL_REVISION=e3cd60e7de98e9867116860d522499a728de1cf9
MOE_BACKEND=flashinfer_b12x
VLLM_LINEAR_BACKEND=triton
VLLM_ENABLE_EXPERT_PARALLEL=0
VLLM_ENFORCE_EAGER=1
VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4
HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=bf16_einsum
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=hopper
HYDRALISK_DEEPSEEK_SPARSE_MLA_PATCH=1
HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK=1
HYDRALISK_B12X_CLAMP_PATCH=1
HYDRALISK_B12X_CLAMP_LIMIT=10.0
TENSOR_PARALLEL_SIZE=8
MAX_MODEL_LEN=2048
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=512
GPU_MEMORY_UTILIZATION=0.95
```

The server did not reach `/v1/models`:

```text
READY=0
completion={"ready":false,"status":"server_not_ready_or_exited"}
```

The model progressed beyond the earlier B12x clamp, `o_proj`, and DSV4 FMHA
unsupported-architecture blockers. The next blocker is now tensor-parallel
collectives during vLLM memory/profile initialization:

```text
torch.distributed.DistBackendError: NCCL error
ncclUnhandledCudaError: Call to CUDA function failed.
Last error:
Cuda failure 800 'operation not permitted'
```

The failing path is `compute_logits -> tensor_model_parallel_all_gather` inside
vLLM's dummy sampler/profile run, before the OpenAI-compatible server becomes
ready.

## Decision

The current 8 x G4 lane is no longer blocked by Google capacity, Hugging Face
artifact access, B12x clamp support, the `o_proj` DeepGEMM path, or the missing
SM120 DSV4 sparse MLA FMHA kernel. It is blocked by NCCL/tensor-parallel
all-gather on the PCIe-only G4 topology.

The next issue should isolate and fix distributed collectives on the same G4
shape before another full-model retry. The first candidate probe is a minimal
8-rank Torch/NCCL all-gather fixture under the same Docker/runtime envelope,
then rerun with safe transport toggles such as disabling NCCL P2P if the stock
fixture reproduces `cudaErrorNotPermitted`.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
