# DeepSeek-V4-Flash B12x full-model G4 smoke

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/38

Date: 2026-06-24

## Scope

This probe retried the pinned DeepSeek-V4-Flash NVFP4 full-model load path on
the private 8 x G4 host after issues #36 and #37 proved patched B12x static and
dynamic clamp kernels on RTX PRO 6000.

The goal was not public serving. The goal was a private load-only vLLM smoke
behind localhost, with `/v1/models` readiness as the first promotion gate.

## Host

```text
project=openagentsgemini
zone=us-central1-b
instance=hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036
machine=g4-standard-384
gpuCount=8
gpuName=NVIDIA RTX PRO 6000 Blackwell Server Edition
driver=580.159.03
cuda=13.0
externalIp=none
```

Post-probe cleanup left no running Docker containers and all GPUs at `0 MiB`.

## Image

```text
baseImage=hydralisk-deepseek-v4-oproj-fallback-g4-vllm:20260624095206
baseDigest=sha256:f606b8120a907951ba311068d95acc533f3ea1235e02156ecc7b0b2799d47890
derivedImage=hydralisk-deepseek-v4-b12x-g4-vllm:20260624112739
derivedDigest=sha256:10c53db54b08d665ef86f52d5e72177b82211fecbc3da574cd4392b167e07441
vllm=0.23.0
torch=2.11.0+cu130
torchCuda=13.0
```

The derived image now builds with:

- an idempotent SM120 NVFP4 guard verifier;
- the CUDA E8M0 Triton upcast patch;
- a B12x `swiglu_limit` wrapper/API surface;
- real CuTe/CUTLASS clamp operations in the B12x static, dynamic, and micro
  gated activation paths;
- vLLM B12x forwarding from `gemm1_clamp_limit`;
- NVFP4 oracle marking B12x as clamp-capable.

## vLLM flags

```text
model=nvidia/DeepSeek-V4-Flash-NVFP4
revision=e3cd60e7de98e9867116860d522499a728de1cf9
moe_backend=flashinfer_b12x
tensor_parallel_size=8
enable_expert_parallel=0
linear_backend=triton
kv_cache_dtype=fp8
block_size=256
max_model_len=2048
max_num_seqs=1
max_num_batched_tokens=512
gpu_memory_utilization=0.95
HF_HUB_DISABLE_XET=1
```

## Result

The direct B12x full-model smoke built and started vLLM successfully, but with
the valid DeepSeek NVIDIA `o_proj` path left on, it stopped at the already-known
DeepGEMM scale-factor layout blocker:

```text
RuntimeError: Assertion error (csrc/apis/layout.hpp:59): Unknown SF transformation
site=vllm/models/deepseek_v4/nvidia/ops/o_proj.py deep_gemm_fp8_o_proj
```

The B12x wrapper was then fixed to propagate the existing default-off
`HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK` control, and the smoke was rerun with the
previously proven correctness-first fallback:

```text
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=hopper
HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS=1
HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE=fp32
HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=bf16_einsum
```

That moved execution past `o_proj` on all eight ranks and into model startup.
The run then failed before `/v1/models` during vLLM cudagraph memory profiling
inside DeepSeek MLA attention metadata construction:

```text
vllm/v1/worker/gpu_worker.py determine_available_memory
vllm/v1/worker/gpu_model_runner.py profile_cudagraph_memory
vllm/v1/worker/gpu_model_runner.py _dummy_run
vllm/v1/attention/backends/mla/indexer.py build
vllm/utils/deep_gemm.py get_paged_mqa_logits_metadata
RuntimeError: Assertion error (csrc/apis/attention.hpp:219): Unsupported architecture
```

Public readiness remained false:

```text
READY	0
completion={"ready":false,"status":"server_not_ready_or_exited"}
```

## Interpretation

Issue #38 clears the B12x image integration boundary:

- the clamp-patched B12x derived image builds reproducibly;
- the image imports on 8 x RTX PRO 6000 with CUDA available;
- vLLM accepts `moe_backend=flashinfer_b12x` for the pinned NVFP4 model;
- the existing `bf16_einsum` `o_proj` fallback still moves the model past the
  DeepGEMM `o_proj` scale-layout blocker;
- the next blocker is not B12x clamp plumbing.

The next executable issue should target the DeepSeek MLA metadata path on
SM120/G4: either find a supported vLLM attention backend/configuration that
avoids DeepGEMM's unsupported `get_paged_mqa_logits_metadata` path, or add a
guarded SM120-safe metadata fallback before another full-model readiness retry.

## Public Safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
