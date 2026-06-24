# DeepSeek-V4-Flash grouped o_proj RHS G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/11

Scripts:

- [`scripts/patch-vllm-deepseek-o-proj-rhs-gce.sh`](../../scripts/patch-vllm-deepseek-o-proj-rhs-gce.sh)
- [`scripts/smoke-deepseek-v4-gce.sh`](../../scripts/smoke-deepseek-v4-gce.sh)

## Result

The grouped RHS layout patch moved the failure.

Before this patch, DeepGEMM failed with:

```text
layout.hpp:39: t.dim() == N
```

After viewing the right-hand side into grouped layout, DeepGEMM receives the
expected 3D grouped RHS tensors and the failure changes to:

```text
layout.hpp:59: Unknown SF transformation
```

The model still does not reach `/v1/models`.

## Host

```text
Instance      hydralisk-deepseek-v4-g4-2g-b-20260624053235
Zone          us-central1-b
Machine       g4-standard-96
GPU           2 x NVIDIA RTX PRO 6000 Blackwell Server Edition
vLLM          0.23.0
Torch         2.11.0+cu130
CUDA          13.0 runtime, CUDA_HOME=/usr/local/cuda-12.9
```

## Patch Applied

The patch is host-local and reversible. It backs up:

```text
/opt/hydralisk-deepseek-v4/.venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py
```

to:

```text
/opt/hydralisk-deepseek-v4/.venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py.hydralisk-o-proj-rhs.bak
```

It adds:

- `HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS=0|1`
- `HYDRALISK_O_PROJ_RHS_TRACE` public-safe shape/dtype logging

Patch receipt:

```json
{"action":"apply","changed":true,"ok":true,"schema":"hydralisk.deepseek-v4.o-proj-rhs-patch.v1"}
```

## Full Smoke

Command shape:

```bash
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=blackwell
HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE=1
HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS=1
VLLM_USE_DEEP_GEMM=1
VLLM_USE_DEEP_GEMM_E8M0=1
VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=1
VLLM_LINEAR_BACKEND=triton
VLLM_ENABLE_EXPERT_PARALLEL=1
FORCE_PYTHON_VLLM=1
REUSE_PYTHON_VENV=1
scripts/smoke-deepseek-v4-gce.sh
```

Engine evidence:

```text
BACKEND	python_vllm
vllm	0.23.0
torch	2.11.0
VLLM_USE_DEEP_GEMM	1
VLLM_USE_DEEP_GEMM_E8M0	1
VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES	1
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE	blackwell
HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE	1
HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS	1
```

Trace before grouping:

```json
{"einsum_recipe":[1,1,128],"o_fp8":{"dtype":"torch.float8_e4m3fn","shape":[1024,4,4096]},"o_scale":{"dtype":"torch.int32","shape":[1024,4,8]},"tma_aligned_scales":true,"wo_a_weight":{"dtype":"torch.float8_e4m3fn","shape":[4096,4096]},"wo_a_weight_scale_inv":{"dtype":"torch.float8_e8m0fnu","shape":[32,32]}}
```

Trace after grouping:

```json
{"group_rhs":true,"rhs_weight":{"dtype":"torch.float8_e4m3fn","shape":[4,1024,4096]},"rhs_scale":{"dtype":"torch.float8_e8m0fnu","shape":[4,8,32]}}
```

Readiness:

```text
READY	0
```

New blocker:

```text
RuntimeError: Assertion error
(/workspace/.deps/deepgemm-src/csrc/apis/layout.hpp:59):
Unknown SF transformation
```

## Interpretation

The grouped RHS shape was necessary: it moved DeepGEMM past the old rank
assertion. The next blocker is now scale-factor format. The grouped
`rhs_scale` is still `torch.float8_e8m0fnu` shaped `[4,8,32]`, and DeepGEMM
does not know which scale-factor transformation to apply for that combination
with the Blackwell recipe.

## Next Step

Open the next issue to test an env-controlled grouped RHS scale mode:

- `raw_e8m0`: current behavior, fails with `Unknown SF transformation`;
- `fp32`: upcast grouped `rhs_scale` to fp32 before `fp8_einsum`;
- `deepgemm_transform`: call DeepGEMM's scale transform helper on the grouped
  scale if the installed API exposes a compatible Python entrypoint.

If none of those moves the failure, stop local hotpatching and pin a known-good
vLLM/DeepGEMM build or use a published-recipe H100/H200/B200 allocation.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
