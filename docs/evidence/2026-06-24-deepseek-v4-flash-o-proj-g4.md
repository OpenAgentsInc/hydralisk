# DeepSeek-V4-Flash o_proj DeepGEMM G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/10

Scripts:

- [`scripts/patch-vllm-deepseek-o-proj-gce.sh`](../../scripts/patch-vllm-deepseek-o-proj-gce.sh)
- [`scripts/smoke-deepseek-v4-gce.sh`](../../scripts/smoke-deepseek-v4-gce.sh)

## Result

The `o_proj` blocker is not fixed by simply changing the SM120
`fp8_einsum` recipe.

The host-local patch made DeepSeek's NVIDIA `o_proj` recipe env-selectable and
added public-safe shape/dtype tracing. Both tested recipes still fail before
`/v1/models` with the same DeepGEMM assertion:

```text
RuntimeError: Assertion error
(/workspace/.deps/deepgemm-src/csrc/apis/../jit_kernels/impls/../heuristics/../../utils/layout.hpp:39):
t.dim() == N
```

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
/opt/hydralisk-deepseek-v4/.venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py.hydralisk-o-proj-recipe.bak
```

It adds:

- `HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=auto|hopper|blackwell`
- `HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE=0|1`

Patch receipt:

```json
{"action":"apply","changed":true,"ok":true,"schema":"hydralisk.deepseek-v4.o-proj-recipe-patch.v1"}
```

## Hopper Recipe Test

Command shape:

```bash
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=hopper
HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE=1
VLLM_LINEAR_BACKEND=triton
VLLM_ENABLE_EXPERT_PARALLEL=1
FORCE_PYTHON_VLLM=1
REUSE_PYTHON_VENV=1
scripts/smoke-deepseek-v4-gce.sh
```

Trace:

```json
{"einsum_recipe":[1,128,128],"o":{"dtype":"torch.bfloat16","shape":[1024,32,512]},"o_fp8":{"dtype":"torch.float8_e4m3fn","shape":[1024,4,4096]},"o_scale":{"dtype":"torch.float32","shape":[1024,4,32]},"tma_aligned_scales":false,"wo_a_weight":{"dtype":"torch.float8_e4m3fn","shape":[4096,4096]},"wo_a_weight_scale_inv":{"dtype":"torch.float8_e8m0fnu","shape":[32,32]}}
```

Result: `READY 0`, same DeepGEMM `layout.hpp:39` assertion.

## Blackwell Recipe Test

Command shape:

```bash
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=blackwell
HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE=1
VLLM_USE_DEEP_GEMM=1
VLLM_USE_DEEP_GEMM_E8M0=1
VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=1
VLLM_LINEAR_BACKEND=triton
VLLM_ENABLE_EXPERT_PARALLEL=1
FORCE_PYTHON_VLLM=1
REUSE_PYTHON_VENV=1
scripts/smoke-deepseek-v4-gce.sh
```

Trace:

```json
{"einsum_recipe":[1,1,128],"o":{"dtype":"torch.bfloat16","shape":[1024,32,512]},"o_fp8":{"dtype":"torch.float8_e4m3fn","shape":[1024,4,4096]},"o_scale":{"dtype":"torch.int32","shape":[1024,4,8]},"tma_aligned_scales":true,"wo_a_weight":{"dtype":"torch.float8_e4m3fn","shape":[4096,4096]},"wo_a_weight_scale_inv":{"dtype":"torch.float8_e8m0fnu","shape":[32,32]}}
```

Result: `READY 0`, same DeepGEMM `layout.hpp:39` assertion.

## Interpretation

The active issue appears to be the grouped `fp8_einsum` input layout, not just
the SM120 recipe tuple.

`deep_gemm_fp8_o_proj` calls:

```python
fp8_einsum(
    "bhr,hdr->bhd",
    (o_fp8, o_scale),
    (wo_a.weight, wo_a.weight_scale_inv),
    z,
    recipe=einsum_recipe,
)
```

The left side is grouped:

```text
o_fp8: [1024, 4, 4096]
```

But the right side is still flat:

```text
wo_a.weight:            [4096, 4096]
wo_a.weight_scale_inv:  [32, 32]
```

For 4 groups and `o_lora_rank=1024`, the next hypothesis is that the right
side should be viewed or transformed into grouped layout before `fp8_einsum`,
for example:

```text
wo_a.weight:            [4, 1024, 4096]
wo_a.weight_scale_inv:  [4, 8, 32]
```

If that fails, this is likely a vLLM/DeepGEMM build mismatch for DeepSeek-V4 on
RTX PRO 6000, and the next useful branch is a known-good image/build pin or
hardware that matches the published H100/H200/B200 recipe.

## Next Step

Open the next issue to test a guarded grouped `o_proj` weight/scale layout
patch:

- keep E8M0/Triton upcast enabled;
- keep the public-safe shape trace;
- view or transform `wo_a.weight` and `wo_a.weight_scale_inv` into the grouped
  shapes DeepGEMM likely expects;
- rerun the full smoke with the official-ish DeepGEMM flags enabled.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
