# DeepSeek-V4-Flash o_proj RHS scale-mode G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/12

Scripts:

- [`scripts/patch-vllm-deepseek-o-proj-rhs-scale-gce.sh`](../../scripts/patch-vllm-deepseek-o-proj-rhs-scale-gce.sh)
- [`scripts/smoke-deepseek-v4-gce.sh`](../../scripts/smoke-deepseek-v4-gce.sh)

## Result

The grouped RHS scale-mode probe did not get DeepSeek-V4-Flash to
`/v1/models`.

All tested scale modes still fail in DeepGEMM before readiness. This is the
point where more local `o_proj` hotpatching is lower-value than pinning or
building a known-good DeepSeek vLLM/DeepGEMM stack.

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
/opt/hydralisk-deepseek-v4/.venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py.hydralisk-o-proj-rhs-scale.bak
```

It adds:

- `HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE=raw_e8m0|fp32|deepgemm_transform|deepgemm_transform_fp32`

Patch receipt:

```json
{"action":"apply","changed":true,"ok":true,"schema":"hydralisk.deepseek-v4.o-proj-rhs-scale-patch.v1"}
```

## Tested Modes

### `raw_e8m0`

Source: issue #11.

```text
rhs_weight  [4,1024,4096] torch.float8_e4m3fn
rhs_scale   [4,8,32]      torch.float8_e8m0fnu
```

Failure:

```text
layout.hpp:59: Unknown SF transformation
```

### `fp32`

```text
rhs_weight  [4,1024,4096] torch.float8_e4m3fn
rhs_scale   [4,8,32]      torch.float32
```

Failure:

```text
layout.hpp:59: Unknown SF transformation
```

### `deepgemm_transform`

Attempted to call DeepGEMM's exposed Python scale-transform helper directly on
the grouped E8M0 RHS scale.

Failure:

```text
layout.hpp:93: sf_dtype == torch::kFloat or sf_dtype == torch::kInt
```

### `deepgemm_transform_fp32`

Upcasted grouped E8M0 RHS scale to fp32, then called DeepGEMM's exposed Python
scale-transform helper.

Trace still showed:

```text
rhs_weight  [4,1024,4096] torch.float8_e4m3fn
rhs_scale   [4,8,32]      torch.float32
```

Failure:

```text
layout.hpp:59: Unknown SF transformation
```

## Interpretation

The local patches proved several useful facts:

- CUTLASS FP8 scaled-mm is not usable on this RTX PRO 6000/vLLM build.
- Triton block FP8 works once E8M0 scales are decoded.
- `o_proj` needs grouped RHS rank; grouping moves the error.
- The grouped RHS scale-factor format still does not match what this DeepGEMM
  build expects.

At this point, the best next move is no longer another ad hoc `o_proj`
hotpatch. The next issue should pin or build a known-good DeepSeek-V4
vLLM/DeepGEMM stack, preferably from the exact recipe/image/revision used by
the model-card guidance. If that cannot run on the G4 host, move to a
published-recipe 8-GPU H100/H200/B200 allocation or stop claiming a plausible
G4 stock-vLLM route.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
