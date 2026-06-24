# DeepSeek-V4-Flash NVFP4 o_proj G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/20

## Target

- Project: `openagentsgemini`
- Instance: `hydralisk-deepseek-v4-nvfp4-g4-2g-b-20260624073921`
- Zone: `us-central1-b`
- Machine: `g4-standard-96`
- GPUs: 2 x `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- VM external IP: none
- Private egress: `hydralisk-default-nat-us-central1`
- Model: `nvidia/DeepSeek-V4-Flash-NVFP4`
- Model revision: `e3cd60e7de98e9867116860d522499a728de1cf9`
- Base image:
  `hydralisk-deepseek-v4-nvfp4-vllm:20260624074009@sha256:35149725614cca7842ba16ea02f5b3765e0b875f85e8f192f1a2ba2a4bcfc9f5`
- Derived image:
  `hydralisk-deepseek-v4-nvfp4-sm120-oproj-vllm:20260624085132@sha256:42ebf94f12e7476d533875409956bde5acb2d77baa98f64c2e4ac0ec469abc07`
- MoE backend: `flashinfer_trtllm`
- Tensor parallel size: `2`

## Script change

The provider-stack and NVFP4 G4 probe scripts now expose default-off DeepSeek
V4 `o_proj` controls:

```text
HYDRALISK_DEEPSEEK_O_PROJ_PATCH
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE
HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE
HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS
HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE
HYDRALISK_DEEPSEEK_O_PROJ_BYPASS
```

The defaults preserve prior behavior:

```text
HYDRALISK_DEEPSEEK_O_PROJ_PATCH=0
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=auto
HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE=0
HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS=0
HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE=raw_e8m0
HYDRALISK_DEEPSEEK_O_PROJ_BYPASS=off
```

No token was passed and no model artifact was committed.

## Grouped RHS run

The first issue #20 run kept the output projection active:

```text
ALLOW_NVFP4_SM120=1
VLLM_LINEAR_BACKEND=triton
VLLM_E8M0_TRITON_UPCAST=1
HYDRALISK_DEEPSEEK_O_PROJ_PATCH=1
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=blackwell
HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE=1
HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS=1
HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE=fp32
HYDRALISK_DEEPSEEK_O_PROJ_BYPASS=off
HF_HUB_DISABLE_XET=1
```

The trace matches the older native-FP8 `o_proj` investigation, now reproduced in
the provider NVFP4 container:

```json
{"einsum_recipe":[1,1,128],"o":{"dtype":"torch.bfloat16","shape":[1024,32,512]},"o_fp8":{"dtype":"torch.float8_e4m3fn","shape":[1024,4,4096]},"o_scale":{"dtype":"torch.int32","shape":[1024,4,8]},"tma_aligned_scales":true,"wo_a_weight":{"dtype":"torch.float8_e4m3fn","shape":[4096,4096]},"wo_a_weight_scale_inv":{"dtype":"torch.float8_e8m0fnu","shape":[32,32]}}
```

After grouped RHS and fp32 scale upcast:

```json
{"group_rhs":true,"rhs_scale":{"dtype":"torch.float32","shape":[4,8,32]},"rhs_scale_mode":"fp32","rhs_weight":{"dtype":"torch.float8_e4m3fn","shape":[4,1024,4096]}}
```

Readiness:

```text
READY	0
```

Blocker:

```text
RuntimeError: Assertion error (csrc/apis/layout.hpp:59):
Unknown SF transformation
```

## Zero o_proj bypass run

The second run used the explicit load-only bypass:

```text
HYDRALISK_DEEPSEEK_O_PROJ_BYPASS=zero
```

This is not a serving-quality path. It replaces the `o_proj` activation with
zeros only to learn whether another startup blocker exists beyond DeepGEMM
`o_proj`.

The bypass trace was:

```json
{"bypass":"zero","n_groups":4,"o":{"dtype":"torch.bfloat16","shape":[1024,32,512]},"o_lora_rank":1024}
```

That moved execution past `o_proj`, but the server still did not reach
`/v1/models`:

```text
READY	0
```

The next blocker is the FlashInfer TRTLLM NVFP4 MoE kernel itself:

```text
tvm.error.InternalError:
Error in function 'run' at /workspace/csrc/trtllm_batched_gemm_runner.cu:286:
Error occurred when running GEMM!
numBatches: 128
GemmMNK: 1024 4096 4096
Kernel: bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x32x512u2_s4_et128x32_m128x32x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_sm100f
```

## Decision

Issue #20 closes the `o_proj` uncertainty:

- The valid grouped-RHS/fp32 path still fails in DeepGEMM scale-factor layout
  handling.
- The invalid zero-`o_proj` load-only bypass proves that `o_proj` is not the
  only remaining G4 blocker. The next blocker is FlashInfer TRTLLM NVFP4 MoE
  GEMM execution on this RTX PRO 6000 SM120 host.

This makes the current two-card G4 stock-vLLM lane a compatibility research
path, not a near-serving path. The next useful issue should either isolate the
FlashInfer TRTLLM NVFP4 MoE kernel on SM120 with a small repro, or stop the
stock-vLLM G4 route and move to a known-good provider stack/hardware target:
H100/H200/B200/GB200 quota, DGX Station-class hardware, or a custom
SGLang/offload engine with owned kernels.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
