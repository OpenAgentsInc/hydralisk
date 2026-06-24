# DeepSeek-V4-Flash o_proj fallback wide-G4 evidence

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/25

Script:
`scripts/probe-deepseek-v4-oproj-fallback-g4-gce.sh`

Generated run directory, not committed:
`.hydralisk/deepseek-v4-oproj-fallback-g4-20260624095153/`

Remote log directory, not committed:
`/var/log/hydralisk/deepseek-provider-stack-20260624095206/`

## Scope

This probe reused the private 8 x RTX PRO 6000 G4 host and tested a
correctness-first DeepSeek V4 `o_proj` fallback in the derived vLLM probe
image. The fallback is explicit/default-off and scoped to the probe image.

- Project: `openagentsgemini`
- Instance: `hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036`
- Zone: `us-central1-b`
- Machine type: `g4-standard-384`
- GPU: 8 x `nvidia-rtx-pro-6000`
- Public ingress: none
- Model: `nvidia/DeepSeek-V4-Flash-NVFP4`
- Revision: `e3cd60e7de98e9867116860d522499a728de1cf9`
- Base image: `vllm/vllm-openai:latest`
- Derived image:
  `hydralisk-deepseek-v4-oproj-fallback-g4-vllm:20260624095206`
- Derived image digest:
  `sha256:f606b8120a907951ba311068d95acc533f3ea1235e02156ecc7b0b2799d47890`
- vLLM: `0.23.0`
- Torch: `2.11.0+cu130`
- CUDA runtime: `13.0`
- DeepGEMM: import succeeded
- Tensor parallel: `8`
- Expert parallel: enabled
- MoE backend: `flashinfer_trtllm`
- Dense linear backend: `triton`
- FP8 KV cache: enabled
- Block size: `256`
- Xet: disabled

## Fallback

The probe added `HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=bf16_einsum`.

The fallback keeps the fused inverse-RoPE FP8 quantization for `o`, forces
non-TMA activation scales with `HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=hopper`,
upcasts E8M0 weight scales to FP32, dequantizes the `wo_a` operands, computes
the `wo_a` projection through `torch.einsum`, and then calls `wo_b`.

Observed trace on the G4 host:

```text
HYDRALISK_O_PROJ_SHAPE_TRACE
o_fp8  [512,1,4096] torch.float8_e4m3fn
o_scale [512,1,32]  torch.float32
wo_a   [1024,4096] torch.float8_e4m3fn
scale  [8,32]      torch.float8_e8m0fnu
tma_aligned_scales false

HYDRALISK_O_PROJ_RHS_TRACE
rhs_weight [1,1024,4096] torch.float8_e4m3fn
rhs_scale  [1,8,32]      torch.float32

HYDRALISK_O_PROJ_FALLBACK_TRACE
lhs [512,1,4096]    torch.float32
rhs [1,1024,4096]   torch.float32
```

The same fallback trace appeared for all eight tensor/expert-parallel ranks.

## Result

The fallback moved the run past the prior DeepGEMM `o_proj` blocker:

```text
RuntimeError:
Assertion error (csrc/apis/layout.hpp:59): Unknown SF transformation
```

The run did not reach `/v1/models`. The next blocker is now the real
FlashInfer TRTLLM NVFP4 MoE GEMM path on RTX PRO 6000:

```text
tvm.error.InternalError:
Error in function 'run' at /workspace/csrc/trtllm_batched_gemm_runner.cu:286:
Error occurred when running GEMM!
numBatches: 32
GemmMNK: 512 4096 4096
Kernel: bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x64x512u2_s4_et128x32_m256x64x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100f
```

## Interpretation

Issue #25 validates the first hard thing after #24: the valid model path can
be moved past `o_proj` on the admitted 8 x G4 host without zeroing the
projection. The current blocker is now the same family as the earlier
synthetic TRTLLM NVFP4 MoE repro, but with the full model and eight GPUs:
FlashInfer/TensorRT-LLM's SM100-family NVFP4 MoE GEMM kernel is not running
successfully on RTX PRO 6000 for this DeepSeek V4 shape.

The next executable issue should stop working on `o_proj` and target the MoE
kernel path directly: either isolate the full-model `numBatches=32`,
`GemmMNK=512 4096 4096` shape in a synthetic repro, patch/avoid the TRTLLM
SM100-family path for SM120, or pivot to a custom B12x/SGLang/offload MoE
kernel lane.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
