# DeepSeek-V4-Flash NVFP4 Triton-linear G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/19

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
  `hydralisk-deepseek-v4-nvfp4-sm120-triton-vllm:20260624084245@sha256:e923fb6576796c166e7d5be017eb012552a1a7b069ded657a2cd51947e16d0b7`
- MoE backend: `flashinfer_trtllm`
- Tensor parallel size: `2`

## Script change

The provider-stack and NVFP4 G4 probe scripts now expose two dense-FP8 controls:

```text
VLLM_LINEAR_BACKEND
VLLM_E8M0_TRITON_UPCAST
```

The defaults preserve prior behavior:

```text
VLLM_LINEAR_BACKEND=auto
VLLM_E8M0_TRITON_UPCAST=0
```

For this run:

```text
ALLOW_NVFP4_SM120=1
VLLM_LINEAR_BACKEND=triton
VLLM_E8M0_TRITON_UPCAST=1
HF_HUB_DISABLE_XET=1
MOE_BACKEND=flashinfer_trtllm
```

No token was passed and no model artifact was committed.

## Result

The derived image built successfully and applied both default-off patches:

```text
patched trtllm_nvfp4_moe.py for NVFP4 SM120 probe
patched fp8_utils.py for CUDA Triton E8M0 upcast probe
```

The engine record confirmed that vLLM received the Triton linear backend:

```text
VLLM_LINEAR_BACKEND	triton
VLLM_E8M0_TRITON_UPCAST	1
PROVIDER_FLAGS	--kv-cache-dtype fp8 --block-size 256 --enable-expert-parallel --tensor-parallel-size 2 --linear-backend triton
```

The run did not reach `/v1/models`:

```text
READY	0
```

It also did not fail in the previous CUTLASS scaled-mm path. The blocker moved
to DeepSeek's NVIDIA `o_proj` path:

```text
vllm/models/deepseek_v4/nvidia/flashmla.py
vllm/models/deepseek_v4/nvidia/ops/o_proj.py
vllm/utils/deep_gemm.py
RuntimeError: Assertion error
(csrc/apis/../jit_kernels/impls/../heuristics/../../utils/layout.hpp:39):
t.dim() == N
```

## Decision

Issue #19 answered its question: forcing dense FP8 linear layers away from
CUTLASS works well enough to remove the `dispatch_scaled_mm` blocker.

The next blocker is now the DeepSeek V4 NVIDIA `o_proj` implementation, which
still routes through DeepGEMM `fp8_einsum` and rejects the current tensor/scale
layout before readiness. The next hard issue should patch or bypass
`deep_gemm_fp8_o_proj` for the NVFP4 G4 path, ideally with a public-safe shape
trace first and then a minimal fallback that preserves the rest of the
FlashInfer TRTLLM NVFP4 expert path.

The pasted provider inventory remains useful as a reference target, not as an
immediate Google solution: the public recipe wants vLLM `0.20.0+`, DeepGEMM,
FP8 KV cache, block size `256`, parser flags, and H100/H200/B200/GB200-class
or DGX Station hardware. The admitted Google lane here remains a two-card G4
compatibility probe.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
