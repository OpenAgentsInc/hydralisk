# DeepSeek-V4-Flash clamp-backend wide-G4 evidence

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/24

Script:
`scripts/probe-deepseek-v4-clamp-backends-g4-gce.sh`

Generated run directory, not committed:
`.hydralisk/deepseek-v4-clamp-backends-g4-20260624093522/`

## Scope

This probe reused the private 8 x RTX PRO 6000 G4 host from the B12x full-model
attempt and tested the provider-recommended clamp-capable NVFP4 MoE backend
lane against the pinned NVIDIA DeepSeek-V4-Flash artifact.

- Project: `openagentsgemini`
- Instance: `hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036`
- Zone: `us-central1-b`
- Machine type: `g4-standard-384`
- GPU: 8 x `nvidia-rtx-pro-6000`
- Public ingress: none
- Model: `nvidia/DeepSeek-V4-Flash-NVFP4`
- Revision: `e3cd60e7de98e9867116860d522499a728de1cf9`
- Engine image lineage: `vllm/vllm-openai:latest`, vLLM `0.23.0`,
  Torch `2.11.0+cu130`, DeepGEMM installed through the vLLM helper
- Shared flags: FP8 KV cache, block size `256`, TP `8`, expert parallel on,
  Triton dense linear backend, E8M0 Triton scale upcast, Xet disabled

## Provider notes cross-check

The provider inventory points at the same basic serving shape for
DeepSeek-V4-Flash: vLLM `0.20.0+`, DeepGEMM, FP8 KV cache, block size `256`,
expert parallel, DeepSeek parser flags, and tensor parallelism equal to GPU
count. It also lists the real published NVIDIA target shapes as 8 x H100,
8 x H200, 8 x B200, 4 x GB200/GB300 class hosts, or DGX Station class
single-GPU hosts.

This G4 probe is therefore a compatibility lane on the Google hardware we can
actually admit today. It is not proof that RTX PRO 6000 is an officially
supported DeepSeek-V4-Flash serving target.

## Result

No backend reached `/v1/models`.

`flashinfer_cutlass` failed during model initialization before weight readiness:

```text
ValueError:
Model sets swiglu_limit=10.0, but the explicitly requested
moe_backend='flashinfer_cutlass' does not apply the SwiGLU clamp.
Use 'flashinfer_trtllm' or 'flashinfer_cutlass' instead.
```

That makes the current vLLM error hint self-contradictory for this image:
`flashinfer_cutlass` is presented as a clamp-capable suggestion, but it is
rejected by the same clamp gate.

`flashinfer_trtllm` progressed further:

- vLLM selected `FLASHINFER_TRTLLM`.
- Expert parallel was active across eight ranks.
- The observed expert split was 32 local experts per rank out of 256 global
  experts.
- Model artifacts loaded far enough to reach engine startup/profiling.

It then failed in DeepSeek V4's NVIDIA attention `o_proj` path:

```text
RuntimeError:
Assertion error (csrc/apis/layout.hpp:59): Unknown SF transformation
```

The public stack path was:

```text
vllm/models/deepseek_v4/attention.py
vllm/models/deepseek_v4/nvidia/flashmla.py
vllm/models/deepseek_v4/nvidia/ops/o_proj.py
vllm/utils/deep_gemm.py fp8_einsum
```

## Interpretation

The next blocker is no longer Google admission, NAT, Hugging Face artifact
transfer, the original dense FP8 CUTLASS failure, B12x expert parallelism, or
B12x missing SwiGLU clamp. On the admitted wide-G4 host, the only currently
credible stock-ish backend is `flashinfer_trtllm`, and that path is now blocked
by DeepGEMM scale-factor layout handling inside `o_proj`.

The next executable step is to implement and validate a correctness-first
serving fallback for DeepSeek V4 `o_proj` on G4 that does not rely on the
failing DeepGEMM scale-factor transform. If that fallback reaches the MoE
kernels, the run will also tell us whether the earlier TRTLLM synthetic MoE
failure remains a full-model blocker on eight G4 GPUs with expert parallel.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
