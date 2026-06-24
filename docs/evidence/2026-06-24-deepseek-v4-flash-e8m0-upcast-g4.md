# DeepSeek-V4-Flash E8M0 upcast G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/9

Scripts:

- [`scripts/patch-vllm-e8m0-triton-upcast-gce.sh`](../../scripts/patch-vllm-e8m0-triton-upcast-gce.sh)
- [`scripts/probe-deepseek-v4-scaled-mm-gce.sh`](../../scripts/probe-deepseek-v4-scaled-mm-gce.sh)
- [`scripts/smoke-deepseek-v4-gce.sh`](../../scripts/smoke-deepseek-v4-gce.sh)

## Result

The E8M0/Triton hypothesis was correct but not sufficient to serve the model.

The host-local vLLM patch removed the previous Triton blocker:
`triton_block_e8m0_m1_k128_n128` now passes on the 2 x RTX PRO 6000 G4 host.
The full DeepSeek-V4-Flash load then advanced to the next startup blocker in
the DeepSeek NVIDIA `o_proj` path, where vLLM calls DeepGEMM `fp8_einsum` and
DeepGEMM asserts on the expected scale/tensor layout.

The model still does not reach `/v1/models`.

## Host

```text
Instance      hydralisk-deepseek-v4-g4-2g-b-20260624053235
Zone          us-central1-b
Machine       g4-standard-96
GPU           2 x NVIDIA RTX PRO 6000 Blackwell Server Edition
Torch         2.11.0+cu130
Torch CUDA    13.0
Capability    12.0
vLLM          0.23.0
```

## Patch Applied

The patch is host-local and reversible. It backs up:

```text
/opt/hydralisk-deepseek-v4/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py
```

to:

```text
/opt/hydralisk-deepseek-v4/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/utils/fp8_utils.py.hydralisk-e8m0-upcast.bak
```

It changes vLLM's Triton block-scaled FP8 helper so E8M0 scale tensors are
decoded to fp32 on CUDA as well as ROCm/XPU before launching Triton.

Patch receipt:

```json
{"action":"apply","changed":true,"ok":true,"schema":"hydralisk.deepseek-v4.e8m0-triton-upcast-patch.v1"}
```

## Microprobe After Patch

| Case | Result | Notes |
| --- | --- | --- |
| `cutlass_fp8_16` | fail | unchanged CUTLASS SM120 issue |
| `cutlass_fp8_m1_k4096_n4096` | fail | unchanged CUTLASS SM120 issue |
| `cutlass_fp8_m16_k4096_n4096` | fail | unchanged CUTLASS SM120 issue |
| `triton_block_fp8_m1_k128_n128` | pass | output `[1, 128]`, `torch.bfloat16` |
| `triton_block_fp8_m16_k4096_n4096` | pass | output `[16, 4096]`, `torch.bfloat16` |
| `triton_block_e8m0_m1_k128_n128` | pass | this was the issue #8 failure |

## Full Smoke After Patch

Command shape:

```bash
ISSUE_NUMBER=9
TARGET_INSTANCE=hydralisk-deepseek-v4-g4-2g-b-20260624053235
TARGET_ZONE=us-central1-b
TARGET_GPU_COUNT=2
FORCE_PYTHON_VLLM=1
REUSE_PYTHON_VENV=1
VLLM_LINEAR_BACKEND=triton
VLLM_ENABLE_EXPERT_PARALLEL=1
VLLM_USE_DEEP_GEMM=0
VLLM_USE_DEEP_GEMM_E8M0=0
VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES=0
VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0
scripts/smoke-deepseek-v4-gce.sh
```

Engine evidence:

```text
BACKEND	python_vllm
vllm	0.23.0
torch	2.11.0
CUDA_HOME	/usr/local/cuda-12.9
VLLM_LINEAR_BACKEND	triton
VLLM_ENABLE_EXPERT_PARALLEL	1
FORCE_PYTHON_VLLM	1
REUSE_PYTHON_VENV	1
```

Readiness:

```text
READY	0
```

New blocker:

```text
RuntimeError: Assertion error
(/workspace/.deps/deepgemm-src/csrc/apis/../jit_kernels/impls/../heuristics/../../utils/layout.hpp:39):
t.dim() == N
```

The vLLM stack points at:

```text
vllm/models/deepseek_v4/nvidia/ops/o_proj.py:61 deep_gemm_fp8_o_proj
vllm/utils/deep_gemm.py:303 fp8_einsum
```

The relevant recipe selector on this host is:

```python
einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
tma_aligned_scales = cap.major >= 10
```

On SM120, that chooses the Blackwell path:

```text
einsum_recipe       (1, 1, 128)
tma_aligned_scales  true
```

## Next Step

Open the next issue to isolate DeepSeek-V4's NVIDIA `o_proj` DeepGEMM
`fp8_einsum` layout failure on RTX PRO 6000:

- capture the exact `o_fp8`, `o_scale`, `wo_a.weight`, and
  `wo_a.weight_scale_inv` shapes/dtypes without dumping model weights;
- test whether forcing the Hopper layout recipe `(1, 128, 128)` with
  `tma_aligned_scales=false` gets past `o_proj`;
- if not, test whether the failure is a DeepGEMM build/kernel mismatch and
  pin a known-good vLLM/DeepGEMM build before writing more hotpatches.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
