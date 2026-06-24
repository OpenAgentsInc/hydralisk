# DeepSeek-V4-Flash G4 backend fallback matrix

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/7

Depends on:
https://github.com/OpenAgentsInc/hydralisk/issues/6

## Result

The first backend fallback matrix did not make DeepSeek-V4-Flash serve on the
admitted Google G4 host. It narrowed the blocker: vLLM `0.23.0` still dispatches
through the same CUTLASS FP8 scaled-mm failure on 2 x RTX PRO 6000 even when
DeepGEMM, UE8M0, TMA-aligned scales, and block-scale FP8 FlashInfer toggles are
disabled.

This means the next implementation target is not more quota discovery. It is a
kernel/runtime compatibility lane: either pin a newer known-good vLLM/NVIDIA
container, force a truly different scaled-mm backend, patch vLLM kernel
selection for this checkpoint and GPU family, or test the same checkpoint on a
Hopper host to distinguish RTX PRO 6000 Blackwell support from model-general
support.

## Matrix

All attempts reused the same fresh non-product probe:

```text
Instance  hydralisk-deepseek-v4-g4-2g-b-20260624053235
Zone      us-central1-b
Machine   g4-standard-96
GPUs      2 x NVIDIA RTX PRO 6000 Blackwell Server Edition
Driver    580.159.03
CUDA_HOME /usr/local/cuda-12.9
vLLM      0.23.0
torch     2.11.0
```

| Attempt | DeepGEMM | UE8M0 | TMA scales | block FP8 FlashInfer | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| default | 1 | 1 | 1 | 1 | `READY 0`; failed in `cutlass_scaled_mm` |
| deepgemm0 | 0 | 0 | 0 | 1 | `READY 0`; same `cutlass_scaled_mm` failure |
| blockfp8off | 0 | 0 | 0 | 0 | `READY 0`; same `cutlass_scaled_mm` failure |

## Backend evidence

The runner now records these values in
`/var/log/hydralisk/deepseek-engine-evidence.txt` before vLLM starts:

```text
VLLM_USE_DEEP_GEMM
VLLM_USE_DEEP_GEMM_E8M0
VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES
VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER
```

The most conservative attempted fallback recorded:

```text
VLLM_USE_DEEP_GEMM                         0
VLLM_USE_DEEP_GEMM_E8M0                    0
VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES      0
VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER        0
READY                                      0
```

Public completion receipt:

```json
{"ready":false,"status":"server_not_ready_or_exited"}
```

## Shared blocker

All model-start attempts failed at the same narrow runtime site:

```text
File "/opt/hydralisk-deepseek-v4/.venv/lib/python3.12/site-packages/vllm/_custom_ops.py",
line 908, in cutlass_scaled_mm
torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)
RuntimeError: dispatch_scaled_mm,
/workspace/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:17
```

The API server never reached `/v1/models`; no public model endpoint was exposed.

## Next patch target

The next hard issue should focus on one of these in order:

1. Pin and test a DeepSeek-V4-specific vLLM image or build newer than the
   current `vllm>=0.20.0` resolver result.
2. Find the supported `KernelConfig` / CLI path for forcing a non-CUTLASS
   scaled-mm backend in vLLM `0.23.0`.
3. Build a tiny isolated `cutlass_scaled_mm` / block-scaled FP8 repro on the G4
   host so the failing shape can be upstreamed or patched without loading the
   full model every time.
4. Re-run the same smoke on `a3-highgpu-2g` if Google admits H100 capacity, to
   separate RTX PRO 6000 Blackwell support from DeepSeek-V4 support generally.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
