# DeepSeek-V4-Flash scaled-mm G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/8

Script:
[`scripts/probe-deepseek-v4-scaled-mm-gce.sh`](../../scripts/probe-deepseek-v4-scaled-mm-gce.sh)

## Result

The G4 blocker is now narrower than "vLLM does not work on RTX PRO 6000."

On the admitted 2 x RTX PRO 6000 host, vLLM reports CUTLASS FP8 support for
SM120, but direct CUTLASS FP8 scaled-mm fails even for tiny matrices. The
Triton block-scaled FP8 path works with ordinary float32 scales. The Triton path
fails when given `float8_e8m0fnu` scale tensors, which matches the full-model
`--linear-backend triton` failure.

This points to the next implementation step: patch or wrap the CUDA Triton
block-scaled FP8 path to upcast E8M0 scale tensors before launch, then retry
the full model with `--linear-backend triton`.

## Host

```text
Instance      hydralisk-deepseek-v4-g4-2g-b-20260624053235
Zone          us-central1-b
Machine       g4-standard-96
GPU           2 x NVIDIA RTX PRO 6000 Blackwell Server Edition
Torch         2.11.0+cu130
Torch CUDA    13.0
Capability    12.0
```

## Capability probe

vLLM support predicates returned:

| Capability | FP8 | Block FP8 | FP4 |
| ---: | --- | --- | --- |
| 80 | false | false | false |
| 90 | true | true | false |
| 100 | true | true | true |
| 120 | true | true | true |

The SM120 predicates claim support, but the direct op does not run.

## Microcases

| Case | Result | Notes |
| --- | --- | --- |
| `cutlass_fp8_16` | fail | `cutlass_scaled_mm ... scaled_mm_entry.cu:203` |
| `cutlass_fp8_m1_k4096_n4096` | fail | `cutlass_scaled_mm ... scaled_mm_entry.cu:203` |
| `cutlass_fp8_m16_k4096_n4096` | fail | `cutlass_scaled_mm ... scaled_mm_entry.cu:203` |
| `triton_block_fp8_m1_k128_n128` | pass | output `[1, 128]`, `torch.bfloat16` |
| `triton_block_fp8_m16_k4096_n4096` | pass | output `[16, 4096]`, `torch.bfloat16` |
| `triton_block_e8m0_m1_k128_n128` | fail | `KeyError: 'float8_e8m0fnu'` |

The `--linear-backend triton` full-model smoke moved the failure from CUTLASS
to the same `float8_e8m0fnu` scale issue:

```text
RuntimeError: Worker failed with error ''float8_e8m0fnu''
```

The provider/model note supplied during this run also says the expected
DeepSeek-V4-Flash vLLM recipe uses:

- vLLM `0.20.0+`;
- DeepGEMM installed via vLLM's `tools/install_deepgemm.sh`;
- `--enable-expert-parallel`;
- TP equal to GPU count;
- validated hardware such as 8 x H100/H200/B200 or single-GPU DGX Station style
  unified-memory systems.

The two-card G4 experiment is therefore a research/debug lane, not a known-good
production recipe.

## Next Step

Open the next issue to test an E8M0 scale upcast patch for vLLM's CUDA Triton
block-scaled FP8 path, then rerun:

```bash
VLLM_LINEAR_BACKEND=triton
VLLM_ENABLE_EXPERT_PARALLEL=1
scripts/smoke-deepseek-v4-gce.sh
```

If that still fails before `/v1/models`, the next branch is not more G4 flag
twiddling. It is either:

- a known-good DeepSeek-V4 vLLM image/build pin with fixed Blackwell kernels;
- an 8-GPU H100/H200/B200 allocation that matches the published recipe;
- or the custom expert-prefetch/offload route described in the DeepSeek-V4
  notes.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
