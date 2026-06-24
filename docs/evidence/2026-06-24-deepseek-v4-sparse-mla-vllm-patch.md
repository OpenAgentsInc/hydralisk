# DeepSeek V4 sparse MLA vLLM fallback patch

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/56

## Summary

Issue #56 added a Hydralisk-owned source patcher for vLLM's DeepSeek V4
FlashInfer sparse MLA path. The patch targets:

```text
vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
DeepseekV4FlashInferMLAAttention._forward
```

It inserts a default-off branch before both
`flashinfer_trtllm_batch_decode_sparse_mla_dsv4` calls:

```text
HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK=1
```

When enabled, the branch runs a correctness-first torch fallback for the narrow
BF16/HND sparse MLA probe contract instead of FlashInfer's TRTLLM-gen FMHA
launcher. It fails closed for unsupported dtypes, KV layouts, cache shapes, and
multi-token sequence-length metadata that the current fallback does not yet
claim to serve.

## Implementation

- `hydralisk/admission/deepseek_v4_sparse_mla_vllm_patch.py`
- `scripts/patch-vllm-deepseek-sparse-mla-fallback.sh`
- console entry point:
  `hydralisk-deepseek-v4-sparse-mla-vllm-patch`

## Dry Run Against vLLM Reference

Command:

```bash
DRY_RUN=1 \
VLLM_ROOT=/Users/christopherdavid/work/projects/repos/vllm \
bash scripts/patch-vllm-deepseek-sparse-mla-fallback.sh
```

Result:

```json
{"already_patched":false,"decode_branch_patched":true,"inserted_helpers":true,"inserted_import":true,"patched":true,"prefill_branch_patched":true,"target":"/Users/christopherdavid/work/projects/repos/vllm/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py"}
```

The local vLLM reference checkout remained clean after the dry run.

## What Remains

The next live run needs a fresh or explicitly provided G4 target. Apply the
patch inside the derived vLLM image, start vLLM with
`HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK=1`, and run the issue #52-sized
synthetic container smoke before attempting another full 8 x G4 model smoke.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
