# DeepSeek FlashMLA sparse-prefill SM120 audit

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/40

## Summary

Issue #39 proved that eager mode can load the pinned DeepSeek-V4-Flash NVFP4
B12x/o_proj-fallback stack to `/v1/models` on 8 x G4, but first generation
fails in `flash_mla_sparse_fwd`:

```text
RuntimeError: Sparse Attention Forward Kernel is only supported on SM90a and SM100f architectures.
```

Issue #40 audited the local vLLM reference source without loading model
weights, prompts, responses, or GPU kernels.

## Decision

Status: `existing_flashinfer_sparse_backend_is_next_sm120_probe`

The next G4 probe should select vLLM's existing DeepSeek V4 FlashInfer sparse
MLA backend before Hydralisk writes or patches a FlashMLA sparse-prefill
kernel.

Use:

```text
VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4
```

Hydralisk's provider-stack wrappers now translate that environment variable
into:

```text
--attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV4"}'
```

## Findings

- `vllm/models/deepseek_v4/nvidia/flashmla.py` implements
  `DeepseekV4FlashMLAAttention`.
- Its prefill path calls `flash_mla_sparse_fwd`.
- `vllm/v1/attention/ops/flashmla.py` exposes a sparse FlashMLA support guard
  for capability families 90 and 100, not 120.
- `vllm/models/deepseek_v4/nvidia/model.py` has an explicit selector:
  `AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4` returns
  `DeepseekV4FlashInferMLAAttention`.
- `vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py` avoids
  `flash_mla_sparse_fwd`, uses plain KV layout instead of FlashMLA's
  `fp8_ds_mla`, builds FlashInfer mixed sparse indices, and calls
  `flashinfer_trtllm_batch_decode_sparse_mla_dsv4`.
- vLLM's CLI exposes `--attention-config`, and `AttentionConfig.backend`
  parses backend names through `AttentionBackendEnum`.

## Fallback Plan

1. Rerun the eager B12x/o_proj-fallback G4 smoke with
   `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4`.
2. If the FlashInfer path reaches generation, keep it as the G4 attention lane
   and collect latency/quality receipts.
3. If FlashInfer fails, capture its exact kernel/runtime blocker before
   touching any FlashMLA SM120 architecture guard.
4. Only implement a correctness-first Python/Triton sparse-prefill fallback
   after proving no existing vLLM DeepSeek V4 backend works on SM120.

## Validation

```text
bash -n scripts/probe-deepseek-v4-provider-stack-gce.sh scripts/probe-deepseek-v4-b12x-g4-gce.sh
.venv/bin/python -m pytest tests/test_deepseek_v4_flashmla_sparse_audit.py tests/test_deepseek_v4_preflight.py -q
.venv/bin/python -m hydralisk.admission.deepseek_v4_flashmla_sparse_audit --vllm-root /Users/christopherdavid/work/projects/repos/vllm --output-dir .hydralisk/flashmla-sparse-audit-issue-40
```

Focused tests: `25 passed`.

## Public Safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
