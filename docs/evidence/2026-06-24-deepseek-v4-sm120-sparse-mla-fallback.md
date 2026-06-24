# DeepSeek V4 SM120 sparse MLA fallback reference

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/54

## Summary

Issue #54 added the first correctness-first sparse MLA fallback contract for
the DeepSeek V4 RTX PRO 6000 / SM120 lane. This does not claim production
serving yet. It gives Hydralisk a deterministic local oracle for the attention
path that issue #53 proved cannot be fixed by a guard-only FlashInfer patch.

Implementation:

- `hydralisk/admission/deepseek_v4_sparse_mla.py`
- `tests/test_deepseek_v4_sparse_mla.py`

The reference covers the issue #52 tensor family:

- BF16-compatible numeric inputs represented as CPU floats;
- query shape `[query, head, dim]`;
- HND KV cache layout `[page, kv_head, page_token, dim]`;
- SWA KV and compressed KV caches with shared shape;
- sparse indices plus `sparse_topk_lens`;
- sequence-length truncation;
- single KV-head broadcast and per-head KV cache modes;
- one-token decode and small batched decode shapes.

## What This Proves

The fallback reference now pins these serving semantics before a GPU/vLLM patch:

- sparse top-k truncation is deterministic;
- sparse routes past `seq_len` are masked;
- empty-route decode returns zeros instead of undefined output;
- sliding-window routes can be bounded independently of sparse routes;
- output is finite and nonzero for a nonzero routed fixture;
- stable softmax handles large logits without overflow.

## What Remains

This is a correctness oracle, not the final runtime path. The next issue should
wire this contract into a derived vLLM DeepSeek V4 attention fallback path and
run the same public-safe synthetic shape in-container. Only after the fallback
survives the in-container synthetic call should we rerun the full 8 x G4 model
smoke.

## Validation

```bash
uv run pytest tests/test_deepseek_v4_sparse_mla.py
```

Result:

```text
9 passed
```

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
