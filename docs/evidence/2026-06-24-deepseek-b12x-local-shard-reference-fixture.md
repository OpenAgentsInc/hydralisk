# DeepSeek B12x local-shard reference fixture

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/29

Commit scope:

- `hydralisk/admission/deepseek_v4_moe.py`
- `tests/test_deepseek_v4_moe.py`

## Why this exists

Issue #28 proved the live B12x kernel can run the exact per-rank shard shape
when global experts are remapped into the local kernel domain. The remaining
G4 path needs a correctness boundary before any GPU patch or wrapper upgrade:

- DeepSeek/vLLM clamp semantics for `swiglu_limit=10.0`.
- Global-to-local expert ID remapping for rank-owned expert intervals.
- Skipping nonlocal experts.
- Accumulating routed local experts with nonzero data.

## Reference semantics

The reference follows the vLLM clamp behavior found in
`projects/repos/vllm/vllm/model_executor/layers/fused_moe/utils.py`:

```text
gate = clamp(gate, max=limit)
up = clamp(up, min=-limit, max=limit)
output = silu(gate) * up
```

The local-shard MoE fixture follows the FlashInfer B12x test convention from
`projects/repos/flashinfer/tests/moe/test_b12x_fused_moe.py`: first half of
GEMM1 output is the linear/up branch, second half is the gate branch.

## Tests

The added tests cover:

- clamp changes saturated positive gate and negative up values;
- global expert IDs map into a local rank interval;
- nonlocal experts are skipped;
- a tiny nonzero routed MoE fixture is deterministic.

Focused validation:

```text
.venv/bin/python -m pytest tests/test_deepseek_v4_moe.py -q
4 passed
```

## Decision

The G4/B12x lane now has a local correctness target independent of GPU kernel
availability. The next implementation issue should either:

- test a newer FlashInfer/vLLM B12x wrapper that can expose local expert offset
  and clamp-compatible semantics; or
- implement a Hydralisk-local dispatcher shim plus GPU/non-GPU comparison that
  proves B12x local-shard output matches this reference for nonzero inputs.

Do not retry the full DeepSeek-V4 model until a tiny nonzero B12x path matches
this reference with the clamp enabled.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
