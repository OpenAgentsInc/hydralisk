# DeepSeek B12x local dispatcher shim

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/32

Module:
[`hydralisk/admission/deepseek_v4_moe.py`](../../hydralisk/admission/deepseek_v4_moe.py)

Tests:
[`tests/test_deepseek_v4_moe.py`](../../tests/test_deepseek_v4_moe.py)

## Result

Hydralisk now owns the pure-Python local-shard dispatch contract that the G4
B12x lane needs before the next GPU attempt.

The shim:

- maps global expert IDs into a rank-local B12x expert domain;
- preserves the fixed `[tokens, top_k]` selected-expert shape expected by B12x;
- replaces nonlocal or invalid global expert IDs with a local fill expert;
- zeros route scales for those masked routes so reference skip semantics are
  preserved;
- exposes a fail-closed DeepSeek clamp gate so Hydralisk does not use a B12x
  backend for serving unless `swiglu_limit` semantics are present.

## Why this matters

The live G4 probes proved the B12x kernel can run a local 32-expert shard when
routes are already remapped into the local domain. They also proved both
FlashInfer `0.6.12` and the matched `0.6.13.dev20260612` nightly wrapper lack
the two surfaces DeepSeek-V4-Flash needs:

- `local_expert_offset` or equivalent global-to-local routing;
- `swiglu_limit=10.0` clamp semantics.

This issue implements the routing half of that missing surface in Hydralisk and
pins the clamp requirement as a fail-closed admission gate.

## Validation

Focused tests:

```text
.venv/bin/python -m pytest tests/test_deepseek_v4_moe.py -q
8 passed
```

Covered behavior:

- local expert remap preserves fixed top-k shape;
- nonlocal and out-of-range routes are masked with zero route scales;
- variable top-k route rows are rejected;
- remapped local-domain execution matches the existing global-reference fixture
  on deterministic nonzero inputs;
- missing `swiglu_limit` support raises before a DeepSeek B12x serving claim can
  proceed.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
