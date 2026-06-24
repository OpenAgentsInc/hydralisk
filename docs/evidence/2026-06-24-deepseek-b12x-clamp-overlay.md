# DeepSeek B12x clamp overlay

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/35

This issue adds a Hydralisk-owned patch overlay for the next FlashInfer B12x
G4 attempt. It does not claim DeepSeek-V4-Flash serving readiness. It makes
the source edit repeatable and source-validated before a GPU compile/runtime
attempt.

## Added tool

```bash
.venv/bin/python -m hydralisk.admission.deepseek_v4_b12x_clamp_patch \
  apply \
  --flashinfer-root /path/to/flashinfer \
  --dry-run
```

Use `validate` after applying to a copied checkout or container site-packages
tree:

```bash
.venv/bin/python -m hydralisk.admission.deepseek_v4_b12x_clamp_patch \
  validate \
  --flashinfer-root /path/to/flashinfer
```

The console entry is:

```bash
hydralisk-deepseek-v4-b12x-clamp-patch
```

## What the overlay changes

The overlay targets:

- `flashinfer/fused_moe/cute_dsl/b12x_moe.py`
- `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py`
- `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py`
- `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py`
- `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py`
- `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py`

It makes the first source-level pass concrete:

- `b12x_fused_moe` accepts `swiglu_limit`.
- `B12xMoEWrapper` accepts and stores `swiglu_limit`.
- both top-level API paths forward `swiglu_limit` into `launch_sm120_moe`;
- `launch_sm120_moe` accepts and normalizes `swiglu_limit`;
- static, micro, dynamic, and W4A16 fused gated-SiLU activation sites are
  marked with `HYDRALISK_B12X_SWIGLU_CLAMP_PATCH_POINT` and the exact vLLM
  clamp expression contract:
  - `gate=min(gate, limit)`;
  - `up=clamp(up, -limit, limit)`.

The current overlay intentionally stops short of claiming a compiled GPU
kernel. The next G4 issue has to convert the marked activation sites into
actual CuTe/CUTLASS clamp operations, compile the patched tree on RTX PRO
6000, and compare a tiny nonzero local-shard output against Hydralisk's
reference fixture.

## Real source dry-run

Command:

```bash
.venv/bin/python -m hydralisk.admission.deepseek_v4_b12x_clamp_patch \
  apply \
  --flashinfer-root /path/to/work/projects/repos/flashinfer \
  --dry-run
```

Result:

- all 12 overlay edits reported `would_change`;
- validation returned `ok: true`;
- no local FlashInfer reference files were modified.

Validation signals:

- `b12xFusedMoeHasSwigluLimit: true`
- `b12xWrapperHasSwigluLimit: true`
- `launchSm120MoeHasSwigluLimit: true`
- `apiForwardsSwigluLimit: true`
- `wrapperForwardsSwigluLimit: true`
- `dispatchNormalizesSwigluLimit: true`
- all four audited kernel files have the patch marker and clamp expression.

## Tests

```bash
.venv/bin/python -m pytest tests/test_deepseek_v4_b12x_clamp_patch.py -q
```

Result: `4 passed`.

Test coverage:

- unpatched fixture validation fails;
- dry-run validates the in-memory patched tree without writing files;
- apply is idempotent;
- missing patch anchors fail closed.

## Public safety

- No secrets.
- No prompts or model responses.
- No model weights or checkpoint data.
- No GPU profiler dumps or private benchmark output.
