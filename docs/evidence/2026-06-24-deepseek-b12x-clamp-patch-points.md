# DeepSeek B12x clamp patch points

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/34

Audit command:

```bash
.venv/bin/python -m hydralisk.admission.deepseek_v4_b12x_clamp_audit \
  --output-dir .hydralisk/b12x-clamp-audit-20260624
```

Generated artifacts:

- `.hydralisk/b12x-clamp-audit-20260624/b12x-clamp-audit.json`
- `.hydralisk/b12x-clamp-audit-20260624/b12x-clamp-audit.md`

The generated artifacts are local-only evidence outputs and are not committed.
This committed note records the public-safe findings.

## Decision

Status: `b12x_clamp_missing_in_api_launch_and_kernel_terms`

The next DeepSeek-V4-Flash G4 step is not another full-model retry. The next
step is a FlashInfer B12x SM120 patch that adds DeepSeek/vLLM
`swiglu_limit=10.0` semantics to the API surface, dispatch path, and fused
SwiGLU activation code, followed by a tiny nonzero local-shard correctness
comparison against Hydralisk's reference fixture.

The user-pasted provider inventory helps by confirming the stock target shape:
vLLM `0.20.0+`, DeepGEMM, FP8 KV cache, block size `256`, tensor parallel set
to local GPU count, expert parallel enabled, and H200/B200/GB-class hardware as
the verified deployment family. It does not remove the G4 / RTX PRO 6000 B12x
clamp blocker.

## FlashInfer B12x surface

Audited reference checkout files:

- `projects/repos/flashinfer/flashinfer/fused_moe/cute_dsl/b12x_moe.py`
- `projects/repos/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py`
- `projects/repos/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py`
- `projects/repos/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py`
- `projects/repos/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py`
- `projects/repos/flashinfer/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py`

Findings:

- `b12x_fused_moe` exposes `num_local_experts`, `activation`,
  `activation_precision`, `quant_mode`, and `source_format`.
- `B12xMoEWrapper.__init__` exposes `num_local_experts`, `activation`,
  `activation_precision`, `quant_mode`, and `source_format`.
- Neither API exposes `swiglu_limit` or `gemm1_clamp_limit`.
- `launch_sm120_moe` threads activation and quantization selection, but does
  not expose or forward a clamp parameter.
- The audited B12x SM120 kernel files contain the fused gated-SiLU activation
  path signals, but no `swiglu_limit`, `gemm1_clamp_limit`, or `clamp_limit`
  term.

That means Hydralisk cannot solve the current blocker with a Python wrapper
alone. The clamp value has to reach the fused B12x activation implementation.

## vLLM contract

Audited reference checkout files:

- `projects/repos/vllm/vllm/model_executor/layers/activation.py`
- `projects/repos/vllm/vllm/model_executor/layers/fused_moe/utils.py`
- `projects/repos/vllm/vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py`
- `projects/repos/vllm/vllm/model_executor/layers/quantization/utils/fp8_utils.py`

The source signals match Hydralisk's reference fixture:

- gate branch clamp: one-sided maximum at `+limit`;
- up branch clamp: symmetric `[-limit, +limit]`;
- output: `silu(gate) * up` after clamping;
- MoE wiring: `gemm1_clamp_limit` feeds the SwiGLU clamp path for SILU
  activation.

## Patch plan

1. Add a compatible clamp keyword, preferably `gemm1_clamp_limit` at the vLLM
   boundary and/or `swiglu_limit` at the direct FlashInfer boundary, to
   `b12x_fused_moe` and `B12xMoEWrapper`.
2. Thread the value through `launch_sm120_moe` and the static, micro, dynamic,
   and W4A16 backend launch helpers.
3. Apply the vLLM-compatible gate/up clamp at the fused B12x activation point
   before FP4 re-quantization and FC2.
4. Keep Hydralisk's existing local-shard remap and zero-scale masking contract
   from issue #32/#33.
5. Validate with a tiny nonzero local-shard B12x GPU fixture before any model
   weight load or full-model retry.

## Validation

```bash
.venv/bin/python -m pytest tests/test_deepseek_v4_b12x_clamp_audit.py -q
```

Result: `4 passed`.

## Public safety

- No secrets.
- No prompts or model responses.
- No model weights or checkpoint data.
- No GPU profiler dumps or private benchmark output.
