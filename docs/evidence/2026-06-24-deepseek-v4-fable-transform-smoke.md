# DeepSeek-V4-Fable packed-LoRA transform smoke

Date: 2026-06-24T19:29:44.876970Z

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/73

Depends on: https://github.com/OpenAgentsInc/hydralisk/issues/72

Profile: `profiles/deepseek-v4-fable-adapter-g4.json`

Status: `blocked_adapter_config_payload_mismatch`

## Decision

- Packed delta can be written now: `false`
- Packed delta writer can be implemented from this manifest: `false`
- Khala general route allowed: `false`
- Public aliases allowed: `false`
- MPP public sale allowed: `false`
- Next step: `map_actual_adapter_payload_contexts_or_pivot_canonical_runtime`

## Adapter inspection

- Adapter path: `.hydralisk/deepseek-v4-fable-issue73/adapter/adapter_model.safetensors`
- Adapter file bytes: `69409832`
- Tensor count: `382`
- LoRA module count: `191`
- Header-only inspection: `true`
- Tensor values read: `false`

## Target shape summary

| Target | Modules | Complete pairs | Contexts | Ranks | Layers | Shape compatible |
| --- | ---: | ---: | --- | --- | --- | --- |
| `q_proj` | 0 | 0 | `none` | `none` | `none` | `true` |
| `k_proj` | 0 | 0 | `none` | `none` | `none` | `true` |
| `v_proj` | 0 | 0 | `none` | `none` | `none` | `true` |
| `o_proj` | 0 | 0 | `none` | `none` | `none` | `true` |
| `gate_proj` | 105 | 105 | `attention_compressor:41, attention_compressor_indexer:21, mlp:43` | `16` | `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42` | `true` |
| `up_proj` | 43 | 43 | `mlp:43` | `16` | `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42` | `true` |
| `down_proj` | 43 | 43 | `mlp:43` | `16` | `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42` | `true` |

## Packed family summary

| Family | Targets | Complete | Layers | Blockers |
| --- | --- | --- | --- | --- |
| `attention_fused_wqa_wkv` | `q_proj, k_proj, v_proj` | `false` | `none` | missing_lora_pairs, missing_lora_pairs, missing_lora_pairs |
| `swiglu_gate_up_proj` | `gate_proj, up_proj` | `false` | `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42` | unsupported_lora_context |
| `attention_output_o_proj` | `o_proj` | `false` | `none` | missing_lora_pairs |
| `direct_down_proj` | `down_proj` | `true` | `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42` | - |

## Blockers

- `missing_lora_pairs` on `q_proj`: No complete LoRA A/B pairs found for q_proj.
- `missing_lora_pairs` on `k_proj`: No complete LoRA A/B pairs found for k_proj.
- `missing_lora_pairs` on `v_proj`: No complete LoRA A/B pairs found for v_proj.
- `unsupported_lora_context` on `gate_proj`: gate_proj appears in unsupported adapter contexts for the current packed-family transform smoke.
- `missing_lora_pairs` on `o_proj`: No complete LoRA A/B pairs found for o_proj.

## Interpretation

This smoke inspects only safetensors header metadata: tensor keys, dtypes,
shapes, and byte ranges. It does not record tensor values and it does not write
transformed model artifacts.

The real Fable adapter payload does not match the published adapter config's
target-module implication. There are no `q_proj`, `k_proj`, `v_proj`, or
`o_proj` LoRA pairs in the safetensors file. The payload contains MLP/shared
expert `gate_proj`, `up_proj`, and `down_proj` tensors, plus
attention-compressor `gate_proj` tensors that need their own runtime mapping.

The next step is not the previously assumed packed q/k/v/o delta writer. It is
to map the actual adapter payload contexts against the runtime, especially
`self_attn.compressor.gate_proj` and
`self_attn.compressor.indexer.gate_proj`, or pivot to a canonical runtime that
can load and validate the adapter exactly as published.

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains tensor values: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
- Contains target details: false
