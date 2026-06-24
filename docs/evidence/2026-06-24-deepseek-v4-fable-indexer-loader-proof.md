# DeepSeek-V4-Fable indexer-compressor loader proof

Date: 2026-06-24T19:41:01.461401Z

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/75

Depends on: https://github.com/OpenAgentsInc/hydralisk/issues/74

Profile: `profiles/deepseek-v4-fable-adapter-g4.json`

Status: `indexer_loader_mapping_proven`

## Decision

- Context transform can be implemented: `true`
- Private load canary can run after transform: `true`
- Khala general route allowed: `false`
- Public aliases allowed: `false`
- MPP public sale allowed: `false`
- Next step: `implement_context_specific_packed_lora_transform_and_private_load_canary`

## Adapter inspection

- Adapter path: `.hydralisk/deepseek-v4-fable-issue73/adapter/adapter_model.safetensors`
- Adapter file bytes: `69409832`
- Tensor count: `382`
- LoRA module count: `191`
- Indexer compressor gate modules: `21`
- Indexer compressor gate layers: `2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42`
- Indexer compressor gate output rows: `256`
- Header-only inspection: `true`
- Tensor values read: `false`

## Loader proof

- Runtime image: `hydralisk-deepseek-v4-b12x-g4-vllm-issue60-vector-v3:20260624v3vector2`
- Loader rule: `name.replace(weight_name, param_name)`
- Stacked mapping: `compressor.wgate` -> `compressor.fused_wkv_wgate` shard `1`
- Indexer instantiation: `DeepseekV4Indexer creates DeepseekCompressor with prefix=f'{prefix}.compressor'.`
- Adapter family: `self_attn.compressor.indexer.gate_proj`
- Transform checkpoint family: `self_attn.compressor.indexer.compressor.wgate`
- Runtime family: `self_attn.compressor.indexer.compressor.fused_wkv_wgate`

| Layer | Adapter module | Transform checkpoint name | Loader runtime name | Shard | OK |
| ---: | --- | --- | --- | ---: | --- |
| 2 | `base_model.model.model.layers.2.self_attn.compressor.indexer.gate_proj` | `model.layers.2.self_attn.compressor.indexer.compressor.wgate.weight` | `model.layers.2.self_attn.compressor.indexer.compressor.fused_wkv_wgate.weight` | 1 | `true` |
| 4 | `base_model.model.model.layers.4.self_attn.compressor.indexer.gate_proj` | `model.layers.4.self_attn.compressor.indexer.compressor.wgate.weight` | `model.layers.4.self_attn.compressor.indexer.compressor.fused_wkv_wgate.weight` | 1 | `true` |

## Blockers

- None

## Interpretation

The nested indexer compressor path is proven well enough to proceed to the
context-specific packed LoRA transform. The transform writer should convert the
published `self_attn.compressor.indexer.gate_proj` LoRA delta into a
checkpoint-style `self_attn.compressor.indexer.compressor.wgate` delta. The
current runtime loader then rewrites that checkpoint family into
`self_attn.compressor.indexer.compressor.fused_wkv_wgate` shard 1, matching
the nested `DeepseekCompressor` created inside `DeepseekV4Indexer`.

This still does not admit Fable for serving. It only removes the last mapping
blocker before implementing the transform and running a private no-public-
ingress load canary.

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains tensor values: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
- Contains target details: false
- Contains full third-party source: false
