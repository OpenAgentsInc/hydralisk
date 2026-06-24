# DeepSeek-V4-Fable adapter context map

Date: 2026-06-24T19:37:44.784678Z

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/74

Depends on: https://github.com/OpenAgentsInc/hydralisk/issues/73

Profile: `profiles/deepseek-v4-fable-adapter-g4.json`

Status: `blocked_indexer_loader_mapping_required`

## Decision

- Packed delta can be written now: `false`
- Context transform can be implemented now: `false`
- Canonical runtime probe required: `false`
- Khala general route allowed: `false`
- Public aliases allowed: `false`
- MPP public sale allowed: `false`
- Next step: `prove_indexer_compressor_loader_mapping_then_write_context_transform`

## Adapter inspection

- Adapter path: `.hydralisk/deepseek-v4-fable-issue73/adapter/adapter_model.safetensors`
- Adapter file bytes: `69409832`
- Tensor count: `382`
- LoRA module count: `191`
- Header-only inspection: `true`
- Tensor values read: `false`

## Runtime inventory

- Image: `hydralisk-deepseek-v4-b12x-g4-vllm-issue60-vector-v3:20260624v3vector2`
- Inspection: `gce_docker_source_summary`
- Sources: `vllm.models.deepseek_v4.nvidia.model, vllm.models.deepseek_v4.attention, vllm.models.deepseek_v4.compressor`

## Context map

| Adapter context | Target | Modules | Layers | Output rows | Runtime candidate | Confidence | Status | Blockers |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `attention_compressor` | `gate_proj` | 41 | `2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42` | `512, 1024` | `layers.*.self_attn.compressor.fused_wkv_wgate` | `probable` | `candidate_transform_ready` | - |
| `attention_compressor_indexer` | `gate_proj` | 21 | `2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42` | `256` | `layers.*.self_attn.compressor.indexer.compressor.fused_wkv_wgate` | `medium` | `blocked` | loader_path_proof_required |
| `mlp_shared_experts` | `down_proj` | 43 | `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42` | `4096` | `layers.*.mlp.shared_experts.down_proj` | `high` | `candidate_transform_ready` | - |
| `mlp_shared_experts` | `gate_proj` | 43 | `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42` | `2048` | `layers.*.mlp.shared_experts.gate_up_proj` | `high` | `candidate_transform_ready` | - |
| `mlp_shared_experts` | `up_proj` | 43 | `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42` | `2048` | `layers.*.mlp.shared_experts.gate_up_proj` | `high` | `candidate_transform_ready` | - |

## Blockers

- `loader_path_proof_required` on `attention_compressor_indexer.gate_proj`: Runtime source shows a nested indexer compressor, but the checkpoint loader mapping for this adapter key family still needs to be proven before writing a transform.

## Interpretation

The real Fable adapter is not a q/k/v/o adapter. The shared-expert MLP LoRA
payload has a high-confidence packed target: `shared_experts.gate_proj` and
`shared_experts.up_proj` map to `shared_experts.gate_up_proj` shards, while
`shared_experts.down_proj` maps directly to `shared_experts.down_proj`.

The plain attention compressor gate path is also a probable packed transform:
runtime loading maps `compressor.wgate` into `compressor.fused_wkv_wgate`
shard 1. The remaining hard blocker is the nested indexer compressor family.
Runtime source shows `indexer.compressor.fused_wkv_wgate`, but Hydralisk still
needs loader-path proof for the published
`self_attn.compressor.indexer.gate_proj` adapter keys before writing a
transform or running another private load canary.

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
