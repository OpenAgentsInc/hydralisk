# DeepSeek-V4-Fable packed LoRA delta artifact

Date: 2026-06-24T19:48:00.669043Z

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/76

Depends on: https://github.com/OpenAgentsInc/hydralisk/issues/75

Profile: `profiles/deepseek-v4-fable-adapter-g4.json`

Status: `packed_delta_artifact_written_zero_delta_payload`

## Decision

- Mechanical load canary can run: `true`
- Semantic adapter canary can run: `false`
- Khala general route allowed: `false`
- Public aliases allowed: `false`
- MPP public sale allowed: `false`
- Next step: `verify_upstream_adapter_payload_or_find_nonzero_revision_before_private_canary`

## Adapter inspection

- Adapter path: `.hydralisk/deepseek-v4-fable-issue73/adapter/adapter_model.safetensors`
- Adapter file bytes: `69409832`
- Tensor count: `382`
- LoRA module count: `191`
- LoRA B tensors: `191`
- LoRA B zero tensors: `191`
- LoRA B nonzero tensors: `0`
- Tensor values read locally: `true`
- Header-only inspection: `false`

## Output artifact

- Path: `.hydralisk/deepseek-v4-fable-issue76/packed-delta-layer2/deepseek-v4-fable-packed-deltas.safetensors`
- File bytes: `121635614`
- SHA-256: `2101cf8e7a74713378b5a5e04673834ec63bd9c1a7b246d6132852ce76c23880`
- Scale: `2`
- Layers: `[2]`
- Tensor count: `5`
- Total tensor bytes: `121634816`

| Family | Tensors |
| --- | ---: |
| `attention_compressor_gate` | 1 |
| `attention_compressor_indexer_gate` | 1 |
| `mlp_shared_experts_down` | 1 |
| `mlp_shared_experts_gate` | 1 |
| `mlp_shared_experts_up` | 1 |

## Tensor manifest

| Key | Shape | Dtype | Bytes | Scale | Nonzero | Abs max | Abs sum | SHA-256 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `model.layers.2.mlp.shared_experts.w1.weight` | `2048x4096` | `F32` | 33554432 | 2 | 0 | 0 | 0 | `83ee47245398adee79bd9c0a8bc57b821e92aba10f5f9ade8a5d1fae4d8c4302` |
| `model.layers.2.mlp.shared_experts.w2.weight` | `4096x2048` | `F32` | 33554432 | 2 | 0 | 0 | 0 | `83ee47245398adee79bd9c0a8bc57b821e92aba10f5f9ade8a5d1fae4d8c4302` |
| `model.layers.2.mlp.shared_experts.w3.weight` | `2048x4096` | `F32` | 33554432 | 2 | 0 | 0 | 0 | `83ee47245398adee79bd9c0a8bc57b821e92aba10f5f9ade8a5d1fae4d8c4302` |
| `model.layers.2.self_attn.compressor.indexer.compressor.wgate.weight` | `256x4096` | `F32` | 4194304 | 2 | 0 | 0 | 0 | `bb9f8df61474d25e71fa00722318cd387396ca1736605e1248821cc0de3d3af8` |
| `model.layers.2.self_attn.compressor.wgate.weight` | `1024x4096` | `F32` | 16777216 | 2 | 0 | 0 | 0 | `080acf35a507ac9849cfcba47dc2ad83e01b75663a516279c8b9d243b719643e` |

## Interpretation

Hydralisk wrote a checkpoint-style packed delta safetensors artifact in ignored
evidence space, but the adapter payload appears to be a zero-delta LoRA
payload: every LoRA B tensor inspected from the adapter is zero. The transform
path is mechanically proven, but applying this artifact should not change model
behavior.

This should not proceed to a semantic Fable canary until the upstream adapter
payload is verified, a nonzero revision is found, or the full merged checkpoint
path is intentionally evaluated.

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
- Tracked evidence only: true
