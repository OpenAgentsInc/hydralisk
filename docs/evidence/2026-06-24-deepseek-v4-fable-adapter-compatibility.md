# DeepSeek-V4-Fable adapter compatibility evidence

Date: 2026-06-24T19:01:19.295266Z

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/67

Profile: `profiles/deepseek-v4-fable-adapter-g4.json`

Status: `rejected_adapter_incompatible`

## Decision

- Private adapter load can be attempted: `false`
- Merged checkpoint load can be attempted: `false`
- Public aliases allowed: `false`
- Khala general route allowed: `false`
- MPP public sale allowed: `false`
- Next step: `stop_before_load_until_adapter_runtime_mapping_exists`

## Model

- Repository: `Chunjiang-Intelligence/DeepSeek-v4-Fable`
- Revision: `999909137c15e0b5539fee887431824fa7cb5b10`
- Base model: `deepseek-ai/DeepSeek-V4-Flash`
- Architecture: `['DeepseekV4ForCausalLM']`
- Model type: `deepseek_v4`
- Context window: `1048576`
- Experts: `256`
- Experts per token: `6`

## Files

| File | Bytes | Payload downloaded | Notes |
| --- | ---: | --- | --- |
| `adapter_config.json` | 1107 | `true` |  |
| `generation_config.json` | 170 | `true` |  |
| `merge_info.json` | 251 | `true` |  |
| `config.json` | 1724 | `true` |  |
| `model.safetensors.index.json` | 2733424 | `true` |  |
| `adapter_model.safetensors` | 69409832 | `false` | adapter payload was not downloaded by the public-safe probe; etag="318276c4103a77954d050bb70ba9eb87a0e91635871edc2ef46f4ea91d2fd57b" |

Merged checkpoint shards detected in index:
`47`

Merged shard fetch policy:
`refuse_model_safetensors_shards_without_explicit_unsafe_flag`

## Adapter

- PEFT type: `LORA`
- Task type: `CAUSAL_LM`
- Rank: `16`
- Alpha: `32`
- Merge output dtype: `torch.bfloat16`
- Merge shard count: `47`

## Runtime target compatibility

Runtime source:
`hydralisk-deepseek-v4-b12x-g4-vllm-issue60-vector-v3:20260624v3vector2 /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py`

Runtime module count:
`6`

| Adapter target | Status | Matched runtime modules |
| --- | --- | --- |
| `v_proj` | `missing` | - |
| `down_proj` | `matched` | `layers.0.mlp.down_proj`, `layers.0.mlp.shared_experts.down_proj` |
| `q_proj` | `missing` | - |
| `up_proj` | `missing` | - |
| `o_proj` | `missing` | - |
| `k_proj` | `missing` | - |
| `gate_proj` | `missing` | - |

Missing targets:
`v_proj, q_proj, up_proj, o_proj, k_proj, gate_proj`

## Interpretation

The Fable adapter is not admitted unless every target module can be mapped to
the exact Hydralisk DeepSeek V4 runtime surface. A missing target means the
probe fails closed before any adapter load smoke. This does not download or
serve the merged Fable checkpoint.

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
