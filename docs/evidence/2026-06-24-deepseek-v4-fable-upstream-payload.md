# DeepSeek-V4-Fable upstream payload verification

Date: 2026-06-24T19:52:00Z

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/77

Depends on: https://github.com/OpenAgentsInc/hydralisk/issues/76

Profile: `profiles/deepseek-v4-fable-adapter-g4.json`

Status: `adapter_payload_zero_delta_merged_checkpoint_only_semantic_path`

## Decision

- Nonzero adapter path found: `false`
- Merged checkpoint path exists: `true`
- Semantic adapter canary allowed: `false`
- Mechanical transform path available: `true`
- Khala general route allowed: `false`
- Public aliases allowed: `false`
- MPP public sale allowed: `false`
- Next step: `evaluate_full_merged_checkpoint_feasibility_or_get_nonzero_adapter`

## Upstream metadata

- Repository: `Chunjiang-Intelligence/DeepSeek-v4-Fable`
- API URL: <https://huggingface.co/api/models/Chunjiang-Intelligence/DeepSeek-v4-Fable>
- Latest upstream SHA: `999909137c15e0b5539fee887431824fa7cb5b10`
- Local pinned SHA tested by Hydralisk: `999909137c15e0b5539fee887431824fa7cb5b10`
- Latest equals pinned: `true`
- Last modified: `2026-06-23T17:56:32.000Z`
- Public siblings: `57`

## Available artifact families

- Adapter config: `adapter_config.json`
- Adapter safetensors: `adapter_model.safetensors`
- Merge metadata: `merge_info.json`
- Merged checkpoint index: `model.safetensors.index.json`
- Merged checkpoint shards: `model-00001-of-00047.safetensors` through
  `model-00047-of-00047.safetensors`

## Adapter metadata

`adapter_config.json`:

- `r`: `16`
- `lora_alpha`: `32`
- `target_modules`: `v_proj`, `down_proj`, `q_proj`, `up_proj`, `o_proj`,
  `k_proj`, `gate_proj`

`merge_info.json`:

- `lora_r`: `8`
- `lora_alpha`: `16.0`
- `scaling`: `2.0`
- base model path records `deepseek-ai/DeepSeek-V4-Flash`

## Adapter payload verification

Hydralisk inspected the current `adapter_model.safetensors` payload from the
same SHA as latest upstream.

- Safetensors entries: `382`
- Non-LoRA tensor entries: `0`
- Complete LoRA module pairs: `191`
- LoRA B tensors: `191`
- LoRA B zero tensors: `191`
- LoRA B nonzero tensors: `0`

This means the public adapter payload is a zero-delta LoRA artifact. Applying
it through the packed transform writer should not change base model behavior.

## Merged checkpoint metadata

`model.safetensors.index.json` reports:

- weight-map entries: `35020`
- total tensor payload size: `298425334924` bytes
- shard count: `47`

The merged checkpoint is therefore the only currently visible upstream artifact
family that could contain semantic Fable behavior. Hydralisk has not downloaded
or admitted those merged shards in this issue.

## Interpretation

There is no current nonzero adapter path for a semantic private canary from the
published `adapter_model.safetensors` file. The mechanical packed transform
writer from issue #76 is valid, but the input adapter is no-op.

The path to get Fable working now is one of:

1. Obtain or find a nonzero adapter revision/artifact and rerun the packed
   transform writer; or
2. Intentionally evaluate the full 47-shard merged checkpoint path against the
   G4 DeepSeek-V4 runtime, with storage/download/load-time gates called out
   explicitly before any canary.

Until one of those happens, do not route Fable through Khala, public aliases,
or MPP.

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
