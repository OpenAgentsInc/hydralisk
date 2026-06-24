# DeepSeek-V4-Fable o_proj ownership evidence

Date: 2026-06-24T19:22:43.013068Z

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/72

Depends on: https://github.com/OpenAgentsInc/hydralisk/issues/71

Profile: `profiles/deepseek-v4-fable-adapter-g4.json`

Status: `o_proj_owner_proven_kernel_provider`

## Decision

- Vanilla PEFT `o_proj` can be used: `false`
- Packed-LoRA transform smoke can proceed: `true`
- Khala general route allowed: `false`
- Public aliases allowed: `false`
- MPP public sale allowed: `false`
- Next step: `implement_offline_packed_lora_delta_transform_smoke`

## Source inventory

- Image: `hydralisk-deepseek-v4-b12x-g4-vllm-issue60-vector-v3:20260624v3vector2`
- Inspection: `gce_docker_ast_summary`
- Module count: `4`

| Module | Relevant functions/classes | Relevant names/attrs/calls |
| --- | --- | --- |
| `vllm.models.deepseek_v4.nvidia.model` | `DeepseekV4Model`, `DeepseekV4ForCausalLM` | `gate_up_proj`, `gate_up_proj` |
| `vllm.models.deepseek_v4.nvidia.flashmla` | `DeepseekV4FlashMLAAttention`, `_o_proj` | `deep_gemm_fp8_o_proj`, `deep_gemm_fp8_o_proj` |
| `vllm.models.deepseek_v4.nvidia.flashinfer_sparse` | `DeepseekV4FlashInferMLAAttention`, `_o_proj` | `deep_gemm_fp8_o_proj`, `deep_gemm_fp8_o_proj` |
| `vllm.models.deepseek_v4.nvidia.ops.o_proj` | `deep_gemm_fp8_o_proj` | - |

## Ownership result

- Classification: `kernel_provider_owned_projection`
- Adapter-addressable module: `false`
- Kernel/provider owned: `true`
- Attention `_o_proj` methods: `_o_proj`
- Attention kernel calls: `deep_gemm_fp8_o_proj`
- Kernel functions: `deep_gemm_fp8_o_proj`
- Model module `o_proj` attrs: `none`

## Interpretation

Fable's `o_proj` target is not vanilla-PEFT-addressable on the current packed
NVIDIA runtime. The attention output projection is owned by backend attention
classes through `_o_proj` methods that call the `deep_gemm_fp8_o_proj` provider
function in `vllm.models.deepseek_v4.nvidia.ops.o_proj`.

That unblocks the retargeting plan's source-inventory question, but it does
not admit Fable for serving. The next implementation step is an offline
packed-LoRA delta transform smoke that proves we can map Fable adapter tensors
into `fused_wqa_wkv`, `gate_up_proj`, and the kernel/provider-owned `o_proj`
path without running public traffic.

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
- Contains target details: false
- Contains full third-party source: false
