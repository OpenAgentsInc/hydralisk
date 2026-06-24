# DeepSeek-V4-Fable packed-runtime retargeting plan

Date: 2026-06-24T19:17:36.775879Z

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/71

Depends on: https://github.com/OpenAgentsInc/hydralisk/issues/67, https://github.com/OpenAgentsInc/hydralisk/issues/68, https://github.com/OpenAgentsInc/hydralisk/issues/70

Profile: `profiles/deepseek-v4-fable-adapter-g4.json`

Status: `blocked_source_inventory_required`

## Decision

- Packed retarget smoke can be attempted: `false`
- Canonical runtime probe can be attempted: `true`
- Khala general route allowed: `false`
- Public aliases allowed: `false`
- MPP public sale allowed: `false`
- Next step: `prove_attention_output_projection_owner_then_build_packed_lora_transform`

## Compatibility input

- Compatibility status: `rejected_adapter_incompatible`
- Runtime source: `hydralisk-deepseek-v4-b12x-g4-vllm-issue60-vector-v3:20260624v3vector2 /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py`
- Runtime module count: `6`
- Missing targets from direct match probe: `v_proj, q_proj, up_proj, o_proj, k_proj, gate_proj`

## Target retargeting plan

| Adapter target | Status | Packed family | Runtime modules | Required work |
| --- | --- | --- | --- | --- |
| `v_proj` | `packed_transform_required` | `attention_fused_wqa_wkv` | `layers.0.attn.fused_wqa_wkv` | derive DeepSeek-V4 MLA projection slice ownership, then repack the LoRA delta into the fused attention input projection |
| `down_proj` | `direct_attachable` | `-` | `layers.0.mlp.down_proj`, `layers.0.mlp.shared_experts.down_proj` | rerun_private_adapter_load_canary |
| `q_proj` | `packed_transform_required` | `attention_fused_wqa_wkv` | `layers.0.attn.fused_wqa_wkv` | derive DeepSeek-V4 MLA projection slice ownership, then repack the LoRA delta into the fused attention input projection |
| `up_proj` | `packed_transform_required` | `swiglu_gate_up_proj` | `layers.0.mlp.gate_up_proj`, `layers.0.mlp.shared_experts.gate_up_proj` | prove gate/up ordering and repack paired SwiGLU LoRA deltas into fused gate_up_proj weights |
| `o_proj` | `source_inventory_required` | `attention_output_o_proj` | - | inspect the runtime source for attention output projection ownership; local evidence shows an o_proj provider/kernel path but no adapter-addressable module inventory entry |
| `k_proj` | `packed_transform_required` | `attention_fused_wqa_wkv` | `layers.0.attn.fused_wqa_wkv` | derive DeepSeek-V4 MLA projection slice ownership, then repack the LoRA delta into the fused attention input projection |
| `gate_proj` | `packed_transform_required` | `swiglu_gate_up_proj` | `layers.0.mlp.gate_up_proj`, `layers.0.mlp.shared_experts.gate_up_proj` | prove gate/up ordering and repack paired SwiGLU LoRA deltas into fused gate_up_proj weights |

Packed transform required targets:
`v_proj, q_proj, up_proj, k_proj, gate_proj`

Source inventory required targets:
`o_proj`

Unknown targets:
`none`

## Interpretation

The path to getting Fable working on the current Google G4 lane is a packed
LoRA retarget, not a vanilla PEFT adapter load. The current runtime can only
claim `down_proj` as directly attachable. Attention `q_proj`, `k_proj`, and
`v_proj` need an architecture-aware transform into `fused_wqa_wkv`; MLP
`gate_proj` and `up_proj` need a paired SwiGLU transform into `gate_up_proj`.
`o_proj` remains blocked until the attention output projection owner is proven
from runtime source or a live module inventory, because current evidence shows
an `o_proj` kernel/provider path but no adapter-addressable module entry.

If `o_proj` ownership cannot be proven quickly, the fallback path is a
canonical DeepSeek-V4-Flash base runtime feasibility probe that exposes the
Fable PEFT target names and reruns admission from scratch.

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
- Contains target details: false
