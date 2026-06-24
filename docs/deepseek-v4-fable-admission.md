# DeepSeek-V4-Fable admission note

Date: 2026-06-24

Model: `Chunjiang-Intelligence/DeepSeek-v4-Fable`

Source:

- Hugging Face model card:
  <https://huggingface.co/Chunjiang-Intelligence/DeepSeek-v4-Fable>
- Hugging Face model API, read on 2026-06-24:
  <https://huggingface.co/api/models/Chunjiang-Intelligence/DeepSeek-v4-Fable>
- Base model card:
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash>
- Current Hydralisk DeepSeek V4 evidence:
  [`docs/deepseek-v4-flash-gce-preflight.md`](deepseek-v4-flash-gce-preflight.md)
- Adapter compatibility evidence:
  [`docs/evidence/2026-06-24-deepseek-v4-fable-adapter-compatibility.md`](evidence/2026-06-24-deepseek-v4-fable-adapter-compatibility.md)
- Load canary evidence:
  [`docs/evidence/2026-06-24-deepseek-v4-fable-load-canary.md`](evidence/2026-06-24-deepseek-v4-fable-load-canary.md)
- Authorized-security policy evidence:
  [`docs/evidence/2026-06-24-deepseek-v4-fable-authorized-security-policy.md`](evidence/2026-06-24-deepseek-v4-fable-authorized-security-policy.md)
- Lab eval decision evidence:
  [`docs/evidence/2026-06-24-deepseek-v4-fable-lab-eval-decision.md`](evidence/2026-06-24-deepseek-v4-fable-lab-eval-decision.md)
- Packed-runtime retargeting evidence:
  [`docs/evidence/2026-06-24-deepseek-v4-fable-retarget-plan.md`](evidence/2026-06-24-deepseek-v4-fable-retarget-plan.md)
- `o_proj` ownership evidence:
  [`docs/evidence/2026-06-24-deepseek-v4-fable-o-proj-ownership.md`](evidence/2026-06-24-deepseek-v4-fable-o-proj-ownership.md)
- Packed-LoRA transform smoke evidence:
  [`docs/evidence/2026-06-24-deepseek-v4-fable-transform-smoke.md`](evidence/2026-06-24-deepseek-v4-fable-transform-smoke.md)
- Adapter context-map evidence:
  [`docs/evidence/2026-06-24-deepseek-v4-fable-context-map.md`](evidence/2026-06-24-deepseek-v4-fable-context-map.md)
- Indexer-compressor loader proof:
  [`docs/evidence/2026-06-24-deepseek-v4-fable-indexer-loader-proof.md`](evidence/2026-06-24-deepseek-v4-fable-indexer-loader-proof.md)
- Packed delta transform evidence:
  [`docs/evidence/2026-06-24-deepseek-v4-fable-packed-delta.md`](evidence/2026-06-24-deepseek-v4-fable-packed-delta.md)

## Summary

Do not admit the published merged `DeepSeek-v4-Fable` checkpoint as a serving
target on the current Google lane today.

The only plausible near-term experiment is an adapter-only probe on top of the
already-admitted DeepSeek-V4-Flash G4 path, with no public alias, no Khala
general routing, and an authorized-security-only policy harness. If that LoRA
adapter does not load cleanly against our pinned DeepSeek-V4-Flash runtime, the
lane should fail closed instead of attempting to serve the larger merged
checkpoint.

## What the model claims to be

The model card describes DeepSeek-V4-Fable as a distilled variant of
Claude-5-Fable built on top of `DeepSeek-V4-Flash`. It is specialized for
autonomous security research workflows: CTFs, authorized security evaluation,
exploitation planning, and multi-step tool-oriented reasoning in controlled
sandboxes.

The card is explicit that it is not a general-purpose assistant. It also states
that deployment should be restricted to authorized, supervised environments
with clear operational boundaries.

That matters for OpenAgents:

- this should not be exposed through `khala` as a general model capability;
- it should not receive public model aliases;
- it should not be sold or routed as an ungated public inference endpoint;
- any experiment needs an explicit authorized-security harness and receipt
  boundary before it is more than a private lab probe.

## Repository facts

Live Hugging Face metadata read on 2026-06-24:

| Field | Value |
| --- | --- |
| Repository | `Chunjiang-Intelligence/DeepSeek-v4-Fable` |
| Revision | `999909137c15e0b5539fee887431824fa7cb5b10` |
| License | `openrail` |
| Library | `transformers` |
| Pipeline | `text-generation` |
| Base model | `deepseek-ai/DeepSeek-V4-Flash` |
| Gated | `false` |
| Tags | `deepseek_v4`, `cybersecurity`, `ctf`, `autonomous-agent`, `lora`, `fp8` |
| Architecture | `DeepseekV4ForCausalLM` |
| Model type | `deepseek_v4` |
| Quantization config | `fp8`, `e4m3`, `ue8m0`, block size `[128, 128]` |
| Experts | 256 routed experts, 6 experts per token |
| Layers | 43 |
| Hidden size | 4096 |
| Context | 1,048,576 tokens |

Artifact inventory:

| Artifact | Observed value |
| --- | ---: |
| Merged safetensor shards | 47 |
| Merged index total size | 298,425,334,924 bytes |
| HF safetensors parameter report | 149,210,695,634 BF16 params |
| Adapter file size | 69,409,832 bytes |
| First shard size | 6,434,269,642 bytes |
| Last shard size | 2,302,335,126 bytes |

## Metadata conflicts

Treat the model as unadmitted until these are resolved:

- The model card says the base is a 284B sparse MoE, but Hugging Face's
  safetensors summary reports 149.2B BF16 parameters for this repository.
- The model card says the LoRA training used rank 64, alpha 128, and about
  0.94B trainable parameters.
- `adapter_config.json` says `r=16`, `lora_alpha=32`, and target modules are
  `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
  `down_proj`.
- `merge_info.json` says `lora_r=8`, `lora_alpha=16`, `scaling=2.0`, output
  dtype `torch.bfloat16`, and 47 output shards.
- The adapter payload is only about 69 MB.

That does not prove the model is bad, but it does mean the published
evaluation claims should not be treated as production evidence. Hydralisk needs
its own load, quality, safety, and policy gates.

## Comparison with our current DeepSeek V4 lane

The current Hydralisk G4 path is not stock DeepSeek-V4-Flash. It is a patched
resident G4 lane for `nvidia/DeepSeek-V4-Flash-NVFP4`, with custom work around
Blackwell/SM120 vLLM, FlashInfer, B12x MoE, sparse MLA fallback, vector gather,
and a single-flight serving envelope.

Current proven envelope:

| Capability | Current evidence |
| --- | --- |
| Hardware | 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition on GCE G4 |
| Engine | vLLM `0.23.0`, Torch `2.11.0+cu130`, patched Hydralisk image |
| Model artifact | `nvidia/DeepSeek-V4-Flash-NVFP4` |
| Short warmed timing | TTFT p95 about 0.289 s, decode p50 about 11.3 tok/s |
| Longer prompt/output | 1,796 prompt tokens, 160 output tokens, warmed decode p50 about 11.1 tok/s |
| First long streaming penalty | about 10.8 s TTFT before explicit stream prewarm |
| Concurrency | `max_num_seqs=2` completed but failed gate, about 3 tok/s/request |
| Safe serving boundary | resident, prewarmed, single-flight canary only |

Fable inherits the hard part of DeepSeek-V4-Flash and adds two new issues:

1. The merged Fable checkpoint is materially larger than the optimized
   DeepSeek-V4-Flash NVFP4 path we made work.
2. The model is an offensive-security specialization, so product exposure is a
   policy and safety problem even if a private load smoke succeeds.

## Can we run it?

### Merged checkpoint

Not today, not honestly, on the current serving lane.

The merged checkpoint's index is about 298.4 GB. It is not the same artifact as
the current NVFP4 DeepSeek-V4-Flash base that Hydralisk has admitted. Serving
it would require a fresh model profile, fresh artifact pin, fresh engine
compatibility work, and the same G4 gates from scratch. It may fit across
8 x 96 GB G4 VRAM in a pure capacity sense, but that is not the blocker we
trust. The blocker is that our current successful lane depends on a specific
NVFP4 artifact, backend patches, eager-mode behavior, sparse MLA fallback, and
single-flight admission. The merged Fable checkpoint has not passed any of
those gates.

Do not route it through Khala or claim it is available from the merged
checkpoint.

### Adapter-only path

Possible enough to validate, but not admitted.

The repo contains a small LoRA adapter and points to
`deepseek-ai/DeepSeek-V4-Flash` as the base. A reasonable first experiment is
to test whether that adapter can be loaded on top of our already-resident
DeepSeek V4 base without downloading or serving the 298 GB merged checkpoint.

This is the only path that could be fast enough to justify a near-term probe:

1. Pin the Fable revision:
   `Chunjiang-Intelligence/DeepSeek-v4-Fable@999909137c15e0b5539fee887431824fa7cb5b10`.
2. Fetch only `adapter_config.json`, `adapter_model.safetensors`,
   `generation_config.json`, `merge_info.json`, `config.json`, and
   `model.safetensors.index.json` for inspection.
3. Compare adapter target module names against the exact model object exposed
   by the patched `nvidia/DeepSeek-V4-Flash-NVFP4` runtime.
4. Run a dry adapter-load smoke with raw prompt/response text excluded from
   tracked artifacts.
5. If the adapter cannot attach to the NVFP4 base, stop. Do not fall back to a
   merged checkpoint load as a shortcut.
6. If the adapter attaches, run the same readiness, quality, long-context, and
   concurrency gates as DeepSeek-V4-Flash, plus a security-policy harness.

Expected performance if the adapter path works should be close to the base G4
lane, because the adapter is tiny relative to the model. A small overhead is
possible, but the real constraints remain the same: prewarm requirement,
single-flight only, narrow quality evidence, and no credible shared serving.

## Safety and product gate

Fable must stay behind a stricter boundary than a normal model profile:

- `publicModelAliases` must stay empty.
- `khala` must not route ordinary user traffic to this model.
- Public MPP/public-sale routing should stay disabled.
- The model should be available only to an authorized security lab harness.
- The harness must require explicit scope metadata for every task.
- Tool execution must be sandboxed, audited, and rate limited.
- Network access should default to deny, with only scoped lab targets allowed.
- Receipts must record model revision, adapter revision, safety profile,
  scope identifier, tool policy, and admission result.
- Evidence must not commit raw prompts, raw responses, secrets, exploit
  payloads, private target data, hidden reasoning, or weights.

## Recommended next issue

Create a narrow Hydralisk issue:

```text
Validate DeepSeek-V4-Fable adapter compatibility against the admitted
DeepSeek-V4-Flash G4 runtime
```

Acceptance criteria:

- Add a fail-closed profile for `deepseek-v4-fable` with no public aliases.
- Pin Fable revision `999909137c15e0b5539fee887431824fa7cb5b10`.
- Add an adapter metadata probe that fetches only small metadata and adapter
  files, never merged shards by default.
- Verify adapter target modules against the live/patched DeepSeek V4 model
  module names.
- Run a no-public-ingress dry load smoke on the existing G4 lane if a live
  host is available.
- Produce a public-safe evidence note with no raw prompts, raw responses, or
  exploit details.
- If target-module matching or adapter load fails, mark the profile
  `rejected_adapter_incompatible` and stop.

## Issue #67 result

Issue #67 added a fail-closed Fable adapter compatibility probe and ran it
against the live cached Hydralisk DeepSeek V4 G4 image. The result is
`rejected_adapter_incompatible`: the patched vLLM runtime exposes packed or
renamed module surfaces such as `fused_wqa_wkv` and `gate_up_proj`, while the
Fable adapter targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, and `down_proj`.

Only `down_proj` matched the inspected runtime inventory by exact suffix. The
adapter load path therefore stops before any private load smoke until there is
an explicit adapter-to-runtime mapping or a different base/runtime path.

Evidence:
[`docs/evidence/2026-06-24-deepseek-v4-fable-adapter-compatibility.md`](evidence/2026-06-24-deepseek-v4-fable-adapter-compatibility.md)

## Issue #68 result

Issue #68 added the private load-canary decision artifact. Because issue #67
rejected the adapter-to-runtime mapping, the load canary status is
`blocked_adapter_incompatible`. Hydralisk did not start vLLM, did not download
adapter payload bytes, did not serve the merged checkpoint, and did not run
timing traffic.

Evidence:
[`docs/evidence/2026-06-24-deepseek-v4-fable-load-canary.md`](evidence/2026-06-24-deepseek-v4-fable-load-canary.md)

## Issue #69 result

Issue #69 added the `authorized_security_lab_only` Hydralisk proxy admission
mode. When this mode is enabled, requests must include explicit scope,
authorization, tool-policy, and network-policy metadata before upstream
inference. Missing or unconfigured metadata fails closed before upstream
construction, and admitted receipts record the policy context.

This policy harness does not admit Fable for serving; it is the minimum gateway
boundary required before any future private Fable lab canary.

Evidence:
[`docs/evidence/2026-06-24-deepseek-v4-fable-authorized-security-policy.md`](evidence/2026-06-24-deepseek-v4-fable-authorized-security-policy.md)

## Issue #70 result

Issue #70 added the final public-safe lab-eval decision artifact. Because issue
#68 never reached an admitted private load canary, Hydralisk did not run lab
traffic, did not record raw prompts or model outputs, did not touch production
or third-party targets, and did not collect timing or quality metrics.

The final Fable profile status is `rejected_runtime_unstable`: the
adapter-backed runtime path is not stable/admitted enough to evaluate. Fable is
not admitted as a private authorized-security canary, a general Khala model, a
public alias, or an MPP/public-sale product.

Evidence:
[`docs/evidence/2026-06-24-deepseek-v4-fable-lab-eval-decision.md`](evidence/2026-06-24-deepseek-v4-fable-lab-eval-decision.md)

## Issue #71 result

Issue #71 added a public-safe packed-runtime retargeting planner. The planner
confirms the next path is not a normal PEFT adapter load. On the current G4
runtime:

| Fable target | Current status |
| --- | --- |
| `down_proj` | directly attachable by suffix |
| `q_proj`, `k_proj`, `v_proj` | require packed attention transform into `fused_wqa_wkv` |
| `gate_proj`, `up_proj` | require paired SwiGLU transform into `gate_up_proj` |
| `o_proj` | blocked pending source-level attention output projection ownership |

So the shortest honest path is:

1. Prove where `o_proj` lives in the NVIDIA runtime source or live module
   inventory.
2. Implement a packed-LoRA delta transform that can repack Fable adapter
   tensors into `fused_wqa_wkv`, `gate_up_proj`, and the proven `o_proj`
   owner.
3. Run a private no-public-ingress transform smoke and then rerun #68/#70.

The fallback path is a canonical `DeepSeek-V4-Flash` base runtime probe that
exposes `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj` directly, then reruns admission from scratch. That path may be
slower or less compatible with the G4 NVFP4 work we already made live, but it
avoids packed tensor surgery.

Evidence:
[`docs/evidence/2026-06-24-deepseek-v4-fable-retarget-plan.md`](evidence/2026-06-24-deepseek-v4-fable-retarget-plan.md)

## Issue #72 result

Issue #72 inspected the live cached G4 image
`hydralisk-deepseek-v4-b12x-g4-vllm-issue60-vector-v3:20260624v3vector2` with
a public-safe AST summary. The result is
`o_proj_owner_proven_kernel_provider`.

`o_proj` is not a vanilla PEFT-addressable module on the current runtime.
Instead, attention backends expose `_o_proj` methods that call
`deep_gemm_fp8_o_proj` in `vllm.models.deepseek_v4.nvidia.ops.o_proj`.

That resolves the source-inventory blocker from #71. The next blocker is now
the actual offline packed-LoRA tensor transform smoke:

- pack `q_proj`, `k_proj`, and `v_proj` deltas into `fused_wqa_wkv`;
- pack `gate_proj` and `up_proj` deltas into `gate_up_proj`;
- route or inject `o_proj` deltas through the proven kernel/provider-owned
  projection path;
- run the transform smoke without public traffic, prompts, outputs, or serving.

Evidence:
[`docs/evidence/2026-06-24-deepseek-v4-fable-o-proj-ownership.md`](evidence/2026-06-24-deepseek-v4-fable-o-proj-ownership.md)

## Issue #73 result

Issue #73 downloaded the real `adapter_model.safetensors` only into ignored
`.hydralisk` evidence space and inspected the safetensors header. It did not
commit adapter bytes, tensor values, prompts, responses, or weights.

The result is `blocked_adapter_config_payload_mismatch`.

The actual adapter payload has:

- 382 tensors;
- 191 complete LoRA module pairs;
- MLP/shared-expert `gate_proj`, `up_proj`, and `down_proj` pairs on layers
  0-42;
- attention-compressor `gate_proj` pairs, including
  `self_attn.compressor.gate_proj` and
  `self_attn.compressor.indexer.gate_proj`;
- no `q_proj`, `k_proj`, `v_proj`, or `o_proj` LoRA pairs.

That means the path is not the previously assumed packed q/k/v/o transform.
The next useful implementation issue is either:

1. Map the actual adapter payload contexts against the runtime, especially
   compressor/indexer `gate_proj` into the runtime's `fused_wkv_wgate` family
   and MLP shared-expert gate/up/down into the packed MLP path; or
2. Pivot to a canonical `DeepSeek-V4-Flash` runtime and see whether the
   published adapter can load there exactly as shipped.

Evidence:
[`docs/evidence/2026-06-24-deepseek-v4-fable-transform-smoke.md`](evidence/2026-06-24-deepseek-v4-fable-transform-smoke.md)

## Issue #74 result

Issue #74 mapped the real Fable adapter contexts against the current packed
DeepSeek-V4 G4 runtime surfaces. It used header-only safetensors inspection and
the live G4 image source inventory; it did not commit adapter bytes, tensor
values, prompts, responses, or weights.

The result is `blocked_indexer_loader_mapping_required`.

Ready context families:

- `mlp_shared_experts.gate_proj`: maps to
  `layers.*.mlp.shared_experts.gate_up_proj` shard 0;
- `mlp_shared_experts.up_proj`: maps to
  `layers.*.mlp.shared_experts.gate_up_proj` shard 1;
- `mlp_shared_experts.down_proj`: maps directly to
  `layers.*.mlp.shared_experts.down_proj`;
- `attention_compressor.gate_proj`: probable map to
  `layers.*.self_attn.compressor.fused_wkv_wgate` shard 1 via
  `compressor.wgate`.

Remaining blocker:

- `attention_compressor_indexer.gate_proj`: runtime source shows a nested
  `indexer.compressor.fused_wkv_wgate`, but the checkpoint loader path for
  the published `self_attn.compressor.indexer.gate_proj` adapter keys still
  needs proof before Hydralisk writes a transform.

This makes the next step narrow: prove the nested indexer compressor loader
mapping, then implement a context-specific packed LoRA transform. If that
loader proof fails, pivot to a canonical DeepSeek-V4-Flash runtime probe.

Evidence:
[`docs/evidence/2026-06-24-deepseek-v4-fable-context-map.md`](evidence/2026-06-24-deepseek-v4-fable-context-map.md)

## Issue #75 result

Issue #75 proved the nested indexer compressor loader path for the published
`self_attn.compressor.indexer.gate_proj` adapter family. It used header-only
adapter inspection plus the current G4 runtime loader rule; it did not commit
adapter bytes, tensor values, prompts, responses, or weights.

The result is `indexer_loader_mapping_proven`.

The transform writer should map:

- adapter family:
  `self_attn.compressor.indexer.gate_proj`;
- transform checkpoint family:
  `self_attn.compressor.indexer.compressor.wgate`;
- runtime family after loader rewrite:
  `self_attn.compressor.indexer.compressor.fused_wkv_wgate`;
- shard: `1`.

The current loader rule is `name.replace(weight_name, param_name)`, with
stacked mapping `compressor.wgate` ->
`compressor.fused_wkv_wgate` shard `1`. Because `DeepseekV4Indexer` creates a
nested `DeepseekCompressor` with `prefix=f"{prefix}.compressor"`, a transformed
checkpoint-style key ending in
`self_attn.compressor.indexer.compressor.wgate.weight` rewrites to
`self_attn.compressor.indexer.compressor.fused_wkv_wgate.weight`.

This removes the final known mapping blocker. It still does not admit Fable for
serving; the next step is the context-specific packed LoRA transform and a
private no-public-ingress load canary.

Evidence:
[`docs/evidence/2026-06-24-deepseek-v4-fable-indexer-loader-proof.md`](evidence/2026-06-24-deepseek-v4-fable-indexer-loader-proof.md)

## Issue #76 result

Issue #76 implemented the first context-specific packed LoRA transform writer.
The writer reads the adapter tensors locally from ignored `.hydralisk`, computes
`B @ A * scale`, and writes checkpoint-style dense delta safetensors into
ignored output space. It emits tracked evidence with names, shapes, byte sizes,
checksums, and scalar zero/nonzero stats only; it does not commit tensor values
or weights.

The result is `packed_delta_artifact_written_zero_delta_payload`.

For the real published adapter payload at the pinned revision:

- LoRA B tensors: `191`;
- LoRA B zero tensors: `191`;
- LoRA B nonzero tensors: `0`;
- no non-LoRA tensors are present in the adapter safetensors file.

Hydralisk wrote a bounded layer-2 packed delta artifact covering:

- `model.layers.2.mlp.shared_experts.w1.weight`;
- `model.layers.2.mlp.shared_experts.w2.weight`;
- `model.layers.2.mlp.shared_experts.w3.weight`;
- `model.layers.2.self_attn.compressor.wgate.weight`;
- `model.layers.2.self_attn.compressor.indexer.compressor.wgate.weight`.

All five generated dense deltas are zero because every selected LoRA B matrix
is zero, and the global adapter scan shows that this is true across all LoRA B
tensors. The mechanical transform path is proven, but this adapter file should
not be used for a semantic Fable canary because applying it should not change
base model behavior.

The next step is to verify the upstream payload: either find a nonzero adapter
revision/artifact or intentionally evaluate the full merged checkpoint path.

Evidence:
[`docs/evidence/2026-06-24-deepseek-v4-fable-packed-delta.md`](evidence/2026-06-24-deepseek-v4-fable-packed-delta.md)

## Decision

Hydralisk should not attempt a merged-checkpoint admission for Fable today.
The adapter compatibility probe now rejects the current G4 runtime path before
load, and the final lab-eval gate rejects the profile because no private load
canary was admitted. `o_proj` ownership is proven as a kernel/provider path,
but the real adapter payload does not include the q/k/v/o tensors implied by
the adapter config. The context map narrowed the path to one hard runtime
question, and the indexer loader proof resolved it. The next technical step is
not a load canary yet: the transform writer works, but the published adapter
payload has zero LoRA B tensors, so the packed deltas are zero. Verify the
upstream adapter payload, find a nonzero revision, or intentionally evaluate
the full merged checkpoint path before a semantic canary. Even if a future
probe succeeds, Fable should remain an authorized-security research
capability, not a general Khala model and not a public inference product.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
