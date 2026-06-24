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

## Decision

Hydralisk should not attempt a merged-checkpoint admission for Fable today.
The adapter compatibility probe now rejects the current G4 runtime path before
load, and the final lab-eval gate rejects the profile because no private load
canary was admitted. The next useful technical step is an explicit
adapter-to-packed-runtime mapping, or a different base/runtime path that
exposes the Fable target module names. Even if a future probe succeeds, Fable
should remain an authorized-security research capability, not a general Khala
model and not a public inference product.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
