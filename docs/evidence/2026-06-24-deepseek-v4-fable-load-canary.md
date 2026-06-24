# DeepSeek-V4-Fable private load canary evidence

Date: 2026-06-24T19:03:36.845308Z

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/68

Depends on: https://github.com/OpenAgentsInc/hydralisk/issues/67

Profile: `profiles/deepseek-v4-fable-adapter-g4.json`

Status: `blocked_adapter_incompatible`

## Decision

- Private adapter load can be attempted: `false`
- Khala general route allowed: `false`
- Public aliases allowed: `false`
- MPP public sale allowed: `false`
- Next step: `do_not_start_load_smoke_until_adapter_mapping_exists`

## Compatibility input

- Compatibility status: `rejected_adapter_incompatible`
- Compatibility issue: https://github.com/OpenAgentsInc/hydralisk/issues/67
- Missing adapter targets: `v_proj, q_proj, up_proj, o_proj, k_proj, gate_proj`

## Load canary

- Attempted: `false`
- No public ingress: `true`
- Merged checkpoint served: `false`
- Adapter payload required by this gate: `false`
- Reason: `blocked_by_adapter_compatibility`

Timing metrics are intentionally empty because the canary did not start.

## Blockers

- `adapter_runtime_targets_missing`: Compatibility issue #67 did not admit the Fable adapter target modules on the current patched G4 runtime.

## Interpretation

The private load canary is blocked before host/model interaction because the
adapter compatibility gate did not admit the Fable LoRA targets on the current
patched G4 runtime. This is the expected safe outcome after issue #67.

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
