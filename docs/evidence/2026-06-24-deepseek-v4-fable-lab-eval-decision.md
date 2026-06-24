# DeepSeek-V4-Fable lab eval decision

Date: 2026-06-24T19:11:10.588110Z

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/70

Depends on: https://github.com/OpenAgentsInc/hydralisk/issues/68, https://github.com/OpenAgentsInc/hydralisk/issues/69

Profile: `profiles/deepseek-v4-fable-adapter-g4.json`

Status: `rejected_runtime_unstable`

## Decision

- Private authorized-security lab canary admitted: `false`
- Khala general route allowed: `false`
- Public aliases allowed: `false`
- MPP public sale allowed: `false`
- Next step: `map_or_retarget_fable_lora_modules_before_any_lab_eval`

## Prerequisites

- Load canary status: `blocked_adapter_incompatible`
- Load canary attempted: `false`
- Private adapter load can be attempted: `false`
- Authorized-security policy harness: `policy_harness_implemented_fail_closed`
- Unscoped requests blocked by policy harness: `true`

## Lab eval

- Attempted: `false`
- Authorized sandbox tasks only: `true`
- Production targets used: `false`
- Third-party targets used: `false`
- Raw prompts committed: `false`
- Raw outputs committed: `false`
- Reason: `blocked_before_eval_by_private_load_canary`

No lab eval traffic was run because the model never reached an admitted private
load canary state.

## Public-safe metrics

- Task categories: `none`
- Verifier results: `none`
- Turn-count summary: `none`
- Tool-call-count summary: `none`
- Timeout/error classes: `none`
- TTFT summary: `none`
- Decode throughput summary: `none`
- Base DeepSeek V4 Flash comparison: `none`

## Blockers

- `adapter_runtime_targets_missing`: Compatibility issue #67 did not admit the Fable adapter target modules on the current patched G4 runtime.

## Interpretation

Issue #70 cannot honestly run or admit a Fable lab eval while issue #68 remains
blocked. The final decision is therefore `rejected_runtime_unstable`: the
adapter-backed runtime path is not stable/admitted enough to evaluate. Fable
remains disallowed for general Khala routing, public aliases, and MPP sale.

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
- Contains target details: false
