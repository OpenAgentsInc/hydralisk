# DeepSeek-V4-Fable authorized-security policy evidence

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/69

Profile: `profiles/deepseek-v4-fable-adapter-g4.json`

Status: `policy_harness_implemented_fail_closed`

## Summary

Issue #69 added a Hydralisk proxy admission mode for models that must run only
inside an authorized security lab. The policy mode is
`authorized_security_lab_only`.

This does not admit Fable for serving. It adds the fail-closed gateway behavior
that must exist before any future Fable adapter or runtime path can be used.

## Required request metadata

When `HYDRALISK_MODEL_POLICY=authorized_security_lab_only`, chat and responses
requests must include a `metadata.hydraliskAuthorizedSecurity` object with:

| Field | Meaning |
| --- | --- |
| `scopeId` | Authorized lab/scope identifier |
| `authorizationRef` | Human/operator authorization or lab-run reference |
| `toolPolicy` | Tool execution policy name |
| `networkPolicy` | Network access policy name |

Snake-case aliases are accepted for bounded field compatibility:
`hydralisk_authorized_security`, `scope_id`, `authorization_ref`,
`lab_run_id`, `tool_policy`, and `network_policy`.

## Fail-closed behavior

The proxy rejects requests before upstream inference when:

- authorized-security metadata is missing;
- any required field is missing or empty;
- `scopeId` is not in `HYDRALISK_AUTHORIZED_SECURITY_SCOPE_IDS`, when that
  allowlist is configured;
- `toolPolicy` is not in `HYDRALISK_AUTHORIZED_SECURITY_TOOL_POLICIES`, when
  that allowlist is configured;
- `networkPolicy` is not in
  `HYDRALISK_AUTHORIZED_SECURITY_NETWORK_POLICIES`, when that allowlist is
  configured.

Rejections use HTTP 403 and public-safe policy error codes.

## Receipts and capabilities

Capabilities now expose public-safe policy shape:

- policy mode;
- adapter revision;
- whether authorized-security scope, tool, and network allowlists are
  configured.

Capabilities intentionally do not publish actual allowed scope ids.

Receipts now include:

- policy mode;
- adapter revision;
- admission result;
- scope id;
- authorization reference;
- tool policy;
- network policy.

## Validation

Focused command:

```text
uv run --extra dev pytest -q tests/test_proxy.py tests/test_receipts.py
```

Result:

```text
14 passed, 1 existing Starlette/httpx warning
```

Full command:

```text
uv run --extra dev pytest -q
```

Result:

```text
92 passed, 1 existing Starlette/httpx warning
```

Covered behavior:

- capabilities publish the authorized-security policy shape without leaking
  allowed scope ids;
- missing authorized-security metadata is rejected before upstream construction;
- unconfigured scope ids are rejected;
- allowed metadata is admitted and recorded in receipts.

## Serving decision

- Fable public aliases allowed: `false`
- Fable Khala general route allowed: `false`
- Fable MPP public sale allowed: `false`
- Fable private lab policy harness present: `true`
- Fable adapter/runtime admitted: `false`

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains hidden reasoning: false
- Contains exploit payloads: false
