# DeepSeek V4 single-flight canary envelope

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/66

## Summary

Issue #66 added a configurable inflight admission envelope to the Hydralisk
proxy. This is the serving-side response to the issue #65 concurrency failure:
the current DeepSeek V4 Flash G4 lane is not a shared serving lane, but it can
be put behind an explicit single-flight canary boundary while scheduler/kernel
work continues.

## Configuration

Default behavior remains unlimited:

```text
HYDRALISK_MAX_INFLIGHT_REQUESTS unset or <=0
HYDRALISK_INFLIGHT_QUEUE_TIMEOUT_SECONDS=0
```

Single-flight canary behavior:

```text
HYDRALISK_MAX_INFLIGHT_REQUESTS=1
HYDRALISK_INFLIGHT_QUEUE_TIMEOUT_SECONDS=0
```

With timeout `0`, saturated requests fail closed immediately with HTTP 429 and
public-safe admission metadata. A positive timeout lets requests wait up to that
many seconds for an inflight slot.

## Proxy Behavior

Implemented behavior:

- the proxy acquires an inflight lease before forwarding chat completions or
  responses requests;
- nonstream requests release the lease after the upstream response is handled;
- streaming requests hold the lease until the stream generator finishes and the
  public-safe receipt is written;
- saturated requests return `hydralisk_inflight_saturated` with the configured
  admission policy;
- capabilities and receipts expose `admission.maxInflightRequests`,
  `admission.queueTimeoutSeconds`, and `admission.singleFlight`.

This does not make DeepSeek V4 concurrency viable. It makes the current
single-flight boundary explicit and enforceable.

## Validation

Commands:

```text
uv run python -m py_compile hydralisk/serve/config.py hydralisk/serve/receipts.py hydralisk/serve/proxy.py
uv run pytest -q
```

Result:

```text
81 passed, 1 existing Starlette/httpx warning
```

New tests cover:

- capabilities include default unlimited admission metadata;
- capabilities include configured single-flight admission metadata;
- the inflight gate rejects saturation with HTTP 429;
- streaming proxy responses release the single-flight slot before the next
  streaming request;
- receipts include admission metadata.

## Serving Boundary

For DeepSeek V4 Flash on the current 8 x G4 lane, this enables only:

```text
resident=true
prewarmed=true
maxInflightRequests=1
publicSharedServing=false
```

It is a narrow internal canary envelope, not a Khala production serving claim.
The remaining hard blockers are true concurrent serving, sparse-indexer
correctness, broader quality, and context beyond the current `max_model_len=2048`.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
