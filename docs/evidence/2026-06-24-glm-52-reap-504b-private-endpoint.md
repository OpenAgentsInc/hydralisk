# GLM-5.2 504B REAP private endpoint

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/87

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Proxy helper:
[`scripts/expose-glm-52-reap-504b-private-proxy-gce.sh`](../../scripts/expose-glm-52-reap-504b-private-proxy-gce.sh)

Receipt:
[`docs/evidence/2026-06-24-glm-52-reap-504b-private-endpoint-receipt.json`](2026-06-24-glm-52-reap-504b-private-endpoint-receipt.json)

Public-safety boundary: this packet contains endpoint metadata, public-safe
health/models output, request hashes, response hashes, token counts, and receipt
metadata only. It contains no bearer token, model-provider credentials, raw
prompts, raw responses, private source, hidden reasoning traces, weights,
checkpoints, compiled engines, profiler dumps, or raw model logs.

## Result

PASS. Hydralisk now exposes a private, bearer-authenticated OpenAI-compatible
proxy for the live GLM-5.2 REAP/vLLM service.

- Proxy bind: `127.0.0.1:8080`
- Raw vLLM bind: `127.0.0.1:8000`
- Public bind: false
- Access path: on-host localhost or IAP/SSH tunnel only
- Auth: bearer required
- Health status: ready
- Readiness blockers: none

The bearer token is generated and stored only on the remote host under the
private proxy state directory. It is not printed by the helper and is not
committed to Git.

## Routes

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /hydralisk/v1/capabilities`
- `GET /hydralisk/v1/metrics`
- `GET /hydralisk/v1/receipts/{runRef}`

`/v1/models` and `/v1/chat/completions` require the bearer token. `/health`,
capabilities, metrics, and receipt reads remain public-safe and do not expose
secrets or raw prompt/response content.

## Model ids

Canonical upstream model:

- `glm-5.2-reap-504b-g4`

Accepted private aliases:

- `openagents/glm-5.2-reap-504b`
- `0xSero/GLM-5.2-504B`

The proxy rewrites accepted aliases to the canonical upstream model before
forwarding to vLLM.

## Default request recipe

Internal clients may omit the GLM guardrails; the proxy injects them before
forwarding:

```json
{
  "min_p": 0.05,
  "repetition_penalty": 1.05,
  "max_tokens": 1024,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

Clients may still set stricter values per request. For direct-answer smoke and
ordinary utility calls, keep `enable_thinking=false`. For reasoning experiments,
set the flag deliberately and keep hidden reasoning out of committed receipts.

Parser metadata exposed in the receipt:

- Tool-call parser: `glm47`
- Reasoning parser: `glm45`

## Fail-closed readiness

The GLM private proxy runs with `HYDRALISK_REQUIRE_PROFILE_EVIDENCE=true`.
Authenticated model/completion routes return 503 until these are pinned:

- model revision
- engine version
- model profile reference
- container image digest/reference
- admission evidence reference
- smoke/evidence reference
- receipt directory

This prevents accidental private traffic through an unpinned or undocumented
model lane.

## Live smoke

Run ID: `20260624231000`

The smoke sent a synthetic request through the private proxy using the alias
`openagents/glm-5.2-reap-504b`. It deliberately omitted:

- `min_p`
- `repetition_penalty`
- `max_tokens`
- `chat_template_kwargs.enable_thinking`

The stored Hydralisk receipt confirms the proxy applied:

- `minP=0.05`
- `repetitionPenalty=1.05`
- `maxTokens=1024`
- `enableThinking=false`

Observed result:

- HTTP status: 200
- Proxy run ref:
  `hydralisk-run-36f1856baa364d7bb42bd08354a2aadb`
- Prompt SHA-256:
  `5f8103cbebaac77e42161be89a636352dafbb3ceda5efef0aa6484d67233dfe2`
- Visible completion SHA-256:
  `deb72954879f318cd0fcb41355e82f54fbed51947d68e71b465fd31aba03f166`
- Visible completion characters: 18
- Prompt tokens: 22
- Completion tokens: 9
- Total tokens: 31
- Proxy wall time: 489 ms in the stored receipt
- Outer smoke wall time: 0.493 s

## Operator command

Start or refresh the private proxy:

```bash
ACTION=start RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-private-proxy-gce.sh
```

Check health and authenticated models:

```bash
ACTION=status RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-private-proxy-gce.sh
```

Run the public-safe proxy smoke:

```bash
ACTION=smoke RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-private-proxy-gce.sh
```

Stop only the Hydralisk proxy:

```bash
ACTION=stop RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-private-proxy-gce.sh
```

For local operator access, create an IAP/SSH tunnel to the VM and forward a
local port to `127.0.0.1:8080` on the host. Keep the bearer token out of shell
history, logs, issue comments, and committed files.

## Next gates

- Issue #88 should tune context, concurrency, and MTP against this private
  proxy and receipt path.
- Issue #89 should run Terminal-Bench only through receipt-safe task metadata,
  never raw prompts/responses or hidden reasoning traces.
