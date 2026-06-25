# GLM-5.2 504B REAP public HTTPS origin

Date: 2026-06-25

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/94

Runbook:
[`docs/glm-5.2-reap-504b-g4-runbook.md`](../glm-5.2-reap-504b-g4-runbook.md)

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Public-safety boundary: this note contains endpoint shape, service state,
hashes, token counts, and aggregate timings only. It contains no public origin
hostname, public IP address, bearer token, raw prompt, raw response, hidden
reasoning trace, provider credential, model weight, checkpoint, compiled
engine, profiler dump, or raw log.

## Result

PASS. The GLM-5.2 REAP proxy now has a Worker-reachable authenticated HTTPS
origin. The tracked OpenAgents Worker should receive the concrete URL and
bearer token only through secret arming:

- `HYDRALISK_GLM_52_REAP_504B_BASE_URL`
- `HYDRALISK_GLM_52_REAP_504B_BEARER_TOKEN`

Raw vLLM remains bound to host-local `127.0.0.1:8000`. The public HTTPS origin
fronts only the Hydralisk bearer-gated proxy routes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /hydralisk/v1/capabilities`
- `GET /hydralisk/v1/metrics`
- receipt lookup routes under `/hydralisk/`

Other paths return `404` at the TLS front.

## Transport Decision

The first preference was a Cloudflare Tunnel so the GCE VM could keep no
external address. The local Cloudflare token available during this work could
list the `openagents.com` zone, but it could not manage DNS records or
Cloudflare Tunnel resources for the account. Rather than block the Khala
integration on an owner-only Cloudflare permission change, Hydralisk used the
same day-zero origin pattern documented for the GPT-OSS lane:

1. reserve a regional static GCE address;
2. attach it to the admitted GLM host;
3. add a target tag to the host;
4. open `80/443` only to that tag;
5. run Caddy on the host for ACME TLS;
6. reverse proxy HTTPS traffic to the bearer-authenticated Hydralisk proxy;
7. keep raw vLLM host-local.

This is an origin, not a public product promise. The concrete origin hostname
is deliberately absent from tracked files and issue comments.

## Live Configuration

- Instance class: admitted 8 x G4 fallback host
- Active GLM GPUs: four selected RTX PRO 6000 GPUs
- Public origin shape: `https://<operator-secret-hostname>`
- Addressing: reserved regional static address, value not tracked
- Firewall: tag-targeted `tcp:80,tcp:443`
- TLS/front: Caddy `v2.11.4`
- Caddy upstream protocol: HTTP/1.1
- Proxy upstream: Hydralisk private proxy on the same VM
- Proxy auth: bearer required for model and generation routes
- Raw vLLM public exposure: false

The HTTP/1.1 upstream setting matters. Caddy's h2c upstream transport produced
502s against this Uvicorn proxy path, so the reusable Caddy example now pins
`versions 1.1`.

## Smoke Evidence

Smoke window: 2026-06-25, after Caddy obtained its ACME certificate.

Public HTTPS health:

- HTTP status: 200
- Health status: `ready`
- Served model: `glm-5.2-reap-504b-g4`
- Auth required: true

Public HTTPS authenticated models:

- HTTP status: 200
- Model count: 1
- Raw vLLM public exposure: false

Public HTTPS authenticated chat completion:

- HTTP status: 200
- Requested alias: `openagents/glm-5.2-reap-504b`
- Wall time: 0.988 s
- Prompt tokens: 11
- Completion tokens: 2
- Total tokens: 13
- Visible completion characters: 5
- Visible completion SHA-256:
  `c2e3ac47f4a325469c1a2d5f117e463ec943c721986d5d9f09ac4540b7d80526`

The reusable operator script also passed the same public HTTPS smoke. Its first
attempts observed the proxy's expected singleflight backpressure while another
request was active, then the completion succeeded on attempt 4:

- Script: `scripts/expose-glm-52-reap-504b-public-https-gce.sh`
- Action: `ACTION=smoke`
- Final HTTP status: 200
- Final attempt: 4
- Final wall time: 0.571 s
- Prompt tokens: 11
- Completion tokens: 2
- Total tokens: 13

The proxy remains singleflight-limited. Busy-window requests can return the
Hydralisk `429 hydralisk_inflight_saturated` backpressure response; that is a
capacity/concurrency policy boundary, not a reachability failure.

## Reproducible Operator Path

Use the public-safe setup script from a clean Hydralisk checkout:

```bash
ACTION=setup RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-public-https-gce.sh

ACTION=smoke RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-public-https-gce.sh
```

The script writes local ignored evidence under `.hydralisk/`, including a
redacted status document and a redacted smoke summary. Do not commit the actual
origin hostname, public IP address, or bearer token.

## Khala Readiness Impact

This issue removes the Cloudflare Worker reachability blocker. The OpenAgents
side still needs owner-only secret arming and deployment before Khala traffic
can use the lane:

- `HYDRALISK_GLM_52_REAP_504B_ENABLED=ready`
- `HYDRALISK_GLM_52_REAP_504B_BASE_URL=<secret origin URL>`
- `HYDRALISK_GLM_52_REAP_504B_BEARER_TOKEN=<secret bearer token>`
- public-safe evidence/reference vars from the OpenAgents integration docs

Durability and keep-warm are tracked separately in Hydralisk issue #95.
