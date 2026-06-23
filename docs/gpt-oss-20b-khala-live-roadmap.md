# GPT-OSS 20B behind Khala ASAP roadmap

Date: 2026-06-23

Status: implementation in progress. The Hydralisk repo now contains the
authenticated proxy scaffold, public-safe capabilities/receipts, systemd units,
smoke script, and a GCE L4 runbook. The service is not yet promoted behind
OpenAgents until the live host smoke and receipt gate pass.

## Goal

Put `gpt-oss-20b` behind the OpenAgents API as an owned Hydralisk supply lane:

- first as an internal Khala dogfood model;
- then as a catalog-visible OpenAgents model once receipts, metering, and
  fallback are proven;
- without pretending this is Psionic-native or full Pylon settlement.

The day-zero serving target is one L4-backed vLLM service. OpenAI's vLLM guide
describes `gpt-oss-20b` as requiring about 16 GB VRAM, and vLLM exposes both
Chat Completions-compatible and Responses-compatible APIs. Our L4 lane has 24 GB
VRAM, so this is the right first live target.

## Product decision

Do not overload the existing Psionic/Pylon `openagents-network` lane for this
first cut.

Use a distinct Hydralisk lane and adapter:

- adapter id: `hydralisk-vllm`
- supply lane: `hydralisk`
- first public-safe model alias: `openagents/khala-oss-20b`
- direct provider-compatible model id: `gpt-oss-20b`
- upstream served model: `openai/gpt-oss-20b`

Why a new lane:

- `openagents-network` currently means admitted Pylon/Psionic fabric, with
  evidence refs for approval, preflight, replay challenge, admitted Pylon, and
  payout readiness. Hydralisk day zero is our owned GPU service, not a paid
  contributor Pylon.
- A separate lane lets the catalog, quote, model-serving policy, telemetry, and
  receipts say the true thing: served by Hydralisk/vLLM on OpenAgents-owned
  infrastructure.
- Psionic can later consume the benchmark/eval packet and replace the lane only
  when it matches the accepted-outcome and receipt strength.

## ASAP architecture

```text
client / agent
  -> https://openagents.com/v1/chat/completions
  -> OpenAgents Worker
     -> auth, balance/free gate, premium gate, pricing/metering
     -> model router selects hydralisk-vllm for openagents/khala-oss-20b
     -> Hydralisk adapter forwards OpenAI-compatible request
  -> Hydralisk HTTPS proxy
     -> bearer auth
     -> vLLM localhost:8000
     -> receipt/capability metadata side channel
  -> NVIDIA L4 VM running openai/gpt-oss-20b
```

Day zero can use Chat Completions only. Responses support should be kept in the
Hydralisk service because vLLM exposes it, but the OpenAgents gateway's live
surface today is `/v1/chat/completions`.

## Phase 0: stand up the Hydralisk L4 service

Target: one GCE L4 instance in `openagentsgemini`, preferably `us-central1`.

2026-06-23 GCP audit:

- `us-central1` has `NVIDIA_L4_GPUS` limit 16, usage 1.
- `us-central1` has `PREEMPTIBLE_NVIDIA_L4_GPUS` limit 16, usage 0.
- The running L4 VM is `gswarm508-clean2-20260325044551-contrib` in
  `us-central1-b`; labels identify it as a Psion training contributor, so it is
  not a Hydralisk target unless an operator explicitly reclaims it.
- Hydralisk should provision a fresh `g2-standard-8` L4 host for day-zero
  inference. See [`gce-l4-vllm-runbook.md`](gce-l4-vllm-runbook.md).

Minimum instance shape:

- `g2-standard-8`
- 1 x NVIDIA L4
- Ubuntu or Deep Learning VM image with NVIDIA drivers
- boot disk large enough for model cache and logs
- service account with least-privilege logging/monitoring only

Runtime:

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match

vllm serve openai/gpt-oss-20b --host 127.0.0.1 --port 8000
```

The public-facing process should not expose raw vLLM directly. Put a Hydralisk
proxy in front of it:

- `GET /health`
- `GET /hydralisk/v1/capabilities`
- `GET /hydralisk/v1/receipts/{runRef}`
- `POST /v1/chat/completions`
- `POST /v1/responses` for local smoke and future gateway support

The proxy must require bearer auth and must not emit raw prompts, hidden
reasoning traces, private source, model cache paths, or local host details in
receipts.

## Phase 1: Hydralisk service contract

Before OpenAgents routes paid traffic, the Hydralisk endpoint must prove:

- `GET /health` returns model id, engine, status, and no secrets.
- `GET /hydralisk/v1/capabilities` returns:
  - `servedModel: openai/gpt-oss-20b`
  - `publicModelAliases: ["openagents/khala-oss-20b", "gpt-oss-20b"]`
  - `engine: vllm`
  - `engineVersion`
  - `gpuClass: l4`
  - `quantization.weights: MXFP4`
  - `chatCompletions: true`
  - `responses: true`
  - `toolCalls: disabled_for_gateway_day_zero`
  - `reasoningVisibility: final_answer_only_for_public_receipts`
- `POST /v1/chat/completions` accepts OpenAI-compatible input with bearer auth.
- A terminal usage object is returned by vLLM and mapped into Hydralisk receipts.
- Streaming sends incremental bytes at least every 15 seconds during long
  generations, or the service refuses long requests until durable/resume support
  is wired.

Day-zero receipt schema can be small:

```json
{
  "schema": "hydralisk.serve.run_receipt.v1",
  "runRef": "hydralisk-run-...",
  "model": "openai/gpt-oss-20b",
  "servedAlias": "openagents/khala-oss-20b",
  "engine": "vllm",
  "engineVersion": "0.10.1+gptoss",
  "gpu": { "name": "NVIDIA L4", "count": 1 },
  "quantization": { "weights": "MXFP4" },
  "usage": {
    "promptTokens": 0,
    "completionTokens": 0,
    "totalTokens": 0
  },
  "latency": {
    "ttftMs": 0,
    "wallMs": 0
  },
  "publicSafe": true
}
```

## Phase 2: OpenAgents Worker code changes

Make the smallest product-side change that can route a model to Hydralisk.

Files to touch:

- `apps/openagents.com/workers/api/src/inference/provider-adapter.ts`
- `apps/openagents.com/workers/api/src/inference/passthrough-adapter.ts`
- `apps/openagents.com/workers/api/src/inference/model-router.ts`
- `apps/openagents.com/workers/api/src/inference/model-serving-policy.ts`
- `apps/openagents.com/workers/api/src/inference/pricing.ts`
- `apps/openagents.com/workers/api/src/inference/model-catalog.ts`
- `apps/openagents.com/workers/api/src/index.ts`
- targeted tests under `apps/openagents.com/workers/api/src/inference/*.test.ts`

Recommended first implementation:

1. Add `HYDRALISK_ADAPTER_ID = "hydralisk-vllm"`.
2. Reuse the OpenAI-wire-format passthrough adapter shape for day zero, but
   construct it under the Hydralisk id with:
   - `HYDRALISK_BASE_URL`
   - `HYDRALISK_BEARER_TOKEN`
   - optional `HYDRALISK_GPT_OSS_20B_ENABLED`
3. Add a `hydralisk` supply lane to pricing/catalog policy.
4. Add `openagents/khala-oss-20b` to the pricing table on the `hydralisk` lane,
   with the same cost basis as `gpt-oss-20b` until real L4 cost accounting lands.
5. Keep direct `gpt-oss-20b` on Fireworks in the catalog until Hydralisk has a
   measured cost basis and fallback policy. Route the Khala alias to Hydralisk
   first.
6. Update the model router:
   - `openagents/khala-oss-20b` -> `hydralisk-vllm`
   - `gpt-oss-20b` remains Fireworks first for public direct calls at day zero
   - later promotion can route `gpt-oss-20b` to Hydralisk first with Fireworks
     fallback
7. Update serving policy so Hydralisk arms only when:
   - `HYDRALISK_GPT_OSS_20B_ENABLED=ready`
   - `HYDRALISK_BASE_URL` is present
   - `HYDRALISK_BEARER_TOKEN` is present
   - `HYDRALISK_GPT_OSS_20B_PREFLIGHT_REF` is public-safe
   - `HYDRALISK_GPT_OSS_20B_RECEIPT_REF` is public-safe

Do not add fuzzy routing. This is a bounded model-id and lane selector.

## Phase 3: OpenAgents deploy config

Worker secrets/config:

```text
INFERENCE_GATEWAY_ENABLED=true
HYDRALISK_GPT_OSS_20B_ENABLED=ready
HYDRALISK_BASE_URL=https://<hydralisk-host>
HYDRALISK_BEARER_TOKEN=<worker-secret>
HYDRALISK_GPT_OSS_20B_PREFLIGHT_REF=preflight.hydralisk.gpt_oss_20b.l4.v1
HYDRALISK_GPT_OSS_20B_RECEIPT_REF=receipt.hydralisk.gpt_oss_20b.l4.smoke.v1
```

Important: `HYDRALISK_BASE_URL` should point at the Hydralisk origin root, for
example `https://hydralisk-gpt-oss-20b.openagents.com`. The OpenAgents
passthrough adapter appends the OpenAI-compatible `/v1/chat/completions` path.

Keep Fireworks configured as fallback for direct open-model traffic. Do not
remove existing Fireworks routing during this rollout.

## Phase 4: smoke tests

Hydralisk host smoke:

```bash
curl -fsS "$HYDRALISK_ORIGIN/health"
curl -fsS "$HYDRALISK_ORIGIN/hydralisk/v1/capabilities"
curl -fsS "$HYDRALISK_ORIGIN/v1/chat/completions" \
  -H "authorization: Bearer $HYDRALISK_BEARER_TOKEN" \
  -H "content-type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [
      { "role": "system", "content": "You are Khala, concise and useful." },
      { "role": "user", "content": "Say READY in one word." }
    ],
    "max_tokens": 8
  }'
```

OpenAgents API smoke:

```bash
curl -fsS "https://openagents.com/v1/models" \
  -H "authorization: Bearer $OPENAGENTS_AGENT_TOKEN"

curl -N "https://openagents.com/v1/chat/completions" \
  -H "authorization: Bearer $OPENAGENTS_AGENT_TOKEN" \
  -H "content-type: application/json" \
  -d '{
    "model": "openagents/khala-oss-20b",
    "stream": true,
    "messages": [
      { "role": "user", "content": "Give a two sentence service status report." }
    ],
    "max_tokens": 120
  }'
```

Expected public response properties:

- `openagents.worker` or equivalent telemetry identifies `hydralisk-vllm`.
- `served_model`/receipt metadata identifies `openai/gpt-oss-20b`.
- No response text names OpenAI, GPT, vLLM, Hydralisk, or provider identity when
  the requested model is a Khala alias.
- Usage tokens are present for non-streaming, and streaming has a terminal usage
  path before metering is enabled for paid traffic.

## Day-zero safety limits

Use conservative limits until load is measured:

- internal/owner allowlist first;
- no tool calls through the gateway on day zero;
- max context lower than model maximum;
- max output cap, for example 1,024 tokens;
- one active L4 worker;
- no public speed claim;
- no accepted-outcome payout claim;
- Fireworks or Vertex fallback remains available for other Khala routes.

## Promotion gates

Internal live:

- Hydralisk host serves 20 successful non-streaming requests.
- Hydralisk host serves 20 successful streaming requests.
- OpenAgents Worker routes `openagents/khala-oss-20b` to `hydralisk-vllm`.
- Metering sees provider usage and can price from receipt-first usage.
- Public-safe receipt is available for at least one smoke run.

Catalog-visible:

- `/v1/models` shows `openagents/khala-oss-20b` only when Hydralisk is armed.
- Unknown/unarmed state returns `model_not_found` or `model_unavailable`, not a
  deep provider error.
- Route remains behind owner/internal allowlist or premium gate until spend cap
  and abuse controls are configured.

Public paid:

- rolling error rate under threshold;
- p95 TTFT and wall time measured;
- cost per accepted outcome measured against Fireworks;
- interrupt/rollback policy tested;
- no hidden reasoning or private prompt leakage in receipts;
- fallback route and customer-facing error copy verified.

## Rollback

Rollback must be instant:

1. Set `HYDRALISK_GPT_OSS_20B_ENABLED=off` or remove the base URL/token.
2. OpenAgents model-serving policy de-arms the Hydralisk catalog alias.
3. `openagents/khala-oss-20b` returns clean unavailability.
4. Existing `gpt-oss-20b` and `khala-code` routes keep using Fireworks/current
   supply.
5. Shut down the L4 VM only after no in-flight requests remain.

## Implementation order

1. Hydralisk: add proxy app and service contract.
2. Hydralisk: add GCE L4 runbook and systemd/container recipe.
3. Hydralisk: smoke `vllm serve openai/gpt-oss-20b` locally on the L4.
4. OpenAgents: add `hydralisk-vllm` adapter wiring.
5. OpenAgents: add `hydralisk` lane and `openagents/khala-oss-20b` catalog row.
6. OpenAgents: add model-router and serving-policy tests.
7. OpenAgents: deploy Worker config with Hydralisk disabled.
8. OpenAgents: set Hydralisk secrets/config and enable for internal traffic.
9. Run smoke tests and capture first public-safe receipt refs.
10. Decide whether direct `gpt-oss-20b` should promote from Fireworks-first to
    Hydralisk-first with Fireworks fallback.

## Immediate code tickets

Hydralisk repo:

- `H1`: FastAPI or Starlette proxy around local vLLM with bearer auth.
- `H2`: capability endpoint and in-memory/file-backed public-safe receipts.
- `H3`: GCE L4 systemd or Docker Compose runbook.
- `H4`: smoke script for health, non-streaming, streaming, and usage.

OpenAgents repo:

- `OA1`: `hydralisk-vllm` adapter registration from env.
- `OA2`: `hydralisk` supply lane in pricing/catalog/serving policy.
- `OA3`: `openagents/khala-oss-20b` model alias routed to Hydralisk.
- `OA4`: route tests for armed/unarmed Hydralisk and streaming pass-through.
- `OA5`: deploy secret/config checklist and rollback note.

## Sources

- OpenAI gpt-oss vLLM guide:
  `https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm`
- OpenAgents Hydralisk spec:
  `openagents/docs/inference/2026-06-23-hydralisk-python-nvidia-inference-stack.md`
- OpenAgents model router:
  `apps/openagents.com/workers/api/src/inference/model-router.ts`
- OpenAgents provider adapter seam:
  `apps/openagents.com/workers/api/src/inference/provider-adapter.ts`
- OpenAgents model serving policy:
  `apps/openagents.com/workers/api/src/inference/model-serving-policy.ts`
