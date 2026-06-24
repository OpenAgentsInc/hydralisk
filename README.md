# Hydralisk

Hydralisk is the standalone OpenAgents Python/NVIDIA inference lane. It owns
conventional serving work such as vLLM, SGLang, TensorRT-LLM, CUDA host
runbooks, model profiles, smoke tests, and public-safe receipts.

Current Khala lane:

- served model: `openai/gpt-oss-20b`
- internal alias: `khala`
- public alias: `openagents/khala`
- compatibility aliases: `openagents/khala-oss-20b`, `gpt-oss-20b`
- engine: vLLM
- first host class: one NVIDIA L4
- proxy port: `127.0.0.1:8012`
- raw vLLM port: `127.0.0.1:8000`

The proxy exposes public-safe health, capabilities, receipt lookup, and
bearer-authenticated Chat Completions forwarding. Raw vLLM stays localhost-only.

## Local Development

```bash
uv sync --extra test
uv run pytest
```

Run the proxy against a local vLLM server:

```bash
export HYDRALISK_BEARER_TOKEN=local-dev-token
uv run hydralisk-proxy --host 127.0.0.1 --port 8012
```

Smoke it:

```bash
uv run hydralisk-smoke \
  --base-url http://127.0.0.1:8012 \
  --bearer-token "$HYDRALISK_BEARER_TOKEN" \
  --model openai/gpt-oss-20b
```

## Host Runbook

Use [docs/gce-l4-vllm-runbook.md](docs/gce-l4-vllm-runbook.md) for the GCE L4
setup, systemd services, start/stop, rollback, and public-safe evidence rules.

## Boundaries

Hydralisk exists beside Psionic, not inside it. Psionic is the Rust-native ML
substrate where we build the runtime ourselves. Hydralisk is the pragmatic
Python stack for environments where conventional NVIDIA-serving practice is the
fastest honest path.

Hydralisk must not claim to be Psionic-native, an admitted Pylon payout lane, or
OpenAgents product authority. Pricing, credits, routing, settlement, payout,
public copy, and product-promise promotion remain in the product repos.

Initial targets:

- `gpt-oss-20b` on L4 with vLLM for the first cheap internal dogfood lane.
- `gpt-oss-120b` on H100/H200/B200/G4-class high-memory GPUs with vLLM.
- GLM-5.2 first as a hosted baseline, then as a high-memory SGLang/Dynamo
  self-hosting campaign.
- DeepSeek-V4-Flash as a Google GPU admission experiment: start with GGUF
  parser validation and GCE lane classification, then try G4/H100 load smoke
  only on fresh non-product hosts.

Hydralisk should produce public-safe capability and run receipts for Khala and
OpenAgents to consume. It should not own pricing, credits, payout, referral,
customer routing, or public product promises.

## Status

The first runtime scaffold has landed: an authenticated Hydralisk proxy for
`openai/gpt-oss-20b`, systemd units for a one-L4 vLLM host, and a GCE runbook.
Live host promotion still requires installing the repo on a fresh or explicitly
reused L4 VM, setting the bearer token out-of-band, smoking the proxy, and
publishing the HTTPS origin to OpenAgents.

The design anchor lives in the OpenAgents inference docs:

- `openagents/docs/inference/2026-06-23-hydralisk-python-nvidia-inference-stack.md`

First execution roadmap:

- [`docs/gpt-oss-20b-khala-live-roadmap.md`](docs/gpt-oss-20b-khala-live-roadmap.md)
- [`docs/gce-l4-vllm-runbook.md`](docs/gce-l4-vllm-runbook.md)
- [`docs/glm-5.2-sglang-preflight-runbook.md`](docs/glm-5.2-sglang-preflight-runbook.md)
- [`docs/deepseek-v4-flash-gce-preflight.md`](docs/deepseek-v4-flash-gce-preflight.md)
- [`profiles/glm-5.2-fp8-sglang.json`](profiles/glm-5.2-fp8-sglang.json)
- [`profiles/deepseek-v4-flash-gce-preflight.json`](profiles/deepseek-v4-flash-gce-preflight.json)

## Early shape

```text
hydralisk/
  hydralisk/
    serve/
    engines/
    models/
    dynamo/
    receipts/
    bench/
    evals/
  deploy/
    gce/
    gke/
    containers/
  docs/
```

## Guardrails

- Do not commit secrets, raw prompts, private source, model credentials, or
  hidden reasoning traces.
- Do not commit model weights, checkpoints, compiled engines, benchmark output,
  or large generated artifacts.
- Keep engine versions, model revisions, container images, GPU shape, CUDA
  runtime, parser behavior, and quantization mode explicit in receipts.
- Fail closed when a model profile, engine pin, GPU admission check,
  quantization eval, or public-safe receipt path is missing.
