# hydralisk

Hydralisk is the OpenAgents Python/NVIDIA inference lane.

It exists beside Psionic, not inside it. Psionic is the Rust-native ML substrate
where we build the runtime ourselves. Hydralisk is the pragmatic Python stack
for environments where conventional NVIDIA-serving practice is the fastest
honest path: vLLM, SGLang, TensorRT-LLM, Triton, NVIDIA Dynamo, CUDA containers,
and model-specific production recipes.

Initial targets:

- `gpt-oss-20b` on L4 with vLLM for the first cheap internal dogfood lane.
- `gpt-oss-120b` on H100/H200/B200/G4-class high-memory GPUs with vLLM.
- GLM-5.2 first as a hosted baseline, then as a high-memory SGLang/Dynamo
  self-hosting campaign.

Hydralisk should produce public-safe capability and run receipts for Khala and
OpenAgents to consume. It should not own pricing, credits, payout, referral,
customer routing, or public product promises.

## Status

Scaffold only. No runtime has landed yet.

The design anchor lives in the OpenAgents inference docs:

- `openagents/docs/inference/2026-06-23-hydralisk-python-nvidia-inference-stack.md`

First execution roadmap:

- [`docs/gpt-oss-20b-khala-live-roadmap.md`](docs/gpt-oss-20b-khala-live-roadmap.md)

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
