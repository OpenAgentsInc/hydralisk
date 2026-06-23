# AGENTS

Hydralisk is the standalone OpenAgents Python/NVIDIA inference lane.

## Scope

- Use this repo for conventional Python ML serving work: vLLM, SGLang,
  TensorRT-LLM, Triton, NVIDIA Dynamo, CUDA containers, GCE/GKE deployment
  recipes, model profiles, benchmarks, evals, and public-safe run receipts.
- Keep Psionic as the Rust-native ML substrate. Hydralisk may produce evidence
  and behavior targets for Psionic, but should not claim to be Psionic-native.
- Keep OpenAgents/Khala product authority outside this repo: pricing, credits,
  payout, referral, customer routing, and public promises live in the product
  repos.

## Invariants

- Never commit secrets, raw prompts, private source, hidden reasoning traces,
  model-provider credentials, or large model artifacts.
- Do not commit model weights, checkpoints, compiled engines, generated
  benchmark outputs, or GPU profiler dumps.
- Make model revision, engine version, image digest, CUDA/runtime versions, GPU
  topology, quantization mode, parser behavior, and eval gate explicit for any
  served capability.
- Fail closed when a lane lacks a model profile, engine pin, GPU admission
  check, quantization eval, or public-safe receipt path.

## Git

- Work on `main` unless the user explicitly asks for a branch or PR flow.
- Keep commits scoped to Hydralisk. Do not stage or commit parent workspace
  files from this repo.
