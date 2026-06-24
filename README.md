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
- DeepSeek-V4-Flash as a Google GPU admission experiment: G4 capacity was
  admitted on 2026-06-24 with 2 x RTX PRO 6000. The current blocker is now
  past the original vLLM `0.23.0` Blackwell FP8 scaled-mm failure: direct
  CUTLASS FP8 cases still fail, Triton block FP8 works after E8M0 scales are
  upcast, and local `o_proj` RHS rank/scale hotpatches still stop in DeepGEMM
  before `/v1/models`. A clean provider-guided vLLM/DeepGEMM container also
  builds and imports successfully on the G4 host, then fails before readiness
  on the original CUTLASS `dispatch_scaled_mm` path. The published-recipe
  GCE probe can see H100/H200/B200/GB200 catalog entries and machine types, but
  this project currently exposes only L4 regional GPU quota. The available
  Google lane today is therefore a custom RTX PRO 6000 kernel/offload path
  unless we obtain H100/H200/B200/GB200 quota. The NVFP4 Blackwell variant
  (`nvidia/DeepSeek-V4-Flash-NVFP4`) also builds and imports on the G4 host,
  but stock vLLM rejects every tested NVFP4 MoE backend before readiness; the
  remaining G4 path starts at the FlashInfer TRTLLM NVFP4 device gate. A
  default-off SM120 gate patch advances the two-card G4 probe into vLLM startup
  with `flashinfer_trtllm`, but the private-only host now blocks on Hugging
  Face artifact access before weight load. Cloud NAT for the `default`
  `us-central1` subnet fixed private config/artifact egress without restoring
  a VM external IP, and the rerun advanced into real model load with
  `FLASHINFER_TRTLLM`; it then stalled in the Hugging Face Xet / vLLM load path
  before `/v1/models`. Disabling Xet for the next private G4 run made snapshot
  acquisition deterministic enough to expose the current blocker again:
  non-expert FP8 layers hit vLLM's CUTLASS `dispatch_scaled_mm` path on SM120
  before `/v1/models`. Forcing vLLM's dense FP8 linear backend to `triton`
  with the derived-image E8M0 upcast patch removed that CUTLASS blocker. The
  active blocker is now DeepSeek V4's NVIDIA `o_proj` DeepGEMM `fp8_einsum`
  scale-factor layout assertion before readiness; an explicit invalid
  zero-`o_proj` load-only bypass then moves to a FlashInfer TRTLLM NVFP4 MoE
  GEMM runtime failure on the same SM120 host. A synthetic FlashInfer repro
  now reproduces that MoE GEMM failure without loading DeepSeek weights,
  Hugging Face artifacts, prompts, or vLLM scheduling. The newer FlashInfer
  B12x SM12x path does run DeepSeek-like synthetic MoE shapes on RTX PRO 6000
  when all experts are local, but it rejects expert parallelism
  (`num_local_experts != num_experts`). That makes the two-card G4 lane a
  compatibility research lane, not a near-serving stock-vLLM path, unless we
  use a wider no-EP G4 shape, add B12x expert parallelism, or build the custom
  offload/prefetch path. The eight-card G4 full-model attempt then proved B12x
  is rejected for DeepSeek-V4 because it lacks the model's required SwiGLU
  clamp. A clamp-backend sweep on the same private 8 x G4 host removed
  `flashinfer_cutlass` for the same reason and advanced `flashinfer_trtllm`
  into expert-parallel startup with 32 local experts per rank, where it now
  blocks in DeepGEMM `o_proj` scale-factor layout handling. The next useful
  G4 issue is therefore a correctness-first DeepSeek V4 `o_proj` fallback or
  scale-factor layout fix that preserves the TRTLLM MoE path. That fallback
  now exists as a default-off `bf16_einsum` probe path and moves the full model
  past `o_proj` on all eight ranks. The active blocker is now the FlashInfer
  TRTLLM NVFP4 MoE GEMM itself on RTX PRO 6000:
  `trtllm_batched_gemm_runner.cu:286`, `numBatches=32`, and
  `GemmMNK=512 4096 4096`. That exact full-model MoE shape is now reproduced
  synthetically without weights or vLLM scheduling, so stock
  `flashinfer_trtllm` is no longer a near-serving G4 lane by wrapper changes
  alone. The follow-up B12x viability probe shows the only positive SM120 MoE
  path is also not ready as-is: FlashInfer B12x has no `swiglu_limit` clamp
  surface, rejects the exact `32 / 256` expert-parallel shard, and only runs
  the DeepSeek-like synthetic shape when all 256 experts are local. The next G4
  work must be real kernel/scheduler work: B12x clamp, B12x expert
  parallelism/offload, or a SGLang-style expert repack plus prefetch lane. A
  follow-up local-shard remap probe proved B12x can run the exact per-rank
  shard when global expert IDs are remapped to a local 32-expert domain
  (`globalNumExperts=256`, `kernelNumExperts=32`, `localNumExperts=32`). That
  leaves clamp semantics plus dispatcher/offload correctness as the next real
  implementation step. Hydralisk now has a pure-Python local-shard reference
  fixture for that boundary: DeepSeek/vLLM SwiGLU clamp, global-to-local expert
  remap, nonlocal expert skipping, and deterministic nonzero routed output. A
  live wrapper-surface probe then found `B12xMoEWrapper` in the installed
  FlashInfer `0.6.12` image, but it only exposes `num_local_experts`; it lacks
  both `local_expert_offset` and `swiglu_limit`, so the G4 path still needs a
  wrapper upgrade or a Hydralisk-local B12x dispatcher/clamp shim before any
  full-model retry. A matched FlashInfer nightly upgrade
  (`flashinfer-python`, `flashinfer-cubin`, and `flashinfer-jit-cache`) reached
  `0.6.13.dev20260612` on the same G4 host, but the B12x wrapper surface was
  unchanged for our blocker: no `local_expert_offset`, no `swiglu_limit`, and
  the direct `256 / 32` expert-parallel call still rejects before launch. The
  next G4 issue should therefore build the Hydralisk-local dispatcher/clamp
  shim against the reference fixture. Hydralisk now has the dispatcher half of
  that shim: fixed-shape global-to-local expert remap, zero-scale masking for
  nonlocal/out-of-range routes, reference-equivalence tests on nonzero inputs,
  and a fail-closed gate for missing DeepSeek `swiglu_limit` support. The
  live B12x kernel also accepts that dispatcher-shaped masked local-domain
  input on RTX PRO 6000 (`maskedRouteCount=1536`, `outShape=[512,4096]`). The
  source audit now maps that remaining blocker to exact FlashInfer B12x SM120
  patch points: API surface, `launch_sm120_moe`, and the fused gated-SiLU
  activation path. Hydralisk now also has a repeatable FlashInfer B12x clamp
  overlay that dry-runs against the local reference checkout and source-marks
  the static, micro, dynamic, and W4A16 activation sites for the next G4
  compile/runtime fixture. A disposable G4 container then converted the static
  marker into real CuTe/CUTLASS clamp ops and ran both zero and nonzero tiny
  B12x fixtures with `swiglu_limit=10.0` on RTX PRO 6000. The static clamp path
  is now a real GPU proof. The follow-up dynamic fixture then patched
  `moe_dynamic_kernel.py` and ran the 512-token DeepSeek-shaped masked
  local-shard case (`kernelNumExperts=32`, `globalNumExperts=256`, `topK=6`)
  with finite nonzero output. The clamp-patched B12x full-model image now
  builds, imports on all eight G4 GPUs, and starts vLLM with
  `moe_backend=flashinfer_b12x`. With the existing `bf16_einsum` `o_proj`
  fallback enabled, execution moves past the old `o_proj` DeepGEMM blocker and
  stops in DeepSeek MLA attention metadata during vLLM cudagraph memory
  profiling: `get_paged_mqa_logits_metadata` fails with
  `attention.hpp:219: Unsupported architecture` on SM120. Enabling vLLM eager
  mode avoids that cudagraph profiling path and brings the full model to a live
  `/v1/models` endpoint on the same 8 x G4 host, but the first public-safe
  generation smoke fails in `flash_mla_sparse_fwd` because the sparse prefill
  kernel only admits SM90a and SM100f. The remaining work is therefore an
  SM120-safe DeepSeek FlashMLA sparse-prefill backend or a correctness-first
  prefill fallback before any generation or serving claim. A source audit then
  found a better next probe already in vLLM: explicit
  `FLASHINFER_MLA_SPARSE_DSV4` backend selection routes DeepSeek V4 to
  `DeepseekV4FlashInferMLAAttention`, which avoids `flash_mla_sparse_fwd` and
  calls FlashInfer's TRTLLM sparse MLA launcher instead. Hydralisk now exposes
  that as `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4` for the next G4
  smoke once gcloud auth is interactive again.

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
- [`docs/evidence/2026-06-24-deepseek-v4-flash-gce-load-smoke.md`](docs/evidence/2026-06-24-deepseek-v4-flash-gce-load-smoke.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-g4-backend-matrix.md`](docs/evidence/2026-06-24-deepseek-v4-flash-g4-backend-matrix.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-scaled-mm-g4-probe.md`](docs/evidence/2026-06-24-deepseek-v4-flash-scaled-mm-g4-probe.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-e8m0-upcast-g4.md`](docs/evidence/2026-06-24-deepseek-v4-flash-e8m0-upcast-g4.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-o-proj-g4.md`](docs/evidence/2026-06-24-deepseek-v4-flash-o-proj-g4.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-o-proj-group-rhs-g4.md`](docs/evidence/2026-06-24-deepseek-v4-flash-o-proj-group-rhs-g4.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-o-proj-rhs-scale-g4.md`](docs/evidence/2026-06-24-deepseek-v4-flash-o-proj-rhs-scale-g4.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-provider-stack-g4.md`](docs/evidence/2026-06-24-deepseek-v4-flash-provider-stack-g4.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-published-recipe-gce-admission.md`](docs/evidence/2026-06-24-deepseek-v4-flash-published-recipe-gce-admission.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-g4-probe.md`](docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-g4-probe.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-sm120-g4-probe.md`](docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-sm120-g4-probe.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-private-egress-g4.md`](docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-private-egress-g4.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-no-xet-g4.md`](docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-no-xet-g4.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-triton-g4.md`](docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-triton-g4.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-oproj-g4.md`](docs/evidence/2026-06-24-deepseek-v4-flash-nvfp4-oproj-g4.md)
- [`docs/evidence/2026-06-24-flashinfer-trtllm-nvfp4-moe-g4.md`](docs/evidence/2026-06-24-flashinfer-trtllm-nvfp4-moe-g4.md)
- [`docs/evidence/2026-06-24-flashinfer-b12x-moe-g4.md`](docs/evidence/2026-06-24-flashinfer-b12x-moe-g4.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-b12x-wide-g4.md`](docs/evidence/2026-06-24-deepseek-v4-flash-b12x-wide-g4.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-clamp-backends-wide-g4.md`](docs/evidence/2026-06-24-deepseek-v4-flash-clamp-backends-wide-g4.md)
- [`docs/evidence/2026-06-24-deepseek-v4-flash-oproj-fallback-wide-g4.md`](docs/evidence/2026-06-24-deepseek-v4-flash-oproj-fallback-wide-g4.md)
- [`docs/evidence/2026-06-24-flashinfer-trtllm-nvfp4-moe-full-shape-g4.md`](docs/evidence/2026-06-24-flashinfer-trtllm-nvfp4-moe-full-shape-g4.md)
- [`docs/evidence/2026-06-24-flashinfer-b12x-clamp-ep-g4.md`](docs/evidence/2026-06-24-flashinfer-b12x-clamp-ep-g4.md)
- [`docs/evidence/2026-06-24-flashinfer-b12x-local-shard-remap-g4.md`](docs/evidence/2026-06-24-flashinfer-b12x-local-shard-remap-g4.md)
- [`docs/evidence/2026-06-24-deepseek-b12x-local-shard-reference-fixture.md`](docs/evidence/2026-06-24-deepseek-b12x-local-shard-reference-fixture.md)
- [`docs/evidence/2026-06-24-flashinfer-b12x-wrapper-surface-g4.md`](docs/evidence/2026-06-24-flashinfer-b12x-wrapper-surface-g4.md)
- [`docs/evidence/2026-06-24-flashinfer-b12x-nightly-wrapper-g4.md`](docs/evidence/2026-06-24-flashinfer-b12x-nightly-wrapper-g4.md)
- [`docs/evidence/2026-06-24-deepseek-b12x-local-dispatcher-shim.md`](docs/evidence/2026-06-24-deepseek-b12x-local-dispatcher-shim.md)
- [`docs/evidence/2026-06-24-flashinfer-b12x-masked-dispatch-g4.md`](docs/evidence/2026-06-24-flashinfer-b12x-masked-dispatch-g4.md)
- [`docs/evidence/2026-06-24-deepseek-b12x-clamp-patch-points.md`](docs/evidence/2026-06-24-deepseek-b12x-clamp-patch-points.md)
- [`docs/evidence/2026-06-24-deepseek-b12x-clamp-overlay.md`](docs/evidence/2026-06-24-deepseek-b12x-clamp-overlay.md)
- [`docs/evidence/2026-06-24-deepseek-b12x-static-clamp-g4.md`](docs/evidence/2026-06-24-deepseek-b12x-static-clamp-g4.md)
- [`docs/evidence/2026-06-24-deepseek-b12x-dynamic-clamp-g4.md`](docs/evidence/2026-06-24-deepseek-b12x-dynamic-clamp-g4.md)
- [`docs/evidence/2026-06-24-deepseek-b12x-full-model-g4.md`](docs/evidence/2026-06-24-deepseek-b12x-full-model-g4.md)
- [`docs/evidence/2026-06-24-deepseek-b12x-eager-mla-g4.md`](docs/evidence/2026-06-24-deepseek-b12x-eager-mla-g4.md)
- [`docs/evidence/2026-06-24-deepseek-flashmla-sparse-audit.md`](docs/evidence/2026-06-24-deepseek-flashmla-sparse-audit.md)
- [`docs/evidence/2026-06-24-deepseek-g4-gcloud-auth-preflight.md`](docs/evidence/2026-06-24-deepseek-g4-gcloud-auth-preflight.md)
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
