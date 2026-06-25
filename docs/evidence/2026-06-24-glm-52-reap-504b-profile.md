# GLM-5.2 504B REAP profile and evidence contract

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/82

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Public-safety boundary: this packet contains public model metadata, serving
plan metadata, and evidence rules only. It contains no bearer tokens, model
credentials, raw prompts, responses, private source, hidden reasoning traces,
weights, checkpoints, compiled engines, profiler dumps, or large generated
logs.

## Goal

Pin the canonical Hydralisk lane for `0xSero/GLM-5.2-504B` before any GCE
allocation, model staging, or serving claim. This lane is the 504B REAP/NVFP4
primary target for 4 x RTX PRO 6000 G4. The related 469B REAP artifact is a
fallback only after the 504B lane fails a documented gate.

## Immutable model metadata

- Model: `0xSero/GLM-5.2-504B`
- Revision:
  `cb6b1e0451b9d560cda864f84187869c9a679712`
- License: MIT
- Architecture: `GlmMoeDsaForCausalLM`
- Model type: `glm_moe_dsa`
- Quantization: ModelOpt NVFP4 / `modelopt_fp4`
- Context window: 1,048,576 tokens
- Layers: 78
- Dense layers: 3
- MoE layers: 75
- MTP layers: 1
- Routed experts per MoE layer: 168
- Experts per token: 8
- Shared experts: 1
- Safetensors payload metadata: 318,247,808,128 bytes
- Expected safetensor shard count: 63

Coherence-critical DSA indexer pattern:

```text
FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
```

## Runtime target

Primary hardware:

- Provider: GCE
- Project: `openagentsgemini`
- Zones: `us-central1-b`, then `us-central1-f`
- Shape: `g4-standard-192`
- Accelerator: 4 x `nvidia-rtx-pro-6000`
- Expected visible memory: 97,887 MiB per GPU, 391,548 MiB total

Runtime family:

- Engine: vLLM, b12x SM120 recipe
- Initial image tag:
  `voipmonitor/vllm:black-benediction-b12xpr11-vllmbb6c5b7-b12xd90d89c-fi3395b41aa8d-dg324aced12c-cu132-20260608`
- Required digest: pending issue #85 image inspection
- Quantization flag: `modelopt_fp4`
- KV cache dtype: `fp8`
- Attention backend: `B12X_MLA_SPARSE`
- MoE backend: `b12x`
- Tool-call parser: `glm47`
- Reasoning parser: `glm45`

First load should start below the final target:

- Tensor parallel size: 4
- Decode-context parallel size: 4
- Max model length: 32K tokens
- Max sequences: 1
- MTP: disabled until the first health and completion smoke pass

Target after smoke:

- Max model length: up to 250K tokens if the context ladder passes
- Max sequences: 2 or higher only after measured memory/concurrency gates
- MTP: enabled only as a separate tuning gate

Issue #88 update: the context ladder passed at 250K on four selected G4 RTX PRO
6000 GPUs with `max_num_seqs=2` and `max_num_batched_tokens=4096`. The
2026-06-25 speed gate then admitted MTP-2/no-`min_p` as the private speed
canary path. See
[`2026-06-24-glm-52-reap-504b-tuning.md`](2026-06-24-glm-52-reap-504b-tuning.md)
and
[`2026-06-25-glm-52-reap-504b-mtp2-speed-gate.md`](2026-06-25-glm-52-reap-504b-mtp2-speed-gate.md).

## Sampler guardrail

The 0xSero report says loop risk is the dominant behavioral regression after
REAP pruning and Router-KD recovery. Hydralisk should use these defaults for
first private smokes and evals:

- `min_p=0.05`
- `repetition_penalty=1.05` or `1.10`
- `temperature=1.0`
- `top_p=0.95`

Any Terminal-Bench or coding-agent comparison must report the exact sampler
settings and retry policy used.

## Evidence contract

Every claim in this lane must attach a public-safe receipt that names:

- Model repository and revision.
- Runtime image tag and digest.
- GCE project, zone, machine type, accelerator type, GPU count, and topology.
- Driver, CUDA runtime, container CUDA, NCCL, engine version, and parser flags.
- Launch flags: TP, DCP, quantization, KV dtype, context, sequence count, MTP,
  attention backend, MoE backend, and index pattern.
- Admission, staging, health, completion, tuning, and eval status.
- Token counts, latency, decode rate, GPU memory, and sanitized error class
  when applicable.

Receipts must not contain:

- Cloud credentials or bearer tokens.
- Hugging Face tokens or model-provider credentials.
- Model weights, checkpoints, compiled engines, or generated benchmark dumps.
- Raw prompts, raw responses, hidden reasoning traces, private source, customer
  data, or private repository contents.
- GPU profiler dumps or large raw logs.

## Claim boundary

This issue creates a profile and evidence contract only. It does not prove
admission, loadability, health, generation, quality, Terminal-Bench
performance, concurrency, or production readiness.

The earliest honest public status after this issue is:

```text
planned profile only; no serving claim
```
