# FlashInfer B12x clamp and expert-shard G4 evidence

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/27

Script:
`scripts/probe-flashinfer-b12x-moe-gce.sh`

Generated run directory, not committed:
`.hydralisk/flashinfer-b12x-moe-20260624100738/`

## Target

- Project: `openagentsgemini`
- Instance: `hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036`
- Zone: `us-central1-b`
- Machine: `g4-standard-384`
- GPUs: 8 x `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- VM external IP: none
- Image: `hydralisk-deepseek-v4-oproj-fallback-g4-vllm:20260624095206`
- FlashInfer: `0.6.12`
- Torch: `2.11.0+cu130`
- CUDA runtime: `13.0`

## Why this probe exists

The provider inventory for DeepSeek-V4-Flash points at a known-good vLLM /
DeepGEMM recipe on H200, B200, GB200, or DGX Station-class hardware. Our
admitted Google lane is 8 x RTX PRO 6000 G4. Stock `flashinfer_trtllm` now
fails the exact full-model NVFP4 MoE GEMM shape on G4, while B12x is the only
FlashInfer path that has shown a positive SM120 MoE kernel result.

This probe answers the two B12x questions without loading DeepSeek weights,
Hugging Face artifacts, prompts, responses, or vLLM scheduling:

- does live FlashInfer B12x expose a DeepSeek-style SwiGLU clamp surface?
- does live FlashInfer B12x accept the exact 8-way full-model expert shard?

## Interface result

The live B12x signature exposes `activation`, `num_local_experts`, and NVFP4
inputs, but not `swiglu_limit`:

```json
{"supportsActivationKwarg":true,"supportsNumLocalExpertsKwarg":true,"supportsSwigluLimitKwarg":false}
```

The source-level signal in the installed package matches that interface:

```json
{"mentionsExpertParallelRejection":true,"mentionsSwigluLimit":false,"mentionsSwigluLimitValue":false}
```

An explicit clamp kwarg probe fails immediately:

```text
TypeError: b12x_fused_moe() got an unexpected keyword argument 'swiglu_limit'
```

## Exact 8-way shard result

Synthetic shape:

```text
seq_len           512
hidden_size       4096
intermediate_size 2048
num_experts       256
local_num_experts 32
top_k             6
swiglu_limit      10.0
```

B12x rejects the exact 8-way expert-parallel shard before kernel launch:

```text
NotImplementedError:
b12x_fused_moe does not yet support Expert Parallelism
(num_local_experts=32 != num_experts=256).
```

## Positive control

The same live image still runs the all-local DeepSeek-like B12x shape:

```json
{"case":"deepseek_shape_no_ep","ok":true,"seqLen":512,"hiddenSize":4096,"intermediateSize":2048,"numExperts":256,"localNumExperts":256,"topK":6,"outShape":[512,4096],"elapsedMs":26542.173828125,"maxMemoryAllocatedBytes":3724553216}
```

## Decision

B12x is the right next G4 direction only if we are willing to modify or replace
the kernel path. Current FlashInfer `0.6.12` cannot serve DeepSeek-V4-Flash on
our G4 host as-is because it is missing both required pieces:

- DeepSeek's required `swiglu_limit=10.0` clamp semantics.
- Expert parallelism or an offload scheduler for `32 / 256` local experts per
  rank.

The next issue should choose one concrete implementation path:

- patch/port B12x clamped SwiGLU support and then test a tiny correctness
  fixture against the clamp-capable TRTLLM semantics;
- add B12x expert-parallel routing for local expert shards;
- or build the SGLang-style expert repack, hot expert cache, and host-RAM
  offload/prefetch lane.

More stock vLLM flag trials on G4 are not a credible path to readiness.

## Cleanup

After the probe, no Docker containers were running and all eight GPUs reported
`0 MiB` used.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
