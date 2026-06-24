# FlashInfer B12x local expert-shard remap G4 evidence

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/28

Script:
`scripts/probe-flashinfer-b12x-moe-gce.sh`

Generated run directory, not committed:
`.hydralisk/flashinfer-b12x-moe-20260624101257/`

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

## What changed

Issue #27 proved the current B12x function rejects a direct global expert
parallel call:

```text
num_experts=256
num_local_experts=32
```

This probe adds a local-shard remap case. It preserves the semantic model shape
as `globalNumExperts=256`, but passes only the per-rank local shard to B12x:

```text
kernelNumExperts=32
localNumExperts=32
routingDomain=local_shard_remapped
```

The synthetic router assignments are local IDs in `[0, 31]`, which is the shape
a Hydralisk or SGLang-style expert dispatcher would feed after routing,
all-to-all/offload selection, and global-to-local expert ID remapping.

The probe still does not load model weights, prompts, responses, Hugging Face
artifacts, vLLM scheduling, or hidden reasoning.

## Result

The direct global expert-parallel call still fails before kernel launch:

```json
{"case":"deepseek_shape_ep","ok":false,"globalNumExperts":256,"kernelNumExperts":256,"localNumExperts":32,"type":"NotImplementedError"}
```

The local-shard remap case succeeds:

```json
{"case":"deepseek_shape_local_shard_remap","ok":true,"globalNumExperts":256,"kernelNumExperts":32,"localNumExperts":32,"routingDomain":"local_shard_remapped","seqLen":512,"hiddenSize":4096,"intermediateSize":2048,"topK":6,"outShape":[512,4096],"elapsedMs":26550.361328125,"maxMemoryAllocatedBytes":487574016}
```

The all-local positive control also still succeeds:

```json
{"case":"deepseek_shape_no_ep","ok":true,"globalNumExperts":256,"kernelNumExperts":256,"localNumExperts":256,"outShape":[512,4096],"elapsedMs":26468.029296875,"maxMemoryAllocatedBytes":4194399744}
```

## Interpretation

This removes one blocker from the custom G4 path. B12x can execute the exact
per-rank expert shard shape once global expert IDs are remapped into the local
32-expert domain.

That does not make DeepSeek-V4-Flash serve-ready on G4. The remaining missing
pieces are now narrower:

- implement DeepSeek's required `swiglu_limit=10.0` clamp semantics for B12x;
- build or reuse the dispatcher/offload layer that maps global experts to local
  expert IDs and supplies only resident local-shard weights to the kernel;
- integrate the path into a vLLM or SGLang serving process with valid model
  weights, all-to-all or host-RAM expert movement, and correctness tests.

The current FlashInfer documentation for the newer API surface describes B12x
wrapper-level local expert controls, including local expert counts and offsets:
https://docs.flashinfer.ai/api/fused_moe.html

The vLLM B12x backend documentation also identifies this as the SM120/SM121
RTX PRO 6000 / DGX Spark path:
https://docs.vllm.ai/en/v0.23.0/api/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe/

The next issue should test a newer FlashInfer/vLLM B12x wrapper or implement a
Hydralisk-local dispatcher shim around the installed B12x kernel. The immediate
success criterion is not full serving yet; it is a tiny correctness fixture that
proves local-shard remapping and clamp semantics match a reference PyTorch MoE
for a nonzero synthetic input.

## Cleanup

After the probe, no Docker containers were running and all eight GPUs reported
`0 MiB` used.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
