# FlashInfer B12x wrapper surface G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/30

Script:
[`scripts/probe-flashinfer-b12x-moe-gce.sh`](../../scripts/probe-flashinfer-b12x-moe-gce.sh)

## Target

- Project: `openagentsgemini`
- Zone: `us-central1-b`
- Instance: `hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036`
- Machine type: `g4-standard-384`
- Accelerator: 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition
- Image: `hydralisk-deepseek-v4-oproj-fallback-g4-vllm:20260624095206`
- Sequence length: `512`
- Hidden size: `4096`
- Intermediate size: `2048`
- Global experts: `256`
- Local experts per rank: `32`
- Top-k: `6`
- Required DeepSeek clamp: `swiglu_limit=10.0`

## Runtime

- FlashInfer: `0.6.12`
- Torch: `2.11.0+cu130`
- CUDA runtime reported by Torch: `13.0`
- CUDA capability: `[12, 0]`
- Device: NVIDIA RTX PRO 6000 Blackwell Server Edition

## Result

The live image does include a B12x wrapper:

```text
B12xMoEWrapper(
  num_experts,
  top_k,
  hidden_size,
  intermediate_size,
  *,
  use_cuda_graph=False,
  max_num_tokens=4096,
  num_local_experts=None,
  output_dtype=torch.bfloat16,
  device='cuda',
  activation='silu',
  activation_precision='fp4',
  quant_mode=None,
  source_format='modelopt'
)
```

That is useful but not sufficient for DeepSeek-V4-Flash on the current G4
lane:

- `supportsNumLocalExpertsKwarg`: `true`
- `supportsLocalExpertOffsetKwarg`: `false`
- `supportsSwigluLimitKwarg`: `false`
- `supportsActivationKwarg`: `true`

The direct `b12x_fused_moe` entry point still has no DeepSeek clamp surface:

```text
b12x_fused_moe() got an unexpected keyword argument 'swiglu_limit'
```

The direct global expert-parallel call still fails before kernel launch:

```text
NotImplementedError:
b12x_fused_moe does not yet support Expert Parallelism
(num_local_experts=32 != num_experts=256)
```

The local-shard remap control still succeeds:

```text
globalNumExperts=256
kernelNumExperts=32
localNumExperts=32
routingDomain=local_shard_remapped
outShape=[512, 4096]
maxMemoryAllocatedBytes=487574016
```

The all-local positive control also succeeds:

```text
globalNumExperts=256
kernelNumExperts=256
localNumExperts=256
outShape=[512, 4096]
maxMemoryAllocatedBytes=4194399744
```

## Interpretation

FlashInfer `0.6.12` on the live G4 image already exposes the wrapper name we
need, but the installed wrapper does not eliminate Hydralisk's next work. It
cannot identify the rank-local expert offset and it cannot apply DeepSeek's
`swiglu_limit=10.0` semantics.

The next hard step is therefore one of:

1. prove a newer FlashInfer/vLLM B12x wrapper exposes the missing local-offset
   surface on RTX PRO 6000, then add or shim the clamp;
2. implement a Hydralisk-local dispatcher/shim that remaps global experts into
   the local 32-expert B12x domain and applies the reference clamp semantics;
3. pivot to the SGLang-style expert repack plus prefetch/offload path.

Given the live probe, a full-model retry on this image without wrapper upgrade
or shim work would repeat the same blockers.

## Cleanup

- Docker containers left running after probe: `0`
- GPU memory in use after probe: `0 MiB`

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
