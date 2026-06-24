# FlashInfer B12x nightly wrapper G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/31

Script:
[`scripts/probe-flashinfer-b12x-moe-gce.sh`](../../scripts/probe-flashinfer-b12x-moe-gce.sh)

## Target

- Project: `openagentsgemini`
- Zone: `us-central1-b`
- Instance: `hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036`
- Machine type: `g4-standard-384`
- Accelerator: 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition
- Base image: `hydralisk-deepseek-v4-oproj-fallback-g4-vllm:20260624095206`
- Container install mode: ephemeral nightly upgrade only

## Install attempts

The probe first tried `flashinfer-python` from the FlashInfer nightly index.
That installed but failed import because the base image still carried
`flashinfer-cubin 0.6.12`.

The second attempt installed `flashinfer-python flashinfer-cubin` from the same
nightly index. That installed but failed import because the base image still
carried `flashinfer-jit-cache 0.6.12+cu130`.

The successful install path matched all three packages:

```text
python3 -m pip install --upgrade --pre \
  --index-url https://flashinfer.ai/whl/nightly/ \
  --extra-index-url https://flashinfer.ai/whl/nightly/cu130 \
  --no-deps \
  flashinfer-python flashinfer-cubin flashinfer-jit-cache
```

The final install completed in `50.061` seconds with return code `0`.

## Runtime

- FlashInfer: `0.6.13.dev20260612`
- Torch: `2.11.0+cu130`
- CUDA runtime reported by Torch: `13.0`
- CUDA capability: `[12, 0]`
- Device: NVIDIA RTX PRO 6000 Blackwell Server Edition

## Result

The nightly wrapper surface is effectively unchanged for the DeepSeek blocker:

- `B12xMoEWrapper`: available
- `supportsNumLocalExpertsKwarg`: `true`
- `supportsLocalExpertOffsetKwarg`: `false`
- `supportsSwigluLimitKwarg`: `false`
- `supportsActivationKwarg`: `true`

The direct `b12x_fused_moe` function still has no clamp kwarg:

```text
b12x_fused_moe() got an unexpected keyword argument 'swiglu_limit'
```

The direct global expert-parallel call still rejects the full-model shard:

```text
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

Upgrading FlashInfer packages inside the container is not enough. The current
nightly moves from `0.6.12` to `0.6.13.dev20260612`, but it does not expose the
two missing surfaces Hydralisk needs for DeepSeek-V4-Flash on G4:

1. `local_expert_offset` or equivalent global-to-local expert routing;
2. DeepSeek/vLLM `swiglu_limit=10.0` clamp semantics.

That removes the wrapper-upgrade shortcut for this image. The next useful issue
is to implement a Hydralisk-local B12x dispatcher/clamp shim against the
existing pure-Python reference fixture, then compare tiny nonzero local-shard
outputs before any full-model retry.

## Cleanup

- Docker containers left running after probe: `0`
- GPU memory in use after probe: `0 MiB`

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
