# FlashInfer B12x masked local-dispatch G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/33

Script:
[`scripts/probe-flashinfer-b12x-moe-gce.sh`](../../scripts/probe-flashinfer-b12x-moe-gce.sh)

## Target

- Project: `openagentsgemini`
- Zone: `us-central1-b`
- Instance: `hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036`
- Machine type: `g4-standard-384`
- Accelerator: 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition
- Image: `hydralisk-deepseek-v4-oproj-fallback-g4-vllm:20260624095206`
- FlashInfer install mode: `none`
- FlashInfer runtime: `0.6.12`

## Result

The probe added a dispatcher-shaped local B12x case that preserves fixed
`[tokens, top_k]` shape while masking nonlocal routes as local expert `0` with
route scale `0.0`.

The live B12x kernel accepted that masked local-domain input:

```text
case=deepseek_shape_local_shard_masked_dispatch
globalNumExperts=256
kernelNumExperts=32
localNumExperts=32
routingDomain=local_shard_masked_zero_scale
zeroEveryNthRoute=2
maskedFillExpert=0
maskedRouteCount=1536
ok=true
outShape=[512, 4096]
maxMemoryAllocatedBytes=940587520
```

The direct global expert-parallel control still fails:

```text
numExperts=256
localNumExperts=32
ok=false
type=NotImplementedError
```

The clamp probe still fails:

```text
b12x_fused_moe() got an unexpected keyword argument 'swiglu_limit'
```

## Interpretation

The fixed-shape zero-scale masking strategy from Hydralisk's local dispatcher
is acceptable to the live B12x kernel on RTX PRO 6000. That means the
global-to-local routing shim is not blocked by the kernel's input contract.

This does not solve DeepSeek-V4-Flash serving yet. The remaining hard blocker
is unchanged: B12x still lacks DeepSeek/vLLM `swiglu_limit=10.0` clamp
semantics. The next useful issue is clamp-capable GPU work, either by adding a
kernel-side clamp path or by finding a B12x-compatible path that can apply the
clamp before/inside the routed MoE operation.

## Cleanup

- Docker containers left running after probe: `0`
- GPU memory in use after probe: `0 MiB`

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
