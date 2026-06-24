# DeepSeek B12x dynamic clamp G4 fixture

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/37

## Target

- Project: `openagentsgemini`
- Zone: `us-central1-b`
- Instance: `hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036`
- Machine type: `g4-standard-384`
- Accelerator: 8 x NVIDIA RTX PRO 6000 Blackwell Server Edition
- Image: `hydralisk-deepseek-v4-oproj-fallback-g4-vllm:20260624095206`
- FlashInfer runtime: `0.6.12`
- Torch: `2.11.0+cu130`
- CUDA runtime: `13.0`

## Patch applied in disposable container

The fixture applied the Hydralisk B12x clamp overlay to the container
`site-packages` tree, then converted the `moe_dynamic_kernel.py` marker into
actual CuTe/CUTLASS clamp operations:

```text
overlayOk=true
dynamicMarkersPatched=1
```

The dynamic activation operation matches the DeepSeek/vLLM clamp shape:

```text
g = min(g, 10.0)
u = min(max(u, -10.0), 10.0)
```

No model weights were loaded. The container was disposable and removed after
the fixture.

## Dynamic masked local-shard fixture

The fixture used the DeepSeek-shaped B12x local-domain route from earlier G4
work:

```text
seqLen=512
hiddenSize=4096
intermediateSize=2048
globalNumExperts=256
kernelNumExperts=32
localNumExperts=32
topK=6
maskedRouteCount=1536
swigluLimit=10.0
backendExpected=dynamic
```

This is the serving-shape synthetic MoE boundary for the custom G4 lane after
Hydralisk remaps global expert IDs into a rank-local 32-expert domain and masks
nonlocal routes with zero route scale.

## Zero fixture result

```json
{
  "schema": "hydralisk.deepseek-v4.b12x.dynamic-clamp-g4-fixture.v1",
  "case": "deepseek_shape_dynamic_masked_zero",
  "ok": true,
  "backendExpected": "dynamic",
  "device": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "capability": [12, 0],
  "outShape": [512, 4096],
  "outDtype": "torch.bfloat16",
  "outAbsSum": 0.0,
  "outMaxAbs": 0.0,
  "finite": true,
  "elapsedMs": 27805.908203125,
  "maxMemoryAllocatedBytes": 501328896
}
```

## Nonzero fixture result

```json
{
  "schema": "hydralisk.deepseek-v4.b12x.dynamic-clamp-g4-fixture.v1",
  "case": "deepseek_shape_dynamic_masked_nonzero",
  "ok": true,
  "backendExpected": "dynamic",
  "device": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "capability": [12, 0],
  "outShape": [512, 4096],
  "outDtype": "torch.bfloat16",
  "outAbsSum": 4375609344.0,
  "outMaxAbs": 10688.0,
  "finite": true,
  "elapsedMs": 0.8352959752082825,
  "maxMemoryAllocatedBytes": 954314752
}
```

The first dynamic call includes JIT compile overhead. The second call reuses
the compiled dynamic path and measures kernel execution only.

## Interpretation

Issue #36 proved the patched static B12x clamp path on G4. Issue #37 proves
the patched dynamic B12x clamp path for the DeepSeek-shaped masked local-shard
fixture. Together, those remove the immediate synthetic B12x clamp blocker for
the custom RTX PRO 6000 lane.

This still is not a DeepSeek-V4-Flash serving claim. The next step is to place
the static and dynamic clamp patches into a derived vLLM/FlashInfer image and
retry a private full-model B12x load smoke on the same 8 x G4 host. That run
must still prove vLLM wiring, model weight load, readiness, and eventually a
public-safe generation receipt.

## Cleanup

- Docker containers left running after probe: `0`
- GPU memory in use after probe: `0 MiB` on all 8 GPUs
- Public ingress created: false

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains model weights: false
- Contains hidden reasoning: false
