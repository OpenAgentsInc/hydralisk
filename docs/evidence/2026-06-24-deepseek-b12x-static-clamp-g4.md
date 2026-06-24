# DeepSeek B12x static clamp G4 fixture

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/36

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

## Provider inventory signal

The user-provided provider inventory card is useful as the public recipe
baseline. It says DeepSeek-V4-Flash expects vLLM `0.20.0+`, DeepGEMM installed
through vLLM's helper, FP8 KV cache, block size `256`, expert parallel,
DeepSeek V4 tokenizer/tool/reasoning parser flags, and tensor parallelism
matching the visible GPU count. Its listed happy-path hardware is H100, H200,
B200, GB200/GB300, DGX Station, or MI300-class AMD hardware.

That does not change the G4 conclusion: our admitted Google lane is still a
custom RTX PRO 6000 compatibility path, not the published-recipe path.

## Patch applied in disposable container

The fixture used the Hydralisk B12x clamp overlay in the container
`site-packages` tree, then converted the static kernel marker into real
CuTe/CUTLASS clamp operations:

```text
overlayOk=true
staticMarkersPatched=1
```

The patched static activation operation matches the DeepSeek/vLLM clamp shape:

```text
g = min(g, 10.0)
u = min(max(u, -10.0), 10.0)
```

No model weights were loaded. The container was disposable and removed after
the fixture.

## Static zero fixture

The first tiny static run used zero tensors as a conservative compile/runtime
probe:

```text
schema=hydralisk.deepseek-v4.b12x.static-clamp-fixture.v1
case=tiny_static_swiglu_limit
backend=static
device=NVIDIA RTX PRO 6000 Blackwell Server Edition
capability=[12,0]
seqLen=8
hiddenSize=256
intermediateSize=256
numExperts=8
localNumExperts=8
topK=2
swigluLimit=10.0
ok=true
outShape=[8,256]
outDtype=torch.bfloat16
outSum=0.0
referenceOutSum=0.0
matchesZeroReference=true
elapsedMs=11183.0380859375
maxMemoryAllocatedBytes=941568
```

## Static nonzero fixture

The follow-up tiny fixture used nonzero hidden states and nonzero synthetic
FP4-packed weights. It proved the patched static B12x path accepts
`swiglu_limit=10.0`, compiles, executes, and returns finite nonzero output on
RTX PRO 6000:

```json
{
  "schema": "hydralisk.deepseek-v4.b12x.static-clamp-nonzero-fixture.v1",
  "case": "tiny_static_swiglu_limit_nonzero",
  "ok": true,
  "device": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "capability": [12, 0],
  "flashinfer": "0.6.12",
  "torch": "2.11.0+cu130",
  "torchCuda": "13.0",
  "seqLen": 8,
  "hiddenSize": 256,
  "intermediateSize": 256,
  "numExperts": 8,
  "localNumExperts": 8,
  "topK": 2,
  "swigluLimit": 10.0,
  "outShape": [8, 256],
  "outDtype": "torch.bfloat16",
  "outAbsSum": 68917232.0,
  "outMaxAbs": 143360.0,
  "finite": true,
  "elapsedMs": 11211.8779296875,
  "maxMemoryAllocatedBytes": 955392
}
```

This is stronger than the zero compile smoke, but it is not full correctness
equivalence for DeepSeek weights. The packed FP4 synthetic weights are not
decoded against Hydralisk's pure-Python reference fixture. The next issue
should either add that exact FP4 decode/reference comparison or move directly
to the DeepSeek-shape backend that the model actually selects.

## Backend selector boundary

The same runtime selects the static backend for small token counts and the
dynamic backend for the DeepSeek-shaped 512-token case:

```text
tokens=1,2,4,8,16 topK=6 -> static
tokens=512 topK=6 -> dynamic
```

So issue #36 proves the first hard B12x clamp compile/runtime boundary on G4,
but it does not prove the DeepSeek-shape dynamic path. The next useful issue is
to apply the same real clamp operation to `moe_dynamic_kernel.py` and run a
masked local-shard DeepSeek-shape fixture on the same private G4 host.

## Cleanup

- Docker containers left running after probe: `0`
- GPU memory in use after probe: `0 MiB` on all 8 GPUs
- Public ingress created: false

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
