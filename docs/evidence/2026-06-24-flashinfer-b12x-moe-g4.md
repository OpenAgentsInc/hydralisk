# FlashInfer B12x SM12x MoE G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/22

Script:
[`scripts/probe-flashinfer-b12x-moe-gce.sh`](../../scripts/probe-flashinfer-b12x-moe-gce.sh)

Generated report:
`.hydralisk/flashinfer-b12x-moe-20260624091339/flashinfer-b12x-moe-probe.md`

## Target

- Project: `openagentsgemini`
- Instance: `hydralisk-deepseek-v4-nvfp4-g4-2g-b-20260624073921`
- Zone: `us-central1-b`
- Machine: `g4-standard-96`
- GPUs: 2 x `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- VM external IP: none
- Image: `hydralisk-deepseek-v4-nvfp4-sm120-oproj-bypass-vllm:20260624085429`
- FlashInfer: `0.6.12`
- Torch: `2.11.0+cu130`
- CUDA runtime: `13.0`

## Why This Probe Exists

Issue #21 proved that the FlashInfer TRTLLM NVFP4 MoE path fails on the G4
RTX PRO 6000 host with `trtllm_batched_gemm_runner.cu:286`.

The local upstream vLLM reference has a newer-looking path:
`FlashInferB12xExperts`, described as an SM12x backend for RTX PRO 6000 /
DGX Spark. That backend calls `flashinfer.fused_moe.b12x_fused_moe`, not the
TRTLLM fused-MoE runner. This probe checks that path directly.

## Availability

The existing probe image already exposes the required B12x functions:

```text
flashinfer.fused_moe.b12x_fused_moe
flashinfer.cute_dsl.utils.convert_sf_to_mma_layout
flashinfer.gemm.Sm120B12xBlockScaledDenseGemmKernel
```

The B12x signature in the live image:

```text
(x, w1_weight, w1_weight_sf, w2_weight, w2_weight_sf,
 token_selected_experts, token_final_scales, num_experts, top_k,
 *, w1_alpha, w2_alpha, fc2_input_scale=None, num_local_experts=None,
 output=None, output_dtype=torch.bfloat16, activation='silu',
 activation_precision='fp4', quant_mode=None, source_format='modelopt')
```

## Synthetic Results

The probe uses synthetic tensors only. It does not load model weights, prompts,
responses, tokens, or Hugging Face artifacts.

Small no-EP kernel smoke:

```json
{"case":"small_no_ep","ok":true,"seqLen":8,"hiddenSize":256,"intermediateSize":256,"numExperts":8,"localNumExperts":8,"topK":2,"outShape":[8,256],"elapsedMs":11164.12109375}
```

DeepSeek-shape expert-parallel case:

```json
{"case":"deepseek_shape_ep","ok":false,"seqLen":1024,"hiddenSize":4096,"intermediateSize":2048,"numExperts":256,"localNumExperts":128,"topK":6,"type":"NotImplementedError","message":"b12x_fused_moe does not yet support Expert Parallelism (num_local_experts=128 != num_experts=256). Use a different MoE backend for EP configurations."}
```

DeepSeek-shape no-EP case:

```json
{"case":"deepseek_shape_no_ep","ok":true,"seqLen":1024,"hiddenSize":4096,"intermediateSize":2048,"numExperts":256,"localNumExperts":256,"topK":6,"outShape":[1024,4096],"elapsedMs":26486.16015625,"maxMemoryAllocatedBytes":3748205568}
```

## Decision

This is the first positive SM120 MoE kernel result for the DeepSeek path on our
G4 host. The B12x FlashInfer kernel exists and can execute DeepSeek-like
synthetic shapes on RTX PRO 6000 when all experts are local.

It does not make the current two-card G4 vLLM path serve-ready. The full
NVFP4 DeepSeek-V4-Flash run needs expert parallelism to split the 256 experts
across the two 96 GB GPUs. B12x currently rejects that mode before launching:

```text
num_local_experts=128 != num_experts=256
```

The next useful step is therefore not another TRTLLM retry. It is one of:

- try a full-model vLLM run that selects B12x with no expert parallel on a
  wider G4 host, where tensor parallelism may reduce per-rank memory enough to
  avoid expert parallelism;
- patch or upstream B12x expert-parallel support;
- build the custom offload/prefetch lane where all 256 experts are not resident
  on every GPU at once.

## Public Safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
