# FlashInfer TRTLLM NVFP4 MoE full-shape G4 repro

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/26

Script:
`scripts/probe-flashinfer-trtllm-nvfp4-moe-gce.sh`

Generated run directory, not committed:
`.hydralisk/flashinfer-trtllm-nvfp4-moe-20260624100105/`

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

## Repro

The probe uses synthetic tensors only. It does not load model weights, prompts,
responses, tokens, or Hugging Face artifacts.

Synthetic shape:

```text
seq_len           512
hidden_size       4096
intermediate_size 2048
num_experts       256
local_num_experts 32
top_k             6
```

Those inputs reproduce the full-model issue #25 failure shape:

```text
numBatches 32
GemmMNK    512 4096 4096
```

## Result

The synthetic call to `flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`
fails with the same runner error observed after the issue #25 `o_proj`
fallback:

```text
RuntimeError:
Error in function 'run' at /workspace/csrc/trtllm_batched_gemm_runner.cu:286:
Error occurred when running GEMM!
numBatches: 32
GemmMNK: 512 4096 4096
Kernel: bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x64x512u2_s4_et128x32_m256x64x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100f
```

The public-safe result record:

```json
{"capability":[12,0],"device":"NVIDIA RTX PRO 6000 Blackwell Server Edition","flashinfer":"0.6.12","hiddenSize":4096,"intermediateSize":2048,"localNumExperts":32,"message":"Error in function 'run' at /workspace/csrc/trtllm_batched_gemm_runner.cu:286: Error occurred when running GEMM! (numBatches:  32 , GemmMNK:  512   4096   4096 , Kernel:  bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x64x512u2_s4_et128x32_m256x64x64_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_tma_tmaSf_rgTma_clmp_swiGlu_dynB_sm100f )","numExperts":256,"ok":false,"publicSafety":{"containsHiddenReasoning":false,"containsPrompts":false,"containsResponses":false,"containsSecrets":false,"containsWeights":false},"schema":"hydralisk.flashinfer.trtllm-nvfp4-moe.synthetic.v1","seqLen":512,"topK":6,"torch":"2.11.0+cu130","torchCuda":"13.0","type":"RuntimeError"}
```

## Decision

The full-model MoE blocker is now isolated from DeepSeek model loading,
Hugging Face transfer, vLLM scheduling, prompts, and the `o_proj` fallback.

This confirms the current `flashinfer_trtllm` NVFP4 MoE kernel path is not a
viable serving path on admitted RTX PRO 6000 G4 by wrapper or vLLM flag changes
alone. The next useful move is to stop rerunning full-model vLLM probes and
choose one of:

- patch or avoid the TRTLLM SM100-family NVFP4 MoE kernel path for SM120;
- add/port a B12x clamped SwiGLU + expert-parallel/offload path;
- build the SGLang/offload expert-prefetch lane;
- obtain known-good H100/H200/B200/GB200/DGX-class hardware.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
