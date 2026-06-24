# FlashInfer TRTLLM NVFP4 MoE G4 repro

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/21

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

## Repro

The probe uses synthetic tensors only. It does not load model weights, prompts,
responses, tokens, or Hugging Face artifacts.

Synthetic shape:

```text
seq_len           1024
hidden_size       4096
intermediate_size 2048
num_experts       256
local_num_experts 128
top_k             6
```

Those inputs reproduce the full-model zero-`o_proj` bypass failure shape:

```text
numBatches 128
GemmMNK    1024 4096 4096
```

## Result

The synthetic call to `flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`
fails with the same runner error observed in the full vLLM startup path:

```text
RuntimeError:
Error in function 'run' at /workspace/csrc/trtllm_batched_gemm_runner.cu:286:
Error occurred when running GEMM!
numBatches: 128
GemmMNK: 1024 4096 4096
Kernel: bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x32x512u2_s4_et128x32_m128x32x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_sm100f
```

The public-safe result record:

```json
{"capability":[12,0],"device":"NVIDIA RTX PRO 6000 Blackwell Server Edition","flashinfer":"0.6.12","hiddenSize":4096,"intermediateSize":2048,"localNumExperts":128,"message":"Error in function 'run' at /workspace/csrc/trtllm_batched_gemm_runner.cu:286: Error occurred when running GEMM! (numBatches:  128 , GemmMNK:  1024   4096   4096 , Kernel:  bmm_E2m1_E2m1E2m1_Fp32_Ab16_Bb16_Cb16_t128x32x512u2_s4_et128x32_m128x32x64_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN_ldgsts_ldgstsSf_rgTma_clmp_swiGlu_dynB_sm100f )","numExperts":256,"ok":false,"schema":"hydralisk.flashinfer.trtllm-nvfp4-moe.synthetic.v1","seqLen":1024,"topK":6,"torch":"2.11.0+cu130","torchCuda":"13.0","type":"RuntimeError"}
```

## Decision

The FlashInfer TRTLLM NVFP4 MoE blocker is now isolated from DeepSeek model
loading, Hugging Face transfer, vLLM scheduling, and `o_proj`.

This confirms that the admitted RTX PRO 6000 G4 host cannot run the current
stock-vLLM + FlashInfer TRTLLM NVFP4 DeepSeek V4 path by simple wrapper changes.
The next useful move is no longer another full-model G4 probe. It is one of:

- upstream/kernel work for FlashInfer TRTLLM NVFP4 on SM120;
- a custom SGLang/offload/kernel path;
- a known-good provider stack on H100/H200/B200/GB200 or DGX-class hardware.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
