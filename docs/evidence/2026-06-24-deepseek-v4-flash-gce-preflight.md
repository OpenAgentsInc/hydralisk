# DeepSeek-V4-Flash GCE admission preflight evidence

Date: 2026-06-24T05:20:37.005616Z

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/5

Profile: `profiles/deepseek-v4-flash-gce-preflight.json`

## Result

Recommendation: `try_g4_standard_96_all_gpu_or_g4_standard_48_offload_prefetch_smoke`

Hydralisk should proceed, but only as an admission/load experiment. Two RTX PRO
6000 GPUs have 195774 MiB aggregate memory and clear the low-context all-GPU
preflight by 81205 MiB. One RTX PRO 6000 misses the conservative all-GPU
reserve by 6893 MiB, but is the cheapest candidate for the custom
hot-expert-cache/offload bridge described in the DeepSeek-V4-Flash thread. Two
H100s also clear the all-GPU memory preflight by 51815 MiB if Google has
capacity. The live single-H100 host is rejected because it is reserved for
GPT-OSS 120B.

Do not disturb the live single-H100 GPT-OSS 120B host for this experiment.
The first useful Hydralisk step is a G4 or multi-H100 admission/load preflight,
not a product route and not a public model selector.

## Parsed local artifact

- Path:
  `~/Downloads/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`
- Size: 86720111200 bytes (82703 MiB)
- GGUF version: 3
- Tensor count: 1328
- Metadata entries: 58

Selected metadata:

```json
{
  "deepseek4.attention.compress_ratios": [
    0,
    0,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    128,
    4,
    0
  ],
  "deepseek4.attention.head_count": 64,
  "deepseek4.attention.head_count_kv": 1,
  "deepseek4.attention.key_length": 512,
  "deepseek4.attention.value_length": 512,
  "deepseek4.block_count": 43,
  "deepseek4.context_length": 1048576,
  "deepseek4.embedding_length": 4096,
  "deepseek4.expert_count": 256,
  "deepseek4.expert_shared_count": 1,
  "deepseek4.expert_used_count": 6,
  "general.architecture": "deepseek4",
  "general.license": "mit",
  "general.name": "DeepSeek V4 Flash",
  "general.size_label": "256x8.4B",
  "tokenizer.ggml.model": "gpt2"
}
```

## Lane classification

| Lane | Decision | Class | GPU memory | Margin | Notes |
| --- | --- | --- | ---: | ---: | --- |
| live-a3-highgpu-1g-h100-gptoss120b | reject_for_deepseek_until_drained | blocked_reserved_live_host | 81559 MiB | -21588 MiB | live host is reserved for GPT-OSS 120B; This is the live Hydralisk GPT-OSS 120B probe host.; Do not use it for DeepSeek smoke work unless it is intentionally drained. |
| g4-standard-48-rtxpro6000-1g | proceed_only_for_custom_offload_prefetch_validation | candidate_offload_prefetch_smoke | 97887 MiB | -6893 MiB | host RAM can hold the artifact while GPU memory is too tight; requires hot expert cache or CPU-offload bridge; Blackwell GPU is directionally aligned with FP4 work but kernel support must be proven; Cheapest plausible Google lane for an Exobyt-style hot expert cache.; Single-card all-GPU is too tight once runtime and KV cache reserve are included. |
| g4-standard-96-rtxpro6000-2g | proceed_if_capacity_admits | candidate_all_gpu_low_context_smoke | 195774 MiB | 81205 MiB | aggregate GPU memory covers weights plus smoke KV reserve; Blackwell GPU is directionally aligned with FP4 work but kernel support must be proven; The most useful next G4 probe if capacity is available.; Still a Blackwell compatibility risk for vLLM/SGLang FP4 kernels. |
| g4-standard-192-rtxpro6000-4g | proceed_if_capacity_admits | candidate_all_gpu_low_context_smoke | 391548 MiB | 257402 MiB | aggregate GPU memory covers weights plus smoke KV reserve; Blackwell GPU is directionally aligned with FP4 work but kernel support must be proven; More headroom than 2x G4; higher burn rate. |
| g4-standard-384-rtxpro6000-8g | proceed_if_capacity_admits | candidate_all_gpu_low_context_smoke | 783096 MiB | 609795 MiB | aggregate GPU memory covers weights plus smoke KV reserve; Blackwell GPU is directionally aligned with FP4 work but kernel support must be proven; G4 admitted previously, but GLM-5.2 hit SGLang/FlashInfer DSA support blockers.; Use only if smaller G4 lanes fail for capacity or topology reasons. |
| a3-highgpu-2g-h100 | proceed_if_capacity_admits | candidate_all_gpu_low_context_smoke | 163118 MiB | 51815 MiB | aggregate GPU memory covers weights plus smoke KV reserve; Hopper path is more mature, but capacity has been volatile. |
| a3-highgpu-4g-h100 | proceed_if_capacity_admits | candidate_all_gpu_low_context_smoke | 326236 MiB | 198621 MiB | aggregate GPU memory covers weights plus smoke KV reserve; Better headroom than 2x H100, but do not assume immediate capacity. |
| a2-highgpu-4g-a100 | do_not_start_here | deprioritized_memory_only_smoke | 162144 MiB | 50938 MiB | aggregate GPU memory is enough only on paper; A100 lacks the Blackwell FP4/NVFP4 path expected by the official recipe; Enough aggregate memory for a low-context smoke.; A100 is not a good target for FP4/NVFP4 Blackwell paths. |
| g2-standard-96-l4-8g | do_not_start_here | deprioritized_memory_only_smoke | 184272 MiB | 70853 MiB | aggregate GPU memory is enough only on paper; multi-L4 has poor bandwidth/interconnect for this MoE path; Aggregate memory is plausible; bandwidth and interconnect make it a poor first lane.; Do not use active Khala/GPT-OSS L4 hosts for this experiment. |

## Live gcloud GPU inventory

- `gswarm508-clean2-20260325044551-coord`: `g2-standard-8`, `RUNNING`,
  `us-central1-a`, `[{'type': 'nvidia-l4', 'count': 1}]`
- `hydralisk-gptoss20b-l4-20260624000550`: `g2-standard-8`, `RUNNING`,
  `us-central1-a`, `[{'type': 'nvidia-l4', 'count': 1}]`
- `gswarm508-clean2-20260325044551-contrib`: `g2-standard-8`, `RUNNING`,
  `us-central1-b`, `[{'type': 'nvidia-l4', 'count': 1}]`
- `hydralisk-gptoss120b-h100-probe-20260623210841`: `a3-highgpu-1g`,
  `RUNNING`, `us-central1-b`,
  `[{'type': 'nvidia-h100-80gb', 'count': 1}]`

## Public safety

- Contains secrets: false
- Contains prompts: false
- Contains responses: false
- Contains weights: false
- Contains hidden reasoning: false
