# DeepSeek-V4-Flash published-recipe GCE admission

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/14

Script:
[`scripts/probe-deepseek-v4-published-recipe-gce.sh`](../../scripts/probe-deepseek-v4-published-recipe-gce.sh)

Generated report:
`.hydralisk/deepseek-v4-published-recipe-gce-20260624073201/published-recipe-gce-probe.md`

## Result

The published-recipe GPU lane is visible in the Google catalog, but it is not
admitted for this project today.

All sampled H100, H100 Mega, H200, B200, and GB200 candidates had matching
accelerator catalog entries and matching machine types. Every candidate then
blocked on missing regional quota metrics for the required accelerator class.
The quota surface for the probed regions exposed only L4 GPU quota.

No 8-GPU H100/H200/B200 or 4-GPU GB200 create attempt was made because
`ATTEMPT_CREATE=0` by default and every candidate was blocked before create by
missing quota.

## Provider Recipe Anchor

The provider card supplied with this investigation matches the lane we probed:

- engine: vLLM 0.20+;
- required helper: vLLM `tools/install_deepgemm.sh`;
- launch shape: `--kv-cache-dtype fp8`, `--block-size 256`,
  `--enable-expert-parallel`, DeepSeek V4 tokenizer/tool/reasoning parsers;
- tensor-parallel size must match GPU count to avoid replicated dense-layer
  OOM;
- advertised NVIDIA hardware: 8 x H100, 8 x H200, 8 x B200, 4 x GB200 NVL4,
  or a DGX Station class single-GPU path.

That makes the GCE result precise: Google lists the required classes, but this
project is not currently quota-admitted to run the published recipe.

## Probed Candidates

```text
h200-us-central1-b       a3-ultragpu-8g  nvidia-h200-141gb      8  blocked quota_missing
b200-us-central1-b       a4-highgpu-8g   nvidia-b200            8  blocked quota_missing
gb200-us-central1-a      a4x-highgpu-4g  nvidia-gb200           4  blocked quota_missing
gb200-us-central1-b      a4x-highgpu-4g  nvidia-gb200           4  blocked quota_missing
h100-us-central1-a       a3-highgpu-8g   nvidia-h100-80gb       8  blocked quota_missing
h100-us-central1-b       a3-highgpu-8g   nvidia-h100-80gb       8  blocked quota_missing
h100-mega-us-central1-a  a3-megagpu-8g   nvidia-h100-mega-80gb  8  blocked quota_missing
h100-mega-us-central1-b  a3-megagpu-8g   nvidia-h100-mega-80gb  8  blocked quota_missing
h200-us-east4-b          a3-ultragpu-8g  nvidia-h200-141gb      8  blocked quota_missing
b200-us-east4-b          a4-highgpu-8g   nvidia-b200            8  blocked quota_missing
gb200-us-east4-b         a4x-highgpu-4g  nvidia-gb200           4  blocked quota_missing
h200-us-west1-c          a3-ultragpu-8g  nvidia-h200-141gb      8  blocked quota_missing
h100-us-west1-a          a3-highgpu-8g   nvidia-h100-80gb       8  blocked quota_missing
b200-us-east1-b          a4-highgpu-8g   nvidia-b200            8  blocked quota_missing
```

The exact missing quota metrics were the expected class metrics:
`NVIDIA_H100_GPUS`, `PREEMPTIBLE_NVIDIA_H100_GPUS`,
`NVIDIA_H100_MEGA_GPUS`, `PREEMPTIBLE_NVIDIA_H100_MEGA_GPUS`,
`NVIDIA_H200_GPUS`, `PREEMPTIBLE_NVIDIA_H200_GPUS`, `NVIDIA_B200_GPUS`,
`PREEMPTIBLE_NVIDIA_B200_GPUS`, `NVIDIA_GB200_GPUS`, and
`PREEMPTIBLE_NVIDIA_GB200_GPUS`.

## Visible Quota

The probed regions exposed L4 quota only:

```text
us-central1  NVIDIA_L4_GPUS              limit 16  usage 3
us-central1  PREEMPTIBLE_NVIDIA_L4_GPUS  limit 16  usage 0
us-east1     NVIDIA_L4_GPUS              limit 16  usage 0
us-east1     PREEMPTIBLE_NVIDIA_L4_GPUS  limit 16  usage 0
us-east4     NVIDIA_L4_GPUS              limit 16  usage 0
us-east4     PREEMPTIBLE_NVIDIA_L4_GPUS  limit 16  usage 0
us-west1     NVIDIA_L4_GPUS              limit 16  usage 0
us-west1     PREEMPTIBLE_NVIDIA_L4_GPUS  limit 16  usage 0
```

## Interpretation

There are now two honest tracks:

1. Request or otherwise obtain quota for an advertised published-recipe class:
   H200 first, B200 second, GB200 if the account can obtain A4X/NVL capacity,
   or H100 as the mature Hopper fallback.
2. Continue the custom RTX PRO 6000 G4 path. That means owning the stock-vLLM
   failure surface: Triton/block-FP8 behavior, DeepGEMM replacement or patching,
   expert repack, hot expert caching, and CPU/RAM-to-VRAM offload/prefetch.

The admitted Google hardware we can actually use today remains the G4
RTX PRO 6000 lane. It is not a published-recipe lane for
DeepSeek-V4-Flash; it is a custom inference-systems lane.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
