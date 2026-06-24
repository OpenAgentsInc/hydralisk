# DeepSeek-V4-Flash B12x Wide-G4 Probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/23

## Result

The 8 x RTX PRO 6000 G4 host admitted successfully, but the full-model
B12x/no-expert-parallel lane did not reach `/v1/models`.

The root blocker is not Google quota, VM capacity, Docker, DeepGEMM install, or
artifact egress. vLLM rejected `flashinfer_b12x` during model initialization
because DeepSeek-V4 sets `swiglu_limit=10.0` and the B12x backend does not
apply the required SwiGLU clamp.

```text
ValueError:
Model sets swiglu_limit=10.0, but the explicitly requested
moe_backend='flashinfer_b12x' does not apply the SwiGLU clamp.
Use 'flashinfer_trtllm' or 'flashinfer_cutlass' instead.
```

## Host

```text
instance: hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036
project: openagentsgemini
zone: us-central1-b
machine: g4-standard-384
accelerator: 8 x nvidia-rtx-pro-6000
gpu: NVIDIA RTX PRO 6000 Blackwell Server Edition
driver: 580.159.03
public ingress: none
```

## Engine

```text
model: nvidia/DeepSeek-V4-Flash-NVFP4
revision: e3cd60e7de98e9867116860d522499a728de1cf9
base image: vllm/vllm-openai:latest@sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f
derived image: hydralisk-deepseek-v4-b12x-g4-vllm:20260624092146@sha256:c7a2932dc027b5ec46107a1a00ac0a9da5f89e1e81a2aaf57cee80b0b893d231
vLLM: 0.23.0
torch: 2.11.0+cu130
CUDA runtime: 13.0
DeepGEMM: 2.5.0+891d57b
MoE backend: flashinfer_b12x
linear backend: triton
tensor parallel: 8
expert parallel: disabled
kv cache: fp8
block size: 256
max model len: 2048
max batched tokens: 512
```

## What this proves

- Google can admit an 8 x RTX PRO 6000 `g4-standard-384` host in
  `us-central1-b` for this project.
- The host can build the provider-style vLLM/DeepGEMM image with CUDA 13.0.
- The derived image imports vLLM, Torch, DeepGEMM, and sees all eight SM120
  GPUs.
- The B12x backend is not a legal full-model DeepSeek-V4 backend in current
  vLLM because it lacks the model-required SwiGLU clamp.

## Next step

Stop trying B12x as the immediate full-model stock-vLLM route. The next
actionable G4 probe should try the clamp-capable provider-supported MoE
backends, starting with `flashinfer_cutlass` and then `flashinfer_trtllm`, on
the same wide-G4 shape with private-only networking, pinned model revision,
Triton dense linear, E8M0 upcast, and the existing `o_proj` patch path.

If both clamp-capable backends fail on SM120, the remaining G4 path is custom
kernel work: either add B12x SwiGLU clamp support, add B12x expert parallel /
expert offload, or build the SGLang-style expert repack plus prefetch lane.

## Public Safety

```text
contains secrets: false
contains prompts: false
contains responses: false
contains weights: false
contains hidden reasoning: false
```
