# DeepSeek-V4-Flash NVFP4 G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/15

Script:
[`scripts/probe-deepseek-v4-nvfp4-g4-gce.sh`](../../scripts/probe-deepseek-v4-nvfp4-g4-gce.sh)

Generated report:
`.hydralisk/deepseek-v4-nvfp4-g4-20260624073921/nvfp4-g4-probe.md`

## Result

The NVFP4 Blackwell variant is real and the clean vLLM/DeepGEMM image can be
built on our admitted G4 host, but it still does not reach `/v1/models` on
2 x RTX PRO 6000.

This probe is materially different from the native-FP8 probe:

- the model was `nvidia/DeepSeek-V4-Flash-NVFP4`;
- the model revision was pinned to
  `e3cd60e7de98e9867116860d522499a728de1cf9`;
- DeepGEMM built successfully with the SM100 FP4/MoE source present;
- vLLM imported successfully on both RTX PRO 6000 devices;
- the failure moved away from `dispatch_scaled_mm`.

The new blocker is vLLM's NVFP4 MoE backend selection:

```text
NotImplementedError:
No NvFp4 MoE backend supports the deployment configuration.
```

The explicit backend matrix clarified the reason:

```text
auto                  blocked: no NvFp4 MoE backend supports the deployment configuration
flashinfer_trtllm     blocked: kernel does not support current device cuda
flashinfer_cutlass    blocked: model swiglu_limit=10.0, backend does not apply SwiGLU clamp
cutlass               blocked: model swiglu_limit=10.0, backend does not apply SwiGLU clamp
flashinfer_cutedsl    blocked: model swiglu_limit=10.0, backend does not apply SwiGLU clamp
flashinfer_b12x       blocked: model swiglu_limit=10.0, backend does not apply SwiGLU clamp
marlin                blocked: model swiglu_limit=10.0, backend does not apply SwiGLU clamp
emulation             blocked: model swiglu_limit=10.0, backend does not apply SwiGLU clamp
```

Interpretation: for this model configuration, the only plausible clamp-capable
NVFP4 backend is FlashInfer TRTLLM, and the current vLLM/FlashInfer stack does
not accept the G4 RTX PRO 6000 device as supported. The two-card G4 lane is
therefore not a stock-vLLM NVFP4 serve path today.

## Host

```text
Instance      hydralisk-deepseek-v4-nvfp4-g4-2g-b-20260624073921
Zone          us-central1-b
Machine       g4-standard-96
GPU           2 x NVIDIA RTX PRO 6000 Blackwell Server Edition
Provisioning  SPOT
Max runtime   21600s
Public NAT    removed after probe; future scripts now pass --no-address
```

The host was created from the GCE default network path, which attached a public
NAT by default. After detection, the access config was removed with
`gcloud compute instances delete-access-config`, and all Hydralisk probe create
scripts were hardened with `--no-address`.

## Stack

```text
Base image      vllm/vllm-openai:latest
Base digest     sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f
Derived image   hydralisk-deepseek-v4-nvfp4-vllm:20260624074009
Derived digest  sha256:35149725614cca7842ba16ea02f5b3765e0b875f85e8f192f1a2ba2a4bcfc9f5
vLLM            0.23.0
Torch           2.11.0+cu130
CUDA runtime    13.0
DeepGEMM        2.5.0+891d57b via vLLM tools/install_deepgemm.sh
Tensor parallel 2
Expert parallel enabled
Max model len   4096
```

Serve flags:

```text
--kv-cache-dtype fp8
--block-size 256
--tensor-parallel-size 2
--enable-expert-parallel
--gpu-memory-utilization 0.90
--max-model-len 4096
--max-num-seqs 1
--max-num-batched-tokens 1024
--tokenizer-mode deepseek_v4
--tool-call-parser deepseek_v4
--enable-auto-tool-choice
--reasoning-parser deepseek_v4
```

## What This Rules Out

This rules out the easy "use the NVFP4 repo on our G4 and stock vLLM will work"
path.

It does not rule out:

- a vLLM/FlashInfer patch that teaches the clamp-capable NVFP4 backend to
  accept RTX PRO 6000 Blackwell Server Edition;
- a different NVIDIA container or upstream revision where that backend already
  accepts this device;
- a custom G4 path that bypasses stock vLLM NVFP4 MoE backend selection;
- the published-recipe H100/H200/B200/GB200 path once quota exists.

## Next Step

The next executable issue should patch or isolate the FlashInfer TRTLLM NVFP4
device gate for RTX PRO 6000. The goal is not a blind monkeypatch to "pretend"
support; it should first identify the exact `_supports_current_device()` guard,
then run the smallest possible NVFP4 MoE kernel smoke on the G4 before retrying
the full model.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
