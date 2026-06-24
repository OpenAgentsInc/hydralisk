# DeepSeek-V4-Flash NVFP4 SM120 G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/16

## Target

- Project: `openagentsgemini`
- Instance: `hydralisk-deepseek-v4-nvfp4-g4-2g-b-20260624073921`
- Zone: `us-central1-b`
- Machine: `g4-standard-96`
- GPUs: 2 x `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Public NAT: removed before this probe; access used IAP
- Model: `nvidia/DeepSeek-V4-Flash-NVFP4`
- Model revision: `e3cd60e7de98e9867116860d522499a728de1cf9`
- Base image: `hydralisk-deepseek-v4-nvfp4-vllm:20260624074009`
- Derived image:
  `hydralisk-deepseek-v4-nvfp4-sm120-vllm:20260624080336`
- MoE backend: `flashinfer_trtllm`
- Tensor parallel size: `2`
- Max model length: `4096`

## Provider-card cross-check

The provider inventory pasted into this investigation matches the Hydralisk
recipe lane:

- `vllm 0.20.0+`
- DeepGEMM installed through vLLM's `tools/install_deepgemm.sh`
- FP8 KV cache
- block size `256`
- expert parallel enabled
- DeepSeek V4 tokenizer, reasoning, and tool-call parsers
- tensor parallel size equal to visible GPU count

It also frames the G4 work correctly. The published serving shapes are still
8 x H100, 8 x H200, 8 x B200, 4 x GB200 NVL4, or a DGX Station class
single-GPU path. The two-card G4 host is therefore a compatibility probe for
Google's admitted Blackwell RTX PRO 6000 lane, not a proven published-recipe
serving shape.

## Patch under test

Issue #15 showed that stock vLLM rejects the NVFP4 model before readiness:

```text
NotImplementedError:
No NvFp4 MoE backend supports the deployment configuration.
```

The explicit backend matrix found `flashinfer_trtllm` was the only plausible
path for this model config because the other backends reject
`swiglu_limit=10.0` without clamp support. The G4 host then failed the
FlashInfer TRTLLM device gate because vLLM checked
`is_device_capability_family(100)` while the RTX PRO 6000 reports SM120.

This probe built a derived image with an explicit, default-off Hydralisk patch
that allows `is_device_capability_family(120)` for the FlashInfer TRTLLM NVFP4
device gate.

## What passed

The derived image built successfully from the prior clean NVFP4 image:

```text
BASE_IMAGE  hydralisk-deepseek-v4-nvfp4-vllm:20260624074009
DERIVED_IMAGE  hydralisk-deepseek-v4-nvfp4-sm120-vllm:20260624080336
INSTALL_DEEPGEMM  0
DOCKER_BUILD_PULL  0
BUILD_RC  0
```

The patch applied inside the derived image:

```text
patched /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/trtllm_nvfp4_moe.py for NVFP4 SM120 probe
```

The import probe still sees the expected GPU/runtime stack:

```json
{"cudaAvailable": true, "deepGemmHasTransformHelper": true, "deepGemmImport": true, "deviceCount": 2, "devices": [{"capability": [12, 0], "index": 0, "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}, {"capability": [12, 0], "index": 1, "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition"}], "schema": "hydralisk.deepseek-v4.provider-stack-import.v1", "torch": "2.11.0+cu130", "torchCuda": "13.0", "vllm": "0.23.0"}
```

vLLM reached API server startup with the patched image and explicit backend:

```text
version 0.23.0
model   nvidia/DeepSeek-V4-Flash-NVFP4
moe_backend='flashinfer_trtllm'
tokenizer_mode='deepseek_v4'
tool_call_parser='deepseek_v4'
reasoning_parser='deepseek_v4'
tensor_parallel_size=2
```

The prior immediate `flashinfer_trtllm` backend selector error did not recur
before the run was stopped.

## What did not pass

The model did not reach `/v1/models`.

```text
READY  0
```

The run was stopped after proving the patched process launch because the
private-only G4 host cannot currently fetch model artifacts:

```text
dns huggingface.co 3.170.185.25
dns_error cdn-lfs.huggingface.co gaierror [Errno -2] Name or service not known
dns google.com 209.85.200.101
fetch https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4/resolve/e3cd60e7de98e9867116860d522499a728de1cf9/config.json
fetch_error URLError <urlopen error [Errno 101] Network is unreachable>
NET_RC=124
```

This is not yet a kernel-load or memory-load result. It is a more precise
blocker: after removing public NAT and switching new GCE probes to
`--no-address`, the host needs either private outbound egress or pre-staged HF
artifacts before the patched SM120 lane can test weight load.

## Decision

The SM120 device-gate patch is worth keeping as an explicit Hydralisk probe
knob, defaulted off. It advances the NVFP4 G4 investigation past the immediate
stock-vLLM backend selector failure and into vLLM startup.

Do not claim DeepSeek-V4-Flash is runnable on the two-card G4 lane yet.

The next hard step is not another vLLM flag trial. It is to make model
artifacts available to the private host, then rerun the same patched probe:

1. Configure private egress, likely Cloud NAT, for the G4 subnet; or
2. Pre-stage the pinned HF snapshot under `/var/lib/hydralisk/huggingface`.

After artifact access is solved, rerun:

```bash
ISSUE_NUMBER=<next> \
TARGET_INSTANCE=hydralisk-deepseek-v4-nvfp4-g4-2g-b-20260624073921 \
TARGET_ZONE=us-central1-b \
CREATE_IF_MISSING=0 \
ALLOW_NVFP4_SM120=1 \
DOCKER_BUILD_PULL=0 \
INSTALL_DEEPGEMM=0 \
BASE_IMAGE=hydralisk-deepseek-v4-nvfp4-vllm:20260624074009 \
MOE_BACKEND=flashinfer_trtllm \
DERIVED_IMAGE=hydralisk-deepseek-v4-nvfp4-sm120-vllm \
READY_TIMEOUT_SECONDS=3600 \
scripts/probe-deepseek-v4-nvfp4-g4-gce.sh
```

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
