# DeepSeek-V4-Flash provider-stack G4 probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/13

Script:
[`scripts/probe-deepseek-v4-provider-stack-gce.sh`](../../scripts/probe-deepseek-v4-provider-stack-gce.sh)

## Result

The provider-guided vLLM/DeepGEMM stack builds and imports on the live two-card
G4 host, but it still does not reach `/v1/models`.

This closes the "maybe our patched Python venv is the problem" branch. A clean
container stack derived from the vLLM image, with DeepGEMM installed through the
vLLM helper, still fails on RTX PRO 6000 Blackwell in stock vLLM's CUTLASS
block-scaled FP8 path:

```text
RuntimeError: dispatch_scaled_mm,
/workspace/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:17
```

## Host

```text
Instance      hydralisk-deepseek-v4-g4-2g-b-20260624053235
Zone          us-central1-b
Machine       g4-standard-96
GPU           2 x NVIDIA RTX PRO 6000 Blackwell Server Edition
Driver        580.159.03
CUDA          13.0 runtime in container
Topology      GPU0/GPU1 PIX
Disk          582G total, 374G free before final run
```

## Provider Stack

Provider-note shape tested:

```text
Base image      vllm/vllm-openai:latest
Base digest     sha256:6d8429e38e3747723ca07ee1b17972e09bb9c51c4032b266f24fb1cc3b22ed8f
Derived image   hydralisk-deepseek-v4-provider-vllm:20260624072257
Derived digest  sha256:0a8292ee291ba53394e953682a00eddc3e243f3e24fdc6e58ac164447eea0b3a
DeepGEMM        built via vLLM tools/install_deepgemm.sh
DeepGEMM ref    891d57b4db1071624b5c8fa0d1e51cb317fa709f
CUDA dev fix    cuda-libraries-dev-13-0 in derived image
uv fix          UV_SYSTEM_PYTHON=1 for helper install
```

The base image did not contain the full build prerequisites for the helper.
The script now installs the CUDA development libraries and uses system-Python
installation for the helper:

```text
apt-get install -y --no-install-recommends ca-certificates git cuda-libraries-dev-13-0
UV_SYSTEM_PYTHON=1 bash /tmp/install_deepgemm.sh
```

## Import Probe

The derived image imports vLLM, Torch, CUDA, and DeepGEMM successfully:

```json
{
  "cudaAvailable": true,
  "deepGemmHasTransformHelper": true,
  "deepGemmImport": true,
  "deviceCount": 2,
  "devices": [
    {
      "capability": [12, 0],
      "index": 0,
      "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    },
    {
      "capability": [12, 0],
      "index": 1,
      "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    }
  ],
  "torch": "2.11.0+cu130",
  "torchCuda": "13.0",
  "vllm": "0.23.0"
}
```

## Serve Attempt

The provider-style low-context serve command used:

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

The server resolved the model and initialized the distributed two-GPU workers,
then failed during available-memory determination before readiness:

```text
BACKEND                 docker_provider_stack
TENSOR_PARALLEL_SIZE    2
LOCAL_SITE_PACKAGES_PATCHES false
READY                   0
```

Relevant public-redacted stack:

```text
vllm/model_executor/layers/quantization/fp8.py:476 apply
vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:132 apply_weights
vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:324 apply_block_scaled_mm
vllm/_custom_ops.py:908 cutlass_scaled_mm
RuntimeError: dispatch_scaled_mm,
/workspace/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:17
```

## Interpretation

The clean provider stack did exactly what it needed to do:

- confirmed the provider-style image can be made reproducible;
- confirmed DeepGEMM builds and imports in that image on the G4 host;
- removed Hydralisk's patched Python venv as the explanation for failure;
- reproduced the original CUTLASS scaled-mm blocker on RTX PRO 6000.

The stock vLLM path on this two-card G4 is not a serving path today. The next
honest work is either:

- run the published-recipe lane on an 8-GPU H100/H200/B200-class node; or
- start the custom G4 path: force/own the Triton/FP8 path, then implement the
  expert offload/prefetch/kernel work needed for RTX PRO 6000.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
