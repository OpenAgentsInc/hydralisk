# DeepSeek-V4-Flash GCE load-smoke evidence

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/6

Runner:
[`scripts/smoke-deepseek-v4-gce.sh`](../../scripts/smoke-deepseek-v4-gce.sh)

## Result

Google admitted the first useful DeepSeek-V4-Flash probe lane:

- Instance: `hydralisk-deepseek-v4-g4-2g-b-20260624053235`
- Project: `openagentsgemini`
- Zone: `us-central1-b`
- Machine: `g4-standard-96`
- GPUs: 2 x `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Provisioning: `SPOT`
- Max run duration: `7200s`
- Public ingress: none

The host had enough disk, RAM, GPU memory, and CUDA toolchain surface to attempt
the model load. The blocker moved from Google capacity to the stock vLLM
Blackwell FP8 scaled-mm path.

## Hardware receipt

The admitted host reported:

```text
OS                 Ubuntu 22.04.5 LTS
Kernel             6.8.0-1060-gcp
Driver             580.159.03
Driver CUDA        13.0
System CUDA_HOME   /usr/local/cuda-12.9
Disk               582G total, 176G used, 407G free
GPU0               NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB
GPU1               NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB
Topology           GPU0 <-> GPU1 PIX
```

The system CUDA 12.9 `nvcc` path compiled a tiny `sm_120a` cubin during the
manual recovery path, so the immediate runtime failure was not a missing
compiler after `CUDA_HOME` was pinned to `/usr/local/cuda-12.9`.

## Runtime receipt

The Deep Learning VM image did not provide a ready Docker path, and the apt
Docker setup was flaky enough that the runner fell back to a Python/uv vLLM
path:

```text
Python     3.12.13
vLLM       0.23.0
torch      2.11.0
g++        12.3.0
cc1plus    /usr/lib/gcc/x86_64-linux-gnu/12/cc1plus
CUDA_HOME  /usr/local/cuda-12.9
```

The runner installed `g++-12` when `cc1plus` was missing and then reused the
system CUDA toolkit so vLLM did not try to compile against the pip CUDA 13.2
headers. The model cache reached roughly 149 GB under
`/var/lib/hydralisk/huggingface`.

## Model-load progress

The load smoke got past the earlier non-capacity blockers:

- resolved `deepseek-ai/DeepSeek-V4-Flash`;
- selected `DeepseekV4ForCausalLM`;
- ran tensor parallel size 2;
- used FP8 KV cache;
- selected `deepseek_v4_fp8` / `Mxfp4` model paths;
- compiled TileLang kernels after the CUDA 12.9 toolchain pin.

It did not reach `/v1/models` and did not produce a synthetic completion.

Public completion receipt:

```json
{"ready":false,"status":"server_not_ready_or_exited"}
```

Smoke summary:

```text
READY 0
```

## Blocker

The vLLM worker failed during engine startup while measuring available GPU
memory:

```text
RuntimeError: dispatch_scaled_mm,
/workspace/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:17
```

The stack went through `vllm/_custom_ops.py` `cutlass_scaled_mm`, then the
engine aborted during `determine_available_memory`.

This is not currently a Google quota/capacity blocker. It is a vLLM / torch /
CUTLASS / Blackwell FP8 kernel-runtime blocker for this checkpoint on the
admitted `g4-standard-96` host.

## Host state

The instance was intentionally left discoverable rather than silently deleted:

```text
hydralisk-deepseek-v4-g4-2g-b-20260624053235 us-central1-b RUNNING
```

It was created with `--instance-termination-action DELETE` and
`--max-run-duration 7200s`.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
