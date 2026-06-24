# GLM-5.2 504B REAP b12x/vLLM launch profile

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/85

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Staging evidence:
[`docs/evidence/2026-06-24-glm-52-reap-504b-staging.md`](2026-06-24-glm-52-reap-504b-staging.md)

Launch command JSON:
[`docs/evidence/2026-06-24-glm-52-reap-504b-b12x-launch-command.json`](2026-06-24-glm-52-reap-504b-b12x-launch-command.json)

Launcher:
[`scripts/launch-glm-52-reap-504b-b12x-gce.sh`](../../scripts/launch-glm-52-reap-504b-b12x-gce.sh)

Public-safety boundary: this packet contains runtime image, launch flags, and
host readiness metadata only. It contains no bearer tokens, model-provider
credentials, raw prompts, responses, private source, hidden reasoning traces,
weights, checkpoints, compiled engines, profiler dumps, or raw model logs.

## Goal

Adapt the upstream SM120 b12x/vLLM recipe into a Hydralisk launch profile for
the staged `0xSero/GLM-5.2-504B` REAP/NVFP4 checkpoint.

This issue prepares the runtime and launch command. It does not run the model
load or make a serving claim; that is issue #86.

## Runtime image

- Image tag:
  `voipmonitor/vllm:black-benediction-b12xpr11-vllmbb6c5b7-b12xd90d89c-fi3395b41aa8d-dg324aced12c-cu132-20260608`
- Image digest:
  `sha256:ce23a9b075bd7138ce3b12ee29609b98606e5050e2def4a29bbb917ad96e5997`
- Pinned image ref:
  `voipmonitor/vllm@sha256:ce23a9b075bd7138ce3b12ee29609b98606e5050e2def4a29bbb917ad96e5997`
- Platform: Linux amd64
- Docker on host: `Docker version 29.1.3`

The G4 host did not have Docker installed at admission time. `ACTION=prepare`
installed `docker.io`, configured the NVIDIA runtime through `nvidia-ctk`, and
pulled the pinned image by digest.

## Container version probe

The pinned container started and reported:

- vLLM:
  `0.11.2.dev279+black.benediction.b12xpr11.cu132.20260608`
- Torch: `2.12.0+cu132`
- FlashInfer Python: `0.6.12+cu132`
- `vllm serve` CLI help returned successfully.

`modelopt` was not exposed as a Python package in the simple metadata probe,
but the b12x recipe still uses vLLM's `--quantization modelopt_fp4` runtime
flag. The first load smoke in issue #86 must treat a missing ModelOpt runtime
surface as a real blocker if vLLM cannot load the checkpoint.

## First-load envelope

The default launch profile intentionally uses only four GPUs on the 8x fallback
host so it preserves the primary 4x fit question:

- `CUDA_VISIBLE_DEVICES=0,1,2,3`
- Tensor parallel size: 4
- Decode-context parallel size: 4
- Max model length: 32,768 tokens
- Max sequences: 1
- Max batched tokens: 4096
- GPU memory utilization: 0.95
- Host: `127.0.0.1`
- Port: `8000`
- MTP: disabled

The 8x host may later be used with `GPU_DEVICES=0,1,2,3,4,5,6,7` and
`TP_SIZE=8` only as an explicit fallback. That must be documented separately
so it does not overwrite the 4x accessibility claim.

## Required launch flags

The launcher preserves the upstream GLM REAP/SM120 requirements:

- `--quantization modelopt_fp4`
- `--kv-cache-dtype fp8`
- `--attention-backend B12X_MLA_SPARSE`
- `--moe-backend b12x`
- `--tool-call-parser glm47`
- `--reasoning-parser glm45`
- `--hf-overrides '{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'`

And the important environment toggles:

- `CUTE_DSL_ARCH=sm_120a`
- `NCCL_P2P_DISABLE=1`
- `NCCL_P2P_LEVEL=SYS`
- `NCCL_IB_DISABLE=1`
- `SAFETENSORS_FAST_GPU=1`
- `VLLM_USE_B12X_SPARSE_INDEXER=1`
- `VLLM_USE_B12X_MOE=1`
- `VLLM_USE_V2_MODEL_RUNNER=1`
- `VLLM_USE_FLASHINFER_SAMPLER=1`
- `VLLM_USE_B12X_FP8_GEMM=1`
- `VLLM_ENABLE_PCIE_ALLREDUCE=0`
- `VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel`
- `B12X_DENSE_SPLITK_TURBO=1`
- `B12X_W4A16_TC_DECODE=1`
- `B12X_MOE_FORCE_A16=1`

## Prepared health status

After preparation, before model load:

- Container status: empty / no server container running
- `/v1/models`: `not_ready`
- GPU memory: 0 MiB used on all eight visible RTX PRO 6000 GPUs

That is the expected issue #85 state. Issue #86 owns the first live load and
completion smoke.

## Commands

Prepare runtime:

```bash
ACTION=prepare RUN_ID=20260624221500 \
  scripts/launch-glm-52-reap-504b-b12x-gce.sh
```

Check status:

```bash
ACTION=status RUN_ID=20260624221500 \
  scripts/launch-glm-52-reap-504b-b12x-gce.sh
```

Start first load smoke for issue #86:

```bash
ACTION=start RUN_ID=<run-id> \
  scripts/launch-glm-52-reap-504b-b12x-gce.sh
```

Stop the server:

```bash
ACTION=stop RUN_ID=<run-id> \
  scripts/launch-glm-52-reap-504b-b12x-gce.sh
```

## Next step

Issue #86 should run `ACTION=start`, wait for `/v1/models`, and submit one
public-safe synthetic completion using the upstream sampler guardrails.
