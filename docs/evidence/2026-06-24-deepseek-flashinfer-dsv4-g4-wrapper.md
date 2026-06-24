# DeepSeek-V4-Flash FlashInfer DSV4 G4 rerun wrapper

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/43

Related live GPU issue: https://github.com/OpenAgentsInc/hydralisk/issues/41

## Summary

Issue #43 added `scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh`, a
dedicated wrapper for the next live DeepSeek-V4-Flash G4 retry.

The wrapper delegates to `scripts/probe-deepseek-v4-b12x-g4-gce.sh` while
defaulting the known issue #41 launch contract:

```text
ISSUE_NUMBER=41
VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4
VLLM_ENFORCE_EAGER=1
HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=bf16_einsum
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=hopper
HYDRALISK_B12X_CLAMP_PATCH=1
HYDRALISK_B12X_CLAMP_LIMIT=10.0
MAX_MODEL_LEN=2048
MAX_NUM_BATCHED_TOKENS=512
GPU_MEMORY_UTILIZATION=0.95
RUN_MODEL_SMOKE=1
```

Advanced debugging can still override those values through environment
variables. The reusable B12x G4 harness remains the implementation owner for
GCE admission, auth preflight, image build, vLLM launch, and public-safe
evidence rendering.

## Dry-run evidence

The wrapper dry-run renders the intended issue #41 launch shape:

```text
Issue: https://github.com/OpenAgentsInc/hydralisk/issues/41
Model: nvidia/DeepSeek-V4-Flash-NVFP4
Model revision: e3cd60e7de98e9867116860d522499a728de1cf9
MoE backend: flashinfer_b12x
vLLM enforce eager: 1
vLLM attention backend: FLASHINFER_MLA_SPARSE_DSV4
DeepSeek o_proj fallback: bf16_einsum
DeepSeek o_proj recipe: hopper
B12x clamp patch: 1
B12x clamp limit: 10.0
Max model length: 2048
Max batched tokens: 512
GPU memory utilization: 0.95
```

Planned G4 candidates:

```tsv
order	zone	machine	accelerator	gpu_count	role
1	us-central1-b	g4-standard-384	nvidia-rtx-pro-6000	8	b12x_no_ep_first
2	us-central1-b	g4-standard-192	nvidia-rtx-pro-6000	4	b12x_no_ep_fallback
```

## Live status

The first wrapper attempts on this Mac stopped at the issue #42 auth preflight:

```text
g4-standard-384 / 8 x nvidia-rtx-pro-6000 / us-central1-b -> blocked_auth
g4-standard-192 / 4 x nvidia-rtx-pro-6000 / us-central1-b -> blocked_auth
```

That was an auth result, not a GPU capacity result.

After gcloud auth was refreshed, the issue #41 wrapper was run against the
existing 8 x G4 target:

```bash
ISSUE_NUMBER=41 \
TARGET_INSTANCE=hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036 \
TARGET_ZONE=us-central1-b \
CREATE_IF_MISSING=0 \
bash scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh
```

The run reached the G4 host, built the derived vLLM image, launched the
container, and failed before `/v1/models` because the wrapper inherited the
B12x harness default `HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=blackwell` while the
selected `HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=bf16_einsum` path requires
non-TMA activation scales:

```text
RuntimeError: HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=bf16_einsum requires non-TMA activation scales; set HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=hopper for this probe
```

The wrapper now defaults `HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=hopper` so the
corrected issue #41 rerun could validate the FlashInfer DSV4 attention backend
rather than a known local launch-shape mismatch.

The corrected rerun reached `/v1/models` on the 8 x G4 host, then the tiny
completion smoke failed in the selected DSV4 attention path:

```text
flashinfer.mla._core.trtllm_batch_decode_sparse_mla_dsv4
tvm.error.InternalError: Error in function 'TllmGenFmhaRunner' at /workspace/include/flashinfer/trtllm/fmha/fmhaRunner.cuh:37: Unsupported architecture
```

The public-safe live receipt is tracked separately:
[`2026-06-24-deepseek-flashinfer-dsv4-g4-live-smoke.md`](2026-06-24-deepseek-flashinfer-dsv4-g4-live-smoke.md).

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
