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

## Current live status

Running the wrapper on this Mac still stops at the issue #42 auth preflight:

```text
g4-standard-384 / 8 x nvidia-rtx-pro-6000 / us-central1-b -> blocked_auth
g4-standard-192 / 4 x nvidia-rtx-pro-6000 / us-central1-b -> blocked_auth
```

This confirms the wrapper is ready, but the live GPU smoke remains blocked
before GCE admission by local gcloud reauthentication.

Next operator action:

```bash
gcloud auth login
gcloud auth application-default login
bash scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh
```

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
