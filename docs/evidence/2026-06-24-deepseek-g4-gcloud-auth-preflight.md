# DeepSeek-V4-Flash G4 gcloud auth preflight

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/42

## Summary

Issue #42 added a fast gcloud credential preflight to
`scripts/probe-deepseek-v4-b12x-g4-gce.sh` so issue #41 can distinguish local
auth failure from Google GPU capacity, GCE admission, and DeepSeek runtime
failures.

The wrapper now checks `gcloud auth print-access-token` before attempting any
GCE instance creation, unless `DRY_RUN=1` or `GCLOUD_AUTH_PREFLIGHT=0` is set.
The access token stdout is discarded. If auth refresh fails, the script records
`blocked_auth` for each planned G4 candidate and stops before calling
`gcloud compute instances create`.

## Live preflight attempt

The issue #41 launch shape was retried with the new preflight:

```bash
ISSUE_NUMBER=42 \
VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4 \
VLLM_ENFORCE_EAGER=1 \
HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=bf16_einsum \
HYDRALISK_B12X_CLAMP_PATCH=1 \
HYDRALISK_B12X_CLAMP_LIMIT=10.0 \
MAX_MODEL_LEN=2048 \
MAX_NUM_BATCHED_TOKENS=512 \
GPU_MEMORY_UTILIZATION=0.95 \
bash scripts/probe-deepseek-v4-b12x-g4-gce.sh
```

The generated local evidence selected the intended backend and flags:

```text
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

Both planned G4 candidates were blocked before creation:

```tsv
order	zone	machine	accelerator	gpu_count	status
1	us-central1-b	g4-standard-384	nvidia-rtx-pro-6000	8	blocked_auth
2	us-central1-b	g4-standard-192	nvidia-rtx-pro-6000	4	blocked_auth
```

## Plain-English read

This is not a DeepSeek kernel result and not evidence that Google lacks G4
capacity. The run did not reach GCE admission. It stopped earlier because the
local `chris@openagents.com` gcloud credentials require interactive
reauthentication.

Next operator action:

```bash
gcloud auth login
gcloud auth application-default login
```

After that, rerun issue #41 with `ISSUE_NUMBER=41` and
`VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4`.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
