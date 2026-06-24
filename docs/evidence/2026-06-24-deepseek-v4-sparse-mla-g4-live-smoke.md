# DeepSeek V4 sparse MLA patched-vLLM G4 live smoke

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/58

## Summary

Issue #58 provisioned a bounded one-GPU G4 target and ran the patched-vLLM
sparse MLA fallback synthetic smoke on real Google RTX PRO 6000 hardware.

The host was created as a spot VM with `maxRunDuration=3600s` and
`instanceTerminationAction=DELETE`:

```text
instance=hydralisk-deepseek-v4-sparse-mla-g4-1g-20260624154209
zone=us-central1-b
machine=g4-standard-48
accelerator=nvidia-rtx-pro-6000 x1
internalIp=10.128.0.30
externalIp=<none>
```

Docker was installed on the host, the already-installed NVIDIA container
toolkit was configured with `nvidia-ctk`, and GPU passthrough was verified with
`nvidia/cuda:13.0.2-base-ubuntu24.04`.

## Patched-vLLM Smoke

Command:

```bash
TARGET_INSTANCE=hydralisk-deepseek-v4-sparse-mla-g4-1g-20260624154209 \
TARGET_ZONE=us-central1-b \
IMAGE=vllm/vllm-openai:latest \
OUTPUT_DIR=.hydralisk/deepseek-v4-sparse-mla-vllm-fallback-g4-live-20260624 \
bash scripts/smoke-deepseek-v4-sparse-mla-vllm-fallback-gce.sh
```

Result:

```json
{"status":"ok","target":{"image":"vllm/vllm-openai:latest","instance":"hydralisk-deepseek-v4-sparse-mla-g4-1g-20260624154209","instanceStatus":"RUNNING","zone":"us-central1-b"},"output":{"checksum":{"count":32768,"l1":2875.557129,"maxAbs":0.453125,"sum":53.662781},"finite":true,"nonzero":true,"shape":[1,64,512]}}
```

The container reported:

```text
vllmVersion=0.23.0
torch=2.11.0+cu130
cuda=13.0
device=NVIDIA RTX PRO 6000 Blackwell Server Edition
envFlag=HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK=1
patchTarget=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py
decodeBranchPatched=true
prefillBranchPatched=true
```

The smoke did not load model weights, prompts, responses, or vLLM scheduling.
After the run, `nvidia-smi` reported `0 MiB` used on the RTX PRO 6000.

## Decision

The fallback branch is now proven inside a real vLLM container on real G4 /
SM120 hardware for the public-safe issue #52 tensor shape. The next honest step
is a full DeepSeek V4 8 x G4 model smoke using the derived vLLM image with
`HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK=1`, not more synthetic attention work.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
