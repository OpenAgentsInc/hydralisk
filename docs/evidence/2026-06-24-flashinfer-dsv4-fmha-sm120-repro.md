# FlashInfer DSV4 FMHA SM120 synthetic repro

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/52

## Summary

Issue #52 isolated the issue #41 generation blocker outside the full model.
The new script `scripts/probe-flashinfer-dsv4-fmha-gce.sh` runs a direct
synthetic call to FlashInfer's DeepSeek V4 sparse MLA TRTLLM launcher on the
same G4 host and container image, without model weights, prompts, vLLM
scheduling, B12x MoE, or `o_proj`.

The repro fails with the same public-safe root error:

```text
Error in function 'TllmGenFmhaRunner' at /workspace/include/flashinfer/trtllm/fmha/fmhaRunner.cuh:37: Unsupported architecture
```

That proves the current stock FlashInfer `0.6.12` DSV4 TRTLLM FMHA runner does
not admit RTX PRO 6000 / SM120 for even a one-token BF16 synthetic call.

## Command

```bash
TARGET_INSTANCE=hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036 \
TARGET_ZONE=us-central1-b \
bash scripts/probe-flashinfer-dsv4-fmha-gce.sh
```

## Environment

```text
targetInstance=hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036
targetZone=us-central1-b
image=hydralisk-deepseek-v4-b12x-g4-vllm:20260624150453
flashinfer=0.6.12
torch=2.11.0+cu130
cuda=13.0
device=NVIDIA RTX PRO 6000 Blackwell Server Edition
capability=[12,0]
```

## Synthetic Inputs

```json
{"compressedKvCache":[1,1,256,512],"containsPrompts":false,"dtype":"bf16","kvLayout":"HND","loadsModelWeights":false,"out":[1,64,512],"query":[1,64,512],"seqLens":[1],"sparseIndices":[1,128],"sparseTopkLens":[1],"swaKvCache":[1,1,256,512],"synthetic":true,"workspaceBytes":134217728}
```

## Result

```json
{"capability":[12,0],"cuda":"13.0","deviceName":"NVIDIA RTX PRO 6000 Blackwell Server Edition","error":"Error in function 'TllmGenFmhaRunner' at /workspace/include/flashinfer/trtllm/fmha/fmhaRunner.cuh:37: Unsupported architecture","errorType":"RuntimeError","flashinfer":"0.6.12","inputs":{"compressedKvCache":[1,1,256,512],"containsPrompts":false,"dtype":"bf16","kvLayout":"HND","loadsModelWeights":false,"out":[1,64,512],"query":[1,64,512],"seqLens":[1],"sparseIndices":[1,128],"sparseTopkLens":[1],"swaKvCache":[1,1,256,512],"synthetic":true,"workspaceBytes":134217728},"schema":"hydralisk.flashinfer-dsv4-fmha-repro.v1","status":"error","torch":"2.11.0+cu130"}
```

## Next Patch Point

Patch or replace the FlashInfer TRTLLM FMHA runner used by
`flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4`.

The narrow code path to inspect next is:

```text
flashinfer/mla/_core.py trtllm_batch_decode_sparse_mla_dsv4
flashinfer/trtllm/fmha/fmhaRunner.cuh TllmGenFmhaRunner
```

The immediate next implementation issue should source-inspect the installed
FlashInfer/TRTLLM FMHA architecture guard and decide whether SM120 can be
admitted by adding a Blackwell server target or whether Hydralisk needs a
correctness-first fallback attention kernel for this lane.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
