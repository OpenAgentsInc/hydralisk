# DeepSeek V4 sparse MLA patched-vLLM fallback smoke

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/57

## Summary

Issue #57 added the final synthetic smoke wrapper needed before a live
DeepSeek G4 retry. The new script injects Hydralisk into a target vLLM Docker
image, patches vLLM's DeepSeek V4 FlashInfer sparse MLA source inside the
ephemeral container, imports the patched fallback helper, and runs the issue
#52 tensor shape with torch tensors.

Implementation:

- `scripts/smoke-deepseek-v4-sparse-mla-vllm-fallback-gce.sh`

The script does not load model weights, prompts, responses, or vLLM scheduling.

## Current Run

Command:

```bash
bash scripts/smoke-deepseek-v4-sparse-mla-vllm-fallback-gce.sh
```

Result:

```json
{"status":"target_missing","target":{"image":"hydralisk-deepseek-v4-b12x-g4-vllm:20260624150453","instance":"hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036","zone":"us-central1-b"},"result":{"ranContainer":false,"reason":"target instance was not found or could not be described"}}
```

No DeepSeek G4 host is currently live. The running Hydralisk hosts are the L4
GPT-OSS 20B lane and the single-H100 GPT-OSS 120B lane; this smoke did not
touch either one.

## Next Live Step

Provision or explicitly provide a DeepSeek G4 host with the target vLLM image,
then run:

```bash
TARGET_INSTANCE=<fresh-g4-host> \
TARGET_ZONE=<zone> \
bash scripts/smoke-deepseek-v4-sparse-mla-vllm-fallback-gce.sh
```

Only if that patched-vLLM synthetic smoke returns finite, nonzero output should
Hydralisk attempt another full 8 x G4 model smoke.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
