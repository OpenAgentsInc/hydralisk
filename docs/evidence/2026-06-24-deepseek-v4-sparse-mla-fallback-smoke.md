# DeepSeek V4 sparse MLA fallback smoke

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/55

## Summary

Issue #55 made the SM120 sparse MLA fallback contract runnable as a public-safe
smoke, with a local CLI and a GCE/container wrapper.

Implementation:

- `hydralisk/admission/deepseek_v4_sparse_mla_smoke.py`
- `scripts/smoke-deepseek-v4-sparse-mla-fallback-gce.sh`
- console entry point: `hydralisk-deepseek-v4-sparse-mla-smoke`

The smoke uses the issue #52 tensor shape family by default:

```json
{"compressedKvCache":[1,1,256,512],"dtypeContract":"bf16-compatible values represented as CPU floats","kvLayout":"HND","query":[1,64,512],"seqLens":[1],"sparseIndices":[1,128],"sparseTopkLens":[1],"swaKvCache":[1,1,256,512]}
```

It does not load model weights, prompts, responses, vLLM scheduling, or GPU
kernels.

## Local Exact-Shape Smoke

Command:

```bash
uv run hydralisk-deepseek-v4-sparse-mla-smoke \
  --output-dir .hydralisk/deepseek-v4-sparse-mla-fallback-local-20260624
```

Result:

```json
{"checksum":{"count":32768,"l1":12619.254309,"maxAbs":0.683994,"sum":3.013336},"elapsedMs":619.917,"finite":true,"nonzero":true,"outputShape":[1,64,512],"stats":{"dim":512,"empty_route_count":0,"head_count":64,"masked_sparse_route_count":0,"page_size":256,"query_count":1,"sliding_window_tokens":128,"sparse_route_count":128,"swa_route_count":128}}
```

## GCE / Container Wrapper

Command:

```bash
bash scripts/smoke-deepseek-v4-sparse-mla-fallback-gce.sh
```

The wrapper first checks the target instance, injects the Hydralisk package into
the target Docker image, and runs the smoke with `python3 -m
hydralisk.admission.deepseek_v4_sparse_mla_smoke --stdout-json`.

Current result:

```json
{"status":"target_missing","target":{"image":"hydralisk-deepseek-v4-b12x-g4-vllm:20260624150453","instance":"hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036","zone":"us-central1-b"},"result":{"ranContainer":false,"reason":"target instance was not found or could not be described"}}
```

Current GCE state has no live DeepSeek G4 host. The live hosts are the L4
GPT-OSS 20B lane and the single-H100 GPT-OSS 120B lane; neither should be
disturbed for this DeepSeek smoke.

## What Remains

The next implementation issue should wire this fallback into vLLM's
`DeepseekV4FlashInferMLAAttention._forward` path under a fail-closed probe flag,
then rerun the container synthetic shape on a fresh or explicitly provided G4
host before any full-model smoke.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
