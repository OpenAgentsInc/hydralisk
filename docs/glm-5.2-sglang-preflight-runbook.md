# GLM-5.2 SGLang preflight runbook

Date: 2026-06-24

Profile:
[`profiles/glm-5.2-fp8-sglang.json`](../profiles/glm-5.2-fp8-sglang.json)

Evidence:
[`docs/evidence/2026-06-24-glm-52-gce-admission-preflight.md`](evidence/2026-06-24-glm-52-gce-admission-preflight.md)

## Boundary

This is not a Khala product route and not a public model selector. It is a
Hydralisk admission and load-smoke lane for GLM-5.2 FP8 on NVIDIA GPUs.

Do not store or commit:

- bearer tokens, HF tokens, OpenAgents secrets, or cloud credentials;
- raw prompts, responses, customer data, private source, or hidden reasoning;
- model weights, checkpoints, compiled engines, profiler dumps, or benchmark
  output.

## Model and engine pins

- Model: `zai-org/GLM-5.2-FP8`
- Model revision:
  `70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1`
- Architecture: `GlmMoeDsaForCausalLM`, `glm_moe_dsa`
- Context window: 1,048,576 tokens advertised by model/SGLang docs
- Engine: SGLang `0.5.13.post1`
- Container:
  `lmsysorg/sglang:v0.5.13.post1@sha256:74084d80c3b7e5649f4b3433b1169db3da26c9b1e31752a43045a34cc26ba5d5`

SGLang GLM-5.2 docs:
https://lmsysorg.mintlify.app/cookbook/autoregressive/GLM/GLM-5.2

## Allocate a probe

The helper script attempts capacity in this order:

1. B200: `a4-highgpu-8g` in `us-central1-b`
2. RTX PRO 6000 Blackwell: `g4-standard-384` in `us-central1-b`
3. H200: `a3-ultragpu-8g` in `us-central1-b`
4. H100: `a3-highgpu-8g` in `us-central1-a`

Run:

```bash
PROJECT_ID=openagentsgemini scripts/probe-glm-52-gce-admission.sh
```

The script writes public-safe evidence under `.hydralisk/`, captures
`nvidia-smi`, topology, and selected instance metadata, then deletes the probe
by default. Set `KEEP_INSTANCE=1` only when you are ready to immediately run
the model-load smoke and have an operator-visible cleanup plan.

## Load-only smoke

Use the admitted instance only after the hardware packet is captured. The first
load smoke should be monolithic SGLang, no Dynamo, no public ingress, and no
OpenAgents routing.

Host setup sketch:

```bash
sudo mkdir -p /var/lib/hydralisk/huggingface
sudo chmod 700 /var/lib/hydralisk/huggingface
```

Set `HF_TOKEN` out of band. Do not echo it into shell history or tracked files.

Container launch:

```bash
docker run --rm --gpus all \
  --ipc=host \
  --shm-size 64g \
  -p 127.0.0.1:30000:30000 \
  -v /var/lib/hydralisk/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN \
  lmsysorg/sglang:v0.5.13.post1@sha256:74084d80c3b7e5649f4b3433b1169db3da26c9b1e31752a43045a34cc26ba5d5 \
  python3 -m sglang.launch_server \
    --model-path zai-org/GLM-5.2-FP8 \
    --revision 70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1 \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 8 \
    --mem-fraction-static 0.80 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6
```

For the first smoke, do not enable HiCache or Dynamo. Record whether SGLang
starts, whether CUDA/NCCL is visible inside the container, peak GPU memory, and
any OOM/parser/runtime blocker.

Tiny public-safe completion smoke:

```bash
curl -fsS http://127.0.0.1:30000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "zai-org/GLM-5.2-FP8",
    "messages": [
      { "role": "user", "content": "Reply with READY." }
    ],
    "max_tokens": 8,
    "temperature": 0
  }' | jq '.usage'
```

Only record public-safe usage, latency, and blocker state. Do not store the raw
prompt or response outside the active terminal.

## Hydralisk capability fields

When proxying a GLM smoke through Hydralisk, set:

```bash
HYDRALISK_SERVED_MODEL=zai-org/GLM-5.2-FP8
HYDRALISK_PUBLIC_MODEL_ALIASES=
HYDRALISK_ENGINE=sglang
HYDRALISK_ENGINE_VERSION=0.5.13.post1
HYDRALISK_GPU_CLASS=g4
HYDRALISK_GPU_NAME='NVIDIA RTX PRO 6000 Blackwell Server Edition'
HYDRALISK_GPU_COUNT=8
HYDRALISK_MODEL_REVISION=zai-org/GLM-5.2-FP8@70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1
HYDRALISK_QUANTIZATION_WEIGHTS=FP8
HYDRALISK_MODEL_PROFILE_REF=profiles/glm-5.2-fp8-sglang.json
HYDRALISK_CONTAINER_IMAGE=lmsysorg/sglang:v0.5.13.post1@sha256:74084d80c3b7e5649f4b3433b1169db3da26c9b1e31752a43045a34cc26ba5d5
HYDRALISK_CONTEXT_WINDOW_TOKENS=1048576
HYDRALISK_ADMITTED_CONTEXT_TOKENS=32768
HYDRALISK_TENSOR_PARALLEL_SIZE=8
HYDRALISK_REASONING_PARSER=glm45
HYDRALISK_TOOL_CALL_PARSER=glm47
HYDRALISK_CACHE_POLICY='prefix-cache-enabled; hicache-planned'
HYDRALISK_KV_CACHE_DTYPE=auto
HYDRALISK_DYNAMO_MODE=disabled_preflight
HYDRALISK_SPECULATIVE_DECODING=EAGLE_MTP_planned
HYDRALISK_ADMISSION_REF=admission.hydralisk.glm52.g4.rtxpro6000.20260624T034345Z
HYDRALISK_EVIDENCE_REF=docs/evidence/2026-06-24-glm-52-gce-admission-preflight.md
```

Keep `HYDRALISK_PUBLIC_MODEL_ALIASES` empty for raw GLM preflight. Any later
Khala consumption must happen behind the product-level `khala` /
`openagents/khala` identity, not a public raw GLM selector.

## Promotion ladder

1. G4 monolithic load-only smoke at 8K to 32K context.
2. Tiny public-safe completion smoke with usage and latency.
3. Repeat on H200 or B200 because those are in the SGLang GLM-5.2 target
   matrix.
4. Add 8K-in / 1K-out, then 32K/128K prefill tests.
5. Add HiCache only after baseline memory and TTFT are understood.
6. Add Dynamo KV-aware routing after multiple replicas exist.
7. Add Dynamo prefill/decode disaggregation only after monolithic bottlenecks
   are measured.

Stop if any lane cannot produce public-safe capability, usage, latency, memory,
and blocker receipts.
