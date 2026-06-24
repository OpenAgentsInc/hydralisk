# GLM-5.2 SGLang load-smoke evidence

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/4

Profile: [`profiles/glm-5.2-fp8-sglang.json`](../../profiles/glm-5.2-fp8-sglang.json)

Prior admission packet:
[`docs/evidence/2026-06-24-glm-52-gce-admission-preflight.md`](2026-06-24-glm-52-gce-admission-preflight.md)

Public-safety boundary: this packet contains hardware, runtime, and blocker
summaries only. It contains no bearer tokens, model-provider credentials, raw
prompts, responses, private source, hidden reasoning traces, weights,
checkpoints, compiled engines, profiler dumps, or large generated logs.

## Goal

Run the first monolithic SGLang load smoke for `zai-org/GLM-5.2-FP8` using the
Hydralisk profile from issue #3. If the model reached a ready OpenAI-compatible
endpoint, run one tiny synthetic completion and persist only usage and latency.

No public OpenAgents or Khala routing changed in this issue.

## Immutable pins

- Model: `zai-org/GLM-5.2-FP8`
- Model revision:
  `70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1`
- Engine: SGLang `0.5.13.post1`
- Container:
  `lmsysorg/sglang:v0.5.13.post1@sha256:74084d80c3b7e5649f4b3433b1169db3da26c9b1e31752a43045a34cc26ba5d5`
- Platform: Linux amd64
- Tensor parallelism: 8
- Public ingress: none; SGLang was bound to localhost on the smoke VM

## Allocation result

Project: `openagentsgemini`

The smoke retried high-memory capacity before falling back to the fastest
learning lane:

| Order | Shape | Accelerator | Zone | Result |
| ---: | --- | --- | --- | --- |
| 1 | `a4-highgpu-8g` | 8 x `nvidia-b200` | `us-central1-b` | capacity blocked |
| 2 | `a3-ultragpu-8g` | 8 x `nvidia-h200-141gb` | `us-central1-b` | capacity blocked |
| 3 | `g4-standard-384` | 8 x `nvidia-rtx-pro-6000` | `us-central1-b` | admitted |

The admitted smoke VM was:

- Instance: `hydralisk-glm52-active-smoke-g4-b-20260624040754`
- Zone: `us-central1-b`
- Scheduling: Spot
- Boot disk: 2500 GB Hyperdisk Balanced
- Cleanup: deleted after evidence capture

Post-cleanup `gcloud compute instances list --filter='name~hydralisk-glm52'`
returned no instances.

## Runtime evidence

Host/runtime facts:

- OS: Ubuntu 22.04.5 LTS
- Kernel: `6.8.0-1060-gcp`
- Driver: `580.159.03`
- CUDA runtime reported by `nvidia-smi`: `13.0`
- Docker was installed on the smoke VM before launch.

Container facts:

- SGLang: `0.5.13.post1`
- PyTorch: `2.11.0+cu130`
- Torch CUDA: `13.0`
- NCCL: SGLang reported `nccl==2.28.9`
- GPUs visible in the container: 8 x
  `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Visible memory per GPU: 97887 MiB
- Aggregate visible GPU memory: 783096 MiB
- Hugging Face snapshot cache after download: 704 GB

The first fully featured profile launch, including parser and EAGLE MTP knobs,
failed before model load while SGLang was reading device memory capacity. A
minimal CUDA probe in the same pinned container immediately afterwards
successfully saw all 8 GPUs and `torch.cuda.mem_get_info()` worked, so the
smoke continued with staged launches.

## Load attempts

### 32K baseline

Command traits:

- `--tp-size 8`
- `--context-length 32768`
- `--mem-fraction-static 0.80`
- no Dynamo
- no HiCache
- no public ingress

Result:

- The model snapshot downloaded successfully.
- SGLang started TP ranks and began FP8 weight load.
- The run failed after load with memory-pool initialization errors.
- Public-safe blocker: SGLang reported not enough memory and suggested
  increasing `--mem-fraction-static`.

### 8K retry

Command traits:

- `--tp-size 8`
- `--context-length 8192`
- `--max-total-tokens 8192`
- `--mem-fraction-static 0.92`
- `--disable-custom-all-reduce`

Result:

- SGLang found the local HF snapshot and skipped download.
- The run failed with the same memory-pool blocker.

### 4K minimized retry

Command traits:

- `--tp-size 8`
- `--context-length 4096`
- `--max-total-tokens 4096`
- `--mem-fraction-static 0.98`
- `--disable-custom-all-reduce`
- `--disable-radix-cache`

Result:

- SGLang loaded all 141 FP8 shards from the cached snapshot.
- Each TP rank reported `type=GlmMoeDsaForCausalLM`, `quant=fp8`, `fmt=e4m3`.
- Per-rank load memory summary: 89.44 GB used, 4.64 GB available.
- KV cache allocated at 4096 tokens with `torch.float8_e4m3fn`.
- Per-rank KV size was 0.21 GB.
- The run did not reach a stable health endpoint.

When piecewise CUDA graph was enabled, SGLang failed with a DeepGEMM unsupported
architecture error. After adding `--disable-piecewise-cuda-graph`, SGLang fell
back to the regular CUDA graph path and hit the same unsupported-architecture
family. After disabling both CUDA graph paths, the scheduler still failed in
the FlashInfer TRTLLM DSA attention path:

```text
TllmGenFmhaRunner ... Unsupported architecture
```

The failed path was SGLang's GLM/DeepSeek DSA attention implementation through
FlashInfer TRTLLM MLA decode. The container was not OOM-killed.

## Completion smoke

No completion smoke was run. The model never reached a stable ready endpoint on
G4 / RTX PRO 6000 under the pinned SGLang image.

## Interpretation

G4 / RTX PRO 6000 is not a viable GLM-5.2 serving lane for this pinned
Hydralisk profile today.

The useful result is narrower:

- The project can admit an 8 x RTX PRO 6000 node.
- The pinned container can see all GPUs and load GLM-5.2 FP8 weights.
- A 32K or 8K context smoke does not fit the SGLang memory-pool plan.
- A 4K minimized smoke can allocate KV cache, but SGLang's DSA kernel stack
  reports unsupported architecture before serving.

This matches the conservative reading from the SGLang GLM-5.2 docs: H200,
B200, B300, and GB300 are the target matrix. G4 was useful as a fast load-risk
probe, not as a production or benchmark-quality GLM-5.2 route.

## Next lane

Open the next implementation issue for a supported-accelerator rerun:

1. Allocate H200 or B200 with enough run time for cached or fresh model load.
2. Reuse the same model revision and SGLang image pin.
3. Start at 32K context with `--tp-size 8`.
4. Keep Dynamo, HiCache, MTP, and public ingress disabled until monolithic load
   and one tiny synthetic completion pass.
5. If capacity remains blocked, record the exact GCE blocker and leave GLM-5.2
   self-hosting gated behind H200/B200 availability.

