# GLM-5.2 GCE admission preflight evidence

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/3

Profile: [`profiles/glm-5.2-fp8-sglang.json`](../../profiles/glm-5.2-fp8-sglang.json)

Public-safety boundary: this packet contains cloud hardware facts and blocker
summaries only. It contains no bearer tokens, model credentials, raw prompts,
responses, private source, hidden reasoning traces, weights, checkpoints, or
large generated logs.

## Source guidance

The OpenAgents Baseten implementation note says GLM-5.2 should start as a
Hydralisk model-profile and high-memory admission preflight, not as a public
Khala model claim:

- https://github.com/OpenAgentsInc/openagents/blob/main/docs/inference/2026-06-24-baseten-glm-52-production-inference-notes.md

The SGLang GLM-5.2 documentation says GLM-5.2 is a DSA MoE model with MTP
speculative decoding and a 1,048,576-token context window, recommends FP8
serving, names `zai-org/GLM-5.2-FP8`, and documents parser/runtime guidance:

- https://lmsysorg.mintlify.app/cookbook/autoregressive/GLM/GLM-5.2

The Hugging Face model API returned this immutable FP8 revision on
2026-06-24:

- `zai-org/GLM-5.2-FP8@70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1`

## GCP catalog and quota facts

Project: `openagentsgemini`

Region: `us-central1`

Visible accelerator catalog in the target zones:

- `us-central1-b`: `nvidia-b200`, `nvidia-h200-141gb`,
  `nvidia-rtx-pro-6000`, `nvidia-h100-80gb`
- `us-central1-a`: `nvidia-h100-80gb`
- `us-central1-f`: `nvidia-rtx-pro-6000`

Relevant machine shapes observed with `gcloud compute machine-types list`:

- B200: `a4-highgpu-8g`, 224 vCPU, 3968 GB RAM, 8 x `nvidia-b200`
- H200: `a3-ultragpu-8g`, 224 vCPU, 2952 GB RAM, 8 x `nvidia-h200-141gb`
- H100: `a3-highgpu-8g`, 208 vCPU, 1872 GB RAM, 8 x `nvidia-h100-80gb`
- RTX PRO 6000: `g4-standard-384`, 384 vCPU, 1440 GB RAM,
  8 x `nvidia-rtx-pro-6000`

Regional quota output did not yet expose named B200/H200/G4 quota metrics, but
the actual allocation call admitted the G4 shape. Older visible GPU quotas in
`us-central1` still include `NVIDIA_L4_GPUS: limit 16, usage 3`, and
`CPUS: limit 3000, usage 47` at preflight time.

## Ordered allocation attempt

The ordered probe used short-lived Spot instances with
`--instance-termination-action DELETE`, `--max-run-duration 600s`,
`--maintenance-policy TERMINATE`, `--boot-disk-type hyperdisk-balanced`, and
labels:

- `lane=hydralisk`
- `workload=glm52-preflight`
- `model=glm-5-2`

Attempt 1:

- Accelerator: `nvidia-b200`
- Shape: `a4-highgpu-8g`
- Zone: `us-central1-b`
- Result: blocked
- Blocker: `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`
- Public-safe message: GCE reported that an `a4-highgpu-8g` VM with
  8 x `nvidia-b200` and 32 local SSDs was unavailable in `us-central1-b`.

Attempt 2:

- Accelerator: `nvidia-rtx-pro-6000`
- Shape: `g4-standard-384`
- Zone: `us-central1-b`
- Result: admitted
- Instance: `hydralisk-glm52-rtx-pro-6000-20260624034345`
- Admission ref:
  `admission.hydralisk.glm52.g4.rtxpro6000.20260624T034345Z`
- Scheduling: Spot, automatic restart disabled, host maintenance terminate,
  max run duration 600 seconds, termination action delete
- Cleanup: instance was deleted after hardware evidence capture

Because the second slot admitted a Blackwell-class high-memory node, the H200
and H100 fallbacks were not attempted in the final ordered pass. An earlier
H100 pass against `us-central1-b` reached real capacity admission and GCE
reported `a3-highgpu-8g` capacity available in `us-central1-a`, but that was
not used as the issue baseline because G4 admitted first in the corrected pass.

## Hardware evidence from admitted G4 node

Host facts:

- OS: Ubuntu 22.04.5 LTS
- Kernel: `6.8.0-1060-gcp`
- `nvidia-smi`: `/usr/bin/nvidia-smi`
- Driver: `580.159.03`
- CUDA runtime reported by `nvidia-smi`: `13.0`
- CUDA toolkit on host: `nvcc` unavailable
- NCCL on host: no host `libnccl` reported by `ldconfig`

GPU facts:

| GPU | Name | Memory MiB | Driver | PCI bus |
| --- | --- | ---: | --- | --- |
| 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | 00000000:05:00.0 |
| 1 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | 00000000:06:00.0 |
| 2 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | 00000000:0A:00.0 |
| 3 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | 00000000:0B:00.0 |
| 4 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | 00000000:84:00.0 |
| 5 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | 00000000:85:00.0 |
| 6 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | 00000000:89:00.0 |
| 7 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | 00000000:8A:00.0 |

Aggregate visible GPU memory: 783096 MiB, about 765 GiB.

Topology summary from `nvidia-smi topo -m`:

- GPUs 0-3 are NUMA node 0, CPU affinity `0-95,192-287`.
- GPUs 4-7 are NUMA node 1, CPU affinity `96-191,288-383`.
- GPUs are paired with `PIX` links: 0-1, 2-3, 4-5, 6-7.
- Same-socket cross-pair links are `NODE`.
- Cross-socket GPU links are `SYS`.
- No NVLink links were reported in this G4 topology.

## Interpretation

The preflight proves that the project can admit an 8 x RTX PRO 6000 Blackwell
Server Edition node in `us-central1-b`. It does not prove GLM-5.2 is loadable,
fast, or production-safe on that node.

Reasons to stay conservative:

- SGLang's GLM-5.2 page names H200, B200, B300, and GB300 as deployment
  targets. It does not list G4 / RTX PRO 6000 as a verified GLM-5.2 matrix row.
- 8 x RTX PRO 6000 gives about 765 GiB aggregate visible memory. That is close
  to GLM-5.2 FP8 weight scale before KV cache, runtime buffers, fragmentation,
  MoE routing overhead, and long-context pressure.
- Full 1M-token context was not admitted. The profile caps the first smoke at
  32K tokens and treats higher context as a benchmark ladder.
- Host NCCL was not present in the base image. The pinned SGLang container must
  provide runtime NCCL evidence during the load smoke.

## Smoke status

No model load or completion smoke was run in this issue pass. The admitted G4
node was used only long enough to capture hardware evidence and was then
deleted.

Smoke blocker:

- `load_smoke_pending`: run a short-lived monolithic SGLang load smoke from the
  pinned profile before any GLM routing, product claim, or Khala integration.

Minimum next smoke:

1. Recreate the G4 probe or allocate a supported H200/B200 node.
2. Pull
   `lmsysorg/sglang:v0.5.13.post1@sha256:74084d80c3b7e5649f4b3433b1169db3da26c9b1e31752a43045a34cc26ba5d5`.
3. Load `zai-org/GLM-5.2-FP8` at revision
   `70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1` with tensor parallel size 8.
4. Start with a small synthetic prompt, `max_tokens <= 32`, and no private
   source material.
5. Record only public-safe capability, usage, latency, memory, and blocker
   receipts.

## Recommendation

Next lane: monolithic SGLang load smoke.

Run it first on the admitted G4 shape only if the goal is fastest learning.
Prefer H200 or B200 for the first benchmark-quality lane because those are
inside SGLang's documented GLM-5.2 target matrix. Dynamo KV-aware routing and
prefill/decode disaggregation should wait until a monolithic load and tiny
completion smoke pass with public-safe receipts.
