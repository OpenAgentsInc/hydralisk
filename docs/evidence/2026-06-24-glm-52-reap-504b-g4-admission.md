# GLM-5.2 504B REAP G4 admission evidence

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/83

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Admission script:
[`scripts/probe-glm-52-reap-504b-g4-gce.sh`](../../scripts/probe-glm-52-reap-504b-g4-gce.sh)

Public-safety boundary: this packet contains cloud allocation, hardware, and
blocker summaries only. It contains no bearer tokens, model credentials, raw
prompts, responses, private source, hidden reasoning traces, weights,
checkpoints, compiled engines, profiler dumps, or large generated logs.

## Goal

Admit a fresh Google Cloud G4 host for the `0xSero/GLM-5.2-504B` REAP/NVFP4
lane without touching the active Fable G4 host.

The primary target remains 4 x RTX PRO 6000 (`g4-standard-192`). Because both
4x zones were capacity-blocked during this run, Hydralisk took the explicit
fallback path to a fresh 8 x RTX PRO 6000 host. That fallback keeps the
integration moving for staging and runtime work, but it does not convert the
4x accessibility claim into an 8x claim.

## GCE target

- Project: `openagentsgemini`
- Primary zones: `us-central1-b`, `us-central1-f`
- Primary machine: `g4-standard-192`
- Primary accelerator: 4 x `nvidia-rtx-pro-6000`
- Fallback machine: `g4-standard-384`
- Fallback accelerator: 8 x `nvidia-rtx-pro-6000`
- Image family: `common-cu129-ubuntu-2204-nvidia-580`
- Image project: `deeplearning-platform-release`
- Boot disk type: `hyperdisk-balanced`
- Boot disk size: 1500 GB
- Public IP: none
- OS Login: enabled
- Labels: `lane=hydralisk`, `workload=glm52-reap-504b`,
  `model=glm-5-2-reap-504b`, `issue=83`

## 4x admission result

Hydralisk attempted the primary 4x G4 target with both Spot and Standard
provisioning.

| Provisioning | Zone | Shape | Accelerator | Result | Blocker |
| --- | --- | --- | --- | --- | --- |
| Spot | `us-central1-b` | `g4-standard-192` | 4 x `nvidia-rtx-pro-6000` | blocked | `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` / stockout |
| Spot | `us-central1-f` | `g4-standard-192` | 4 x `nvidia-rtx-pro-6000` | blocked | `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` / stockout |
| Standard | `us-central1-b` | `g4-standard-192` | 4 x `nvidia-rtx-pro-6000` | blocked | `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` / stockout |
| Standard | `us-central1-f` | `g4-standard-192` | 4 x `nvidia-rtx-pro-6000` | blocked | `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` / stockout |

The latest Spot 4x run used local evidence directory:

```text
.hydralisk/glm52-reap-504b-g4-admission-20260624214500
```

The Standard 4x confirmation used local evidence directory:

```text
.hydralisk/glm52-reap-504b-g4-admission-20260624213600
```

Those local directories are intentionally not committed because they contain
raw cloud command logs. The public-safe result is summarized here.

## Fallback admission result

The explicit 8x fallback admitted successfully:

- Instance: `hydralisk-glm52-reap-504b-g4-8g-b-20260624214500`
- Zone: `us-central1-b`
- Machine: `g4-standard-384`
- Accelerator: 8 x `nvidia-rtx-pro-6000`
- Network IP: `10.128.0.38`
- Provisioning: Spot
- Termination action: Stop
- Max run duration: 21,600 seconds
- Status at evidence capture: running
- Boot disk: 1500 GB Hyperdisk Balanced
- Boot disk auto-delete: false
- Public ingress: none

This is a fresh GLM REAP host. It is distinct from the active Fable host.

## Hardware evidence

Host facts:

- OS: Ubuntu 22.04.5 LTS
- Kernel: `6.8.0-1060-gcp`
- Driver: `580.159.03`
- CUDA runtime reported by `nvidia-smi`: `13.0`
- CUDA toolkit on host: `nvcc` unavailable
- Host NCCL: no host `libnccl` line reported

GPU facts:

| GPU | Name | Memory MiB | Driver | PCI bus |
| --- | --- | ---: | --- | --- |
| 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | `00000000:05:00.0` |
| 1 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | `00000000:06:00.0` |
| 2 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | `00000000:0A:00.0` |
| 3 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | `00000000:0B:00.0` |
| 4 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | `00000000:84:00.0` |
| 5 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | `00000000:85:00.0` |
| 6 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | `00000000:89:00.0` |
| 7 | NVIDIA RTX PRO 6000 Blackwell Server Edition | 97887 | 580.159.03 | `00000000:8A:00.0` |

Aggregate visible GPU memory: 783,096 MiB, about 765 GiB.

Topology summary from `nvidia-smi topo -m`:

- GPUs 0-3 are NUMA node 0, CPU affinity `0-95,192-287`.
- GPUs 4-7 are NUMA node 1, CPU affinity `96-191,288-383`.
- GPUs are paired with `PIX` links: 0-1, 2-3, 4-5, 6-7.
- Same-socket cross-pair links are `NODE`.
- Cross-socket GPU links are `SYS`.
- No NVLink links were reported.

## Interpretation

This issue proves that Hydralisk has a fresh G4 host for the GLM-5.2 504B REAP
integration path, but only through the 8x fallback. The primary 4x target is
capacity-blocked at the time of this run, not technically falsified.

Next issue #84 should stage the pinned HF checkpoint onto:

```text
hydralisk-glm52-reap-504b-g4-8g-b-20260624214500
```

in:

```text
/opt/hydralisk/models/glm-5.2-504b
```

The model profile must continue to show the 4x path as blocked/pending rather
than served.
