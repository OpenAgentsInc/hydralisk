# GLM-5.2 504B REAP checkpoint staging evidence

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/84

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Admission evidence:
[`docs/evidence/2026-06-24-glm-52-reap-504b-g4-admission.md`](2026-06-24-glm-52-reap-504b-g4-admission.md)

Public manifest:
[`docs/evidence/2026-06-24-glm-52-reap-504b-staging-manifest.json`](2026-06-24-glm-52-reap-504b-staging-manifest.json)

Staging script:
[`scripts/stage-glm-52-reap-504b-gce.sh`](../../scripts/stage-glm-52-reap-504b-gce.sh)

Public-safety boundary: this packet contains reduced checkpoint metadata only.
It contains no bearer tokens, model-provider credentials, raw prompts,
responses, private source, hidden reasoning traces, weights, checkpoints,
compiled engines, profiler dumps, or raw transfer logs.

## Goal

Stage `0xSero/GLM-5.2-504B` at the pinned revision onto durable G4 storage and
produce a resumable, public-safe manifest proving the local checkpoint is
complete.

## Target

- Model: `0xSero/GLM-5.2-504B`
- Revision:
  `cb6b1e0451b9d560cda864f84187869c9a679712`
- Instance: `hydralisk-glm52-reap-504b-g4-8g-b-20260624214500`
- Zone: `us-central1-b`
- Machine: `g4-standard-384`
- Accelerator: 8 x `nvidia-rtx-pro-6000`
- Model directory: `/opt/hydralisk/models/glm-5.2-504b`
- HF cache directory: `/var/lib/hydralisk/huggingface`
- Remote reduced evidence directory:
  `/var/log/hydralisk/glm52-reap-504b-staging-20260624220000`

The host is the 8x G4 fallback admitted in issue #83 because the primary 4x
G4 target was capacity-blocked in both target zones. The boot disk is a 1500 GB
Hyperdisk Balanced disk with `autoDelete=false`, so the staged checkpoint
survives instance deletion/recreation workflows that preserve the disk.

## Staging command

The transfer was started with:

```bash
ACTION=start RUN_ID=20260624220000 \
  scripts/stage-glm-52-reap-504b-gce.sh
```

The script installed host-side Python staging tools in:

```text
/opt/hydralisk/venvs/hf-staging
```

and ran:

```bash
hf download 0xSero/GLM-5.2-504B \
  --revision cb6b1e0451b9d560cda864f84187869c9a679712 \
  --local-dir /opt/hydralisk/models/glm-5.2-504b
```

The transfer is resumable by rerunning the same script with the same model
directory and revision.

## Result

Status: `complete`

Completion time: `2026-06-24T21:51:52Z`

Reduced manifest summary:

| Field | Value |
| --- | ---: |
| Complete | `true` |
| Local safetensor shard count | `63` |
| Expected shard count | `63` |
| Local safetensor shard bytes | `308829060264` |
| Index metadata total size bytes | `318247808128` |
| Weight-map entries | `154433` |
| All file count | `147` |
| All file bytes | `308862030135` |
| Missing shard count | `0` |
| Unexpected shard count | `0` |

The index metadata total is the safetensors tensor payload value from
`model.safetensors.index.json`. The local safetensor shard byte total is the
observed on-disk file size total. Both values are preserved in the public
manifest because they answer different verification questions.

## Metadata hashes

The public manifest includes SHA-256 hashes for small metadata files only:

- `README.md`
- `config.json`
- `generation_config.json`
- `model.safetensors.index.json`
- `tokenizer.json`
- `tokenizer_config.json`

Full shard hashes were intentionally not computed by default in this pass:
rerunning a full 309 GB hash sweep is possible, but the current acceptance gate
only requires the pinned HF revision, complete shard set, index total, and
reduced metadata proof. The shard files themselves remain private runtime
artifacts and are not committed.

## Config confirmation

The staged config confirms:

- Architecture: `GlmMoeDsaForCausalLM`
- Model type: `glm_moe_dsa`
- Context window: 1,048,576 tokens
- Layers: 78
- Routed experts: 168
- Experts per token: 8
- Quantization method: `modelopt`
- Quantization algorithm: `NVFP4`

## Next step

Issue #85 can now build the b12x/vLLM launch profile against the staged local
path:

```text
/opt/hydralisk/models/glm-5.2-504b
```

No public serving claim is made by this staging result.
