# DeepSeek-V4-Fable merged checkpoint staging

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/79

Depends on: https://github.com/OpenAgentsInc/hydralisk/issues/78

Status: `staged_private_g4_ready_for_canary`

## Decision

- Private merged-checkpoint load can be attempted: `true`
- Khala general route allowed: `false`
- Public aliases allowed: `false`
- MPP public sale allowed: `false`
- Next step: `run_fable_merged_checkpoint_private_g4_canary`

## Target

```text
project=openagentsgemini
zone=us-central1-b
instance=hydralisk-deepseek-v4-b12x-g4-8g-b-20260624155352
machine=g4-standard-384
gpuCount=8
gpuName=NVIDIA RTX PRO 6000 Blackwell Server Edition
modelPath=/opt/hydralisk/models/deepseek-v4-fable-merged
```

The instance is a Spot VM with `maxRunDuration=21600s`. During staging,
the boot disk was changed from auto-delete to `autoDelete=false` so a Spot
instance deletion would not destroy the staged checkpoint.

## Staging

The checkpoint was staged into private GCE runtime storage, not Git.

```text
sourceRepo=Chunjiang-Intelligence/DeepSeek-v4-Fable
revision=999909137c15e0b5539fee887431824fa7cb5b10
initialSequentialStart=2026-06-24T20:14:36Z
parallelManifestStart=2026-06-24T20:35:03Z
completedAt=2026-06-24T21:31:24Z
parallelJobs=8
manifestRows=52
metadataFiles=5
mergedShardFiles=47
stagedFileBytes=298428071600
manifestSha256=0610e0fc3f79512a9cc11b6ce93e48e1bdf6c25e0e694d52f4046f43c06a8833
```

The tracked manifest contains file names, byte counts, SHA-256 hashes, and
completion timestamps for all staged artifacts:

[`docs/evidence/2026-06-24-deepseek-v4-fable-merged-staging-manifest.tsv`](2026-06-24-deepseek-v4-fable-merged-staging-manifest.tsv)

## Final host state

At `2026-06-24T21:32:39Z`:

```text
/dev/root size=873G used=488G avail=385G use=56%
runningDockerContainers=0
gpuMemoryUsedMiB=0 on all 8 GPUs
```

The staged file total is slightly larger than the index metadata's tensor
payload size (`298425334924`) because safetensors files include container
metadata as well as tensor payload bytes.

## Interpretation

Issue #79 proves artifact staging is not the blocker. The full merged Fable
checkpoint now exists on the private G4 host with resumable-download integrity
evidence and enough remaining disk for a private load canary.

This does not admit Fable for serving. The next risk is runtime compatibility:
the staged artifact is an FP8 merged DeepSeek-V4-Fable checkpoint, while the
currently proven Hydralisk G4 lane is a patched NVIDIA NVFP4 DeepSeek-V4-Flash
runtime with multiple SM120 fallbacks.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
