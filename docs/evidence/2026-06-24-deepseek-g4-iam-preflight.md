# DeepSeek-V4-Flash G4 IAM preflight

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/46

Related live GPU issue: https://github.com/OpenAgentsInc/hydralisk/issues/41

## Summary

Issue #46 added a default-on `GCLOUD_IAM_PREFLIGHT` check to
`scripts/probe-deepseek-v4-b12x-g4-gce.sh`. The check runs after auth preflight
and before any `gcloud compute instances create` attempt.

The preflight uses the selected `GCLOUD_ACCOUNT` / `CLOUDSDK_CORE_ACCOUNT`
token to call Cloud Resource Manager's `projects.testIamPermissions` endpoint.
Access tokens are never printed or committed. The wrapper records only the
returned permission names and the missing permission list.

## Required permissions checked

```text
compute.instances.create
compute.instances.get
compute.instances.setLabels
compute.instances.setMetadata
compute.instances.setTags
compute.disks.create
compute.subnetworks.use
```

## Live preflight result

The canonical DeepSeek FlashInfer DSV4 wrapper was run with the
`oa-vertex-inference` service account:

```bash
GCLOUD_ACCOUNT=oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com \
ISSUE_NUMBER=46 \
bash scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh
```

The auth preflight passed:

```text
gcloud Auth Preflight: ok
```

The IAM preflight failed before any GCE create attempt:

```text
gcloud IAM Preflight: blocked_iam
```

Both planned G4 candidates were marked `blocked_iam` with the same missing
core permissions:

```tsv
machine	accelerator	gpu_count	zone	status
g4-standard-384	nvidia-rtx-pro-6000	8	us-central1-b	blocked_iam
g4-standard-192	nvidia-rtx-pro-6000	4	us-central1-b	blocked_iam
```

## Plain-English read

This proves the service-account route is no longer blocked by token minting.
It is blocked by IAM before GCE admission. We still have not tested G4 capacity
or DeepSeek runtime with the FlashInfer DSV4 backend.

Next action:

```bash
gcloud projects add-iam-policy-binding openagentsgemini \
  --member=serviceAccount:oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com \
  --role=<compute-capable-role>
```

Use a narrowly scoped custom role if possible. At minimum it must cover the
permissions listed above, and follow-on SSH/SCP may require additional OS Login
or instance-access permissions. After the IAM grant, rerun:

```bash
GCLOUD_ACCOUNT=oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com \
bash scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh
```

Alternative: refresh the user account interactively and run the same wrapper
without `GCLOUD_ACCOUNT`.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
