# DeepSeek-V4-Flash gcloud account override

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/45

Related live GPU issue: https://github.com/OpenAgentsInc/hydralisk/issues/41

## Summary

Issue #45 added an explicit `GCLOUD_ACCOUNT` override to the DeepSeek G4 smoke
wrappers so a single run can use a service account from the local Cloud SDK
auth store without changing the user's active gcloud config.

The override is used for local gcloud operations through `CLOUDSDK_CORE_ACCOUNT`
and is rendered in public evidence. Access-token output is discarded and never
committed.

Affected scripts:

- `scripts/probe-deepseek-v4-b12x-g4-gce.sh`
- `scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh`
- `scripts/probe-deepseek-v4-provider-stack-gce.sh`

## Service-account probes

Two locally configured service accounts can pass the auth preflight:

```text
nexus-mainnet@openagentsgemini.iam.gserviceaccount.com -> auth preflight ok
oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com -> auth preflight ok
```

Both accounts can list matching Hydralisk DeepSeek instances, but neither can
create the planned G4 probe hosts:

```tsv
account	machine	accelerator	gpu_count	zone	status	blocker
oa-vertex-inference	g4-standard-384	nvidia-rtx-pro-6000	8	us-central1-b	blocked_iam	missing compute.instances.create
oa-vertex-inference	g4-standard-192	nvidia-rtx-pro-6000	4	us-central1-b	blocked_iam	missing compute.instances.create
nexus-mainnet	g4-standard-384	nvidia-rtx-pro-6000	8	us-central1-b	blocked_iam	missing compute.instances.create
nexus-mainnet	g4-standard-192	nvidia-rtx-pro-6000	4	us-central1-b	blocked_iam	missing compute.instances.create
```

## Plain-English read

This moves us one step past the user-account reauth blocker: service-account
token minting works noninteractively. The next blocker is IAM, not DeepSeek and
not Google GPU capacity. The service account used for the run needs permission
to create the private G4 probe VM, plus whatever follow-on permissions are
needed for SSH/SCP, service-account attachment, labels, disks, and networking.

Immediate next options:

1. Grant an appropriate compute role to one of the existing service accounts
   for the `openagentsgemini` project, then rerun:

   ```bash
   GCLOUD_ACCOUNT=oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com \
   bash scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh
   ```

2. Or refresh the user account interactively and run:

   ```bash
   gcloud auth login
   gcloud auth application-default login
   bash scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh
   ```

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
