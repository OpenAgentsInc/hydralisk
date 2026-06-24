# DeepSeek-V4-Flash gcloud credential authority probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/49

Script:
[`scripts/probe-deepseek-v4-gcloud-credentials.sh`](../../scripts/probe-deepseek-v4-gcloud-credentials.sh)

Profile:
[`profiles/deepseek-v4-flash-gce-preflight.json`](../../profiles/deepseek-v4-flash-gce-preflight.json)

## Why

Issue #41 is the next live DeepSeek-V4-Flash G4 smoke. The local user account
requires interactive reauthentication, while the two locally configured service
accounts can mint access tokens but previously failed compute IAM checks. This
probe inventories configured gcloud accounts and classifies each account by:

- access-token status;
- required G4 runner permissions;
- required IAM grant-authority permissions;
- the next operator action.

The probe does not write bearer tokens into evidence. It uses a temporary token
file outside the evidence directory and removes it during the run.

## Provider-card note

The pasted DeepSeek-V4-Flash provider card helps with the target recipe, but not
with the current Google blocker. It repeats the published path Hydralisk already
tracked:

- vLLM 0.20.0+ with DeepGEMM installed through vLLM's helper;
- FP8 KV cache, block size 256, expert parallelism, DeepSeek V4 tokenizer,
  reasoning, and tool-call parser flags;
- tensor parallel size must match GPU count to avoid replicated dense-layer
  OOM;
- ordinary hardware target is 8 x H100, 8 x H200, 8 x B200, GB200 NVL4, or a
  DGX Station class single-GPU lane.

Our sampled Google project still has no visible H100/H200/B200/GB200 quota for
that ordinary path. The current executable Google path is therefore still the G4
RTX PRO 6000 FlashInfer DSV4 smoke, but the local identities must be fixed
before that smoke can create a host.

## Required runner permissions

```text
compute.instances.create
compute.instances.get
compute.instances.setLabels
compute.instances.setMetadata
compute.instances.setTags
compute.disks.create
compute.subnetworks.use
```

## Required grant permissions

```text
iam.roles.create
iam.roles.update
resourcemanager.projects.get
resourcemanager.projects.getIamPolicy
resourcemanager.projects.setIamPolicy
iam.serviceAccounts.get
iam.serviceAccounts.getIamPolicy
iam.serviceAccounts.setIamPolicy
```

## Result

The real local run wrote sanitized output under:

```text
.hydralisk/deepseek-v4-gcloud-credentials-issue-49/gcloud-credentials.md
```

Summary:

```tsv
account	auth_status	runner_status	grant_status	next
chris@openagents.com	failed	skipped	skipped	refresh gcloud auth or choose a working service account
nexus-mainnet@openagentsgemini.iam.gserviceaccount.com	ok	missing_permissions	missing_permissions	use an IAM-capable account to grant runner permissions
oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com	ok	missing_permissions	missing_permissions	use an IAM-capable account to grant runner permissions
```

Both service accounts lack every required runner permission:

```text
compute.instances.create
compute.instances.get
compute.instances.setLabels
compute.instances.setMetadata
compute.instances.setTags
compute.disks.create
compute.subnetworks.use
```

Both service accounts also lack the grant-authority permissions needed to repair
themselves:

```text
iam.roles.create
iam.roles.update
resourcemanager.projects.getIamPolicy
resourcemanager.projects.setIamPolicy
iam.serviceAccounts.get
iam.serviceAccounts.getIamPolicy
iam.serviceAccounts.setIamPolicy
```

## Decision

No configured local gcloud account can currently run issue #41 or grant the
permissions needed to run it. This is not DeepSeek runtime evidence and not GPU
capacity evidence; it is an IAM/auth admission blocker before any GCE create
attempt.

The next real operator step is one of:

```bash
gcloud auth login
gcloud auth application-default login
bash scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh
```

or:

```bash
APPLY=1 bash scripts/plan-deepseek-v4-g4-iam-grant.sh
GCLOUD_ACCOUNT=oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com \
bash scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh
```

The second path must be run by an IAM-capable account.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
