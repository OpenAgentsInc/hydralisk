# DeepSeek-V4-Flash alternate Google credential probe

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/50

Script:
[`scripts/probe-deepseek-v4-google-alt-credentials.sh`](../../scripts/probe-deepseek-v4-google-alt-credentials.sh)

Profile:
[`profiles/deepseek-v4-flash-gce-preflight.json`](../../profiles/deepseek-v4-flash-gce-preflight.json)

## Why

Issue #49 proved that the configured `gcloud auth list` accounts cannot run or
grant the issue #41 DeepSeek G4 smoke. This probe checks the nearby credential
surfaces that are easy to miss:

- Application Default Credentials via
  `gcloud auth application-default print-access-token`;
- any configured `auth/impersonate_service_account` value;
- explicit `IMPERSONATE_ACCOUNTS` targets supplied by the operator.

The probe uses the same DeepSeek G4 runner and grant-authority permission sets
as issue #49. It does not write bearer tokens, refresh tokens, credential file
contents, prompts, responses, weights, or hidden reasoning into evidence.

## Real local run

Command shape:

```bash
IMPERSONATE_ACCOUNTS='oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com,nexus-mainnet@openagentsgemini.iam.gserviceaccount.com' \
OUTPUT_DIR=.hydralisk/deepseek-v4-google-alt-credentials-issue-50 \
bash scripts/probe-deepseek-v4-google-alt-credentials.sh
```

Sanitized output was written under:

```text
.hydralisk/deepseek-v4-google-alt-credentials-issue-50/google-alt-credentials.md
```

Summary:

```tsv
surface	identity	token_status	runner_status	grant_status	next
adc	application-default	failed	skipped	skipped	refresh ADC with gcloud auth application-default login or use another credential surface
explicit_impersonation	oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com	failed	skipped	skipped	grant serviceAccountTokenCreator or use a credential that can mint this impersonated token
explicit_impersonation	nexus-mainnet@openagentsgemini.iam.gserviceaccount.com	failed	skipped	skipped	grant serviceAccountTokenCreator or use a credential that can mint this impersonated token
```

No configured impersonation target was returned by local gcloud config.

## Decision

No alternate Google credential surface available in this shell can run issue
#41, grant issue #41 runner permissions, or mint an impersonated token for the
two known service accounts.

The DeepSeek G4 path is now externally blocked on one of these operator actions:

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

The second path must be run from an IAM-capable account.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
