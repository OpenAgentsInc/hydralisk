# DeepSeek-V4-Flash G4 grant-authority preflight

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/48

Related live GPU issue: https://github.com/OpenAgentsInc/hydralisk/issues/41

## Summary

Issue #48 added a grant-authority preflight to
`scripts/plan-deepseek-v4-g4-iam-grant.sh`.

The helper still prints the IAM grant plan by default. When `APPLY=1` is set,
it now checks that the current gcloud account can apply the grant before it
creates/updates the custom role or adds any IAM bindings.

## Permissions checked

Project-level grant authority:

```text
iam.roles.create
iam.roles.update
resourcemanager.projects.get
resourcemanager.projects.getIamPolicy
resourcemanager.projects.setIamPolicy
```

Default Compute Engine service-account policy authority:

```text
iam.serviceAccounts.get
iam.serviceAccounts.getIamPolicy
iam.serviceAccounts.setIamPolicy
```

## Live checks

Active user account:

```text
blocked_grant_auth
```

The active `chris@openagents.com` gcloud account still requires interactive
reauthentication and cannot resolve the default Compute Engine service account.

Service-account override:

```bash
GCLOUD_ACCOUNT=oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com \
APPLY=1 \
bash scripts/plan-deepseek-v4-g4-iam-grant.sh
```

Result:

```text
blocked_grant_iam
```

Missing grant permissions:

```text
iam.roles.create
iam.roles.update
resourcemanager.projects.getIamPolicy
resourcemanager.projects.setIamPolicy
iam.serviceAccounts.get
iam.serviceAccounts.getIamPolicy
iam.serviceAccounts.setIamPolicy
```

No IAM mutation was performed in either check.

## Plain-English read

Hydralisk now has a safe IAM grant helper, but the accounts available from this
shell still cannot apply the grant. The active user account needs interactive
reauth, or an IAM-capable account must run:

```bash
APPLY=1 bash scripts/plan-deepseek-v4-g4-iam-grant.sh
```

Then rerun:

```bash
GCLOUD_ACCOUNT=oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com \
bash scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh
```

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
