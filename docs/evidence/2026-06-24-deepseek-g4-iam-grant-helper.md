# DeepSeek-V4-Flash G4 IAM grant helper

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/47

Related live GPU issue: https://github.com/OpenAgentsInc/hydralisk/issues/41

## Summary

Issue #47 added a public-safe IAM grant helper for the service-account route
to the DeepSeek G4 smoke.

New files:

- `deploy/gce/deepseek-v4-g4-runner-role.yaml`
- `scripts/plan-deepseek-v4-g4-iam-grant.sh`

The helper prints a plan by default and does not mutate IAM unless `APPLY=1` is
set. It creates or updates a custom project role, binds it to
`oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com`, adds OS Login,
adds `roles/iam.serviceAccountUser` on the default Compute Engine service
account, and then prints the canonical DeepSeek G4 rerun command.

## Default command

```bash
bash scripts/plan-deepseek-v4-g4-iam-grant.sh
```

## Apply command

Run this only from a shell with project IAM authority:

```bash
APPLY=1 bash scripts/plan-deepseek-v4-g4-iam-grant.sh
```

The helper does not run the DeepSeek smoke automatically. After the IAM grant,
run:

```bash
GCLOUD_ACCOUNT=oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com \
bash scripts/probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh
```

## Plain-English read

Hydralisk now has the exact service-account IAM plan needed to unblock issue
#41's service-account route. This still has not created a G4 instance. It gives
an IAM-capable operator a safe, repeatable command path.

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
