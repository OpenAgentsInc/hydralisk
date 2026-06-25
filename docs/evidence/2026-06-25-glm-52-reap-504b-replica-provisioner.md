# GLM-5.2 504B REAP replica provisioner

Date: 2026-06-25

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/97

Script:
[`scripts/provision-glm-52-reap-504b-replica-gce.sh`](../../scripts/provision-glm-52-reap-504b-replica-gce.sh)

Runbook:
[`docs/glm-5.2-reap-504b-g4-runbook.md`](../glm-5.2-reap-504b-g4-runbook.md)

Public-safety boundary: this note describes an operator workflow, public-safe
resource refs, and reduced evidence paths only. It contains no endpoint
hostname, public IP address, private VPC address, bearer token, prompt,
response, hidden reasoning trace, provider credential, model weight,
checkpoint, compiled engine, raw log, or profiler dump.

## Result

PASS for a repeatable operator path. Hydralisk now has a single command wrapper
for adding a GLM-5.2 REAP replica without touching an active benchmark lane.
The wrapper defaults to `ACTION=plan`, which emits a public-safe plan and
cleanup instructions without creating Google Cloud resources.

The live `ACTION=run` path composes the already-proven primitives:

1. admit a new G4 host with `scripts/probe-glm-52-reap-504b-g4-gce.sh`;
2. stage the model by either same-zone cloned disk (`STAGING_MODE=clone-disk`)
   or fresh download (`STAGING_MODE=download`);
3. verify the cloned disk is mounted read-only and the safetensors index exists;
4. launch the b12x/vLLM 4 x TP, MTP-2, no-`min_p`, 250K profile;
5. install the bearer-gated private proxy with replica routing metadata;
6. expose the authenticated public HTTPS origin;
7. install distinct watchdog and keep-warm resources;
8. emit a public-safe evidence bundle and cleanup plan.

## Safety Guards

- `DO_NOT_TOUCH_EXISTING_LANES=1` is required.
- `REPLICA_REF` may not be the primary benchmark ref
  `glm52-reap-primary-g4-tp4`.
- The generated watchdog service account, custom role, Cloud Run job, and Cloud
  Scheduler job include a short hash of `REPLICA_REF` and are rejected if they
  collide with the primary lane names.
- The output evidence directory must not already exist for `ACTION=run`.
- A cloned model disk must not already exist.
- Raw endpoint values, IPs, bearer tokens, prompts, responses, weights, and raw
  logs stay out of tracked evidence.

## Dry Plan Smoke

Command:

```bash
RUN_ID=issue97-plan2 \
REPLICA_REF=glm52-reap-replica-test \
SOURCE_MODEL_DISK=operator-source-model-disk \
ACTION=plan \
OUTPUT_DIR=.hydralisk/issue97-plan2 \
  scripts/provision-glm-52-reap-504b-replica-gce.sh
```

Public-safe output included:

- `replicaRef`: `glm52-reap-replica-test`
- `replicaProfileRef`: `glm-reap-504b-g4-tp4-mtp2-rp105`
- `stagingMode`: `clone-disk`
- `sourceModelDiskRef`: `operator-provided-same-zone-disk`
- `watchdogRunJob`: `hydralisk-glm52-reap-watchdog-glm52-r-6ac9f0`
- `watchdogSchedulerJob`: `hydralisk-glm52-reap-watchdog-glm52-r-6ac9f0-5m`
- `watchdogServiceAccount`: `hydra-glm52-wd-glm52-r-6ac9f0`

The dry plan made no GCE calls and created only ignored local evidence under
`.hydralisk/`.

## Operator Examples

Plan a replica using a same-zone model-disk clone:

```bash
REPLICA_REF=glm52-reap-replica-c \
SOURCE_MODEL_DISK=<operator-known-source-disk-name> \
ACTION=plan \
  scripts/provision-glm-52-reap-504b-replica-gce.sh
```

Run the full bring-up:

```bash
REPLICA_REF=glm52-reap-replica-c \
SOURCE_MODEL_DISK=<operator-known-source-disk-name> \
RUN_MODEL_SMOKES=1 \
ALLOW_MODEL_KEEPWARM_SMOKE=1 \
ACTION=run \
  scripts/provision-glm-52-reap-504b-replica-gce.sh
```

Use `RUN_MODEL_SMOKES=0` and `ALLOW_MODEL_KEEPWARM_SMOKE=0` when a benchmark or
operator reservation should avoid model-calling smokes.

## Cleanup Boundary

The generated cleanup plan is intentionally scoped to resources named by the
replica evidence bundle:

- the new GCE instance, if created by the run;
- the cloned model disk, if `STAGING_MODE=clone-disk`;
- the reserved address, firewall rule, and tag, if created by the run and not
  reused by another replica;
- the replica-specific Cloud Scheduler job, Cloud Run job, custom role, and
  service account.

Do not delete or restart the primary GLM lane, active Harbor benchmark lanes,
or resources whose names are not present in the evidence bundle.

## Validation

- `bash -n scripts/provision-glm-52-reap-504b-replica-gce.sh`
- `ACTION=plan ... scripts/provision-glm-52-reap-504b-replica-gce.sh`
- `uv run --extra dev pytest`: 120 passed, with one upstream
  Starlette/httpx deprecation warning.
