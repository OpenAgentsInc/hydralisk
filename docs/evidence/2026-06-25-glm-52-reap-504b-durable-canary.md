# GLM-5.2 504B REAP durable canary

Date: 2026-06-25

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/95

Runbook:
[`docs/glm-5.2-reap-504b-g4-runbook.md`](../glm-5.2-reap-504b-g4-runbook.md)

Public-safety boundary: this note contains service shape, lifecycle status,
cost estimates, hashes, token counts, and aggregate timings only. It contains
no endpoint hostname, public IP address, private VPC address, bearer token, raw
prompt, raw response, hidden reasoning trace, provider credential, model
weight, checkpoint, compiled engine, profiler dump, or raw log.

## Result

PASS for the non-model durability control plane, with keep-warm installed and
temporarily deferred during the live #6253 benchmark. The durable choice is
explicit:

- Google Cloud still did not provide a fresh standalone 4 x G4 host in the
  original capacity attempts.
- The current admitted `g4-standard-384` host remains Spot, so it is not the
  same reliability class as on-demand or a reservation.
- The current instance no longer reports a 6-hour max-run-duration field in
  `scheduling`, but it can still be stopped by Spot preemption.
- Hydralisk therefore installed an external GCP watchdog that conditionally
  starts the VM after a STOP, plus on-host keep-warm units that can be enabled
  when the decision-grade Terminal-Bench run no longer owns the singleflight
  lane.

This is durable enough for a labeled canary lane, not an enterprise SLA. During
the official 89-task Harbor run for OpenAgents issue #6253, the benchmark owner
temporarily disabled the keep-warm timer so warm probes do not contend with or
contaminate the benchmark. Do not re-enable warm probes until that run finishes.

## What Was Armed

On the GLM host:

- raw vLLM container restart policy: `unless-stopped`
- raw vLLM profile: 4 x TP, DCP=4, 250K context, MTP-2 speculative decoding,
  default `min_p` omitted
- Docker service: enabled and active
- Hydralisk proxy service: enabled and active
- Caddy HTTPS front: enabled and active
- boot disk auto-delete: false
- keep-warm service: `hydralisk-glm52-reap-keepwarm.service`
- keep-warm timer: `hydralisk-glm52-reap-keepwarm.timer`
- keep-warm cadence when enabled: every 4 minutes, with `OnBootSec=2min`

In Google Cloud:

- Cloud Run job: `hydralisk-glm52-reap-watchdog`
- Cloud Scheduler job: `hydralisk-glm52-reap-watchdog-5m`
- Schedule: every 5 minutes, UTC
- Watchdog identity:
  `hydralisk-glm52-reap-watchdog@openagentsgemini.iam.gserviceaccount.com`
- Watchdog permissions:
  - custom project role with `compute.instances.get` and
    `compute.instances.start`
  - `roles/run.invoker` so Cloud Scheduler can execute the Cloud Run job

The watchdog first describes the target instance. If the VM is already
`RUNNING`, it exits without calling `instances.start`. If it sees any other
state, it requests a start. This avoids a noisy Scheduler job that repeatedly
fails while the instance is already up.

## Live Status Evidence

Checked after setup and after the #6253 coordination note:

- Instance status: `RUNNING`
- Machine: `g4-standard-384`
- Provisioning model: `SPOT`
- Termination action: `STOP`
- Max run duration present: false
- Automatic restart: false
- Boot disk auto-delete: false
- Boot disk size: 1500 GB
- Docker: enabled and active
- Caddy: enabled and active
- Private proxy: enabled and active
- Keep-warm units: installed
- Keep-warm timer: disabled and inactive during the live #6253 Harbor run
- Watchdog Cloud Run job: configured
- Watchdog Cloud Scheduler job: configured

Follow-up status at `2026-06-25T19:02Z`, using
`ACTION=status scripts/install-glm-52-reap-504b-durable-canary-gce.sh` without
calling the model lane:

- Instance status remained `RUNNING`.
- Health stayed `ready` for `glm-5.2-reap-504b-g4`.
- Machine, lifecycle, and disk shape were unchanged:
  `g4-standard-384`, `SPOT`, `STOP`, no `maxRunDuration`, boot disk
  auto-delete false.
- Docker, Caddy, and the Hydralisk private proxy stayed enabled and active.
- The keep-warm timer stayed disabled and inactive because the public Gym
  projection still showed the official GLM baseline Terminal-Bench run in
  progress (`19/89` completed at that check). This preserves the singleflight
  benchmark lane instead of adding warm-probe contention.
- The watchdog Cloud Run job and Cloud Scheduler job remained configured on
  the five-minute cadence.
- The latest host-local keep-warm record was `busy`/HTTP `429`, which is the
  expected fail-closed response when a singleflight benchmark request owns the
  lane; it did not reveal prompts, responses, endpoints, or tokens.

Earlier keep-warm probes showed both the useful and honest behavior:

- A busy/error warm attempt can fail without taking the timer down.
- A later keep-warm attempt passed with HTTP 200.
- Latest passing warm probe:
  - HTTP status: 200
  - Prompt tokens: 21
  - Completion tokens: 13
  - Total tokens: 34
  - Visible completion characters: 60
  - Visible completion SHA-256:
    `2e0bb12107e5e27b35801bb8f8e27c3632ea545303f599442be8f14789d9531f`

After the official Terminal-Bench run finishes, re-enable the timer with:

```bash
ENABLE_KEEPWARM_TIMER=1 ACTION=setup-keepwarm RUN_ID=<run-id> \
  scripts/install-glm-52-reap-504b-durable-canary-gce.sh
```

Then run the model-calling warm smoke only after the benchmark owner clears the
singleflight lane:

```bash
ALLOW_MODEL_KEEPWARM_SMOKE=1 ACTION=smoke RUN_ID=<run-id> \
  scripts/install-glm-52-reap-504b-durable-canary-gce.sh
```

The timer uses public-safe JSON output under the host-local Hydralisk log
directory; tracked docs keep only hashes, counts, and status.

The watchdog was also executed manually while the VM was already running. The
latest control-plane-only smoke skipped the model warm probe and executed only
the watchdog. It completed without requesting a start, proving the conditional
check path works without causing a disruptive restart:

- Execution: `hydralisk-glm52-reap-watchdog-7lqgx`
- Observed VM status: `RUNNING`
- Action: `none`
- Exit: 0

## Cost

Monthly estimate uses the same 730-hour pricing table from the canary status
evidence:

| Shape | GPUs | Spot / month | On-demand / month | DWS flex / month |
| --- | ---: | ---: | ---: | ---: |
| `g4-standard-192` | 4 | $2,696.21 | $13,139.80 | $6,570.00 |
| `g4-standard-384` | 8 | $5,392.42 | $26,279.59 | $13,140.00 |

Plain-language translation:

- The live service behaves like a 4-GPU model server because it routes one
  4-GPU TP process.
- It costs like the 8-GPU fallback host while that host is up.
- Moving to on-demand on the same 8-GPU shape would roughly quintuple the VM
  cost versus Spot.
- A true standalone 4-GPU host would be roughly half the fallback-host cost,
  but the attempted 4-GPU shapes were capacity-exhausted.
- If the 8-GPU fallback must remain allocated, the best capacity upgrade is
  still two independent 4-GPU replicas behind a router, not 8-GPU TP.

## Remaining Boundary

The lane is now self-starting after a VM start, and the warm timer is installed
for post-benchmark re-enable, but Spot can still be capacity-blocked during
re-admission. If `instances.start` is rejected because the zone has no G4
stock, the watchdog will keep retrying on the next schedule. The honest
production upgrade from here is either:

1. on-demand or reservation-backed G4 capacity, accepting the cost, or
2. a two-replica router on the current 8-GPU fallback while it remains
   admitted.

## Reproducible Operator Path

```bash
ACTION=setup RUN_ID=<run-id> \
  scripts/install-glm-52-reap-504b-durable-canary-gce.sh

ACTION=smoke RUN_ID=<run-id> \
  scripts/install-glm-52-reap-504b-durable-canary-gce.sh

ACTION=status RUN_ID=<run-id> \
  scripts/install-glm-52-reap-504b-durable-canary-gce.sh
```

The script writes ignored public-safe artifacts under `.hydralisk/`.
