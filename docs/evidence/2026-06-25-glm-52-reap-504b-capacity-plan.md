# GLM-5.2 504B REAP multi-replica capacity plan

Date: 2026-06-25

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/98

Capacity model:
[`scripts/plan-glm-52-reap-504b-capacity.py`](../../scripts/plan-glm-52-reap-504b-capacity.py)

Public-safety boundary: this note contains public-safe resource shape,
catalog/quota observations, prices already recorded in Hydralisk evidence, and
aggregate capacity math only. It contains no endpoint hostname, public IP
address, private VPC address, bearer token, prompt, response, hidden reasoning
trace, provider credential, model weight, checkpoint, compiled engine, raw log,
or profiler dump.

## Result

The reliable product unit should be modeled as:

> one warmed 4 x G4 GLM replica = one fast interactive slot

Do not model one endpoint as many simultaneous developers. The measured second
endpoint keeps one request near `0.281s` TTFT and about `46.7` completion
tokens/s including TTFT on the 160-token streaming case, while same-endpoint
concurrency rejects the second request with HTTP 429. Multiple developers need
multiple replicas behind Khala pool routing.

## Live Read-Only GCE Check

Checked with read-only `gcloud` commands on 2026-06-25:

- Running GLM hosts:
  - `g4-standard-384` Spot in `us-central1-b`, termination `STOP`, no
    `maxRunDuration` field reported.
  - `g4-standard-192` Spot in `us-central1-b`, termination `STOP`,
    `maxRunDuration=604800s`.
- G4 catalog visibility:
  - `nvidia-rtx-pro-6000` visible in `us-central1-b`.
  - `nvidia-rtx-pro-6000` visible in `us-central1-f`.
  - `g4-standard-192` and `g4-standard-384` machine types visible in both
    `us-central1-b` and `us-central1-f`.
- Regional quota signal:
  - `CPUS`: limit `3000`, usage `47`.
  - `PREEMPTIBLE_CPUS`: limit `5000`, usage `0`.
  - The `regions describe us-central1` quota output did not expose a named
    RTX PRO 6000 / G4 GPU quota metric.
- Current GLM storage footprint in the project:
  - two 1500 GB Hyperdisk Balanced boot disks;
  - one 1500 GB Hyperdisk Balanced cloned model disk;
  - total visible GLM disk allocation: about 4.5 TB before future replicas.

Interpretation: catalog and CPU quota are not the blocker. The earlier
capacity failures were zonal resource-pool stockout (`ZONE_RESOURCE_POOL_...`),
which cannot be ruled out by quota reads. A real admission attempt is the only
honest stockout probe, and it can allocate spend if it succeeds.

## Cost Scenarios

Assumptions:

- one slot = one `g4-standard-192` replica with 4 x RTX PRO 6000;
- per-slot decode throughput = `46.7` output tokens/s;
- month = 730 hours;
- costs below are VM only, before storage, egress, logging, Cloud Run/Scheduler,
  public IP, warm probes, and operator time;
- token-cost rows use output/decode tokens only, not long-prefill wall time.

| Slots | Replicas | GPUs | Pricing | VM $/mo | $/M output tok @10% | @25% | @50% | Reliability |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2 | 2 | 8 | Spot | $5,392.42 | $219.69 | $87.88 | $43.94 | cheap, interruptible |
| 2 | 2 | 8 | DWS Flex-start | $13,140.00 | $535.33 | $214.13 | $107.07 | queued/flexible start |
| 2 | 2 | 8 | On-demand | $26,279.60 | $1,070.65 | $428.26 | $214.13 | durable only if stock exists |
| 2 | 2 | 8 | 1-year CUD/reserved stock | $18,133.20 | $738.76 | $295.50 | $147.75 | procured capacity |
| 2 | 2 | 8 | 3-year CUD/reserved stock | $11,559.98 | $470.96 | $188.38 | $94.19 | procured capacity |
| 4 | 4 | 16 | Spot | $10,784.84 | $219.69 | $87.88 | $43.94 | cheap, interruptible |
| 4 | 4 | 16 | DWS Flex-start | $26,280.00 | $535.33 | $214.13 | $107.07 | queued/flexible start |
| 4 | 4 | 16 | On-demand | $52,559.20 | $1,070.65 | $428.26 | $214.13 | durable only if stock exists |
| 4 | 4 | 16 | 1-year CUD/reserved stock | $36,266.40 | $738.76 | $295.50 | $147.75 | procured capacity |
| 4 | 4 | 16 | 3-year CUD/reserved stock | $23,119.96 | $470.96 | $188.38 | $94.19 | procured capacity |
| 8 | 8 | 32 | Spot | $21,569.68 | $219.69 | $87.88 | $43.94 | cheap, interruptible |
| 8 | 8 | 32 | DWS Flex-start | $52,560.00 | $535.33 | $214.13 | $107.07 | queued/flexible start |
| 8 | 8 | 32 | On-demand | $105,118.40 | $1,070.65 | $428.26 | $214.13 | durable only if stock exists |
| 8 | 8 | 32 | 1-year CUD/reserved stock | $72,532.80 | $738.76 | $295.50 | $147.75 | procured capacity |
| 8 | 8 | 32 | 3-year CUD/reserved stock | $46,239.92 | $470.96 | $188.38 | $94.19 | procured capacity |

Why the per-token cost does not change by slot count in this table: this is a
linear pool model. Twice the replicas means twice the spend and twice the
maximum decode token capacity, so unit cost stays the same at the same
utilization. The business question is utilization and reliability, not whether
four replicas are magically cheaper than two.

## Storage Placeholder

The current clone path allocates about 1.5 TB of Hyperdisk Balanced per
boot/model disk. A clone-staged replica can add another 1.5 TB model disk unless
the model lives on the boot disk. The VM tables above intentionally exclude this
because storage price should come from the active Cloud Billing SKU at purchase
time.

Planning placeholder:

- current visible GLM disks: about 4.5 TB;
- additional clone-staged replica: add about 1.5 TB;
- track storage as `hyperdisk_balanced_gb_month * provisioned_gb` in owner cost
  analytics, separate from VM cost profile refs.

## Procurement Recommendation

1. Keep Spot replicas as the cheap burst/dev lane, but label them honestly as
   interruptible. Watchdogs reduce manual restart pain; they do not guarantee
   zonal re-admission.
2. For reliable Khala developer traffic, procure at least two durable 4 x G4
   slots via reservation/CUD or DWS Flex-start. Two slots are the minimum
   practical pool: one busy developer should not block all GLM traffic.
3. Do not buy an 8 x tensor-parallel lane for speed by default. The measured
   capacity win is two independent 4 x replicas, not one 8 x TP process.
4. Scale to 4 and 8 slots only after Khala routing consumes the `replica`
   health/capacity metadata from #96 and owner analytics can show per-replica
   utilization, 429/busy rejects, warm/cold state, and cost per accepted
   outcome.
5. If on-demand stockout continues in `us-central1-b`/`f`, use DWS/reservation
   procurement for the durable floor and Spot for overflow.

## Visibility Needed In Khala / Owner Analytics

Each served request should be attributable to:

- `replicaRef` and `profileRef`;
- `provisioningClass` (`spot`, `on_demand`, `dws`, `reservation`);
- `costProfileRef`;
- prompt, completion, and total token counts;
- wall time, TTFT, and completion tokens/s;
- route result: served, queued, busy/429, failed, fallback;
- benchmark reservation/draining state;
- accepted outcome / verifier result when available.

Owner views should roll this up by hour/day/month:

- VM cost burn by cost profile;
- storage cost placeholder by disk class and GB-month;
- tokens served per dollar;
- accepted outcomes per dollar;
- warm-idle burn when no traffic is using a replica;
- stockout/preemption events and watchdog recovery time.

## Non-Disruptive Probe Path

Read-only probes are safe:

```bash
gcloud compute instances list --filter='name~hydralisk-glm52-reap-504b'
gcloud compute accelerator-types list --filter='name=nvidia-rtx-pro-6000'
gcloud compute machine-types list --filter='name=(g4-standard-192 OR g4-standard-384)'
gcloud compute regions describe us-central1
```

Actual stockout probes require an admission attempt and can create a billable
VM if they succeed. Use the #97 provisioner with an explicit budget/operator
window:

```bash
REPLICA_REF=glm52-reap-replica-c \
PROVISIONING_MODEL=STANDARD \
STAGING_MODE=download \
ACTION=plan \
  scripts/provision-glm-52-reap-504b-replica-gce.sh
```

Switch to `ACTION=run` only when the owner wants the VM admitted if capacity is
available.

## Validation

- Read-only `gcloud` catalog/current-resource checks listed above.
- `scripts/plan-glm-52-reap-504b-capacity.py --slots 2,4,8`

