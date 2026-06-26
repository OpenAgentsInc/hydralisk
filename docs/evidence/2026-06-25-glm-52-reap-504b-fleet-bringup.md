# GLM-5.2 504B REAP multi-replica fleet bring-up

Date: 2026-06-25

Follow-on to: hydralisk #95 (single second endpoint) and #99 (prebake weights)

Public-safety boundary: this note contains cloud allocation, hardware shapes,
per-zone admission results, aggregate timings, and endpoint-count metadata
only. It contains no public origin hostnames, public IP addresses, bearer
tokens, raw prompts, raw responses, hidden reasoning traces, provider
credentials, weights, checkpoints, compiled engines, or raw logs. The concrete
per-replica origins and bearer tokens live only in the gitignored operator
secrets file `~/work/.secrets/hydralisk-glm-endpoints.env`.

## Goal

Bring up as many durable GLM-5.2-REAP-504B serving replicas as Google Cloud
will grant right now, make each HTTPS-reachable + health-checked + serving an
OpenAI-compatible `/v1/chat/completions`, and publish the roster for the Khala
arming agent. Prior state was a single borrowed-Spot canary; this moves the
lane to a multi-region replica pool.

## Result

PASS. **10 healthy GLM-5.2-REAP-504B replicas serving** across **5 regions**,
each verified with a real authenticated completion (HTTP 200) over its public
HTTPS origin. Total **44 x NVIDIA RTX PRO 6000** (G4) GPUs.

| Region | Hosts | GPUs |
| --- | ---: | ---: |
| us-central1 | 2 (4x + 8x) | 16 |
| us-east1 | 2 (4x) | 8 |
| us-east5 | 3 (4x) | 12 |
| us-south1 | 1 (4x) | 4 |
| us-west1 | 1 (4x) | 4 |
| **Total** | **10** | **44** |

Two of the ten are the pre-existing baseline hosts (the reserved 4x canary and
the 8x host); eight are newly admitted on this run. Every host serves the same
pinned `0xSero/GLM-5.2-504B` REAP/NVFP4 checkpoint on the b12x MTP-2 profile.

Each replica: raw vLLM on `127.0.0.1:8000`, a bearer-gated singleflight
Hydralisk proxy bound to the VM private interface on `:8080`, and a Caddy
HTTPS origin (`https://hydralisk-glm52-reap-504b.<ip>.sslip.io`) fronted by a
per-host static IP and an 80/443 firewall rule scoped by a per-host tag.

## GCP capacity: granted vs refused

All capacity was requested as **Spot 4x `g4-standard-192`** (RTX PRO 6000)
first, with an on-demand and an 8x fallback. The honest per-attempt result:

Admitted and now serving (Spot 4x): `us-central1-f`, `us-east1-b`,
`us-east1-d`, `us-east5-a`, `us-east5-b`, `us-east5-c`, `us-south1-b`,
`us-west1-a`.

Refused / lost:

| Attempt | Shape | Result |
| --- | --- | --- |
| `us-central1-b` Spot 4x | `g4-standard-192` | admitted then Spot-preempted within ~minutes (deleted) |
| `us-central1-f` Spot 8x | `g4-standard-384` | `QUOTA_EXCEEDED` (regional RTX PRO 6000 quota ceiling) |
| `us-east4-b/-c` Spot 4x | `g4-standard-192` | `ZONE_RESOURCE_POOL_EXHAUSTED` (stockout, both passes) |
| `us-west1-b`, `us-west3-a`, `us-west4-a/-c` Spot 4x | `g4-standard-192` | `ZONE_RESOURCE_POOL_EXHAUSTED` (stockout) |
| `us-south1-a` Spot 4x | `g4-standard-192` | admitted but host became SSH-unreachable mid-stage; deleted (redundant with `us-south1-b`) |
| `us-central1-b` on-demand 4x | `g4-standard-192` | not retried after the regional quota signal |

Plain read: GCP had ample **Spot** 4x G4 across central1-f, east1, east5,
south1, and west1; us-east4 and most of us-west were stocked out at request
time; and us-central1 hit its **regional RTX PRO 6000 quota** at 16 GPUs (the
8x add was refused). On-demand 4x G4 has been stockout-prone for this project
in prior runs and was not the path that produced capacity here.

## Per-replica performance

On-host warmed proxy benchmark (prior evidence,
`2026-06-25-glm-52-reap-504b-second-endpoint.md`): ~0.28s TTFT and ~47
completion tok/s including TTFT for 160-token streams on a 4x replica.

Client-perceived spot check over the public HTTPS origin from an off-cloud
operator laptop (includes WAN round-trips, so slower than on-host):

| Replica | TTFT | decode tok/s (client) |
| --- | ---: | ---: |
| us-central1-f | 0.55s | 27.0 |
| us-east5-a | 0.72s | 23.6 |
| us-south1-b | 0.53s | 25.2 |

The decode rate seen by a client far from the region is dominated by
per-token streaming WAN latency; Khala routing from in-cloud or in-region
callers will see numbers closer to the ~47 tok/s on-host figure. Every replica
in the roster returned a correct completion (HTTP 200) at collection time.

## Durability

Within the Spot constraint, the lane now has:

- **Multi-region spread (5 regions, 10 hosts).** A single zone or region Spot
  reclaim cannot take the lane down; Khala routes across the replica pool.
- **Boot disks preserved on STOP** on every host (`--no-auto-delete`), so a
  Spot STOP keeps the ~288 GB staged model on disk and a restart skips
  re-staging.
- **Docker `--restart unless-stopped`** on each vLLM container, so an in-place
  vLLM crash self-heals.
- **systemd-managed private proxy** and **enabled Caddy** per host.

Residual risk, stated honestly:

- All serving hosts are **Spot**. Google can reclaim any of them at any time.
  Spot does not guarantee re-admission after preemption.
- These new hosts do **not** yet have the per-host Cloud Scheduler -> Cloud Run
  STOP-watchdog or keep-warm timers that the original two canaries have. After
  a STOP they need a manual or future-watchdog restart; the boot disk is
  preserved so that restart is fast, but it is not yet automatic.
- us-central1 is at its **regional RTX PRO 6000 quota** (16 GPUs). Widening the
  central1 footprint or adding 8x hosts there needs a quota increase.
- Re-staging any host that loses its disk pays the full ~288 GB Hugging Face
  download; #99 tracks prebaking weights into an image to remove that.

## Tooling added this run

- `scripts/admit-glm-52-reap-504b-fleet-gce.sh` — multi-zone admission sweep.
- `scripts/arm-glm-52-reap-504b-replica-gce.sh` — end-to-end per-host arming
  (stage, launch, proxy, HTTPS, smoke) with per-replica TMPDIR isolation.
- `scripts/arm-glm-52-reap-504b-fleet-driver.sh` — parallel arm driver that
  waits on all children.
- `scripts/collect-glm-52-reap-504b-roster.sh` — writes the secret-free-stdout
  roster to the gitignored operator secrets file.

## Network prerequisite

The newly admitted no-external-IP hosts had no outbound internet until a Cloud
Router + Cloud NAT was created in each new region (`us-east1`, `us-east5`,
`us-south1`, `us-west1`), matching the pre-existing `us-central1` default-network
NAT. Without NAT, host bootstrap (apt, Hugging Face, NVIDIA repo) fails with
"Network is unreachable".

## Roster handoff

`scripts/collect-glm-52-reap-504b-roster.sh` wrote **10** healthy replicas to
`~/work/.secrets/hydralisk-glm-endpoints.env` (gitignored, mode 600). Each
record carries `REPLICA_<n>_{ID,BASE_URL,MODEL_ID,BEARER,ZONE,MACHINE,
PROVISIONING}` plus `REPLICA_COUNT`. This is the file the OpenAgents Khala
arming agent reads. Bearer tokens are written to that file only and were never
printed to stdout, logs, or this evidence.

## Claim boundary

Admitted: 10 multi-region GLM-5.2-REAP-504B serving replicas, each HTTPS,
bearer-gated, singleflight, health-checked, and serving an OpenAI-compatible
chat endpoint, with preserved Spot boot disks and a secret-free roster for
arming.

Not admitted: a public product SLA; on-demand or reservation-backed G4
capacity; multi-request concurrency within a single replica (each stays
singleflight with 429 backpressure); automatic STOP-recovery watchdogs on the
eight new hosts; Worker-side Khala arming, billing, settlement, or any public
product promise.
