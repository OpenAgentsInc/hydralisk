# GLM-5.2 504B REAP replica routing metadata

Date: 2026-06-25

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/96

Runbook:
[`docs/glm-5.2-reap-504b-g4-runbook.md`](../glm-5.2-reap-504b-g4-runbook.md)

Public-safety boundary: this note describes public-safe JSON fields and
operator-visible refs only. It contains no endpoint hostname, public IP
address, private VPC address, bearer token, prompt, response, hidden reasoning
trace, provider credential, model weight, checkpoint, compiled engine, raw log,
or profiler dump.

## Result

PASS. Hydralisk now exposes a stable public-safe replica view for Khala pool
routing on both:

- `GET /hydralisk/v1/capabilities`
- `GET /hydralisk/v1/metrics`

The static capability surface now includes:

```json
{
  "replica": {
    "replicaRef": "glm52-reap-primary-g4-tp4",
    "profileRef": "glm-reap-504b-g4-tp4-mtp2-rp105",
    "routing": {
      "draining": false,
      "reserved": false,
      "reservationRef": "optional-public-safe-ref"
    },
    "lifecycle": {
      "provisioningClass": "spot",
      "maxRunDurationPresent": false,
      "watchdog": {
        "watchdogRef": "hydralisk-glm52-reap-watchdog-5m",
        "status": "configured"
      }
    }
  }
}
```

The dynamic metrics surface includes the same replica identity plus route
eligibility:

```json
{
  "replica": {
    "capacity": {
      "inflight": 0,
      "maxInflight": 1,
      "queueTimeoutSeconds": 0.0,
      "singleFlight": true,
      "backpressure": {
        "busy": false,
        "busyRejectsTotal": 0,
        "lastBusyAt": null,
        "lastBusyStatus": null
      }
    },
    "warmState": {
      "configured": true,
      "statusPathRef": "host-local-public-json",
      "lastKeepWarmAt": "2026-06-25T18:00:00Z",
      "lastKeepWarmStatus": "passed",
      "lastKeepWarmHttpStatus": 200,
      "lastKeepWarmWallSeconds": 0.428,
      "lastKeepWarmTokens": {
        "promptTokens": 21,
        "completionTokens": 4,
        "totalTokens": 25
      }
    }
  }
}
```

## Khala Routing Read

Khala can treat a replica as route-eligible when:

- health is `ready`;
- `replica.routing.draining` is false;
- `replica.routing.reserved` is false, or the reservation matches the caller's
  benchmark/operator lane;
- `replica.capacity.inflight < replica.capacity.maxInflight`;
- `replica.capacity.backpressure.busy` is false;
- `replica.warmState.lastKeepWarmStatus` is `passed` or the operator has
  intentionally paused keep-warm for an active benchmark.

The important singleflight signal is `maxInflight=1`: under the current GLM
profile, one warmed 4 x G4 replica is one fast interactive slot. A second
developer needs a second replica, not a second request on the same endpoint.

## Implementation

New public-safe settings:

- `HYDRALISK_REPLICA_REF`
- `HYDRALISK_REPLICA_PROFILE_REF`
- `HYDRALISK_REPLICA_DRAINING`
- `HYDRALISK_REPLICA_RESERVED`
- `HYDRALISK_REPLICA_RESERVATION_REF`
- `HYDRALISK_PROVISIONING_CLASS`
- `HYDRALISK_MAX_RUN_DURATION_PRESENT`
- `HYDRALISK_WATCHDOG_REF`
- `HYDRALISK_WATCHDOG_STATUS`
- `HYDRALISK_WATCHDOG_CHECKED_AT`
- `HYDRALISK_KEEPWARM_STATUS_PATH`

The GLM private-proxy installer now sets defaults for the primary replica and
points `HYDRALISK_KEEPWARM_STATUS_PATH` at the host-local public JSON emitted by
the durable keep-warm unit. The proxy reads only the public summary fields from
that file and does not export prompt hashes, completion hashes, endpoint paths,
or raw log text.

## Validation

- `bash -n scripts/expose-glm-52-reap-504b-private-proxy-gce.sh`
- `python3 -m compileall -q hydralisk/serve`
- `uv run --extra dev pytest`: 120 passed, with one upstream
  Starlette/httpx deprecation warning.

