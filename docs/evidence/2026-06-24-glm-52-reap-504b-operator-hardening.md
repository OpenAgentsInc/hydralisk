# GLM-5.2 504B REAP operator hardening

Date: 2026-06-24

Issue: https://github.com/OpenAgentsInc/hydralisk/issues/91

Profile:
[`profiles/glm-5.2-reap-504b-b12x-g4.json`](../../profiles/glm-5.2-reap-504b-b12x-g4.json)

Raw launcher:
[`scripts/launch-glm-52-reap-504b-b12x-gce.sh`](../../scripts/launch-glm-52-reap-504b-b12x-gce.sh)

Private proxy helper:
[`scripts/expose-glm-52-reap-504b-private-proxy-gce.sh`](../../scripts/expose-glm-52-reap-504b-private-proxy-gce.sh)

Public-safety boundary: this packet contains service metadata, run IDs,
hashes, token counts, aggregate timings, GPU memory snapshots, disk lifecycle
snapshots, and public-safe service status only. It contains no bearer token,
model-provider credentials, raw prompts, raw responses, private source, hidden
reasoning traces, weights, checkpoints, compiled engines, profiler dumps, or
raw model logs.

## Result

PASS. The GLM-5.2 504B REAP private lane now has an operator-ready serving
path on the admitted G4 fallback host:

- Raw vLLM container restart policy: `unless-stopped`
- Raw vLLM bind: `127.0.0.1:8000`
- Private proxy unit: `hydralisk-glm52-reap-private-proxy.service`
- Private proxy restart policy: systemd `Restart=on-failure`
- Private proxy bind: `127.0.0.1:8080`
- Public bind: false
- Auth: bearer required for `/v1/models` and generation routes
- Public-safe observability: `/health`, `/hydralisk/v1/metrics`,
  `/hydralisk/v1/capabilities`, and receipt lookup
- Durable model path: `/opt/hydralisk/models/glm-5.2-504b`
- Durable Hugging Face cache path: `/var/lib/hydralisk/huggingface`
- Durable proxy state and receipts: `/var/lib/hydralisk/glm52-reap-private-proxy`
- Log root: `/var/log/hydralisk`

## Live validation

Raw vLLM was relaunched once from the existing durable model directory because
the pre-hardening container had been started with `--rm`; Docker refused to
change its restart policy in place while `AutoRemove` was enabled.

Raw vLLM restart:

- Run ID: `20260624234859`
- Command: `ACTION=start scripts/launch-glm-52-reap-504b-b12x-gce.sh`
- Restart policy after relaunch: `unless-stopped`
- `/v1/models` ready at: `2026-06-24T23:52:54Z`
- Active GPUs: `0,1,2,3`
- GPU memory at readiness: `93397 MiB` used on each active GPU
- Idle GPUs in fallback host: `4,5,6,7`
- Disk snapshot: `/dev/root` `1.5T`, `338G` used, `1.1T` available
- Model directory: `/opt/hydralisk/models/glm-5.2-504b`
- HF cache directory: `/var/lib/hydralisk/huggingface`

Private proxy service install:

- Run ID: `20260624235325`
- Command: `ACTION=install-systemd scripts/expose-glm-52-reap-504b-private-proxy-gce.sh`
- Started at: `2026-06-24T23:53:41Z`
- Unit: `hydralisk-glm52-reap-private-proxy.service`
- Unit state: enabled and active
- Process mode after install: systemd, not PID-file process mode

Private proxy status:

- Checked at: `2026-06-25T00:08:30Z`
- Health: ready
- Models endpoint: ready
- Metrics endpoint: ready
- Proxy bind: `127.0.0.1:8080`
- Public bind: false
- Systemd active: active
- Proxy memory in systemd status: `32.4M`
- State directory: `/var/lib/hydralisk/glm52-reap-private-proxy`
- Receipt directory: `/var/lib/hydralisk/glm52-reap-private-proxy/receipts`

Private proxy smoke:

- Checked at: `2026-06-25T00:12:28Z`
- HTTP status: 200
- Proxy run ref: `hydralisk-run-91b8c576959c4e6d8633626c7adf9d36`
- Prompt SHA-256:
  `5f8103cbebaac77e42161be89a636352dafbb3ceda5efef0aa6484d67233dfe2`
- Visible completion SHA-256:
  `deb72954879f318cd0fcb41355e82f54fbed51947d68e71b465fd31aba03f166`
- Visible completion characters: 18
- Prompt tokens: 22
- Completion tokens: 9
- Total tokens: 31
- Receipt blockers: none
- Receipt wall time: `25845 ms`

Metrics after the smoke:

- Requests: 1
- Responses: 1
- Errors: 0
- Status counts: `200=1`
- Inflight current: 0
- Inflight limit: 1
- Latency count: 1
- Latency average/max: `25845 ms`

## Operator lifecycle

Start or restart raw vLLM from the durable model directory:

```bash
ACTION=start RUN_ID=<run-id> \
  scripts/launch-glm-52-reap-504b-b12x-gce.sh
```

Check raw vLLM health, restart policy, GPU memory, and disk lifecycle:

```bash
ACTION=status RUN_ID=<run-id> \
  scripts/launch-glm-52-reap-504b-b12x-gce.sh
```

Apply the configured Docker restart policy to a non-`--rm` running container:

```bash
ACTION=apply-restart-policy RUN_ID=<run-id> \
  scripts/launch-glm-52-reap-504b-b12x-gce.sh
```

If that action fails with Docker `AutoRemove` enabled, the running container was
started before this hardening pass. Use `ACTION=start` once to relaunch it with
the restart policy and local model/cache mounts.

Install or refresh the systemd-managed private proxy:

```bash
ACTION=install-systemd RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-private-proxy-gce.sh
```

Restart the systemd-managed private proxy after code or env changes:

```bash
ACTION=restart-systemd RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-private-proxy-gce.sh
```

Check proxy health, authenticated models, metrics, systemd status, and disk
lifecycle:

```bash
ACTION=status RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-private-proxy-gce.sh
```

Run the public-safe private proxy smoke:

```bash
ACTION=smoke RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-private-proxy-gce.sh
```

Stop only the proxy service:

```bash
ACTION=stop RUN_ID=<run-id> \
  scripts/expose-glm-52-reap-504b-private-proxy-gce.sh
```

Stop only the raw vLLM container:

```bash
ACTION=stop RUN_ID=<run-id> \
  scripts/launch-glm-52-reap-504b-b12x-gce.sh
```

## Stop, start, and preemption recovery

After a host stop, Spot preemption, kernel update, or operator refresh:

1. Confirm the GCE instance is running and the 1.5T boot disk is attached.
2. Run raw vLLM `ACTION=status`.
3. If `/v1/models` is not ready, run raw vLLM `ACTION=start`.
4. Poll raw vLLM `ACTION=status` until `modelsEndpoint=ready`.
5. Run proxy `ACTION=install-systemd` or `ACTION=restart-systemd`.
6. Run proxy `ACTION=status` and verify health, models, metrics, and systemd
   are ready.
7. Run proxy `ACTION=smoke`.
8. Attach the public-safe status and smoke summary to the issue or run log.

The service-level restart from durable model disk was tested live in this
packet. A full Spot VM stop/start was not forced during this gate because GCE
Spot capacity re-admission can fail independently of Hydralisk. The runbook
above is the required recovery path after a real preemption or planned VM stop.

## Model revision refresh

To refresh the model revision:

1. Stage the new revision into a new durable model directory or verify the
   existing directory atomically.
2. Update `MODEL_REVISION`, `MODEL_DIR`, and the profile evidence refs in the
   launcher/proxy environment.
3. Do not mutate the old directory in place while raw vLLM is running.
4. Relaunch raw vLLM with `ACTION=start`.
5. Refresh the proxy with `ACTION=restart-systemd`.
6. Run status and smoke.
7. Update this profile and evidence docs before promoting any serving claim.

## Secrets and logs

- The bearer token lives only in
  `/var/lib/hydralisk/glm52-reap-private-proxy/bearer-token`.
- The systemd wrapper reads the bearer token from that file at process start.
- Do not put the bearer token in tracked files, issue comments, shell history,
  systemd unit files, or public logs.
- Public-safe artifacts may include `status-public.txt`, `health.json`,
  `models.json`, `metrics.json`, `systemd-status-public.txt`, and smoke JSON.
- Do not commit raw `vllm.log`, full journals, raw prompts, raw responses,
  hidden reasoning, weights, compiled engines, profiler dumps, or benchmark
  dumps.

## Cost and capacity notes

- The current host is the admitted `g4-standard-384` fallback using four active
  RTX PRO 6000 GPUs on an eight-GPU VM because standalone 4x G4 capacity was
  blocked during admission.
- Spot reduces runtime cost but can stop without service-level warning; keep
  all model artifacts, cache, receipts, and evidence on durable disk paths.
- Standard provisioning can reduce interruption risk when the service is needed
  for a long eval or customer-facing internal workload.
- The 1.5T boot disk carries ongoing storage cost while retained. Stop the VM
  when idle, but keep the disk if the goal is fast recovery without a model
  re-download.
- Snapshot before destructive model directory changes or major revision swaps.

## Claim boundary

This gate hardens the private operator surface. It does not change the tuned
serving limits: 250K admitted context, proxy single-flight, `max_num_seqs=2`,
`max_num_batched_tokens=4096`, MTP disabled by default, and standalone 4x G4
capacity still blocked at admission time.
