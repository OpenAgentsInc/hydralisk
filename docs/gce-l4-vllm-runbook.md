# GCE L4 vLLM runbook for GPT-OSS 20B

Date: 2026-06-23

## 2026-06-24 live promotion

Hydralisk GPT-OSS 20B is live on a dedicated GCE L4 host:

- project: `openagentsgemini`
- instance: `hydralisk-gptoss20b-l4-20260624000550`
- zone: `us-central1-a`
- shape: `g2-standard-8`, 1 x NVIDIA L4
- image family used: `common-cu129-ubuntu-2204-nvidia-580`
- public HTTPS origin: stored as the OpenAgents Worker
  `HYDRALISK_BASE_URL` secret
- service stack: vLLM `0.23.0` serving `openai/gpt-oss-20b`; Hydralisk proxy on
  `127.0.0.1:8080`; Caddy `2.11.4` fronting HTTPS
- public-safe smoke refs:
  - `preflight.hydralisk.gpt_oss_20b.l4.20260624T002313Z`
  - `receipt.hydralisk.gpt_oss_20b.l4.hydralisk-run-88ccd454ea4f4fc7baee9a72c4894527`

The public-origin smoke passed through HTTPS and exercised health,
capabilities, non-streaming chat completion, streaming chat completion, and
receipt fetch. The latest public-safe receipts observed:

- `hydralisk-run-0e65f1ef6281413eab28f8aa71580c7b`: non-streaming,
  81 total tokens, 2578 ms wall time.
- `hydralisk-run-88ccd454ea4f4fc7baee9a72c4894527`: streaming,
  195 total tokens, 77 ms TTFT, 2050 ms wall time.

No bearer token, prompt, response body, local cache path, or private host
material belongs in this repo, GitHub issues, or public readiness payloads.

## Current Google state

Project: `openagentsgemini`

Region: `us-central1`

Live quota observed with `gcloud compute regions describe us-central1`:

- `NVIDIA_L4_GPUS`: limit 16, usage 3 after the Hydralisk promotion
- `PREEMPTIBLE_NVIDIA_L4_GPUS`: limit 16, usage 0
- `CPUS`: limit 3000, usage 47
- `SSD_TOTAL_GB`: limit 40960, usage 6532

Existing L4 VMs observed:

- `gswarm508-clean2-20260325044551-contrib`, `us-central1-b`, running,
  `g2-standard-8`, 1 x L4, labels identify it as a Psion training contributor.
- `gswarm508-clean2-20260325044551-coord`, `us-central1-a`, running,
  `g2-standard-8`, 1 x L4, labels identify it as a Psion training coordinator.
- `hydralisk-gptoss20b-l4-20260624000550`, `us-central1-a`, running,
  `g2-standard-8`, 1 x L4, labels identify it as Hydralisk inference for
  GPT-OSS 20B.

Do not repurpose the running Psion hosts without an explicit operator decision.
Hydralisk has enough L4 quota for additional fresh `g2-standard-8` lanes.

The terminated coordinator-shaped L4 VM can be repurposed if the operator wants
to reuse existing infrastructure instead of creating a fresh host. Clear the old
startup script before starting it so the historical Psion bootstrap does not
resume:

```bash
gcloud compute instances remove-metadata \
  gswarm508-clean2-20260325044551-coord \
  --zone us-central1-a \
  --keys startup-script

gcloud compute instances add-tags \
  gswarm508-clean2-20260325044551-coord \
  --zone us-central1-a \
  --tags hydralisk-host,gpt-oss-20b,l4

gcloud compute instances start \
  gswarm508-clean2-20260325044551-coord \
  --zone us-central1-a
```

## Provision a fresh L4 host

```bash
export PROJECT_ID=openagentsgemini
export ZONE=us-central1-a
export INSTANCE=hydralisk-gptoss20b-l4-$(date -u +%Y%m%d%H%M%S)

gcloud compute instances create "$INSTANCE" \
  --project "$PROJECT_ID" \
  --zone "$ZONE" \
  --machine-type g2-standard-8 \
  --accelerator type=nvidia-l4,count=1 \
  --maintenance-policy TERMINATE \
  --provisioning-model STANDARD \
  --boot-disk-size 250GB \
  --boot-disk-type pd-ssd \
  --image-family common-cu129-ubuntu-2204-nvidia-580 \
  --image-project deeplearning-platform-release \
  --metadata enable-oslogin=TRUE \
  --tags hydralisk-host,gpt-oss-20b,l4 \
  --labels lane=hydralisk,workload=inference,model=gpt-oss-20b,environment=internal
```

## Install Hydralisk and vLLM

```bash
sudo useradd --system --create-home --home-dir /var/lib/hydralisk hydralisk || true
sudo mkdir -p /opt/hydralisk /etc/hydralisk /var/lib/hydralisk/huggingface
sudo chown -R "$USER":hydralisk /opt/hydralisk
sudo chown -R hydralisk:hydralisk /var/lib/hydralisk

git clone https://github.com/OpenAgentsInc/hydralisk.git /opt/hydralisk
cd /opt/hydralisk

sudo deploy/gce/install-hydralisk-l4.sh
```

Create `/etc/hydralisk/hydralisk.env` from
`deploy/systemd/hydralisk.env.example`. Set a real `HYDRALISK_BEARER_TOKEN`;
do not commit it. Set `HYDRALISK_ENGINE_VERSION` to the output of
`vllm --version`; the helper installer does this automatically for new env
files.

```bash
sudo install -m 0644 deploy/systemd/vllm-gpt-oss-20b.service /etc/systemd/system/
sudo install -m 0644 deploy/systemd/hydralisk-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-gpt-oss-20b.service
sudo systemctl enable --now hydralisk-proxy.service
```

## HTTPS fronting

Do not expose raw vLLM. The systemd proxy binds to `127.0.0.1:8080`; put a TLS
terminator in front of that listener. The repo includes
`deploy/caddy/hydralisk.Caddyfile.example` for Caddy-based hosts:

```bash
sudo apt-get update -y
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update -y
sudo apt-get install -y caddy
sudo install -m 0644 deploy/caddy/hydralisk.Caddyfile.example /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

DNS must point the chosen host name at the Hydralisk VM before Caddy can issue a
certificate. For private day-zero smoke, use SSH port forwarding and keep the
VM firewall closed to public ingress.

For a public Worker origin, add a tag-targeted HTTPS firewall rule. The proxy
still requires bearer auth, and raw vLLM stays bound to localhost:

```bash
gcloud compute firewall-rules create hydralisk-host-https \
  --project openagentsgemini \
  --direction INGRESS \
  --priority 1000 \
  --network default \
  --action ALLOW \
  --rules tcp:80,tcp:443 \
  --source-ranges 0.0.0.0/0 \
  --target-tags hydralisk-host
```

## Host-local smoke

```bash
curl -fsS http://127.0.0.1:8080/health | jq .
curl -fsS http://127.0.0.1:8080/hydralisk/v1/capabilities | jq .

curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H "authorization: Bearer $HYDRALISK_BEARER_TOKEN" \
  -H "content-type: application/json" \
  -d '{
    "model": "openagents/khala-oss-20b",
    "messages": [
      { "role": "user", "content": "Say READY in one word." }
    ],
    "max_tokens": 8
  }' | jq .
```

The non-streaming response includes `x-hydralisk-run-ref` and
`x-hydralisk-receipt-ref` headers. Fetch the receipt:

```bash
curl -fsS "http://127.0.0.1:8080/hydralisk/v1/receipts/$RUN_REF" | jq .
```

Streaming smoke:

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "authorization: Bearer $HYDRALISK_BEARER_TOKEN" \
  -H "content-type: application/json" \
  -d '{
    "model": "openagents/khala-oss-20b",
    "stream": true,
    "messages": [
      { "role": "user", "content": "Write a short status report." }
    ],
    "max_tokens": 120
  }'
```

The full day-zero smoke script fails closed if health, capabilities,
non-streaming usage, streaming SSE frames, or receipts are missing:

```bash
HYDRALISK_ORIGIN=http://127.0.0.1:8080 \
HYDRALISK_BEARER_TOKEN="$HYDRALISK_BEARER_TOKEN" \
scripts/smoke-gpt-oss-20b.sh
```

OpenAgents arming refs should point at public-safe evidence generated by that
smoke:

- `HYDRALISK_GPT_OSS_20B_PREFLIGHT_REF`: the capabilities/preflight evidence
  identifier for this deployed host, for example
  `preflight.hydralisk.gpt_oss_20b.l4.v1`.
- `HYDRALISK_GPT_OSS_20B_RECEIPT_REF`: the public-safe smoke receipt id or
  durable evidence URI proving at least one non-streaming and one streaming run,
  for example `receipt.hydralisk.gpt_oss_20b.l4.smoke.v1`.

For a smaller local CLI smoke that checks health, non-streaming, and streaming
byte flow from the installed console script:

```bash
hydralisk-smoke \
  --base-url http://127.0.0.1:8080 \
  --bearer-token "$HYDRALISK_BEARER_TOKEN" \
  --model openai/gpt-oss-20b
```

## Start, stop, restart, logs

```bash
sudo systemctl start vllm-gpt-oss-20b.service hydralisk-proxy.service
sudo systemctl stop hydralisk-proxy.service vllm-gpt-oss-20b.service
sudo systemctl restart vllm-gpt-oss-20b.service hydralisk-proxy.service

journalctl -u vllm-gpt-oss-20b.service -f
journalctl -u hydralisk-proxy.service -f
```

## Rollback

Hydralisk is a separate supply lane. Rollback is therefore:

1. Set `HYDRALISK_GPT_OSS_20B_ENABLED` away from `ready` in OpenAgents.
2. Restart or redeploy the OpenAgents Worker.
3. Stop Hydralisk services:

```bash
sudo systemctl stop hydralisk-proxy.service vllm-gpt-oss-20b.service
```

4. Keep the VM for inspection or stop it to halt GPU spend:

```bash
gcloud compute instances stop "$INSTANCE" --zone "$ZONE"
```

Do not delete logs or local cache until any public-safe receipt and smoke
artifacts have been copied out.
