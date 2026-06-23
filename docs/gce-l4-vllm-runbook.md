# GCE L4 vLLM runbook for GPT-OSS 20B

Date: 2026-06-23

## Current Google state

Project: `openagentsgemini`

Region: `us-central1`

Live quota observed with `gcloud compute regions describe us-central1`:

- `NVIDIA_L4_GPUS`: limit 16, usage 1
- `PREEMPTIBLE_NVIDIA_L4_GPUS`: limit 16, usage 0
- `CPUS`: limit 3000, usage 47
- `SSD_TOTAL_GB`: limit 40960, usage 6532

Existing L4 VMs observed:

- `gswarm508-clean2-20260325044551-contrib`, `us-central1-b`, running,
  `g2-standard-8`, 1 x L4, labels identify it as a Psion training contributor.
- `gswarm508-clean2-20260325044551-coord`, `us-central1-a`, terminated,
  `g2-standard-8`, 1 x L4, labels identify it as a Psion training coordinator.

Do not repurpose the running Psion contributor without an explicit operator
decision. Hydralisk has enough L4 quota for a fresh `g2-standard-8` lane.

## Provision a fresh L4 host

```bash
export PROJECT_ID=openagentsgemini
export ZONE=us-central1-b
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
  --image-family common-cu128-ubuntu-2204-py310 \
  --image-project deeplearning-platform-release \
  --metadata enable-oslogin=TRUE \
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

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 --seed
uv pip install -e .
uv pip install --pre vllm==0.10.1+gptoss \
  --extra-index-url https://wheels.vllm.ai/gpt-oss/ \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
  --index-strategy unsafe-best-match
```

Create `/etc/hydralisk/hydralisk.env` from
`deploy/systemd/hydralisk.env.example`. Set a real `HYDRALISK_BEARER_TOKEN`;
do not commit it.

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

## Host-local smoke

```bash
curl -fsS http://127.0.0.1:8080/health | jq .

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
