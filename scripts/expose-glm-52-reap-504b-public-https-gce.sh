#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-${CLOUDSDK_CORE_ACCOUNT:-}}"
TARGET_INSTANCE="${TARGET_INSTANCE:-hydralisk-glm52-reap-504b-g4-8g-b-20260624214500}"
TARGET_ZONE="${TARGET_ZONE:-us-central1-b}"
REGION="${REGION:-us-central1}"
ADDRESS_NAME="${ADDRESS_NAME:-hydralisk-glm52-reap-504b-ingress}"
TARGET_TAG="${TARGET_TAG:-hydralisk-glm52-reap-https}"
FIREWALL_RULE="${FIREWALL_RULE:-hydralisk-glm52-reap-https}"
PROXY_PORT="${PROXY_PORT:-8080}"
CADDY_ADMIN_EMAIL="${CADDY_ADMIN_EMAIL:-}"
ACTION="${ACTION:-setup}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/glm52-reap-public-https-$RUN_ID}"

MODEL_ALIAS="${MODEL_ALIAS:-openagents/glm-5.2-reap-504b}"
COMPLETION_TIMEOUT_SECONDS="${COMPLETION_TIMEOUT_SECONDS:-120}"
COMPLETION_RETRIES="${COMPLETION_RETRIES:-6}"
COMPLETION_RETRY_SLEEP_SECONDS="${COMPLETION_RETRY_SLEEP_SECONDS:-10}"

if [[ -z "$TARGET_INSTANCE" || -z "$TARGET_ZONE" ]]; then
  echo "error: TARGET_INSTANCE and TARGET_ZONE are required" >&2
  exit 2
fi

if [[ "$TARGET_INSTANCE" != hydralisk-glm52-reap-504b-* ]]; then
  echo "error: refusing non-GLM target instance: $TARGET_INSTANCE" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

run_gcloud() {
  if [[ -n "$GCLOUD_ACCOUNT" ]]; then
    CLOUDSDK_CORE_ACCOUNT="$GCLOUD_ACCOUNT" gcloud "$@"
  else
    gcloud "$@"
  fi
}

curl_bin() {
  if command -v curl >/dev/null 2>&1; then
    command -v curl
  elif [[ -x /usr/bin/curl ]]; then
    printf '/usr/bin/curl\n'
  else
    echo "error: curl is required" >&2
    exit 2
  fi
}

describe_instance() {
  run_gcloud compute instances describe "$TARGET_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --format=json
}

instance_value() {
  local fmt="$1"
  run_gcloud compute instances describe "$TARGET_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --format="value($fmt)"
}

ensure_address() {
  if ! run_gcloud compute addresses describe "$ADDRESS_NAME" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --format='value(address)' > "$OUTPUT_DIR/address.txt" 2>/dev/null; then
    run_gcloud compute addresses create "$ADDRESS_NAME" \
      --project "$PROJECT_ID" \
      --region "$REGION" \
      --network-tier PREMIUM \
      --format=json > "$OUTPUT_DIR/address-create.json"
    run_gcloud compute addresses describe "$ADDRESS_NAME" \
      --project "$PROJECT_ID" \
      --region "$REGION" \
      --format='value(address)' > "$OUTPUT_DIR/address.txt"
  fi
}

public_address() {
  if [[ ! -s "$OUTPUT_DIR/address.txt" ]]; then
    ensure_address
  fi
  tr -d '\r\n' < "$OUTPUT_DIR/address.txt"
}

public_hostname() {
  if [[ -n "${PUBLIC_HOSTNAME:-}" ]]; then
    printf '%s\n' "$PUBLIC_HOSTNAME"
  else
    printf 'hydralisk-glm52-reap-504b.%s.sslip.io\n' "$(public_address)"
  fi
}

private_proxy_upstream() {
  local private_ip
  private_ip="$(instance_value 'networkInterfaces[0].networkIP' | tr -d '\r\n')"
  if [[ -z "$private_ip" ]]; then
    echo "error: could not resolve instance private IP" >&2
    exit 1
  fi
  printf 'http://%s:%s\n' "$private_ip" "$PROXY_PORT"
}

network_name() {
  instance_value 'networkInterfaces[0].network.basename()' | tr -d '\r\n'
}

ensure_access_config() {
  local desired current
  desired="$(public_address)"
  current="$(instance_value 'networkInterfaces[0].accessConfigs[0].natIP' | tr -d '\r\n')"
  if [[ -z "$current" ]]; then
    run_gcloud compute instances add-access-config "$TARGET_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --access-config-name 'External NAT' \
      --address "$desired" \
      --network-tier PREMIUM \
      --format=json > "$OUTPUT_DIR/add-access-config.json"
    return 0
  fi
  if [[ "$current" != "$desired" ]]; then
    echo "error: instance already has a different external address; set up manually or remove it first" >&2
    exit 1
  fi
}

ensure_tag() {
  local tags
  tags="$(instance_value 'tags.items.list()' | tr ';' '\n' || true)"
  if ! printf '%s\n' "$tags" | grep -Fx "$TARGET_TAG" >/dev/null 2>&1; then
    run_gcloud compute instances add-tags "$TARGET_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --tags "$TARGET_TAG" \
      --format=json > "$OUTPUT_DIR/add-tags.json"
  fi
}

ensure_firewall() {
  local network
  network="$(network_name)"
  if [[ -z "$network" ]]; then
    echo "error: could not resolve instance network" >&2
    exit 1
  fi
  if ! run_gcloud compute firewall-rules describe "$FIREWALL_RULE" \
    --project "$PROJECT_ID" \
    --format=json > "$OUTPUT_DIR/firewall-existing.json" 2>/dev/null; then
    run_gcloud compute firewall-rules create "$FIREWALL_RULE" \
      --project "$PROJECT_ID" \
      --direction INGRESS \
      --priority 1000 \
      --network "$network" \
      --action ALLOW \
      --rules tcp:80,tcp:443 \
      --source-ranges 0.0.0.0/0 \
      --target-tags "$TARGET_TAG" \
      --format=json > "$OUTPUT_DIR/firewall-create.json"
  fi
}

install_caddy() {
  local host upstream remote_script
  host="$(public_hostname)"
  upstream="$(private_proxy_upstream)"
  remote_script="$(mktemp "${TMPDIR:-/tmp}/hydralisk-glm52-public-https.XXXXXXXXXX")"
  trap 'rm -f "$remote_script"' RETURN

  cat > "$remote_script" <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update -y >/tmp/hydralisk-caddy-apt-update.log 2>&1 || true
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  debian-keyring debian-archive-keyring apt-transport-https curl gpg \
  >/tmp/hydralisk-caddy-apt-prereqs.log 2>&1

if ! command -v caddy >/dev/null 2>&1; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -y >/tmp/hydralisk-caddy-apt-caddy-update.log 2>&1
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y caddy \
    >/tmp/hydralisk-caddy-apt-caddy.log 2>&1
fi

tmp="$(mktemp)"
if [[ -n "${CADDY_ADMIN_EMAIL:-}" ]]; then
  {
    printf '{\n'
    printf '  email %s\n' "$CADDY_ADMIN_EMAIL"
    printf '}\n\n'
  } > "$tmp"
else
  : > "$tmp"
fi

cat >> "$tmp" <<EOF
$PUBLIC_HOSTNAME {
  encode zstd gzip

  header {
    -Server
    X-Content-Type-Options nosniff
  }

  @hydralisk_origin {
    path /health /v1/* /hydralisk/*
  }

  reverse_proxy @hydralisk_origin $PROXY_UPSTREAM {
    transport http {
      versions 1.1
    }
  }

  respond 404
}
EOF

sudo install -m 0644 "$tmp" /etc/caddy/Caddyfile
rm -f "$tmp"
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl enable caddy >/dev/null
sudo systemctl reload caddy >/dev/null 2>&1 || sudo systemctl restart caddy
systemctl is-active caddy
caddy version
REMOTE

  run_gcloud compute scp "$remote_script" "$TARGET_INSTANCE:/tmp/hydralisk-glm52-public-https.sh" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet >/dev/null
  run_gcloud compute ssh "$TARGET_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet \
    --command "PUBLIC_HOSTNAME='$host' PROXY_UPSTREAM='$upstream' CADDY_ADMIN_EMAIL='$CADDY_ADMIN_EMAIL' bash /tmp/hydralisk-glm52-public-https.sh" \
    > "$OUTPUT_DIR/caddy-install-public.txt"
}

write_status() {
  local host address upstream scheduling boot_auto_delete caddy_status public_sha
  host="$(public_hostname)"
  address="$(public_address)"
  upstream="$(private_proxy_upstream)"
  scheduling="$(instance_value 'scheduling.provisioningModel,scheduling.instanceTerminationAction,scheduling.maxRunDuration')"
  boot_auto_delete="$(instance_value 'disks[0].autoDelete')"
  caddy_status="$(
    run_gcloud compute ssh "$TARGET_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --quiet \
      --command='set -euo pipefail; printf "active=%s\n" "$(systemctl is-active caddy || true)"; printf "version=%s\n" "$(caddy version 2>/dev/null || true)"; sudo caddy validate --config /etc/caddy/Caddyfile >/dev/null && printf "config=valid\n"' \
      2>/dev/null || true
  )"
  public_sha="$(printf '%s' "$host" | python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.read().encode()).hexdigest())')"
  python3 - <<'PY' "$OUTPUT_DIR/public-https-status.json" "$TARGET_INSTANCE" "$TARGET_ZONE" "$REGION" "$ADDRESS_NAME" "$TARGET_TAG" "$FIREWALL_RULE" "$upstream" "$scheduling" "$boot_auto_delete" "$caddy_status" "$public_sha"
import json
import sys

(
    out,
    instance,
    zone,
    region,
    address_name,
    target_tag,
    firewall_rule,
    upstream,
    scheduling,
    boot_auto_delete,
    caddy_status,
    public_sha,
) = sys.argv[1:]

doc = {
    "schema": "hydralisk.glm52_reap.public_https_status.v1",
    "publicSafe": True,
    "instance": instance,
    "zone": zone,
    "region": region,
    "addressName": address_name,
    "targetTag": target_tag,
    "firewallRule": firewall_rule,
    "endpointShape": "https://<operator-secret-hostname>",
    "endpointValueTracked": False,
    "endpointHostSha256": public_sha,
    "proxyUpstreamShape": upstream.rsplit(":", 1)[0].split("//", 1)[0] + "//<instance-private-address>:8080",
    "rawVllmPublic": False,
    "proxyBearerAuthRequired": True,
    "gceScheduling": scheduling,
    "bootDiskAutoDelete": boot_auto_delete,
    "caddy": dict(
        line.split("=", 1)
        for line in caddy_status.splitlines()
        if "=" in line
    ),
    "publicSafety": {
        "containsEndpoint": False,
        "containsBearerToken": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
    },
}
with open(out, "w") as f:
    json.dump(doc, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(doc, indent=2, sort_keys=True))
PY
}

read_bearer_token() {
  run_gcloud compute ssh "$TARGET_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet \
    --command='sudo cat /var/lib/hydralisk/glm52-reap-private-proxy/bearer-token' \
    2>/dev/null | tr -d '\r\n'
}

smoke_https() {
  local host base token curl http_code start_ms end_ms attempt
  host="$(public_hostname)"
  base="https://$host"
  token="$(read_bearer_token)"
  curl="$(curl_bin)"
  if [[ -z "$token" ]]; then
    echo "error: bearer token was empty" >&2
    exit 1
  fi
  "$curl" -fsS "$base/health" > "$OUTPUT_DIR/health.json"
  "$curl" -fsS -H "authorization: Bearer $token" "$base/v1/models" > "$OUTPUT_DIR/models.json"
  for attempt in $(seq 1 "$COMPLETION_RETRIES"); do
    start_ms="$(python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
)"
    http_code="$(
      "$curl" -sS \
        --max-time "$COMPLETION_TIMEOUT_SECONDS" \
        -w '%{http_code}' \
        -o "$OUTPUT_DIR/completion.json" \
        -H "authorization: Bearer $token" \
        -H 'content-type: application/json' \
        "$base/v1/chat/completions" \
        --data-binary "{\"model\":\"$MODEL_ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with READY only.\"}],\"max_tokens\":8,\"temperature\":0}"
    )"
    end_ms="$(python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
)"
    if [[ "$http_code" != "429" || "$attempt" == "$COMPLETION_RETRIES" ]]; then
      break
    fi
    sleep "$COMPLETION_RETRY_SLEEP_SECONDS"
  done
  python3 - <<'PY' "$OUTPUT_DIR" "$http_code" "$start_ms" "$end_ms" "$MODEL_ALIAS" "$attempt"
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
http_code = sys.argv[2]
start = int(sys.argv[3])
end = int(sys.argv[4])
model_alias = sys.argv[5]
attempt = int(sys.argv[6])
health = json.loads((root / "health.json").read_text())
models = json.loads((root / "models.json").read_text())
completion_text = (root / "completion.json").read_text()
try:
    completion = json.loads(completion_text)
except json.JSONDecodeError:
    completion = {"rawLen": len(completion_text), "rawSha256": hashlib.sha256(completion_text.encode()).hexdigest()}

summary = {
    "schema": "hydralisk.glm52_reap.public_https_smoke.v1",
    "publicSafe": True,
    "endpointShape": "https://<operator-secret-hostname>",
    "endpointValueTracked": False,
    "requestedModelAlias": model_alias,
    "health": {
        "status": health.get("status"),
        "servedModel": health.get("servedModel"),
        "authRequired": health.get("authRequired"),
    },
    "models": {
        "httpStatus": 200,
        "count": len(models.get("data", [])),
    },
    "completion": {
        "httpStatus": int(http_code),
        "wallMs": end - start,
        "attempt": attempt,
    },
    "publicSafety": {
        "containsEndpoint": False,
        "containsBearerToken": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
    },
}
if isinstance(completion, dict) and completion.get("choices"):
    content = completion.get("choices", [{}])[0].get("message", {}).get("content", "")
    summary["completion"].update(
        {
            "visibleCompletionChars": len(content),
            "visibleCompletionSha256": hashlib.sha256(content.encode()).hexdigest(),
            "usage": completion.get("usage"),
        }
    )
else:
    summary["completion"]["error"] = completion

(root / "public-https-smoke.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
if http_code != "200":
    raise SystemExit(1)
PY
}

case "$ACTION" in
  setup)
    ensure_address
    ensure_access_config
    ensure_tag
    ensure_firewall
    install_caddy
    write_status
    ;;
  status)
    ensure_address
    write_status
    ;;
  smoke)
    ensure_address
    smoke_https
    ;;
  *)
    echo "error: unknown ACTION=$ACTION; expected setup, status, or smoke" >&2
    exit 2
    ;;
esac
