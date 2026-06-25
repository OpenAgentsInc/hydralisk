#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-${CLOUDSDK_CORE_ACCOUNT:-}}"
TARGET_INSTANCE="${TARGET_INSTANCE:-hydralisk-glm52-reap-504b-g4-8g-b-20260624214500}"
TARGET_ZONE="${TARGET_ZONE:-us-central1-b}"
ACTION="${ACTION:-start}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d%H%M%S)}"

PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-8080}"
UPSTREAM_BASE_URL="${UPSTREAM_BASE_URL:-http://127.0.0.1:8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.2-reap-504b-g4}"
MODEL_ALIASES="${MODEL_ALIASES:-openagents/glm-5.2-reap-504b,0xSero/GLM-5.2-504B}"
REMOTE_SRC_DIR="${REMOTE_SRC_DIR:-/opt/hydralisk/glm52-reap-private-proxy-src}"
REMOTE_VENV_DIR="${REMOTE_VENV_DIR:-/opt/hydralisk/glm52-reap-private-proxy-venv}"
REMOTE_STATE_DIR="${REMOTE_STATE_DIR:-/var/lib/hydralisk/glm52-reap-private-proxy}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/var/log/hydralisk/glm52-reap-private-proxy-$RUN_ID}"
SYSTEMD_UNIT_NAME="${SYSTEMD_UNIT_NAME:-hydralisk-glm52-reap-private-proxy.service}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/glm52-reap-private-proxy-$RUN_ID}"

MODEL_REVISION="${MODEL_REVISION:-0xSero/GLM-5.2-504B@cb6b1e0451b9d560cda864f84187869c9a679712}"
ENGINE_VERSION="${ENGINE_VERSION:-0.11.2.dev279+black.benediction.b12xpr11.cu132.20260608}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-voipmonitor/vllm@sha256:ce23a9b075bd7138ce3b12ee29609b98606e5050e2def4a29bbb917ad96e5997}"
MODEL_PROFILE_REF="${MODEL_PROFILE_REF:-profiles/glm-5.2-reap-504b-b12x-g4.json}"
ADMISSION_REF="${ADMISSION_REF:-docs/evidence/2026-06-24-glm-52-reap-504b-g4-admission.md}"
EVIDENCE_REF="${EVIDENCE_REF:-docs/evidence/2026-06-25-glm-52-reap-504b-mtp2-speed-gate.md}"
ADMITTED_CONTEXT_TOKENS="${ADMITTED_CONTEXT_TOKENS:-250000}"

DEFAULT_MIN_P="${DEFAULT_MIN_P-}"
DEFAULT_REPETITION_PENALTY="${DEFAULT_REPETITION_PENALTY:-1.05}"
DEFAULT_MAX_TOKENS="${DEFAULT_MAX_TOKENS:-1024}"
DEFAULT_ENABLE_THINKING="${DEFAULT_ENABLE_THINKING:-false}"
MAX_INFLIGHT_REQUESTS="${MAX_INFLIGHT_REQUESTS:-1}"
SPECULATIVE_DECODING="${SPECULATIVE_DECODING:-mtp2}"

if [[ -z "$TARGET_INSTANCE" || -z "$TARGET_ZONE" ]]; then
  echo "error: TARGET_INSTANCE and TARGET_ZONE are required" >&2
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

bundle="$(mktemp "${TMPDIR:-/tmp}/hydralisk-glm52-proxy.XXXXXXXXXX")"
remote_script="$(mktemp "${TMPDIR:-/tmp}/hydralisk-glm52-proxy.XXXXXXXXXX")"
trap 'rm -f "$bundle" "$remote_script"' EXIT

COPYFILE_DISABLE=1 tar -czf "$bundle" hydralisk pyproject.toml README.md

cat > "$remote_script" <<'REMOTE'
#!/usr/bin/env bash
set -Eeuo pipefail

TOKEN_FILE="$REMOTE_STATE_DIR/bearer-token"
PID_FILE="$REMOTE_STATE_DIR/proxy.pid"
SYSTEMD_WRAPPER="$REMOTE_STATE_DIR/run-proxy-service.sh"

sudo install -d -m 0755 "$REMOTE_LOG_DIR" /var/log/hydralisk "$REMOTE_STATE_DIR"
sudo chown "$(whoami):$(id -gn)" "$REMOTE_LOG_DIR" "$REMOTE_STATE_DIR"

unit_exists() {
  sudo systemctl cat "$SYSTEMD_UNIT_NAME" >/dev/null 2>&1
}

ensure_token() {
  if [[ ! -s "$TOKEN_FILE" ]]; then
    umask 077
    python3 - <<'PY' > "$TOKEN_FILE"
import secrets
print(secrets.token_urlsafe(48))
PY
  fi
  chmod 0600 "$TOKEN_FILE"
}

write_public_env() {
  cat > "$REMOTE_LOG_DIR/proxy.env.public" <<EOF
PROXY_HOST=$PROXY_HOST
PROXY_PORT=$PROXY_PORT
UPSTREAM_BASE_URL=$UPSTREAM_BASE_URL
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
MODEL_ALIASES=$MODEL_ALIASES
MODEL_REVISION=$MODEL_REVISION
ENGINE_VERSION=$ENGINE_VERSION
CONTAINER_IMAGE=$CONTAINER_IMAGE
MODEL_PROFILE_REF=$MODEL_PROFILE_REF
ADMISSION_REF=$ADMISSION_REF
EVIDENCE_REF=$EVIDENCE_REF
ADMITTED_CONTEXT_TOKENS=$ADMITTED_CONTEXT_TOKENS
DEFAULT_MIN_P=$DEFAULT_MIN_P
DEFAULT_REPETITION_PENALTY=$DEFAULT_REPETITION_PENALTY
DEFAULT_MAX_TOKENS=$DEFAULT_MAX_TOKENS
DEFAULT_ENABLE_THINKING=$DEFAULT_ENABLE_THINKING
MAX_INFLIGHT_REQUESTS=$MAX_INFLIGHT_REQUESTS
SPECULATIVE_DECODING=$SPECULATIVE_DECODING
AUTH_REQUIRED=true
PUBLIC_BIND=false
SYSTEMD_UNIT_NAME=$SYSTEMD_UNIT_NAME
EOF
}

write_service_wrapper() {
  cat > "$SYSTEMD_WRAPPER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

export HYDRALISK_BEARER_TOKEN="\$(cat "$TOKEN_FILE")"
export HYDRALISK_VLLM_BASE_URL="$UPSTREAM_BASE_URL"
export HYDRALISK_SERVED_MODEL="$SERVED_MODEL_NAME"
export HYDRALISK_PUBLIC_MODEL_ALIASES="$MODEL_ALIASES"
export HYDRALISK_ENGINE="vllm"
export HYDRALISK_ENGINE_VERSION="$ENGINE_VERSION"
export HYDRALISK_GPU_CLASS="g4-rtx-pro-6000"
export HYDRALISK_GPU_NAME="NVIDIA RTX PRO 6000 Blackwell Server Edition"
export HYDRALISK_GPU_COUNT="4"
export HYDRALISK_MODEL_REVISION="$MODEL_REVISION"
export HYDRALISK_QUANTIZATION_WEIGHTS="NVFP4/REAP"
export HYDRALISK_MODEL_PROFILE_REF="$MODEL_PROFILE_REF"
export HYDRALISK_CONTAINER_IMAGE="$CONTAINER_IMAGE"
export HYDRALISK_CONTEXT_WINDOW_TOKENS="1048576"
export HYDRALISK_ADMITTED_CONTEXT_TOKENS="$ADMITTED_CONTEXT_TOKENS"
export HYDRALISK_TENSOR_PARALLEL_SIZE="4"
export HYDRALISK_REASONING_PARSER="glm45"
export HYDRALISK_TOOL_CALL_PARSER="glm47"
export HYDRALISK_KV_CACHE_DTYPE="fp8"
export HYDRALISK_SPECULATIVE_DECODING="$SPECULATIVE_DECODING"
export HYDRALISK_ADMISSION_REF="$ADMISSION_REF"
export HYDRALISK_EVIDENCE_REF="$EVIDENCE_REF"
export HYDRALISK_RECEIPT_DIR="$REMOTE_STATE_DIR/receipts"
export HYDRALISK_REQUIRE_PROFILE_EVIDENCE="true"
export HYDRALISK_DEFAULT_MIN_P="$DEFAULT_MIN_P"
export HYDRALISK_DEFAULT_REPETITION_PENALTY="$DEFAULT_REPETITION_PENALTY"
export HYDRALISK_DEFAULT_MAX_TOKENS="$DEFAULT_MAX_TOKENS"
export HYDRALISK_DEFAULT_ENABLE_THINKING="$DEFAULT_ENABLE_THINKING"
export HYDRALISK_MAX_INFLIGHT_REQUESTS="$MAX_INFLIGHT_REQUESTS"
export HYDRALISK_INFLIGHT_QUEUE_TIMEOUT_SECONDS="0"
export PYTHONPATH="$REMOTE_SRC_DIR"

exec "$REMOTE_VENV_DIR/bin/uvicorn" hydralisk.serve.proxy:app \
  --host "$PROXY_HOST" \
  --port "$PROXY_PORT" \
  --proxy-headers
EOF
  chmod 0750 "$SYSTEMD_WRAPPER"
}

prepare_action() {
  sudo apt-get update >/tmp/hydralisk-glm52-proxy-apt.log 2>&1 || true
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-venv python3-pip curl jq >/tmp/hydralisk-glm52-proxy-apt-install.log 2>&1 || true
  sudo rm -rf "$REMOTE_SRC_DIR"
  sudo install -d -m 0755 "$REMOTE_SRC_DIR"
  sudo chown "$(whoami):$(id -gn)" "$REMOTE_SRC_DIR"
  tar -xzf "$BUNDLE_PATH" -C "$REMOTE_SRC_DIR"
  sudo rm -rf "$REMOTE_VENV_DIR"
  sudo install -d -m 0755 "$REMOTE_VENV_DIR"
  sudo chown "$(whoami):$(id -gn)" "$REMOTE_VENV_DIR"
  python3 -m venv "$REMOTE_VENV_DIR"
  "$REMOTE_VENV_DIR/bin/python" -m pip install --upgrade pip >/tmp/hydralisk-glm52-proxy-pip.log 2>&1
  "$REMOTE_VENV_DIR/bin/python" -m pip install \
    'fastapi>=0.115.0,<1' \
    'httpx>=0.28.0,<1' \
    'numpy>=2.0,<3' \
    'pydantic>=2.10.0,<3' \
    'uvicorn[standard]>=0.34.0,<1' >>/tmp/hydralisk-glm52-proxy-pip.log 2>&1
  ensure_token
  write_public_env
  printf 'status=prepared\npreparedAt=%s\n' "$(date -u +%FT%TZ)" > "$REMOTE_LOG_DIR/proxy.status"
}

start_action() {
  prepare_action
  if unit_exists; then
    sudo systemctl stop "$SYSTEMD_UNIT_NAME" >/dev/null 2>&1 || true
  fi
  if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; then
    kill "$(cat "$PID_FILE")" >/dev/null 2>&1 || true
    sleep 1
  fi
  HYDRALISK_BEARER_TOKEN="$(cat "$TOKEN_FILE")" \
  HYDRALISK_VLLM_BASE_URL="$UPSTREAM_BASE_URL" \
  HYDRALISK_SERVED_MODEL="$SERVED_MODEL_NAME" \
  HYDRALISK_PUBLIC_MODEL_ALIASES="$MODEL_ALIASES" \
  HYDRALISK_ENGINE="vllm" \
  HYDRALISK_ENGINE_VERSION="$ENGINE_VERSION" \
  HYDRALISK_GPU_CLASS="g4-rtx-pro-6000" \
  HYDRALISK_GPU_NAME="NVIDIA RTX PRO 6000 Blackwell Server Edition" \
  HYDRALISK_GPU_COUNT="4" \
  HYDRALISK_MODEL_REVISION="$MODEL_REVISION" \
  HYDRALISK_QUANTIZATION_WEIGHTS="NVFP4/REAP" \
  HYDRALISK_MODEL_PROFILE_REF="$MODEL_PROFILE_REF" \
  HYDRALISK_CONTAINER_IMAGE="$CONTAINER_IMAGE" \
  HYDRALISK_CONTEXT_WINDOW_TOKENS="1048576" \
  HYDRALISK_ADMITTED_CONTEXT_TOKENS="$ADMITTED_CONTEXT_TOKENS" \
  HYDRALISK_TENSOR_PARALLEL_SIZE="4" \
  HYDRALISK_REASONING_PARSER="glm45" \
  HYDRALISK_TOOL_CALL_PARSER="glm47" \
  HYDRALISK_KV_CACHE_DTYPE="fp8" \
  HYDRALISK_SPECULATIVE_DECODING="$SPECULATIVE_DECODING" \
  HYDRALISK_ADMISSION_REF="$ADMISSION_REF" \
  HYDRALISK_EVIDENCE_REF="$EVIDENCE_REF" \
  HYDRALISK_RECEIPT_DIR="$REMOTE_STATE_DIR/receipts" \
  HYDRALISK_REQUIRE_PROFILE_EVIDENCE="true" \
  HYDRALISK_DEFAULT_MIN_P="$DEFAULT_MIN_P" \
  HYDRALISK_DEFAULT_REPETITION_PENALTY="$DEFAULT_REPETITION_PENALTY" \
  HYDRALISK_DEFAULT_MAX_TOKENS="$DEFAULT_MAX_TOKENS" \
  HYDRALISK_DEFAULT_ENABLE_THINKING="$DEFAULT_ENABLE_THINKING" \
  HYDRALISK_MAX_INFLIGHT_REQUESTS="$MAX_INFLIGHT_REQUESTS" \
  HYDRALISK_INFLIGHT_QUEUE_TIMEOUT_SECONDS="0" \
  PYTHONPATH="$REMOTE_SRC_DIR" \
  nohup "$REMOTE_VENV_DIR/bin/uvicorn" hydralisk.serve.proxy:app \
    --host "$PROXY_HOST" \
    --port "$PROXY_PORT" \
    --proxy-headers \
    > "$REMOTE_LOG_DIR/proxy.log" 2>&1 &
  echo "$!" > "$PID_FILE"
  printf 'status=starting\nstartedAt=%s\npid=%s\nbind=%s:%s\n' \
    "$(date -u +%FT%TZ)" "$(cat "$PID_FILE")" "$PROXY_HOST" "$PROXY_PORT" \
    > "$REMOTE_LOG_DIR/proxy.status"
}

install_systemd_action() {
  if unit_exists; then
    sudo systemctl stop "$SYSTEMD_UNIT_NAME" >/dev/null 2>&1 || true
  fi
  prepare_action
  write_service_wrapper
  if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; then
    kill "$(cat "$PID_FILE")" >/dev/null 2>&1 || true
    sleep 1
  fi
  sudo tee "/etc/systemd/system/$SYSTEMD_UNIT_NAME" >/dev/null <<EOF
[Unit]
Description=Hydralisk GLM-5.2 REAP private proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$REMOTE_SRC_DIR
ExecStart=$SYSTEMD_WRAPPER
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
PrivateTmp=true
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable "$SYSTEMD_UNIT_NAME" >/dev/null
  sudo systemctl restart "$SYSTEMD_UNIT_NAME"
  printf 'status=systemd_started\nstartedAt=%s\nunit=%s\nbind=%s:%s\n' \
    "$(date -u +%FT%TZ)" "$SYSTEMD_UNIT_NAME" "$PROXY_HOST" "$PROXY_PORT" \
    > "$REMOTE_LOG_DIR/proxy.status"
}

restart_systemd_action() {
  if ! unit_exists; then
    install_systemd_action
    return
  fi
  sudo systemctl stop "$SYSTEMD_UNIT_NAME" >/dev/null 2>&1 || true
  prepare_action
  write_service_wrapper
  sudo systemctl restart "$SYSTEMD_UNIT_NAME"
  printf 'status=systemd_restarted\nrestartedAt=%s\nunit=%s\nbind=%s:%s\n' \
    "$(date -u +%FT%TZ)" "$SYSTEMD_UNIT_NAME" "$PROXY_HOST" "$PROXY_PORT" \
    > "$REMOTE_LOG_DIR/proxy.status"
}

status_action() {
  ensure_token
  token="$(cat "$TOKEN_FILE")"
  {
    printf "checkedAt=%s\n" "$(date -u +%FT%TZ)"
    printf "bind=%s:%s\n" "$PROXY_HOST" "$PROXY_PORT"
    printf "publicBind=false\n"
    printf "pidRunning="
    if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; then
      printf "true\n"
    else
      printf "false\n"
    fi
    printf "systemdUnit=%s\n" "$SYSTEMD_UNIT_NAME"
    printf "systemdActive="
    sudo systemctl is-active "$SYSTEMD_UNIT_NAME" 2>/dev/null || true
    printf "healthStatus="
    if curl -fsS "http://$PROXY_HOST:$PROXY_PORT/health" > "$REMOTE_LOG_DIR/health.json" 2>/tmp/hydralisk-glm52-proxy-health.stderr; then
      printf "ready\n"
    else
      printf "not_ready\n"
    fi
    printf "modelsStatus="
    if curl -fsS -H "authorization: Bearer $token" \
      "http://$PROXY_HOST:$PROXY_PORT/v1/models" \
      > "$REMOTE_LOG_DIR/models.json" 2>/tmp/hydralisk-glm52-proxy-models.stderr; then
      printf "ready\n"
    else
      printf "not_ready\n"
    fi
    printf "metricsStatus="
    if curl -fsS "http://$PROXY_HOST:$PROXY_PORT/hydralisk/v1/metrics" \
      > "$REMOTE_LOG_DIR/metrics.json" 2>/tmp/hydralisk-glm52-proxy-metrics.stderr; then
      printf "ready\n"
    else
      printf "not_ready\n"
    fi
    printf "diskLifecycleBegin\n"
    printf "stateDir=%s\n" "$REMOTE_STATE_DIR"
    printf "receiptDir=%s\n" "$REMOTE_STATE_DIR/receipts"
    printf "logDir=%s\n" "$REMOTE_LOG_DIR"
    df -h "$REMOTE_STATE_DIR" "$REMOTE_LOG_DIR" 2>/dev/null || true
    printf "diskLifecycleEnd\n"
  } > "$REMOTE_LOG_DIR/status-public.txt"
  if unit_exists; then
    sudo systemctl status "$SYSTEMD_UNIT_NAME" --no-pager \
      > "$REMOTE_LOG_DIR/systemd-status-public.txt" 2>&1 || true
    sudo journalctl -u "$SYSTEMD_UNIT_NAME" -n 120 --no-pager \
      > "$REMOTE_LOG_DIR/systemd-journal-public.txt" 2>&1 || true
  fi
}

smoke_action() {
  status_action
  token="$(cat "$TOKEN_FILE")"
  python3 - <<'PY'
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

base_url = f"http://{os.environ['PROXY_HOST']}:{os.environ['PROXY_PORT']}"
log_dir = os.environ["REMOTE_LOG_DIR"]
token = open(os.environ["TOKEN_FILE"]).read().strip()
prompt = "Answer in one short final sentence only: what is 2 plus 2?"
payload = {
    "model": "openagents/glm-5.2-reap-504b",
    "messages": [{"role": "user", "content": prompt}],
}
body = json.dumps(payload).encode()
request = urllib.request.Request(
    f"{base_url}/v1/chat/completions",
    data=body,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    },
    method="POST",
)
started = time.perf_counter()
status = "unknown"
http_status = None
run_ref = None
receipt_ref = None
visible_sha = hashlib.sha256()
visible_chars = 0
usage = None
error_type = None
error_summary = None
try:
    with urllib.request.urlopen(request, timeout=900) as response:
        http_status = response.status
        run_ref = response.headers.get("x-hydralisk-run-ref")
        receipt_ref = response.headers.get("x-hydralisk-receipt-ref")
        parsed = json.loads(response.read().decode("utf-8"))
        usage = parsed.get("usage")
        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str) and content:
                visible_chars = len(content)
                visible_sha.update(content.encode("utf-8"))
        status = "passed" if visible_chars > 0 and run_ref else "failed"
except Exception as exc:
    if isinstance(exc, urllib.error.HTTPError):
        http_status = exc.code
        try:
            error_summary = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            error_summary = str(exc)[:300]
    else:
        error_summary = str(exc)[:300]
    error_type = type(exc).__name__
    status = "failed"
ended = time.perf_counter()
receipt = None
if receipt_ref:
    try:
        with urllib.request.urlopen(f"{base_url}{receipt_ref}", timeout=30) as response:
            receipt = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        receipt = {"receiptFetchError": type(exc).__name__}
public = {
    "runId": os.environ["RUN_ID"],
    "checkedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "status": status,
    "httpStatus": http_status,
    "proxyRunRef": run_ref,
    "proxyReceiptRef": receipt_ref,
    "servedModelName": os.environ["SERVED_MODEL_NAME"],
    "requestedModelAlias": payload["model"],
    "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    "requestOmittedDefaults": [
        "min_p",
        "repetition_penalty",
        "max_tokens",
        "chat_template_kwargs.enable_thinking",
    ],
    "visibleCompletionSha256": visible_sha.hexdigest(),
    "visibleCompletionChars": visible_chars,
    "usage": usage,
    "wallSeconds": ended - started,
    "receipt": receipt,
    "errorType": error_type,
    "errorSummary": error_summary,
    "publicSafety": {
        "containsSecrets": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
        "containsHiddenReasoning": False,
    },
}
with open(os.path.join(log_dir, "private-proxy-smoke-public.json"), "w") as f:
    json.dump(public, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(public, indent=2, sort_keys=True))
PY
  status_action
}

stop_action() {
  if unit_exists; then
    sudo systemctl stop "$SYSTEMD_UNIT_NAME" >/dev/null 2>&1 || true
  fi
  if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; then
    kill "$(cat "$PID_FILE")" >/dev/null 2>&1 || true
  fi
  printf 'status=stopped\nstoppedAt=%s\n' "$(date -u +%FT%TZ)" > "$REMOTE_LOG_DIR/proxy.status"
}

case "$ACTION" in
  prepare) prepare_action ;;
  start) start_action ;;
  install-systemd) install_systemd_action ;;
  restart-systemd) restart_systemd_action ;;
  status) status_action ;;
  smoke) smoke_action ;;
  stop) stop_action ;;
  *) echo "bad ACTION: $ACTION" >&2; exit 2 ;;
esac
REMOTE

copy_artifacts() {
  for name in \
    proxy.env.public \
    proxy.status \
    status-public.txt \
    metrics.json \
    health.json \
    models.json \
    systemd-status-public.txt \
    systemd-journal-public.txt \
    private-proxy-smoke-public.json; do
    run_gcloud compute scp \
      "$TARGET_INSTANCE:$REMOTE_LOG_DIR/$name" \
      "$OUTPUT_DIR/$name" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --quiet >/dev/null 2>&1 || true
  done
}

run_gcloud compute scp "$bundle" \
  "$TARGET_INSTANCE:/tmp/hydralisk-glm52-proxy-$RUN_ID.tar.gz" \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --quiet

run_gcloud compute scp "$remote_script" \
  "$TARGET_INSTANCE:/tmp/hydralisk-glm52-proxy-$RUN_ID.sh" \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --quiet

run_gcloud compute ssh "$TARGET_INSTANCE" \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --quiet \
  --command="ACTION='$ACTION' RUN_ID='$RUN_ID' BUNDLE_PATH='/tmp/hydralisk-glm52-proxy-$RUN_ID.tar.gz' REMOTE_SRC_DIR='$REMOTE_SRC_DIR' REMOTE_VENV_DIR='$REMOTE_VENV_DIR' REMOTE_STATE_DIR='$REMOTE_STATE_DIR' REMOTE_LOG_DIR='$REMOTE_LOG_DIR' SYSTEMD_UNIT_NAME='$SYSTEMD_UNIT_NAME' PROXY_HOST='$PROXY_HOST' PROXY_PORT='$PROXY_PORT' UPSTREAM_BASE_URL='$UPSTREAM_BASE_URL' SERVED_MODEL_NAME='$SERVED_MODEL_NAME' MODEL_ALIASES='$MODEL_ALIASES' MODEL_REVISION='$MODEL_REVISION' ENGINE_VERSION='$ENGINE_VERSION' CONTAINER_IMAGE='$CONTAINER_IMAGE' MODEL_PROFILE_REF='$MODEL_PROFILE_REF' ADMISSION_REF='$ADMISSION_REF' EVIDENCE_REF='$EVIDENCE_REF' ADMITTED_CONTEXT_TOKENS='$ADMITTED_CONTEXT_TOKENS' DEFAULT_MIN_P='$DEFAULT_MIN_P' DEFAULT_REPETITION_PENALTY='$DEFAULT_REPETITION_PENALTY' DEFAULT_MAX_TOKENS='$DEFAULT_MAX_TOKENS' DEFAULT_ENABLE_THINKING='$DEFAULT_ENABLE_THINKING' MAX_INFLIGHT_REQUESTS='$MAX_INFLIGHT_REQUESTS' SPECULATIVE_DECODING='$SPECULATIVE_DECODING' TOKEN_FILE='$REMOTE_STATE_DIR/bearer-token' bash /tmp/hydralisk-glm52-proxy-$RUN_ID.sh"

copy_artifacts
echo "OUTPUT_DIR=$OUTPUT_DIR"
