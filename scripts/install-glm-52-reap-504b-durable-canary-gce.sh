#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-${CLOUDSDK_CORE_ACCOUNT:-}}"
TARGET_INSTANCE="${TARGET_INSTANCE:-hydralisk-glm52-reap-504b-g4-8g-b-20260624214500}"
TARGET_ZONE="${TARGET_ZONE:-us-central1-b}"
REGION="${REGION:-us-central1}"
ACTION="${ACTION:-setup}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/glm52-reap-durable-canary-$RUN_ID}"

PROXY_PORT="${PROXY_PORT:-8080}"
TOKEN_FILE="${TOKEN_FILE:-/var/lib/hydralisk/glm52-reap-private-proxy/bearer-token}"
KEEPWARM_SERVICE="${KEEPWARM_SERVICE:-hydralisk-glm52-reap-keepwarm.service}"
KEEPWARM_TIMER="${KEEPWARM_TIMER:-hydralisk-glm52-reap-keepwarm.timer}"
KEEPWARM_LOG_DIR="${KEEPWARM_LOG_DIR:-/var/log/hydralisk/glm52-reap-keepwarm}"
KEEPWARM_CADENCE="${KEEPWARM_CADENCE:-4min}"
KEEPWARM_TIMEOUT_SECONDS="${KEEPWARM_TIMEOUT_SECONDS:-90}"
KEEPWARM_MAX_TOKENS="${KEEPWARM_MAX_TOKENS:-16}"
ENABLE_KEEPWARM_TIMER="${ENABLE_KEEPWARM_TIMER:-0}"
ALLOW_MODEL_KEEPWARM_SMOKE="${ALLOW_MODEL_KEEPWARM_SMOKE:-0}"

WATCHDOG_SERVICE_ACCOUNT_NAME="${WATCHDOG_SERVICE_ACCOUNT_NAME:-hydralisk-glm52-reap-watchdog}"
WATCHDOG_ROLE_ID="${WATCHDOG_ROLE_ID:-hydraliskGlm52Watchdog}"
WATCHDOG_RUN_JOB="${WATCHDOG_RUN_JOB:-hydralisk-glm52-reap-watchdog}"
WATCHDOG_SCHEDULER_JOB="${WATCHDOG_SCHEDULER_JOB:-hydralisk-glm52-reap-watchdog-5m}"
WATCHDOG_SCHEDULE="${WATCHDOG_SCHEDULE:-*/5 * * * *}"
WATCHDOG_IMAGE="${WATCHDOG_IMAGE:-gcr.io/google.com/cloudsdktool/google-cloud-cli:slim}"
WATCHDOG_TASK_TIMEOUT="${WATCHDOG_TASK_TIMEOUT:-600s}"

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

service_account_email() {
  printf '%s@%s.iam.gserviceaccount.com\n' "$WATCHDOG_SERVICE_ACCOUNT_NAME" "$PROJECT_ID"
}

private_proxy_base_url() {
  local private_ip
  private_ip="$(
    run_gcloud compute instances describe "$TARGET_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --format='value(networkInterfaces[0].networkIP)'
  )"
  if [[ -z "$private_ip" ]]; then
    echo "error: could not resolve private proxy address" >&2
    exit 1
  fi
  printf 'http://%s:%s\n' "$private_ip" "$PROXY_PORT"
}

install_keepwarm() {
  local base_url remote_script
  base_url="$(private_proxy_base_url)"
  remote_script="$(mktemp "${TMPDIR:-/tmp}/hydralisk-glm52-keepwarm.XXXXXXXXXX")"
  trap 'rm -f "$remote_script"' RETURN

  cat > "$remote_script" <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

sudo install -d -m 0755 /opt/hydralisk/bin "$KEEPWARM_LOG_DIR"

sudo tee /opt/hydralisk/bin/glm52-reap-keepwarm.py >/dev/null <<'PY'
#!/usr/bin/env python3
import datetime as dt
import hashlib
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

base_url = os.environ["HYDRALISK_KEEPWARM_BASE_URL"].rstrip("/")
token_file = pathlib.Path(os.environ["HYDRALISK_TOKEN_FILE"])
log_dir = pathlib.Path(os.environ["HYDRALISK_KEEPWARM_LOG_DIR"])
timeout = float(os.environ.get("HYDRALISK_KEEPWARM_TIMEOUT_SECONDS", "90"))
max_tokens = int(os.environ.get("HYDRALISK_KEEPWARM_MAX_TOKENS", "16"))

log_dir.mkdir(parents=True, exist_ok=True)
token = token_file.read_text().strip()
prompt = "Return the word WARM and one short readiness phrase."
payload = {
    "model": "openagents/glm-5.2-reap-504b",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0,
}
data = json.dumps(payload).encode("utf-8")
request = urllib.request.Request(
    base_url + "/v1/chat/completions",
    data=data,
    headers={
        "authorization": "Bearer " + token,
        "content-type": "application/json",
    },
    method="POST",
)

start = time.perf_counter()
http_status = None
body_text = ""
error_type = None
error_summary = None
try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        http_status = response.status
        body_text = response.read().decode("utf-8", errors="replace")
except urllib.error.HTTPError as exc:
    http_status = exc.code
    body_text = exc.read().decode("utf-8", errors="replace")
    error_type = type(exc).__name__
    try:
        error_summary = json.loads(body_text).get("detail")
    except Exception:
        error_summary = body_text[:180]
except Exception as exc:
    error_type = type(exc).__name__
    error_summary = str(exc)[:180]
wall = time.perf_counter() - start

usage = None
content = ""
proxy_run_ref = None
proxy_receipt_ref = None
if body_text:
    try:
        body = json.loads(body_text)
        usage = body.get("usage")
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        if isinstance(message.get("content"), str):
            content = message["content"]
        metadata = body.get("hydralisk") or {}
        proxy_run_ref = metadata.get("runRef") or body.get("id")
        proxy_receipt_ref = metadata.get("receiptRef")
        if http_status and http_status >= 400 and error_summary is None:
            error_summary = body.get("detail") or body.get("error")
    except Exception:
        if http_status and http_status >= 400 and error_summary is None:
            error_summary = body_text[:180]

if http_status == 200:
    status = "passed"
elif http_status == 429:
    status = "busy"
else:
    status = "failed"

summary = {
    "schema": "hydralisk.keepwarm.v1",
    "status": status,
    "checkedAt": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "baseUrlRef": "gce-internal-proxy:8080",
    "httpStatus": http_status,
    "errorType": error_type,
    "errorSummary": error_summary if isinstance(error_summary, str) else json.dumps(error_summary, sort_keys=True)[:240] if error_summary is not None else None,
    "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    "visibleCompletionChars": len(content),
    "visibleCompletionSha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    "usage": usage,
    "timing": {"wallSeconds": wall},
    "proxyRunRef": proxy_run_ref,
    "proxyReceiptRef": proxy_receipt_ref,
    "publicSafety": {
        "containsSecrets": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsHiddenReasoning": False,
        "containsWeights": False,
    },
}

line = json.dumps(summary, sort_keys=True)
(log_dir / "latest-public.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
with (log_dir / "keepwarm-public.jsonl").open("a") as f:
    f.write(line + "\n")
print(line)
PY
sudo chmod 0755 /opt/hydralisk/bin/glm52-reap-keepwarm.py

sudo tee "/etc/systemd/system/$KEEPWARM_SERVICE" >/dev/null <<EOF
[Unit]
Description=Hydralisk GLM-5.2 REAP private keep-warm probe
After=network-online.target hydralisk-glm52-reap-private-proxy.service
Wants=network-online.target hydralisk-glm52-reap-private-proxy.service

[Service]
Type=oneshot
Environment=HYDRALISK_KEEPWARM_BASE_URL=$KEEPWARM_BASE_URL
Environment=HYDRALISK_TOKEN_FILE=$TOKEN_FILE
Environment=HYDRALISK_KEEPWARM_LOG_DIR=$KEEPWARM_LOG_DIR
Environment=HYDRALISK_KEEPWARM_TIMEOUT_SECONDS=$KEEPWARM_TIMEOUT_SECONDS
Environment=HYDRALISK_KEEPWARM_MAX_TOKENS=$KEEPWARM_MAX_TOKENS
ExecStart=/usr/bin/python3 /opt/hydralisk/bin/glm52-reap-keepwarm.py
Nice=10
EOF

sudo tee "/etc/systemd/system/$KEEPWARM_TIMER" >/dev/null <<EOF
[Unit]
Description=Run Hydralisk GLM-5.2 REAP keep-warm probe every $KEEPWARM_CADENCE

[Timer]
OnBootSec=2min
OnUnitActiveSec=$KEEPWARM_CADENCE
AccuracySec=15s
Unit=$KEEPWARM_SERVICE
Persistent=false

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
if [[ "$ENABLE_KEEPWARM_TIMER" == "1" ]]; then
  sudo systemctl enable --now "$KEEPWARM_TIMER"
else
  sudo systemctl disable --now "$KEEPWARM_TIMER" >/dev/null 2>&1 || true
fi
if [[ "${RUN_KEEPWARM_ON_INSTALL:-0}" == "1" ]]; then
  sudo systemctl start "$KEEPWARM_SERVICE" || true
fi
systemctl is-enabled "$KEEPWARM_TIMER"
systemctl is-active "$KEEPWARM_TIMER"
REMOTE

  run_gcloud compute scp "$remote_script" "$TARGET_INSTANCE:/tmp/hydralisk-glm52-keepwarm.sh" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet >/dev/null
  run_gcloud compute ssh "$TARGET_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet \
    --command "KEEPWARM_BASE_URL='$base_url' TOKEN_FILE='$TOKEN_FILE' KEEPWARM_SERVICE='$KEEPWARM_SERVICE' KEEPWARM_TIMER='$KEEPWARM_TIMER' KEEPWARM_LOG_DIR='$KEEPWARM_LOG_DIR' KEEPWARM_CADENCE='$KEEPWARM_CADENCE' KEEPWARM_TIMEOUT_SECONDS='$KEEPWARM_TIMEOUT_SECONDS' KEEPWARM_MAX_TOKENS='$KEEPWARM_MAX_TOKENS' ENABLE_KEEPWARM_TIMER='$ENABLE_KEEPWARM_TIMER' bash /tmp/hydralisk-glm52-keepwarm.sh" \
    > "$OUTPUT_DIR/keepwarm-install-public.txt"
}

ensure_watchdog_iam() {
  local sa role_name
  sa="$(service_account_email)"
  role_name="projects/$PROJECT_ID/roles/$WATCHDOG_ROLE_ID"

  if ! run_gcloud iam service-accounts describe "$sa" \
    --project "$PROJECT_ID" >/dev/null 2>&1; then
    run_gcloud iam service-accounts create "$WATCHDOG_SERVICE_ACCOUNT_NAME" \
      --project "$PROJECT_ID" \
      --display-name "Hydralisk GLM-5.2 REAP watchdog" \
      > "$OUTPUT_DIR/watchdog-service-account-create.txt"
  fi

  if ! run_gcloud iam roles describe "$WATCHDOG_ROLE_ID" \
    --project "$PROJECT_ID" >/dev/null 2>&1; then
    run_gcloud iam roles create "$WATCHDOG_ROLE_ID" \
      --project "$PROJECT_ID" \
      --title "Hydralisk GLM-5.2 REAP watchdog" \
      --description "Can read and start the GLM-5.2 REAP GCE canary instance" \
      --permissions compute.instances.get,compute.instances.start \
      --stage GA \
      > "$OUTPUT_DIR/watchdog-role-create.txt"
  else
    run_gcloud iam roles update "$WATCHDOG_ROLE_ID" \
      --project "$PROJECT_ID" \
      --title "Hydralisk GLM-5.2 REAP watchdog" \
      --description "Can read and start the GLM-5.2 REAP GCE canary instance" \
      --permissions compute.instances.get,compute.instances.start \
      --stage GA \
      > "$OUTPUT_DIR/watchdog-role-update.txt"
  fi

  run_gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:$sa" \
    --role "$role_name" \
    --condition=None \
    > "$OUTPUT_DIR/watchdog-compute-iam-binding.txt"

  run_gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:$sa" \
    --role roles/run.invoker \
    --condition=None \
    > "$OUTPUT_DIR/watchdog-run-invoker-binding.txt"
}

watchdog_command() {
  cat <<'EOF'
set -euo pipefail; checked_at="$(date -u +%FT%TZ)"; status="$(gcloud compute instances describe "$TARGET_INSTANCE" --project "$PROJECT_ID" --zone "$TARGET_ZONE" --format="value(status)")"; action="none"; if [[ "$status" != "RUNNING" ]]; then gcloud compute instances start "$TARGET_INSTANCE" --project "$PROJECT_ID" --zone "$TARGET_ZONE" --quiet >/tmp/start.log 2>&1; action="start_requested"; fi; printf 'schema=hydralisk.glm52_reap.watchdog.v1 checkedAt=%s targetInstanceRef=%s targetZone=%s observedStatus=%s action=%s publicSafe=true\n' "$checked_at" "$TARGET_INSTANCE" "$TARGET_ZONE" "$status" "$action"
EOF
}

ensure_watchdog_job() {
  local sa command
  sa="$(service_account_email)"
  command="$(watchdog_command)"

  if run_gcloud run jobs describe "$WATCHDOG_RUN_JOB" \
    --project "$PROJECT_ID" \
    --region "$REGION" >/dev/null 2>&1; then
    run_gcloud run jobs update "$WATCHDOG_RUN_JOB" \
      --project "$PROJECT_ID" \
      --region "$REGION" \
      --image "$WATCHDOG_IMAGE" \
      --service-account "$sa" \
      --set-env-vars "PROJECT_ID=$PROJECT_ID,TARGET_ZONE=$TARGET_ZONE,TARGET_INSTANCE=$TARGET_INSTANCE" \
      --command /bin/bash \
      --args "-lc,$command" \
      --task-timeout "$WATCHDOG_TASK_TIMEOUT" \
      --max-retries 0 \
      > "$OUTPUT_DIR/watchdog-run-job-update.txt"
  else
    run_gcloud run jobs create "$WATCHDOG_RUN_JOB" \
      --project "$PROJECT_ID" \
      --region "$REGION" \
      --image "$WATCHDOG_IMAGE" \
      --service-account "$sa" \
      --set-env-vars "PROJECT_ID=$PROJECT_ID,TARGET_ZONE=$TARGET_ZONE,TARGET_INSTANCE=$TARGET_INSTANCE" \
      --command /bin/bash \
      --args "-lc,$command" \
      --task-timeout "$WATCHDOG_TASK_TIMEOUT" \
      --max-retries 0 \
      --labels lane=hydralisk,workload=glm52-reap-504b,role=watchdog \
      > "$OUTPUT_DIR/watchdog-run-job-create.txt"
  fi
}

ensure_scheduler_job() {
  local sa uri
  sa="$(service_account_email)"
  uri="https://run.googleapis.com/v2/projects/$PROJECT_ID/locations/$REGION/jobs/$WATCHDOG_RUN_JOB:run"
  if run_gcloud scheduler jobs describe "$WATCHDOG_SCHEDULER_JOB" \
    --project "$PROJECT_ID" \
    --location "$REGION" >/dev/null 2>&1; then
    run_gcloud scheduler jobs update http "$WATCHDOG_SCHEDULER_JOB" \
      --project "$PROJECT_ID" \
      --location "$REGION" \
      --schedule "$WATCHDOG_SCHEDULE" \
      --time-zone Etc/UTC \
      --uri "$uri" \
      --http-method POST \
      --message-body '{}' \
      --update-headers 'Content-Type=application/json' \
      --oauth-service-account-email "$sa" \
      --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
      --attempt-deadline 180s \
      --max-retry-attempts 3 \
      --min-backoff 10s \
      --max-backoff 60s \
      --description "Hydralisk GLM-5.2 REAP Spot canary watchdog; executes a Cloud Run job that conditionally starts the VM." \
      > "$OUTPUT_DIR/watchdog-scheduler-update.txt"
  else
    run_gcloud scheduler jobs create http "$WATCHDOG_SCHEDULER_JOB" \
      --project "$PROJECT_ID" \
      --location "$REGION" \
      --schedule "$WATCHDOG_SCHEDULE" \
      --time-zone Etc/UTC \
      --uri "$uri" \
      --http-method POST \
      --message-body '{}' \
      --headers 'Content-Type=application/json' \
      --oauth-service-account-email "$sa" \
      --oauth-token-scope https://www.googleapis.com/auth/cloud-platform \
      --attempt-deadline 180s \
      --max-retry-attempts 3 \
      --min-backoff 10s \
      --max-backoff 60s \
      --description "Hydralisk GLM-5.2 REAP Spot canary watchdog; executes a Cloud Run job that conditionally starts the VM." \
      > "$OUTPUT_DIR/watchdog-scheduler-create.txt"
  fi
}

install_watchdog() {
  ensure_watchdog_iam
  ensure_watchdog_job
  ensure_scheduler_job
}

smoke_keepwarm() {
  if [[ "$ALLOW_MODEL_KEEPWARM_SMOKE" != "1" ]]; then
    python3 - <<'PY' "$OUTPUT_DIR/keepwarm-smoke-public.json"
import json
import sys
from datetime import datetime, timezone

payload = {
    "schema": "hydralisk.glm52_reap.keepwarm_smoke.v1",
    "status": "skipped",
    "reason": "model_keepwarm_smoke_requires_explicit_opt_in",
    "checkedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "publicSafe": True,
    "publicSafety": {
        "containsSecrets": False,
        "containsEndpoint": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
    },
}
with open(sys.argv[1], "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
    return 0
  fi
  run_gcloud compute ssh "$TARGET_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet \
    --command "sudo systemctl start '$KEEPWARM_SERVICE' || true; sudo sed -n '1,220p' '$KEEPWARM_LOG_DIR/latest-public.json'" \
    > "$OUTPUT_DIR/keepwarm-smoke-public.json"
}

smoke_watchdog() {
  run_gcloud run jobs execute "$WATCHDOG_RUN_JOB" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --wait \
    > "$OUTPUT_DIR/watchdog-execute-public.txt"
}

write_status() {
  local sa instance_json scheduler_json run_job_json host_status keepwarm_latest
  sa="$(service_account_email)"
  instance_json="$OUTPUT_DIR/instance-public-source.json"
  scheduler_json="$OUTPUT_DIR/scheduler-public-source.json"
  run_job_json="$OUTPUT_DIR/run-job-public-source.json"

  run_gcloud compute instances describe "$TARGET_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --format=json > "$instance_json"
  run_gcloud scheduler jobs describe "$WATCHDOG_SCHEDULER_JOB" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --format=json > "$scheduler_json" 2>/dev/null || printf '{}\n' > "$scheduler_json"
  run_gcloud run jobs describe "$WATCHDOG_RUN_JOB" \
    --project "$PROJECT_ID" \
    --region "$REGION" \
    --format=json > "$run_job_json" 2>/dev/null || printf '{}\n' > "$run_job_json"
  host_status="$(
    run_gcloud compute ssh "$TARGET_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --quiet \
      --command "set -euo pipefail; printf 'docker=%s:%s\n' \"\$(systemctl is-enabled docker 2>/dev/null || true)\" \"\$(systemctl is-active docker 2>/dev/null || true)\"; printf 'caddy=%s:%s\n' \"\$(systemctl is-enabled caddy 2>/dev/null || true)\" \"\$(systemctl is-active caddy 2>/dev/null || true)\"; printf 'proxy=%s:%s\n' \"\$(systemctl is-enabled hydralisk-glm52-reap-private-proxy.service 2>/dev/null || true)\" \"\$(systemctl is-active hydralisk-glm52-reap-private-proxy.service 2>/dev/null || true)\"; printf 'keepwarmTimer=%s:%s:%s\n' \"\$(systemctl is-enabled '$KEEPWARM_TIMER' 2>/dev/null || true)\" \"\$(systemctl is-active '$KEEPWARM_TIMER' 2>/dev/null || true)\" \"\$(systemctl show '$KEEPWARM_TIMER' -p NextElapseUSecRealtime --value 2>/dev/null || true)\"; printf 'health='; curl -fsS \$(hostname -I | awk '{print \"http://\" \$1 \":$PROXY_PORT/health\"}') || true" \
      2>/dev/null || true
  )"
  keepwarm_latest="$(
    run_gcloud compute ssh "$TARGET_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --quiet \
      --command "sudo sed -n '1,220p' '$KEEPWARM_LOG_DIR/latest-public.json' 2>/dev/null || true" \
      2>/dev/null || true
  )"

  python3 - <<'PY' "$OUTPUT_DIR/durable-canary-status-public.json" "$instance_json" "$scheduler_json" "$run_job_json" "$host_status" "$keepwarm_latest" "$PROJECT_ID" "$TARGET_INSTANCE" "$TARGET_ZONE" "$REGION" "$WATCHDOG_RUN_JOB" "$WATCHDOG_SCHEDULER_JOB" "$sa"
import json
import pathlib
import sys

(
    out,
    instance_path,
    scheduler_path,
    run_job_path,
    host_status,
    keepwarm_latest,
    project_id,
    target_instance,
    target_zone,
    region,
    run_job_name,
    scheduler_job_name,
    service_account,
) = sys.argv[1:]

instance = json.loads(pathlib.Path(instance_path).read_text())
scheduler = json.loads(pathlib.Path(scheduler_path).read_text())
run_job = json.loads(pathlib.Path(run_job_path).read_text())
try:
    keepwarm = json.loads(keepwarm_latest) if keepwarm_latest.strip() else None
except Exception:
    keepwarm = {"parseError": True}

scheduling = instance.get("scheduling") or {}
disks = instance.get("disks") or []
boot = next((d for d in disks if d.get("boot")), disks[0] if disks else {})
host_lines = dict(
    line.split("=", 1)
    for line in host_status.splitlines()
    if "=" in line
)

doc = {
    "schema": "hydralisk.glm52_reap.durable_canary_status.v1",
    "publicSafe": True,
    "targetInstance": target_instance,
    "targetZone": target_zone,
    "region": region,
    "machineType": (instance.get("machineType") or "").split("/")[-1],
    "status": instance.get("status"),
    "provisioningModel": scheduling.get("provisioningModel"),
    "terminationAction": scheduling.get("instanceTerminationAction"),
    "maxRunDurationPresent": "maxRunDuration" in scheduling,
    "automaticRestart": scheduling.get("automaticRestart"),
    "bootDiskAutoDelete": boot.get("autoDelete"),
    "bootDiskSizeGb": boot.get("diskSizeGb"),
    "services": host_lines,
    "keepwarmLatest": keepwarm,
    "watchdog": {
        "serviceAccount": service_account,
        "runJob": run_job_name,
        "runJobConfigured": bool(run_job),
        "schedulerJob": scheduler_job_name,
        "schedulerConfigured": bool(scheduler),
        "schedule": scheduler.get("schedule"),
        "uriShape": "https://run.googleapis.com/v2/projects/<project>/locations/<region>/jobs/<job>:run",
        "oauthToken": "service-account",
    },
    "costModelMonthlyUsd": {
        "g4-standard-384": {
            "spot": 5392.42,
            "onDemand": 26279.59,
            "dwsFlexStart": 13140.00,
        },
        "g4-standard-192": {
            "spot": 2696.21,
            "onDemand": 13139.80,
            "dwsFlexStart": 6570.00,
        },
    },
    "publicSafety": {
        "containsSecrets": False,
        "containsEndpoint": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
    },
}
pathlib.Path(out).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
print(json.dumps(doc, indent=2, sort_keys=True))
PY
}

case "$ACTION" in
  setup)
    install_keepwarm
    install_watchdog
    write_status
    ;;
  setup-keepwarm)
    install_keepwarm
    write_status
    ;;
  setup-watchdog)
    install_watchdog
    write_status
    ;;
  status)
    write_status
    ;;
  smoke)
    smoke_keepwarm
    smoke_watchdog
    write_status
    ;;
  *)
    echo "error: unknown ACTION=$ACTION; expected setup, setup-keepwarm, setup-watchdog, status, or smoke" >&2
    exit 2
    ;;
esac
