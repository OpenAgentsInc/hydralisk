#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-${CLOUDSDK_CORE_ACCOUNT:-}}"
REGION="${REGION:-us-central1}"
ACTION="${ACTION:-plan}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d%H%M%S)}"
REPLICA_REF="${REPLICA_REF:-glm52-reap-replica-$RUN_ID}"
REPLICA_PROFILE_REF="${REPLICA_PROFILE_REF:-glm-reap-504b-g4-tp4-mtp2-rp105}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/glm52-reap-replica-$REPLICA_REF-$RUN_ID}"

DO_NOT_TOUCH_EXISTING_LANES="${DO_NOT_TOUCH_EXISTING_LANES:-1}"
CREATE_IF_MISSING="${CREATE_IF_MISSING:-1}"
ALLOW_8G_FALLBACK="${ALLOW_8G_FALLBACK:-0}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
INSTANCE_TERMINATION_ACTION="${INSTANCE_TERMINATION_ACTION:-STOP}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-604800s}"
ENABLE_KEEPWARM_TIMER="${ENABLE_KEEPWARM_TIMER:-1}"
ALLOW_MODEL_KEEPWARM_SMOKE="${ALLOW_MODEL_KEEPWARM_SMOKE:-0}"
RUN_MODEL_SMOKES="${RUN_MODEL_SMOKES:-0}"

STAGING_MODE="${STAGING_MODE:-clone-disk}"
SOURCE_MODEL_DISK="${SOURCE_MODEL_DISK:-}"
MODEL_DISK_CLONE="${MODEL_DISK_CLONE:-hydralisk-glm52-model-${REPLICA_REF}-${RUN_ID}}"
MODEL_DISK_DEVICE_NAME="${MODEL_DISK_DEVICE_NAME:-hydralisk-glm52-model}"
MODEL_CLONE_MOUNT="${MODEL_CLONE_MOUNT:-/mnt/hydralisk-glm52-model-clone}"
MODEL_DIR="${MODEL_DIR:-/opt/hydralisk/models/glm-5.2-504b}"

WATCHDOG_SUFFIX="${WATCHDOG_SUFFIX:-$(python3 - "$REPLICA_REF" <<'PY'
import hashlib
import re
import sys

ref = sys.argv[1].lower()
safe = re.sub(r"[^a-z0-9-]+", "-", ref).strip("-") or "replica"
print(f"{safe[:7]}-{hashlib.sha1(ref.encode()).hexdigest()[:6]}")
PY
)}"
WATCHDOG_ROLE_SUFFIX="${WATCHDOG_ROLE_SUFFIX:-$(printf '%s' "$WATCHDOG_SUFFIX" | tr -cd '[:alnum:]' | cut -c1-16)}"
WATCHDOG_SERVICE_ACCOUNT_NAME="${WATCHDOG_SERVICE_ACCOUNT_NAME:-hydra-glm52-wd-${WATCHDOG_SUFFIX}}"
WATCHDOG_ROLE_ID="${WATCHDOG_ROLE_ID:-hydraliskGlm52${WATCHDOG_ROLE_SUFFIX}Wd}"
WATCHDOG_RUN_JOB="${WATCHDOG_RUN_JOB:-hydralisk-glm52-reap-watchdog-${WATCHDOG_SUFFIX}}"
WATCHDOG_SCHEDULER_JOB="${WATCHDOG_SCHEDULER_JOB:-hydralisk-glm52-reap-watchdog-${WATCHDOG_SUFFIX}-5m}"
KEEPWARM_LOG_DIR="${KEEPWARM_LOG_DIR:-/var/log/hydralisk/glm52-reap-keepwarm-${WATCHDOG_SUFFIX}}"

ADDRESS_NAME="${ADDRESS_NAME:-hydralisk-glm52-reap-${WATCHDOG_SUFFIX}-ingress}"
TARGET_TAG="${TARGET_TAG:-hydralisk-glm52-reap-${WATCHDOG_SUFFIX}-https}"
FIREWALL_RULE="${FIREWALL_RULE:-hydralisk-glm52-reap-${WATCHDOG_SUFFIX}-https}"
PUBLIC_HOSTNAME="${PUBLIC_HOSTNAME:-}"

TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"

run_gcloud() {
  if [[ -n "$GCLOUD_ACCOUNT" ]]; then
    CLOUDSDK_CORE_ACCOUNT="$GCLOUD_ACCOUNT" gcloud "$@"
  else
    gcloud "$@"
  fi
}

fail() {
  echo "error: $*" >&2
  exit 2
}

script_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

repo_root() {
  cd "$(script_dir)/.." && pwd
}

public_json() {
  python3 - "$@" <<'PY'
import json
import sys
from datetime import datetime, timezone

out = sys.argv[1]
pairs = sys.argv[2:]
doc = {
    "schema": "hydralisk.glm52_reap.replica_provision.v1",
    "generatedAt": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "publicSafe": True,
    "publicSafety": {
        "containsEndpoint": False,
        "containsBearerToken": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
        "containsRawLogs": False,
    },
}
for pair in pairs:
    key, _, value = pair.partition("=")
    if value.lower() in {"true", "false"}:
        doc[key] = value.lower() == "true"
    else:
        doc[key] = value
with open(out, "w") as f:
    json.dump(doc, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(doc, indent=2, sort_keys=True))
PY
}

validate_common() {
  [[ "$DO_NOT_TOUCH_EXISTING_LANES" == "1" ]] || fail "set DO_NOT_TOUCH_EXISTING_LANES=1; this script only creates new lanes"
  [[ -n "$REPLICA_REF" ]] || fail "REPLICA_REF is required"
  [[ "$REPLICA_REF" != *" "* ]] || fail "REPLICA_REF must not contain spaces"
  [[ "$REPLICA_REF" != "glm52-reap-primary-g4-tp4" ]] || fail "refusing to target the primary benchmark lane"
  [[ "$WATCHDOG_RUN_JOB" != "hydralisk-glm52-reap-watchdog" ]] || fail "watchdog run job must be distinct from the primary lane"
  [[ "$WATCHDOG_SCHEDULER_JOB" != "hydralisk-glm52-reap-watchdog-5m" ]] || fail "watchdog scheduler job must be distinct from the primary lane"
  [[ "$WATCHDOG_SERVICE_ACCOUNT_NAME" != "hydralisk-glm52-reap-watchdog" ]] || fail "watchdog service account must be distinct from the primary lane"
  ((${#WATCHDOG_SERVICE_ACCOUNT_NAME} <= 30)) || fail "watchdog service account name must be 30 chars or shorter"
  [[ "$WATCHDOG_ROLE_ID" =~ ^[A-Za-z0-9_\\.]{3,64}$ ]] || fail "watchdog role id must be a valid custom role id"
  case "$STAGING_MODE" in
    clone-disk|download) ;;
    *) fail "STAGING_MODE must be clone-disk or download" ;;
  esac
  if [[ "$STAGING_MODE" == "clone-disk" && -z "$SOURCE_MODEL_DISK" ]]; then
    fail "SOURCE_MODEL_DISK is required when STAGING_MODE=clone-disk"
  fi
}

ensure_output_dir() {
  if [[ -e "$OUTPUT_DIR" ]]; then
    fail "OUTPUT_DIR already exists; choose a new RUN_ID or OUTPUT_DIR: $OUTPUT_DIR"
  fi
  mkdir -p "$OUTPUT_DIR"
}

write_plan() {
  mkdir -p "$OUTPUT_DIR"
  public_json "$OUTPUT_DIR/replica-plan-public.json" \
    "action=$ACTION" \
    "runId=$RUN_ID" \
    "replicaRef=$REPLICA_REF" \
    "replicaProfileRef=$REPLICA_PROFILE_REF" \
    "stagingMode=$STAGING_MODE" \
    "sourceModelDiskRef=${SOURCE_MODEL_DISK:+operator-provided-same-zone-disk}" \
    "provisioningModel=$PROVISIONING_MODEL" \
    "maxRunDuration=$MAX_RUN_DURATION" \
    "watchdogRunJob=$WATCHDOG_RUN_JOB" \
    "watchdogSchedulerJob=$WATCHDOG_SCHEDULER_JOB" \
    "watchdogServiceAccount=$WATCHDOG_SERVICE_ACCOUNT_NAME" \
    "keepwarmLogDirRef=host-local-public-json" \
    "doNotTouchExistingLanes=$DO_NOT_TOUCH_EXISTING_LANES"
}

admit_replica() {
  local out="$OUTPUT_DIR/01-admission"
  ISSUE_NUMBER=97 \
  CREATE_IF_MISSING="$CREATE_IF_MISSING" \
  ALLOW_8G_FALLBACK="$ALLOW_8G_FALLBACK" \
  PROVISIONING_MODEL="$PROVISIONING_MODEL" \
  INSTANCE_TERMINATION_ACTION="$INSTANCE_TERMINATION_ACTION" \
  MAX_RUN_DURATION="$MAX_RUN_DURATION" \
  TS="$RUN_ID" \
  OUTPUT_DIR="$out" \
    "$(repo_root)/scripts/probe-glm-52-reap-504b-g4-gce.sh"
  TARGET_INSTANCE="$(cat "$out/admitted_instance")"
  TARGET_ZONE="$(cat "$out/admitted_zone")"
}

clone_and_mount_model_disk() {
  local out="$OUTPUT_DIR/02-model-disk-clone"
  mkdir -p "$out"
  if run_gcloud compute disks describe "$MODEL_DISK_CLONE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" >/dev/null 2>&1; then
    fail "model disk clone already exists; choose a new MODEL_DISK_CLONE: $MODEL_DISK_CLONE"
  fi
  run_gcloud compute disks create "$MODEL_DISK_CLONE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --source-disk "$SOURCE_MODEL_DISK" \
    --format=json > "$out/disk-create-public.json"
  run_gcloud compute instances attach-disk "$TARGET_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --disk "$MODEL_DISK_CLONE" \
    --device-name "$MODEL_DISK_DEVICE_NAME" \
    --mode=ro \
    --format=json > "$out/disk-attach-public.json"

  local remote_script
  remote_script="$(mktemp "${TMPDIR:-/tmp}/hydralisk-glm52-mount-clone.XXXXXXXXXX")"
  trap 'rm -f "$remote_script"' RETURN
  cat > "$remote_script" <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

device="$(readlink -f "/dev/disk/by-id/google-$MODEL_DISK_DEVICE_NAME")"
[[ -n "$device" ]] || { echo "missing attached model disk"; exit 1; }
partition="$device"
if [[ -e "${device}-part1" ]]; then
  partition="${device}-part1"
fi

sudo install -d -m 0755 "$MODEL_CLONE_MOUNT" /opt/hydralisk/models
if ! findmnt -rno TARGET "$MODEL_CLONE_MOUNT" >/dev/null 2>&1; then
  sudo mount -o ro "$partition" "$MODEL_CLONE_MOUNT"
fi

candidate="$MODEL_CLONE_MOUNT/opt/hydralisk/models/glm-5.2-504b"
if [[ ! -d "$candidate" ]]; then
  candidate="$MODEL_CLONE_MOUNT/glm-5.2-504b"
fi
[[ -f "$candidate/model.safetensors.index.json" ]] || {
  echo "model index not found on mounted clone";
  exit 1;
}

sudo rm -rf "$MODEL_DIR"
sudo ln -s "$candidate" "$MODEL_DIR"
mount_options="$(findmnt -rno OPTIONS "$MODEL_CLONE_MOUNT" || true)"
[[ "$mount_options" == *ro* ]] || {
  echo "model clone is not mounted read-only";
  exit 1;
}

python3 - <<'PY'
import json
import os
from pathlib import Path
model_dir = Path(os.environ["MODEL_DIR"])
shards = sorted(model_dir.glob("*.safetensors"))
doc = {
    "schema": "hydralisk.glm52_reap.model_clone_mount.v1",
    "publicSafe": True,
    "mountRef": "host-local-read-only-model-clone",
    "modelDir": str(model_dir),
    "indexExists": (model_dir / "model.safetensors.index.json").exists(),
    "safetensorShardCount": len(shards),
    "readOnlyMount": True,
    "publicSafety": {
        "containsEndpoint": False,
        "containsBearerToken": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
        "containsRawLogs": False
    }
}
print(json.dumps(doc, indent=2, sort_keys=True))
PY
REMOTE
  run_gcloud compute scp "$remote_script" "$TARGET_INSTANCE:/tmp/hydralisk-glm52-mount-clone.sh" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet
  run_gcloud compute ssh "$TARGET_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet \
    --command="MODEL_DISK_DEVICE_NAME='$MODEL_DISK_DEVICE_NAME' MODEL_CLONE_MOUNT='$MODEL_CLONE_MOUNT' MODEL_DIR='$MODEL_DIR' bash /tmp/hydralisk-glm52-mount-clone.sh" \
    > "$out/model-clone-mount-public.json"
}

stage_model_download() {
  ACTION=run \
  TARGET_INSTANCE="$TARGET_INSTANCE" \
  TARGET_ZONE="$TARGET_ZONE" \
  MODEL_DIR="$MODEL_DIR" \
  RUN_ID="$RUN_ID" \
  OUTPUT_DIR="$OUTPUT_DIR/02-model-download" \
    "$(repo_root)/scripts/stage-glm-52-reap-504b-gce.sh"
}

launch_vllm() {
  ACTION=start \
  TARGET_INSTANCE="$TARGET_INSTANCE" \
  TARGET_ZONE="$TARGET_ZONE" \
  MODEL_DIR="$MODEL_DIR" \
  RUN_ID="$RUN_ID" \
  OUTPUT_DIR="$OUTPUT_DIR/03-vllm-launch" \
    "$(repo_root)/scripts/launch-glm-52-reap-504b-b12x-gce.sh"
  ACTION=status \
  TARGET_INSTANCE="$TARGET_INSTANCE" \
  TARGET_ZONE="$TARGET_ZONE" \
  MODEL_DIR="$MODEL_DIR" \
  RUN_ID="$RUN_ID" \
  OUTPUT_DIR="$OUTPUT_DIR/04-vllm-status" \
    "$(repo_root)/scripts/launch-glm-52-reap-504b-b12x-gce.sh"
}

private_proxy_host() {
  run_gcloud compute instances describe "$TARGET_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --format='value(networkInterfaces[0].networkIP)'
}

install_private_proxy() {
  local proxy_host
  proxy_host="$(private_proxy_host)"
  ACTION=install-systemd \
  TARGET_INSTANCE="$TARGET_INSTANCE" \
  TARGET_ZONE="$TARGET_ZONE" \
  PROXY_HOST="$proxy_host" \
  RUN_ID="$RUN_ID" \
  OUTPUT_DIR="$OUTPUT_DIR/05-private-proxy" \
  REPLICA_REF="$REPLICA_REF" \
  REPLICA_PROFILE_REF="$REPLICA_PROFILE_REF" \
  PROVISIONING_CLASS="$(printf '%s' "$PROVISIONING_MODEL" | tr '[:upper:]' '[:lower:]')" \
  MAX_RUN_DURATION_PRESENT=true \
  WATCHDOG_REF="$WATCHDOG_SCHEDULER_JOB" \
  WATCHDOG_STATUS=configured \
  KEEPWARM_STATUS_PATH="$KEEPWARM_LOG_DIR/latest-public.json" \
    "$(repo_root)/scripts/expose-glm-52-reap-504b-private-proxy-gce.sh"
  if [[ "$RUN_MODEL_SMOKES" == "1" ]]; then
    ACTION=smoke \
    TARGET_INSTANCE="$TARGET_INSTANCE" \
    TARGET_ZONE="$TARGET_ZONE" \
    PROXY_HOST="$proxy_host" \
    RUN_ID="$RUN_ID" \
    OUTPUT_DIR="$OUTPUT_DIR/06-private-proxy-smoke" \
      "$(repo_root)/scripts/expose-glm-52-reap-504b-private-proxy-gce.sh"
  fi
}

install_public_https() {
  ACTION=setup \
  TARGET_INSTANCE="$TARGET_INSTANCE" \
  TARGET_ZONE="$TARGET_ZONE" \
  REGION="$REGION" \
  ADDRESS_NAME="$ADDRESS_NAME" \
  TARGET_TAG="$TARGET_TAG" \
  FIREWALL_RULE="$FIREWALL_RULE" \
  PUBLIC_HOSTNAME="$PUBLIC_HOSTNAME" \
  RUN_ID="$RUN_ID" \
  OUTPUT_DIR="$OUTPUT_DIR/07-public-https" \
    "$(repo_root)/scripts/expose-glm-52-reap-504b-public-https-gce.sh"
  if [[ "$RUN_MODEL_SMOKES" == "1" ]]; then
    ACTION=smoke \
    TARGET_INSTANCE="$TARGET_INSTANCE" \
    TARGET_ZONE="$TARGET_ZONE" \
    REGION="$REGION" \
    ADDRESS_NAME="$ADDRESS_NAME" \
    PUBLIC_HOSTNAME="$PUBLIC_HOSTNAME" \
    RUN_ID="$RUN_ID" \
    OUTPUT_DIR="$OUTPUT_DIR/08-public-https-smoke" \
      "$(repo_root)/scripts/expose-glm-52-reap-504b-public-https-gce.sh"
  fi
}

install_durable_watchdog() {
  ACTION=setup \
  TARGET_INSTANCE="$TARGET_INSTANCE" \
  TARGET_ZONE="$TARGET_ZONE" \
  REGION="$REGION" \
  WATCHDOG_SERVICE_ACCOUNT_NAME="$WATCHDOG_SERVICE_ACCOUNT_NAME" \
  WATCHDOG_ROLE_ID="$WATCHDOG_ROLE_ID" \
  WATCHDOG_RUN_JOB="$WATCHDOG_RUN_JOB" \
  WATCHDOG_SCHEDULER_JOB="$WATCHDOG_SCHEDULER_JOB" \
  KEEPWARM_LOG_DIR="$KEEPWARM_LOG_DIR" \
  ENABLE_KEEPWARM_TIMER="$ENABLE_KEEPWARM_TIMER" \
  RUN_ID="$RUN_ID" \
  OUTPUT_DIR="$OUTPUT_DIR/09-durable-watchdog" \
    "$(repo_root)/scripts/install-glm-52-reap-504b-durable-canary-gce.sh"
  ACTION=smoke \
  TARGET_INSTANCE="$TARGET_INSTANCE" \
  TARGET_ZONE="$TARGET_ZONE" \
  REGION="$REGION" \
  WATCHDOG_SERVICE_ACCOUNT_NAME="$WATCHDOG_SERVICE_ACCOUNT_NAME" \
  WATCHDOG_ROLE_ID="$WATCHDOG_ROLE_ID" \
  WATCHDOG_RUN_JOB="$WATCHDOG_RUN_JOB" \
  WATCHDOG_SCHEDULER_JOB="$WATCHDOG_SCHEDULER_JOB" \
  KEEPWARM_LOG_DIR="$KEEPWARM_LOG_DIR" \
  ALLOW_MODEL_KEEPWARM_SMOKE="$ALLOW_MODEL_KEEPWARM_SMOKE" \
  RUN_ID="$RUN_ID" \
  OUTPUT_DIR="$OUTPUT_DIR/10-durable-smoke" \
    "$(repo_root)/scripts/install-glm-52-reap-504b-durable-canary-gce.sh"
}

write_evidence() {
  public_json "$OUTPUT_DIR/replica-evidence-public.json" \
    "action=run" \
    "runId=$RUN_ID" \
    "replicaRef=$REPLICA_REF" \
    "replicaProfileRef=$REPLICA_PROFILE_REF" \
    "targetInstanceRef=$TARGET_INSTANCE" \
    "targetZone=$TARGET_ZONE" \
    "stagingMode=$STAGING_MODE" \
    "modelDiskCloneRef=${STAGING_MODE/clone-disk/created-read-only-clone}" \
    "privateProxyEvidenceDir=05-private-proxy" \
    "publicHttpsEvidenceDir=07-public-https" \
    "durableEvidenceDir=09-durable-watchdog" \
    "cleanupPlan=delete only resources named by this evidence bundle"
}

status_existing() {
  [[ -n "$TARGET_INSTANCE" && -n "$TARGET_ZONE" ]] || fail "TARGET_INSTANCE and TARGET_ZONE are required for ACTION=status"
  ACTION=status TARGET_INSTANCE="$TARGET_INSTANCE" TARGET_ZONE="$TARGET_ZONE" RUN_ID="$RUN_ID" OUTPUT_DIR="$OUTPUT_DIR/status-vllm" \
    "$(repo_root)/scripts/launch-glm-52-reap-504b-b12x-gce.sh"
  ACTION=status TARGET_INSTANCE="$TARGET_INSTANCE" TARGET_ZONE="$TARGET_ZONE" RUN_ID="$RUN_ID" OUTPUT_DIR="$OUTPUT_DIR/status-private-proxy" \
    "$(repo_root)/scripts/expose-glm-52-reap-504b-private-proxy-gce.sh"
  ACTION=status TARGET_INSTANCE="$TARGET_INSTANCE" TARGET_ZONE="$TARGET_ZONE" REGION="$REGION" WATCHDOG_RUN_JOB="$WATCHDOG_RUN_JOB" WATCHDOG_SCHEDULER_JOB="$WATCHDOG_SCHEDULER_JOB" KEEPWARM_LOG_DIR="$KEEPWARM_LOG_DIR" RUN_ID="$RUN_ID" OUTPUT_DIR="$OUTPUT_DIR/status-durable" \
    "$(repo_root)/scripts/install-glm-52-reap-504b-durable-canary-gce.sh"
}

cleanup_plan() {
  mkdir -p "$OUTPUT_DIR"
  cat > "$OUTPUT_DIR/cleanup-plan-public.txt" <<EOF
Public-safe cleanup plan for replica $REPLICA_REF:

1. Do not touch primary GLM or active benchmark lanes.
2. Stop/delete only TARGET_INSTANCE=$TARGET_INSTANCE if it was created by this run.
3. Delete MODEL_DISK_CLONE=$MODEL_DISK_CLONE only if STAGING_MODE=clone-disk and this run created it.
4. Delete ADDRESS_NAME=$ADDRESS_NAME only if this run created it and no DNS/operator secret points at it.
5. Delete FIREWALL_RULE=$FIREWALL_RULE and remove TARGET_TAG=$TARGET_TAG only if no other replica uses them.
6. Delete Cloud Scheduler job $WATCHDOG_SCHEDULER_JOB, Cloud Run job $WATCHDOG_RUN_JOB, custom role $WATCHDOG_ROLE_ID, and service account $WATCHDOG_SERVICE_ACCOUNT_NAME only for this replica.
7. Keep public-safe evidence under $OUTPUT_DIR; do not commit raw GCE logs, bearer tokens, endpoint values, prompts, responses, or weights.
EOF
  cat "$OUTPUT_DIR/cleanup-plan-public.txt"
}

validate_common

case "$ACTION" in
  plan)
    write_plan
    cleanup_plan
    ;;
  run)
    ensure_output_dir
    write_plan
    admit_replica
    if [[ "$STAGING_MODE" == "clone-disk" ]]; then
      clone_and_mount_model_disk
    else
      stage_model_download
    fi
    launch_vllm
    install_private_proxy
    install_public_https
    install_durable_watchdog
    write_evidence
    cleanup_plan
    ;;
  status)
    mkdir -p "$OUTPUT_DIR"
    status_existing
    ;;
  cleanup-plan)
    cleanup_plan
    ;;
  *)
    fail "ACTION must be plan, run, status, or cleanup-plan"
    ;;
esac

echo "OUTPUT_DIR=$OUTPUT_DIR"
