#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
ISSUE_NUMBER="${ISSUE_NUMBER:-83}"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-${CLOUDSDK_CORE_ACCOUNT:-}}"
TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"
CREATE_IF_MISSING="${CREATE_IF_MISSING:-1}"
ALLOW_8G_FALLBACK="${ALLOW_8G_FALLBACK:-0}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
INSTANCE_TERMINATION_ACTION="${INSTANCE_TERMINATION_ACTION:-STOP}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-21600s}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-hyperdisk-balanced}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-1500GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
KEEP_INSTANCE="${KEEP_INSTANCE:-1}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/glm52-reap-504b-g4-admission-$TS}"

mkdir -p "$OUTPUT_DIR"

PLAN_TSV="$OUTPUT_DIR/glm52-reap-g4-admission-plan.tsv"
ATTEMPTS_TSV="$OUTPUT_DIR/glm52-reap-g4-admission-attempts.tsv"

cat > "$PLAN_TSV" <<'EOF'
order	label	zone	machine	accelerator	gpu_count	role
1	g4-4g-b	us-central1-b	g4-standard-192	nvidia-rtx-pro-6000	4	primary_4x_g4
2	g4-4g-f	us-central1-f	g4-standard-192	nvidia-rtx-pro-6000	4	primary_4x_g4_alt_zone
3	g4-8g-b	us-central1-b	g4-standard-384	nvidia-rtx-pro-6000	8	explicit_fallback_only
4	g4-8g-f	us-central1-f	g4-standard-384	nvidia-rtx-pro-6000	8	explicit_fallback_only
EOF

cat > "$ATTEMPTS_TSV" <<'EOF'
order	instance	zone	machine	accelerator	gpu_count	status	blocker
EOF

run_gcloud() {
  if [[ -n "$GCLOUD_ACCOUNT" ]]; then
    CLOUDSDK_CORE_ACCOUNT="$GCLOUD_ACCOUNT" gcloud "$@"
  else
    gcloud "$@"
  fi
}

sanitize_blocker() {
  tr '\n\t' '  ' | sed 's/  */ /g' | cut -c1-1800
}

refuse_unsafe_target() {
  if [[ -n "$TARGET_INSTANCE" && "$TARGET_INSTANCE" != hydralisk-glm52-reap-504b-* ]]; then
    echo "error: TARGET_INSTANCE must be a hydralisk-glm52-reap-504b-* host" >&2
    exit 2
  fi
  if [[ -n "$TARGET_INSTANCE" && -z "$TARGET_ZONE" ]]; then
    echo "error: TARGET_ZONE is required when TARGET_INSTANCE is set" >&2
    exit 2
  fi
}

attempt_create() {
  local order="$1" label="$2" zone="$3" machine="$4" accelerator="$5" count="$6" role="$7"
  local instance="hydralisk-glm52-reap-504b-${label}-${TS}"
  local log="$OUTPUT_DIR/create-${label}.log"
  local scheduling_args=()

  if [[ "$role" == "explicit_fallback_only" && "$ALLOW_8G_FALLBACK" != "1" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tskipped\tset ALLOW_8G_FALLBACK=1 after a documented 4x G4 blocker\n' \
      "$order" "$instance" "$zone" "$machine" "$accelerator" "$count" >> "$ATTEMPTS_TSV"
    return 1
  fi

  if [[ "$CREATE_IF_MISSING" != "1" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tskipped\tCREATE_IF_MISSING is not 1\n' \
      "$order" "$instance" "$zone" "$machine" "$accelerator" "$count" >> "$ATTEMPTS_TSV"
    return 1
  fi

  echo "Attempt $order: $instance $zone $machine $accelerator x$count"
  if [[ "$PROVISIONING_MODEL" == "SPOT" ]]; then
    scheduling_args=(
      --provisioning-model "$PROVISIONING_MODEL"
      --instance-termination-action "$INSTANCE_TERMINATION_ACTION"
      --max-run-duration "$MAX_RUN_DURATION"
    )
  else
    scheduling_args=(--provisioning-model "$PROVISIONING_MODEL")
  fi

  if run_gcloud compute instances create "$instance" \
    --project "$PROJECT_ID" \
    --zone "$zone" \
    --machine-type "$machine" \
    --maintenance-policy TERMINATE \
    "${scheduling_args[@]}" \
    --boot-disk-size "$BOOT_DISK_SIZE" \
    --boot-disk-type "$BOOT_DISK_TYPE" \
    --image-family "$IMAGE_FAMILY" \
    --image-project "$IMAGE_PROJECT" \
    --accelerator "type=$accelerator,count=$count" \
    --no-address \
    --metadata enable-oslogin=TRUE \
    --tags hydralisk-probe,glm-52,glm-52-reap \
    --labels lane=hydralisk,workload=glm52-reap-504b,model=glm-5-2-reap-504b,probe="$label",issue="$ISSUE_NUMBER" \
    --format=json > "$log" 2>&1; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tadmitted\t\n' \
      "$order" "$instance" "$zone" "$machine" "$accelerator" "$count" >> "$ATTEMPTS_TSV"
    TARGET_INSTANCE="$instance"
    TARGET_ZONE="$zone"
    echo "$TARGET_INSTANCE" > "$OUTPUT_DIR/admitted_instance"
    echo "$TARGET_ZONE" > "$OUTPUT_DIR/admitted_zone"
    return 0
  fi

  local blocker
  blocker="$(tail -n 80 "$log" | sanitize_blocker)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\tblocked\t%s\n' \
    "$order" "$instance" "$zone" "$machine" "$accelerator" "$count" "$blocker" >> "$ATTEMPTS_TSV"
  echo "blocked: $blocker"
  return 1
}

preserve_boot_disk() {
  local boot_disk
  boot_disk="$(
    run_gcloud compute instances describe "$TARGET_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --format='value(disks[0].source.basename())'
  )"
  if [[ -z "$boot_disk" ]]; then
    echo "warning: could not resolve boot disk for $TARGET_INSTANCE" >&2
    return 0
  fi
  echo "$boot_disk" > "$OUTPUT_DIR/boot_disk"
  run_gcloud compute instances set-disk-auto-delete "$TARGET_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --disk "$boot_disk" \
    --no-auto-delete \
    > "$OUTPUT_DIR/set-disk-auto-delete.log" 2>&1 || true
}

wait_for_ssh() {
  for _ in $(seq 1 90); do
    if run_gcloud compute ssh "$TARGET_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --quiet \
      --command='command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null' \
      > "$OUTPUT_DIR/ssh-ready.log" 2>&1; then
      return 0
    fi
    sleep 10
  done
  return 1
}

capture_evidence() {
  run_gcloud compute instances describe "$TARGET_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --format=json > "$OUTPUT_DIR/instance-describe.json"

  wait_for_ssh || true

  run_gcloud compute ssh "$TARGET_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet \
    --command='set +e
printf "HOSTNAME\t%s\n" "$(hostname)"
printf "KERNEL\t%s\n" "$(uname -r)"
if [ -r /etc/os-release ]; then . /etc/os-release; printf "OS\t%s\n" "${PRETTY_NAME:-unknown}"; fi
printf "NVIDIA_SMI_PATH\t%s\n" "$(command -v nvidia-smi || true)"
printf "NVIDIA_SMI_HEADER_BEGIN\n"
nvidia-smi | sed -n "1,4p"
printf "NVIDIA_SMI_HEADER_EXIT\t%s\n" "$?"
printf "GPU_QUERY_BEGIN\n"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv,noheader,nounits
printf "GPU_QUERY_EXIT\t%s\n" "$?"
printf "TOPOLOGY_BEGIN\n"
nvidia-smi topo -m
printf "TOPOLOGY_EXIT\t%s\n" "$?"
printf "TOPOLOGY_END\n"
printf "CUDA_TOOLKIT\t"
if command -v nvcc >/dev/null 2>&1; then nvcc --version | tr "\n" " "; else printf "nvcc-unavailable"; fi
printf "\n"
printf "NCCL_LIBRARY\t"
(ldconfig -p 2>/dev/null | grep -m1 -i libnccl || true) | sed "s/^ *//"
' > "$OUTPUT_DIR/hardware-evidence.txt" 2> "$OUTPUT_DIR/hardware-evidence.stderr" || true

  run_gcloud compute disks describe "$(cat "$OUTPUT_DIR/boot_disk" 2>/dev/null || true)" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --format=json > "$OUTPUT_DIR/boot-disk-describe.json" 2> "$OUTPUT_DIR/boot-disk-describe.stderr" || true
}

cleanup() {
  if [[ -n "$TARGET_INSTANCE" && "$KEEP_INSTANCE" != "1" ]]; then
    run_gcloud compute instances delete "$TARGET_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --quiet > "$OUTPUT_DIR/delete.log" 2>&1 || true
  fi
}
trap cleanup EXIT

refuse_unsafe_target

if [[ -z "$TARGET_INSTANCE" ]]; then
  while IFS=$'\t' read -r order label zone machine accelerator count role; do
    [[ "$order" == "order" ]] && continue
    if attempt_create "$order" "$label" "$zone" "$machine" "$accelerator" "$count" "$role"; then
      break
    fi
  done < "$PLAN_TSV"
fi

if [[ -n "$TARGET_INSTANCE" ]]; then
  preserve_boot_disk
  capture_evidence
fi

echo "OUTPUT_DIR=$OUTPUT_DIR"
cat "$ATTEMPTS_TSV"
