#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-hyperdisk-balanced}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-250GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
KEEP_INSTANCE="${KEEP_INSTANCE:-0}"
TS="$(date -u +%Y%m%d%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/glm52-preflight-$TS}"

mkdir -p "$OUTPUT_DIR"
cat > "$OUTPUT_DIR/attempts.tsv" <<'EOF'
order	name	zone	machine	gpu	gpu_count	status	blocker
EOF

ADMITTED_INSTANCE=""
ADMITTED_ZONE=""

cleanup() {
  if [[ -n "$ADMITTED_INSTANCE" && "$KEEP_INSTANCE" != "1" ]]; then
    gcloud compute instances delete "$ADMITTED_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$ADMITTED_ZONE" \
      --quiet > "$OUTPUT_DIR/delete.log" 2>&1 || true
  fi
}
trap cleanup EXIT

attempt() {
  local order="$1" label="$2" zone="$3" machine="$4" gpu="$5" count="$6"
  local instance="hydralisk-glm52-${label}-${TS}"
  local log="$OUTPUT_DIR/${order}-${label}.log"
  echo "Attempt $order: $label $zone $machine $gpu x$count"
  set +e
  gcloud compute instances create "$instance" \
    --project "$PROJECT_ID" \
    --zone "$zone" \
    --machine-type "$machine" \
    --maintenance-policy TERMINATE \
    --provisioning-model SPOT \
    --instance-termination-action DELETE \
    --max-run-duration 600s \
    --boot-disk-size "$BOOT_DISK_SIZE" \
    --boot-disk-type "$BOOT_DISK_TYPE" \
    --image-family "$IMAGE_FAMILY" \
    --image-project "$IMAGE_PROJECT" \
    --no-address \
    --metadata enable-oslogin=TRUE \
    --tags hydralisk-probe,glm-52 \
    --labels lane=hydralisk,workload=glm52-preflight,model=glm-5-2,probe="$label" \
    --format=json > "$log" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    local summary
    summary="$(tail -n 30 "$log" | tr '\n\t' '  ' | sed 's/  */ /g' | cut -c1-1200)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\tblocked\t%s\n' \
      "$order" "$instance" "$zone" "$machine" "$gpu" "$count" "$summary" \
      >> "$OUTPUT_DIR/attempts.tsv"
    echo "blocked: $summary"
    return 1
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\tadmitted\t\n' \
    "$order" "$instance" "$zone" "$machine" "$gpu" "$count" \
    >> "$OUTPUT_DIR/attempts.tsv"
  ADMITTED_INSTANCE="$instance"
  ADMITTED_ZONE="$zone"
  echo "$instance" > "$OUTPUT_DIR/admitted_instance"
  echo "$zone" > "$OUTPUT_DIR/admitted_zone"
  echo "$label" > "$OUTPUT_DIR/admitted_label"
  return 0
}

capture_hardware() {
  gcloud compute instances describe "$ADMITTED_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$ADMITTED_ZONE" \
    --format=json > "$OUTPUT_DIR/instance-describe.json"

  for _ in $(seq 1 24); do
    if gcloud compute ssh "$ADMITTED_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$ADMITTED_ZONE" \
      --quiet \
      --command='command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null' \
      > "$OUTPUT_DIR/ssh-ready.log" 2>&1; then
      break
    fi
    sleep 10
  done

  gcloud compute ssh "$ADMITTED_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$ADMITTED_ZONE" \
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
}

attempt 1 b200 us-central1-b a4-highgpu-8g nvidia-b200 8 || \
attempt 2 h200 us-central1-b a3-ultragpu-8g nvidia-h200-141gb 8 || \
attempt 3 rtx-pro-6000 us-central1-b g4-standard-384 nvidia-rtx-pro-6000 8 || \
attempt 4 h100 us-central1-a a3-highgpu-8g nvidia-h100-80gb 8 || true

if [[ -n "$ADMITTED_INSTANCE" ]]; then
  capture_hardware
fi

echo "OUTPUT_DIR=$OUTPUT_DIR"
cat "$OUTPUT_DIR/attempts.tsv"
