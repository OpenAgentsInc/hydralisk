#!/usr/bin/env bash
# Admit as many GLM-5.2-REAP-504B G4 (RTX PRO 6000) serving hosts as GCP grants.
# Tries 4x Spot across US zones first, then on-demand, then an 8x fallback.
# Records per-attempt result (admitted / blocked + sanitized blocker) to a TSV.
# Public-safe: records only cloud allocation + blocker summaries, no secrets.
set -uo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-${CLOUDSDK_CORE_ACCOUNT:-}}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/glm52-reap-504b-fleet-admission-$TS}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-hyperdisk-balanced}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-1500GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-21600s}"
# Space-separated "zone:machine:gpus:prov" tuples. prov is SPOT or STANDARD.
PLAN_TUPLES="${PLAN_TUPLES:-\
us-central1-b:g4-standard-192:4:SPOT \
us-central1-f:g4-standard-192:4:SPOT \
us-east1-b:g4-standard-192:4:SPOT \
us-east1-d:g4-standard-192:4:SPOT \
us-east4-b:g4-standard-192:4:SPOT \
us-east4-c:g4-standard-192:4:SPOT \
us-east5-a:g4-standard-192:4:SPOT \
us-east5-b:g4-standard-192:4:SPOT \
us-west1-a:g4-standard-192:4:SPOT \
us-west1-b:g4-standard-192:4:SPOT \
us-west4-a:g4-standard-192:4:SPOT \
us-south1-a:g4-standard-192:4:SPOT \
us-central1-b:g4-standard-192:4:STANDARD \
us-central1-f:g4-standard-192:4:STANDARD \
us-central1-f:g4-standard-384:8:SPOT}"

mkdir -p "$OUTPUT_DIR"
ATTEMPTS="$OUTPUT_DIR/attempts.tsv"
printf 'order\tinstance\tzone\tmachine\tgpus\tprov\tstatus\tblocker\n' > "$ATTEMPTS"

run_gcloud() {
  if [[ -n "$GCLOUD_ACCOUNT" ]]; then
    CLOUDSDK_CORE_ACCOUNT="$GCLOUD_ACCOUNT" gcloud "$@"
  else
    gcloud "$@"
  fi
}
sanitize() { tr '\n\t' '  ' | sed 's/  */ /g' | cut -c1-400; }

order=0
for tuple in $PLAN_TUPLES; do
  order=$((order+1))
  IFS=':' read -r zone machine gpus prov <<< "$tuple"
  label="g4-${gpus}g-$(echo "$zone" | sed 's/.*-//')-$(echo "$prov" | tr 'A-Z' 'a-z')"
  instance="hydralisk-glm52-reap-504b-${label}-${TS}"
  log="$OUTPUT_DIR/create-${order}-${label}.log"
  sched=(--maintenance-policy TERMINATE)
  if [[ "$prov" == "SPOT" ]]; then
    sched+=(--provisioning-model SPOT --instance-termination-action STOP --max-run-duration "$MAX_RUN_DURATION")
  else
    sched+=(--provisioning-model STANDARD)
  fi
  echo ">>> [$order] $instance ($zone $machine ${gpus}x $prov)"
  if run_gcloud compute instances create "$instance" \
      --project "$PROJECT_ID" --zone "$zone" --machine-type "$machine" \
      "${sched[@]}" \
      --boot-disk-size "$BOOT_DISK_SIZE" --boot-disk-type "$BOOT_DISK_TYPE" \
      --image-family "$IMAGE_FAMILY" --image-project "$IMAGE_PROJECT" \
      --accelerator "type=nvidia-rtx-pro-6000,count=$gpus" \
      --no-address --metadata enable-oslogin=TRUE \
      --tags hydralisk-probe,glm-52,glm-52-reap \
      --labels lane=hydralisk,workload=glm52-reap-504b,model=glm-5-2-reap-504b,probe="$label" \
      --format=json > "$log" 2>&1; then
    # Preserve boot disk on STOP for Spot durability.
    bd="$(run_gcloud compute instances describe "$instance" --project "$PROJECT_ID" --zone "$zone" --format='value(disks[0].source.basename())' 2>/dev/null)"
    [[ -n "$bd" ]] && run_gcloud compute instances set-disk-auto-delete "$instance" --project "$PROJECT_ID" --zone "$zone" --disk "$bd" --no-auto-delete >/dev/null 2>&1 || true
    printf '%s\t%s\t%s\t%s\t%s\t%s\tadmitted\t\n' "$order" "$instance" "$zone" "$machine" "$gpus" "$prov" >> "$ATTEMPTS"
    echo "    ADMITTED"
  else
    blk="$(tail -n 40 "$log" | sanitize)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\tblocked\t%s\n' "$order" "$instance" "$zone" "$machine" "$gpus" "$prov" "$blk" >> "$ATTEMPTS"
    echo "    BLOCKED: $blk"
  fi
done

echo "OUTPUT_DIR=$OUTPUT_DIR"
column -t -s$'\t' "$ATTEMPTS" 2>/dev/null || cat "$ATTEMPTS"
