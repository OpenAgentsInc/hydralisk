#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-1800s}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-hyperdisk-balanced}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-600GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
ATTEMPT_CREATE="${ATTEMPT_CREATE:-0}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-published-recipe-gce-$TS}"

mkdir -p "$OUTPUT_DIR"

PLAN_TSV="$OUTPUT_DIR/published-recipe-candidates.tsv"
RESULTS_TSV="$OUTPUT_DIR/published-recipe-results.tsv"
QUOTA_TSV="$OUTPUT_DIR/published-recipe-quotas.tsv"
CATALOG_TSV="$OUTPUT_DIR/published-recipe-catalog.tsv"

cat > "$PLAN_TSV" <<'EOF'
order	label	region	zone	machine	accelerator	gpu_count	role	quota_metrics
1	h200-us-central1-b	us-central1	us-central1-b	a3-ultragpu-8g	nvidia-h200-141gb	8	published_h200_8g	NVIDIA_H200_GPUS,PREEMPTIBLE_NVIDIA_H200_GPUS
2	b200-us-central1-b	us-central1	us-central1-b	a4-highgpu-8g	nvidia-b200	8	published_b200_8g	NVIDIA_B200_GPUS,PREEMPTIBLE_NVIDIA_B200_GPUS
3	gb200-us-central1-a	us-central1	us-central1-a	a4x-highgpu-4g	nvidia-gb200	4	published_gb200_nvl4	NVIDIA_GB200_GPUS,PREEMPTIBLE_NVIDIA_GB200_GPUS
4	gb200-us-central1-b	us-central1	us-central1-b	a4x-highgpu-4g	nvidia-gb200	4	published_gb200_nvl4	NVIDIA_GB200_GPUS,PREEMPTIBLE_NVIDIA_GB200_GPUS
5	h100-us-central1-a	us-central1	us-central1-a	a3-highgpu-8g	nvidia-h100-80gb	8	published_h100_8g	NVIDIA_H100_GPUS,PREEMPTIBLE_NVIDIA_H100_GPUS
6	h100-us-central1-b	us-central1	us-central1-b	a3-highgpu-8g	nvidia-h100-80gb	8	published_h100_8g	NVIDIA_H100_GPUS,PREEMPTIBLE_NVIDIA_H100_GPUS
7	h100-mega-us-central1-a	us-central1	us-central1-a	a3-megagpu-8g	nvidia-h100-mega-80gb	8	published_h100_mega_8g	NVIDIA_H100_MEGA_GPUS,PREEMPTIBLE_NVIDIA_H100_MEGA_GPUS
8	h100-mega-us-central1-b	us-central1	us-central1-b	a3-megagpu-8g	nvidia-h100-mega-80gb	8	published_h100_mega_8g	NVIDIA_H100_MEGA_GPUS,PREEMPTIBLE_NVIDIA_H100_MEGA_GPUS
9	h200-us-east4-b	us-east4	us-east4-b	a3-ultragpu-8g	nvidia-h200-141gb	8	published_h200_8g	NVIDIA_H200_GPUS,PREEMPTIBLE_NVIDIA_H200_GPUS
10	b200-us-east4-b	us-east4	us-east4-b	a4-highgpu-8g	nvidia-b200	8	published_b200_8g	NVIDIA_B200_GPUS,PREEMPTIBLE_NVIDIA_B200_GPUS
11	gb200-us-east4-b	us-east4	us-east4-b	a4x-highgpu-4g	nvidia-gb200	4	published_gb200_nvl4	NVIDIA_GB200_GPUS,PREEMPTIBLE_NVIDIA_GB200_GPUS
12	h200-us-west1-c	us-west1	us-west1-c	a3-ultragpu-8g	nvidia-h200-141gb	8	published_h200_8g	NVIDIA_H200_GPUS,PREEMPTIBLE_NVIDIA_H200_GPUS
13	h100-us-west1-a	us-west1	us-west1-a	a3-highgpu-8g	nvidia-h100-80gb	8	published_h100_8g	NVIDIA_H100_GPUS,PREEMPTIBLE_NVIDIA_H100_GPUS
14	b200-us-east1-b	us-east1	us-east1-b	a4-highgpu-8g	nvidia-b200	8	published_b200_8g	NVIDIA_B200_GPUS,PREEMPTIBLE_NVIDIA_B200_GPUS
EOF

cat > "$RESULTS_TSV" <<'EOF'
order	label	region	zone	machine	accelerator	gpu_count	catalog	machine_type	quota	status	blocker
EOF

quota_json_for_region() {
  local region="$1"
  local cache="$OUTPUT_DIR/quota-$region.json"
  if [[ ! -f "$cache" ]]; then
    gcloud compute regions describe "$region" \
      --project "$PROJECT_ID" \
      --format='json(quotas)' > "$cache" 2> "$cache.stderr" || echo '{"quotas":[]}' > "$cache"
  fi
  cat "$cache"
}

quota_state() {
  local region="$1" metrics="$2" need="$3"
  local cache="$OUTPUT_DIR/quota-$region.json"
  quota_json_for_region "$region" >/dev/null
  python3 - "$metrics" "$need" "$cache" <<'PY'
import json
import sys

metrics = [m for m in sys.argv[1].split(",") if m]
need = float(sys.argv[2])
with open(sys.argv[3], "r", encoding="utf-8") as f:
    payload = json.load(f)
seen = []
for quota in payload.get("quotas", []):
    metric = quota.get("metric", "")
    if metric in metrics:
        limit = float(quota.get("limit", 0))
        usage = float(quota.get("usage", 0))
        remaining = limit - usage
        seen.append(f"{metric}:limit={limit:g}:usage={usage:g}:remaining={remaining:g}")
        if remaining >= need:
            print("ok:" + ";".join(seen))
            sys.exit(0)
if seen:
    print("insufficient:" + ";".join(seen))
else:
    print("missing:" + ",".join(metrics))
PY
}

record_catalog() {
  {
    echo "type	name	zone	description"
    gcloud compute accelerator-types list \
      --project "$PROJECT_ID" \
      --format='csv[no-heading](name,zone,description)' \
      | grep -Ei 'h100|h200|b200|gb200|rtx.pro|rtx-pro|6000' \
      | sed 's/^/accelerator\t/' || true
    gcloud compute machine-types list \
      --project "$PROJECT_ID" \
      --format='csv[no-heading](name,zone,guestCpus,memoryMb)' \
      | grep -Ei '^(a3|a4|g4)' \
      | sed 's/^/machine\t/' || true
  } > "$CATALOG_TSV"
}

record_quotas() {
  {
    echo "region	metric	limit	usage"
    cut -f3 "$PLAN_TSV" | tail -n +2 | sort -u | while read -r region; do
      quota_json_for_region "$region" >/dev/null
      python3 - "$region" "$OUTPUT_DIR/quota-$region.json" <<'PY'
import json
import re
import sys

region = sys.argv[1]
with open(sys.argv[2], "r", encoding="utf-8") as f:
    payload = json.load(f)
for quota in payload.get("quotas", []):
    metric = quota.get("metric", "")
    if re.search(r"H100|H200|B200|GB200|A3|A4|RTX|L4", metric, re.I):
        print(f"{region}\t{metric}\t{quota.get('limit')}\t{quota.get('usage')}")
PY
    done
  } > "$QUOTA_TSV"
}

catalog_exists() {
  local accelerator="$1" zone="$2"
  gcloud compute accelerator-types describe "$accelerator" \
    --project "$PROJECT_ID" \
    --zone "$zone" \
    --format='value(name)' >/dev/null 2>&1
}

machine_exists() {
  local machine="$1" zone="$2"
  gcloud compute machine-types describe "$machine" \
    --project "$PROJECT_ID" \
    --zone "$zone" \
    --format='value(name)' >/dev/null 2>&1
}

attempt_create() {
  local label="$1" zone="$2" machine="$3" accelerator="$4" count="$5"
  local instance="hydralisk-deepseek-v4-pub-${label}-${TS}"
  local log="$OUTPUT_DIR/create-$label.log"

  gcloud compute instances create "$instance" \
    --project "$PROJECT_ID" \
    --zone "$zone" \
    --machine-type "$machine" \
    --maintenance-policy TERMINATE \
    --provisioning-model "$PROVISIONING_MODEL" \
    --instance-termination-action DELETE \
    --max-run-duration "$MAX_RUN_DURATION" \
    --boot-disk-size "$BOOT_DISK_SIZE" \
    --boot-disk-type "$BOOT_DISK_TYPE" \
    --image-family "$IMAGE_FAMILY" \
    --image-project "$IMAGE_PROJECT" \
    --accelerator "type=$accelerator,count=$count" \
    --no-address \
    --metadata enable-oslogin=TRUE \
    --tags hydralisk-probe,deepseek-v4,published-recipe \
    --labels lane=hydralisk,workload=deepseek-v4-published-recipe,model=deepseek-v4,probe="$label",issue=14 \
    --format=json > "$log" 2>&1
}

render_markdown() {
  local md="$OUTPUT_DIR/published-recipe-gce-probe.md"
  {
    echo "# DeepSeek-V4 published-recipe GCE admission probe"
    echo
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "Issue: https://github.com/OpenAgentsInc/hydralisk/issues/14"
    echo
    echo "- Project: \`$PROJECT_ID\`"
    echo "- Attempt create: \`$ATTEMPT_CREATE\`"
    echo "- Provisioning model: \`$PROVISIONING_MODEL\`"
    echo "- Max run duration: \`$MAX_RUN_DURATION\`"
    echo
    if [[ "$DRY_RUN" = "1" ]]; then
      echo "DRY_RUN=1"
      echo
    fi
    echo "## Candidate Plan"
    echo
    echo '```tsv'
    cat "$PLAN_TSV"
    echo '```'
    echo
    echo "## Catalog"
    echo
    echo '```tsv'
    sed -n '1,220p' "$CATALOG_TSV" 2>/dev/null || true
    echo '```'
    echo
    echo "## Relevant Quotas"
    echo
    echo '```tsv'
    sed -n '1,220p' "$QUOTA_TSV" 2>/dev/null || true
    echo '```'
    echo
    echo "## Results"
    echo
    echo '```tsv'
    cat "$RESULTS_TSV"
    echo '```'
    echo
    echo "## Public safety"
    echo
    echo "- Contains secrets: false"
    echo "- Contains private prompts: false"
    echo "- Contains private responses: false"
    echo "- Contains weights: false"
    echo "- Contains hidden reasoning: false"
  } > "$md"
  echo "Wrote $md"
}

if [[ "$DRY_RUN" = "1" ]]; then
  {
    echo "type	name	zone	description"
    echo "dry_run	skipped	skipped	gcloud catalog lookup skipped"
  } > "$CATALOG_TSV"
  {
    echo "region	metric	limit	usage"
    echo "dry_run	skipped	skipped	skipped"
  } > "$QUOTA_TSV"
  while IFS=$'\t' read -r order label region zone machine accelerator count role metrics; do
    [[ "$order" == "order" ]] && continue
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\tdry_run\tdry_run\tdry_run\tdry_run\t\n' \
      "$order" "$label" "$region" "$zone" "$machine" "$accelerator" "$count" >> "$RESULTS_TSV"
  done < "$PLAN_TSV"
  render_markdown
  exit 0
fi

record_catalog
record_quotas

while IFS=$'\t' read -r order label region zone machine accelerator count role metrics; do
  [[ "$order" == "order" ]] && continue
  catalog="missing"
  machine_state="missing"
  quota="unknown"
  status="blocked"
  blocker=""

  if catalog_exists "$accelerator" "$zone"; then
    catalog="ok"
  else
    blocker="accelerator_not_in_zone"
  fi

  if machine_exists "$machine" "$zone"; then
    machine_state="ok"
  else
    blocker="${blocker:+$blocker; }machine_type_not_in_zone"
  fi

  quota="$(quota_state "$region" "$metrics" "$count")"
  if [[ "$quota" != ok:* ]]; then
    blocker="${blocker:+$blocker; }quota_${quota%%:*}"
  fi

  if [[ "$catalog" = "ok" && "$machine_state" = "ok" && "$quota" = ok:* ]]; then
    if [[ "$ATTEMPT_CREATE" = "1" ]]; then
      if attempt_create "$label" "$zone" "$machine" "$accelerator" "$count"; then
        status="admitted"
        blocker=""
      else
        status="blocked"
        blocker="$(tail -n 40 "$OUTPUT_DIR/create-$label.log" | tr '\n\t' '  ' | sed 's/  */ /g' | cut -c1-1600)"
      fi
    else
      status="not_attempted"
      blocker="ATTEMPT_CREATE=0"
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$order" "$label" "$region" "$zone" "$machine" "$accelerator" "$count" \
    "$catalog" "$machine_state" "$quota" "$status" "$blocker" >> "$RESULTS_TSV"
done < "$PLAN_TSV"

render_markdown
echo "OUTPUT_DIR=$OUTPUT_DIR"
