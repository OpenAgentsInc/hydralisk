#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
ISSUE_NUMBER="${ISSUE_NUMBER:-15}"
MODEL_ID="${MODEL_ID:-nvidia/DeepSeek-V4-Flash-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-e3cd60e7de98e9867116860d522499a728de1cf9}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
ALLOW_NVFP4_SM120="${ALLOW_NVFP4_SM120:-0}"
DOCKER_BUILD_PULL="${DOCKER_BUILD_PULL:-1}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:latest}"
INSTALL_DEEPGEMM="${INSTALL_DEEPGEMM:-1}"
DERIVED_IMAGE="${DERIVED_IMAGE:-hydralisk-deepseek-v4-nvfp4-vllm}"
TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"
CREATE_IF_MISSING="${CREATE_IF_MISSING:-1}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-21600s}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-hyperdisk-balanced}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-700GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-5400}"
STACK_BUILD_TIMEOUT_SECONDS="${STACK_BUILD_TIMEOUT_SECONDS:-1800}"
RUN_MODEL_SMOKE="${RUN_MODEL_SMOKE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-nvfp4-g4-$TS}"

mkdir -p "$OUTPUT_DIR"

PLAN_TSV="$OUTPUT_DIR/nvfp4-g4-plan.tsv"
ATTEMPTS_TSV="$OUTPUT_DIR/nvfp4-g4-attempts.tsv"
PROVIDER_OUTPUT_DIR="$OUTPUT_DIR/provider-stack"

cat > "$PLAN_TSV" <<'EOF'
order	label	zone	machine	accelerator	gpu_count	role
1	g4-2g-b	us-central1-b	g4-standard-96	nvidia-rtx-pro-6000	2	blackwell_nvfp4_first
2	g4-2g-f	us-central1-f	g4-standard-96	nvidia-rtx-pro-6000	2	blackwell_nvfp4_fallback
EOF

cat > "$ATTEMPTS_TSV" <<'EOF'
order	instance	zone	machine	accelerator	gpu_count	status	blocker
EOF

refuse_unsafe_target() {
  if [[ -n "$TARGET_INSTANCE" && "$TARGET_INSTANCE" != hydralisk-deepseek-v4-* ]]; then
    echo "error: TARGET_INSTANCE must be a fresh hydralisk-deepseek-v4-* probe host" >&2
    exit 2
  fi
  if [[ -n "$TARGET_INSTANCE" && -z "$TARGET_ZONE" ]]; then
    echo "error: TARGET_ZONE is required when TARGET_INSTANCE is set" >&2
    exit 2
  fi
}

attempt_create() {
  local order="$1" label="$2" zone="$3" machine="$4" accelerator="$5" count="$6"
  local instance="hydralisk-deepseek-v4-nvfp4-${label}-${TS}"
  local log="$OUTPUT_DIR/create-$label.log"

  if [[ "$DRY_RUN" = "1" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tdry_run\t\n' \
      "$order" "$instance" "$zone" "$machine" "$accelerator" "$count" >> "$ATTEMPTS_TSV"
    return 1
  fi

  if gcloud compute instances create "$instance" \
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
    --tags hydralisk-probe,deepseek-v4,nvfp4 \
    --labels lane=hydralisk,workload=deepseek-v4-nvfp4,model=deepseek-v4,probe="$label",issue="$ISSUE_NUMBER" \
    --format=json > "$log" 2>&1; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tadmitted\t\n' \
      "$order" "$instance" "$zone" "$machine" "$accelerator" "$count" >> "$ATTEMPTS_TSV"
    TARGET_INSTANCE="$instance"
    TARGET_ZONE="$zone"
    return 0
  fi

  local blocker
  blocker="$(tail -n 40 "$log" | tr '\n\t' '  ' | sed 's/  */ /g' | cut -c1-1500)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\tblocked\t%s\n' \
    "$order" "$instance" "$zone" "$machine" "$accelerator" "$count" "$blocker" >> "$ATTEMPTS_TSV"
  return 1
}

wait_for_ssh() {
  for _ in $(seq 1 72); do
    if gcloud compute ssh "$TARGET_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --quiet \
      --command='true' > "$OUTPUT_DIR/ssh-ready.log" 2>&1; then
      return 0
    fi
    sleep 10
  done
  return 1
}

run_provider_probe() {
  ISSUE_NUMBER="$ISSUE_NUMBER" \
  MODEL_ID="$MODEL_ID" \
  MODEL_REVISION="$MODEL_REVISION" \
  MOE_BACKEND="$MOE_BACKEND" \
  ALLOW_NVFP4_SM120="$ALLOW_NVFP4_SM120" \
  DOCKER_BUILD_PULL="$DOCKER_BUILD_PULL" \
  BASE_IMAGE="$BASE_IMAGE" \
  INSTALL_DEEPGEMM="$INSTALL_DEEPGEMM" \
  DERIVED_IMAGE="$DERIVED_IMAGE" \
  TARGET_INSTANCE="$TARGET_INSTANCE" \
  TARGET_ZONE="$TARGET_ZONE" \
  READY_TIMEOUT_SECONDS="$READY_TIMEOUT_SECONDS" \
  STACK_BUILD_TIMEOUT_SECONDS="$STACK_BUILD_TIMEOUT_SECONDS" \
  RUN_MODEL_SMOKE="$RUN_MODEL_SMOKE" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  OUTPUT_DIR="$PROVIDER_OUTPUT_DIR" \
  "$PWD/scripts/probe-deepseek-v4-provider-stack-gce.sh" \
    > "$OUTPUT_DIR/provider-stack-wrapper.stdout" \
    2> "$OUTPUT_DIR/provider-stack-wrapper.stderr" || true
}

render_markdown() {
  local md="$OUTPUT_DIR/nvfp4-g4-probe.md"
  {
    echo "# DeepSeek-V4-Flash NVFP4 G4 probe"
    echo
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "- Issue: https://github.com/OpenAgentsInc/hydralisk/issues/$ISSUE_NUMBER"
    echo "- Project: \`$PROJECT_ID\`"
    echo "- Model: \`$MODEL_ID\`"
    echo "- Model revision: \`$MODEL_REVISION\`"
    echo "- MoE backend: \`$MOE_BACKEND\`"
    echo "- Allow NVFP4 SM120 guard patch: \`$ALLOW_NVFP4_SM120\`"
    echo "- Docker build pull: \`$DOCKER_BUILD_PULL\`"
    echo "- Base image: \`$BASE_IMAGE\`"
    echo "- Install DeepGEMM helper: \`$INSTALL_DEEPGEMM\`"
    echo "- Target instance: \`${TARGET_INSTANCE:-unadmitted}\`"
    echo "- Target zone: \`${TARGET_ZONE:-unadmitted}\`"
    echo "- Create if missing: \`$CREATE_IF_MISSING\`"
    echo "- Run model smoke: \`$RUN_MODEL_SMOKE\`"
    echo
    if [[ "$DRY_RUN" = "1" ]]; then
      echo "DRY_RUN=1"
      echo
    fi
    echo "## Plan"
    echo
    echo '```tsv'
    cat "$PLAN_TSV"
    echo '```'
    echo
    echo "## Admission"
    echo
    echo '```tsv'
    cat "$ATTEMPTS_TSV"
    echo '```'
    echo
    if [[ -f "$PROVIDER_OUTPUT_DIR/provider-stack-probe.md" ]]; then
      echo "## Provider Stack Summary"
      echo
      echo "Nested provider-stack report:"
      echo "\`$PROVIDER_OUTPUT_DIR/provider-stack-probe.md\`"
      echo
      echo "Smoke summary:"
      echo
      echo '```text'
      sed -n '1,80p' "$PROVIDER_OUTPUT_DIR/provider-stack-smoke-summary.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "Engine:"
      echo
      echo '```text'
      sed -n '1,120p' "$PROVIDER_OUTPUT_DIR/provider-stack-engine.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "vLLM tail, public redacted:"
      echo
      echo '```text'
      sed -n '1,180p' "$PROVIDER_OUTPUT_DIR/provider-stack-vllm-tail-public.txt" 2>/dev/null || true
      echo '```'
    fi
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

refuse_unsafe_target

if [[ "$DRY_RUN" = "1" ]]; then
  while IFS=$'\t' read -r order label zone machine accelerator count role; do
    [[ "$order" == "order" ]] && continue
    attempt_create "$order" "$label" "$zone" "$machine" "$accelerator" "$count" || true
  done < "$PLAN_TSV"
  render_markdown
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  exit 0
fi

if [[ -z "$TARGET_INSTANCE" ]]; then
  if [[ "$CREATE_IF_MISSING" != "1" ]]; then
    echo "error: TARGET_INSTANCE is required when CREATE_IF_MISSING!=1" >&2
    exit 2
  fi
  while IFS=$'\t' read -r order label zone machine accelerator count role; do
    [[ "$order" == "order" ]] && continue
    if attempt_create "$order" "$label" "$zone" "$machine" "$accelerator" "$count"; then
      break
    fi
  done < "$PLAN_TSV"
fi

if [[ -z "$TARGET_INSTANCE" || -z "$TARGET_ZONE" ]]; then
  render_markdown
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  exit 0
fi

if wait_for_ssh; then
  run_provider_probe
else
  echo "ssh_not_ready" > "$OUTPUT_DIR/blocker.txt"
fi

render_markdown
echo "OUTPUT_DIR=$OUTPUT_DIR"
