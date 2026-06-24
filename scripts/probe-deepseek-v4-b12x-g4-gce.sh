#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
ISSUE_NUMBER="${ISSUE_NUMBER:-23}"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-${CLOUDSDK_CORE_ACCOUNT:-}}"
GCLOUD_AUTH_PREFLIGHT="${GCLOUD_AUTH_PREFLIGHT:-1}"
MODEL_ID="${MODEL_ID:-nvidia/DeepSeek-V4-Flash-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-e3cd60e7de98e9867116860d522499a728de1cf9}"
MOE_BACKEND="${MOE_BACKEND:-flashinfer_b12x}"
VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-0}"
ALLOW_NVFP4_SM120="${ALLOW_NVFP4_SM120:-1}"
DOCKER_BUILD_PULL="${DOCKER_BUILD_PULL:-1}"
VLLM_LINEAR_BACKEND="${VLLM_LINEAR_BACKEND:-triton}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-auto}"
VLLM_E8M0_TRITON_UPCAST="${VLLM_E8M0_TRITON_UPCAST:-1}"
HYDRALISK_DEEPSEEK_O_PROJ_PATCH="${HYDRALISK_DEEPSEEK_O_PROJ_PATCH:-1}"
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE="${HYDRALISK_DEEPSEEK_O_PROJ_RECIPE:-blackwell}"
HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE="${HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE:-1}"
HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS="${HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS:-1}"
HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE="${HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE:-fp32}"
HYDRALISK_DEEPSEEK_O_PROJ_BYPASS="${HYDRALISK_DEEPSEEK_O_PROJ_BYPASS:-off}"
HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK="${HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK:-off}"
HYDRALISK_B12X_CLAMP_PATCH="${HYDRALISK_B12X_CLAMP_PATCH:-1}"
HYDRALISK_B12X_CLAMP_LIMIT="${HYDRALISK_B12X_CLAMP_LIMIT:-10.0}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-0}"
HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:latest}"
INSTALL_DEEPGEMM="${INSTALL_DEEPGEMM:-1}"
DERIVED_IMAGE="${DERIVED_IMAGE:-hydralisk-deepseek-v4-b12x-g4-vllm}"
TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"
CREATE_IF_MISSING="${CREATE_IF_MISSING:-1}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-21600s}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-hyperdisk-balanced}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-900GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-7200}"
COMPLETION_TIMEOUT_SECONDS="${COMPLETION_TIMEOUT_SECONDS:-180}"
CONTAINER_START_TIMEOUT_SECONDS="${CONTAINER_START_TIMEOUT_SECONDS:-180}"
STACK_BUILD_TIMEOUT_SECONDS="${STACK_BUILD_TIMEOUT_SECONDS:-2400}"
RUN_MODEL_SMOKE="${RUN_MODEL_SMOKE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-b12x-g4-$TS}"

mkdir -p "$OUTPUT_DIR"

PLAN_TSV="$OUTPUT_DIR/b12x-g4-plan.tsv"
ATTEMPTS_TSV="$OUTPUT_DIR/b12x-g4-attempts.tsv"
PROVIDER_OUTPUT_DIR="$OUTPUT_DIR/provider-stack"

cat > "$PLAN_TSV" <<'EOF'
order	label	zone	machine	accelerator	gpu_count	role
1	g4-8g-b	us-central1-b	g4-standard-384	nvidia-rtx-pro-6000	8	b12x_no_ep_first
2	g4-4g-b	us-central1-b	g4-standard-192	nvidia-rtx-pro-6000	4	b12x_no_ep_fallback
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

check_gcloud_auth() {
  local log="$OUTPUT_DIR/gcloud-auth-preflight.log"
  if [[ "$GCLOUD_AUTH_PREFLIGHT" != "1" ]]; then
    echo "skipped" > "$OUTPUT_DIR/gcloud-auth-preflight-status.txt"
    return 0
  fi

  if run_gcloud auth print-access-token > /dev/null 2> "$log"; then
    echo "ok" > "$OUTPUT_DIR/gcloud-auth-preflight-status.txt"
    return 0
  fi

  echo "blocked_auth" > "$OUTPUT_DIR/gcloud-auth-preflight-status.txt"
  return 1
}

record_auth_blocker_for_plan() {
  local log="$OUTPUT_DIR/gcloud-auth-preflight.log"
  local blocker
  blocker="$(sanitize_blocker < "$log")"

  if [[ -n "$TARGET_INSTANCE" ]]; then
    printf '0\t%s\t%s\tmanual\tmanual\t0\tblocked_auth\t%s\n' \
      "$TARGET_INSTANCE" "$TARGET_ZONE" "$blocker" >> "$ATTEMPTS_TSV"
    return 0
  fi

  while IFS=$'\t' read -r order label zone machine accelerator count role; do
    [[ "$order" == "order" ]] && continue
    printf '%s\t%s\t%s\t%s\t%s\t%s\tblocked_auth\t%s\n' \
      "$order" "hydralisk-deepseek-v4-b12x-${label}-${TS}" "$zone" "$machine" \
      "$accelerator" "$count" "$blocker" >> "$ATTEMPTS_TSV"
  done < "$PLAN_TSV"
}

attempt_create() {
  local order="$1" label="$2" zone="$3" machine="$4" accelerator="$5" count="$6"
  local instance="hydralisk-deepseek-v4-b12x-${label}-${TS}"
  local log="$OUTPUT_DIR/create-$label.log"

  if [[ "$DRY_RUN" = "1" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tdry_run\t\n' \
      "$order" "$instance" "$zone" "$machine" "$accelerator" "$count" >> "$ATTEMPTS_TSV"
    return 1
  fi

  if run_gcloud compute instances create "$instance" \
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
    --tags hydralisk-probe,deepseek-v4,nvfp4,b12x \
    --labels lane=hydralisk,workload=deepseek-v4-b12x,model=deepseek-v4,probe="$label",issue="$ISSUE_NUMBER" \
    --format=json > "$log" 2>&1; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tadmitted\t\n' \
      "$order" "$instance" "$zone" "$machine" "$accelerator" "$count" >> "$ATTEMPTS_TSV"
    TARGET_INSTANCE="$instance"
    TARGET_ZONE="$zone"
    return 0
  fi

  local blocker
  blocker="$(tail -n 60 "$log" | sanitize_blocker)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\tblocked\t%s\n' \
    "$order" "$instance" "$zone" "$machine" "$accelerator" "$count" "$blocker" >> "$ATTEMPTS_TSV"
  return 1
}

wait_for_ssh() {
  for _ in $(seq 1 90); do
    if run_gcloud compute ssh "$TARGET_INSTANCE" \
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
  GCLOUD_ACCOUNT="$GCLOUD_ACCOUNT" \
  MODEL_ID="$MODEL_ID" \
  MODEL_REVISION="$MODEL_REVISION" \
  MOE_BACKEND="$MOE_BACKEND" \
  VLLM_ENABLE_EXPERT_PARALLEL="$VLLM_ENABLE_EXPERT_PARALLEL" \
  ALLOW_NVFP4_SM120="$ALLOW_NVFP4_SM120" \
  DOCKER_BUILD_PULL="$DOCKER_BUILD_PULL" \
  VLLM_LINEAR_BACKEND="$VLLM_LINEAR_BACKEND" \
  VLLM_ENFORCE_EAGER="$VLLM_ENFORCE_EAGER" \
  VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND" \
  VLLM_E8M0_TRITON_UPCAST="$VLLM_E8M0_TRITON_UPCAST" \
  HYDRALISK_DEEPSEEK_O_PROJ_PATCH="$HYDRALISK_DEEPSEEK_O_PROJ_PATCH" \
  HYDRALISK_DEEPSEEK_O_PROJ_RECIPE="$HYDRALISK_DEEPSEEK_O_PROJ_RECIPE" \
  HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE="$HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE" \
  HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS="$HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS" \
  HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE="$HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE" \
  HYDRALISK_DEEPSEEK_O_PROJ_BYPASS="$HYDRALISK_DEEPSEEK_O_PROJ_BYPASS" \
  HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK="$HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK" \
  HYDRALISK_B12X_CLAMP_PATCH="$HYDRALISK_B12X_CLAMP_PATCH" \
  HYDRALISK_B12X_CLAMP_LIMIT="$HYDRALISK_B12X_CLAMP_LIMIT" \
  HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
  HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE" \
  HF_XET_NUM_CONCURRENT_RANGE_GETS="$HF_XET_NUM_CONCURRENT_RANGE_GETS" \
  BASE_IMAGE="$BASE_IMAGE" \
  INSTALL_DEEPGEMM="$INSTALL_DEEPGEMM" \
  DERIVED_IMAGE="$DERIVED_IMAGE" \
  TARGET_INSTANCE="$TARGET_INSTANCE" \
  TARGET_ZONE="$TARGET_ZONE" \
  READY_TIMEOUT_SECONDS="$READY_TIMEOUT_SECONDS" \
  COMPLETION_TIMEOUT_SECONDS="$COMPLETION_TIMEOUT_SECONDS" \
  CONTAINER_START_TIMEOUT_SECONDS="$CONTAINER_START_TIMEOUT_SECONDS" \
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
  local md="$OUTPUT_DIR/deepseek-v4-b12x-g4-probe.md"
  {
    echo "# DeepSeek-V4-Flash B12x no-EP G4 full-model probe"
    echo
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "- Issue: https://github.com/OpenAgentsInc/hydralisk/issues/$ISSUE_NUMBER"
    echo "- Project: \`$PROJECT_ID\`"
    echo "- gcloud account override: \`${GCLOUD_ACCOUNT:-default}\`"
    echo "- gcloud auth preflight: \`$GCLOUD_AUTH_PREFLIGHT\`"
    echo "- Model: \`$MODEL_ID\`"
    echo "- Model revision: \`$MODEL_REVISION\`"
    echo "- MoE backend: \`$MOE_BACKEND\`"
    echo "- vLLM expert parallel: \`$VLLM_ENABLE_EXPERT_PARALLEL\`"
    echo "- vLLM linear backend: \`$VLLM_LINEAR_BACKEND\`"
    echo "- vLLM enforce eager: \`$VLLM_ENFORCE_EAGER\`"
    echo "- vLLM attention backend: \`$VLLM_ATTENTION_BACKEND\`"
    echo "- HF Hub disable Xet: \`$HF_HUB_DISABLE_XET\`"
    echo "- DeepSeek o_proj fallback: \`$HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK\`"
    echo "- B12x clamp patch: \`$HYDRALISK_B12X_CLAMP_PATCH\`"
    echo "- B12x clamp limit: \`$HYDRALISK_B12X_CLAMP_LIMIT\`"
    echo "- Target instance: \`${TARGET_INSTANCE:-unadmitted}\`"
    echo "- Target zone: \`${TARGET_ZONE:-unadmitted}\`"
    echo "- Create if missing: \`$CREATE_IF_MISSING\`"
    echo "- Run model smoke: \`$RUN_MODEL_SMOKE\`"
    echo "- Completion timeout seconds: \`$COMPLETION_TIMEOUT_SECONDS\`"
    echo "- Container start timeout seconds: \`$CONTAINER_START_TIMEOUT_SECONDS\`"
    echo "- Max model length: \`$MAX_MODEL_LEN\`"
    echo "- Max batched tokens: \`$MAX_NUM_BATCHED_TOKENS\`"
    echo "- GPU memory utilization: \`$GPU_MEMORY_UTILIZATION\`"
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
    if [[ -f "$OUTPUT_DIR/gcloud-auth-preflight-status.txt" ]]; then
      echo "## gcloud Auth Preflight"
      echo
      echo "Status: \`$(cat "$OUTPUT_DIR/gcloud-auth-preflight-status.txt")\`"
      if [[ "$(cat "$OUTPUT_DIR/gcloud-auth-preflight-status.txt")" == "blocked_auth" ]]; then
        echo
        echo "No GCE instance creation was attempted because local gcloud credentials"
        echo "require interactive reauthentication. Next operator action:"
        echo
        echo '```bash'
        echo "gcloud auth login"
        echo "gcloud auth application-default login"
        echo '```'
      fi
      echo
    fi
    if [[ -f "$PROVIDER_OUTPUT_DIR/provider-stack-probe.md" ]]; then
      echo "## Provider Stack Summary"
      echo
      echo "Nested provider-stack report:"
      echo "\`$PROVIDER_OUTPUT_DIR/provider-stack-probe.md\`"
      echo
      echo "Smoke summary:"
      echo
      echo '```text'
      sed -n '1,100p' "$PROVIDER_OUTPUT_DIR/provider-stack-smoke-summary.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "Engine:"
      echo
      echo '```text'
      sed -n '1,140p' "$PROVIDER_OUTPUT_DIR/provider-stack-engine.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "Completion, public redacted:"
      echo
      echo '```json'
      sed -n '1,80p' "$PROVIDER_OUTPUT_DIR/provider-stack-completion-public.json" 2>/dev/null || true
      echo '```'
      echo
      echo "vLLM tail, public redacted:"
      echo
      echo '```text'
      sed -n '1,220p' "$PROVIDER_OUTPUT_DIR/provider-stack-vllm-tail-public.txt" 2>/dev/null || true
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

if ! check_gcloud_auth; then
  record_auth_blocker_for_plan
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
