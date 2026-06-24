#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
ISSUE_NUMBER="${ISSUE_NUMBER:-25}"
MODEL_ID="${MODEL_ID:-nvidia/DeepSeek-V4-Flash-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-e3cd60e7de98e9867116860d522499a728de1cf9}"
MOE_BACKEND="${MOE_BACKEND:-flashinfer_trtllm}"
VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-1}"
ALLOW_NVFP4_SM120="${ALLOW_NVFP4_SM120:-1}"
DOCKER_BUILD_PULL="${DOCKER_BUILD_PULL:-1}"
VLLM_LINEAR_BACKEND="${VLLM_LINEAR_BACKEND:-triton}"
VLLM_E8M0_TRITON_UPCAST="${VLLM_E8M0_TRITON_UPCAST:-1}"
HYDRALISK_DEEPSEEK_O_PROJ_PATCH="${HYDRALISK_DEEPSEEK_O_PROJ_PATCH:-1}"
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE="${HYDRALISK_DEEPSEEK_O_PROJ_RECIPE:-hopper}"
HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE="${HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE:-1}"
HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS="${HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS:-1}"
HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE="${HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE:-fp32}"
HYDRALISK_DEEPSEEK_O_PROJ_BYPASS="${HYDRALISK_DEEPSEEK_O_PROJ_BYPASS:-off}"
HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK="${HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK:-bf16_einsum}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-0}"
HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:latest}"
INSTALL_DEEPGEMM="${INSTALL_DEEPGEMM:-1}"
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
STACK_BUILD_TIMEOUT_SECONDS="${STACK_BUILD_TIMEOUT_SECONDS:-2400}"
RUN_MODEL_SMOKE="${RUN_MODEL_SMOKE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-oproj-fallback-g4-$TS}"

mkdir -p "$OUTPUT_DIR"

HOST_PLAN_TSV="$OUTPUT_DIR/oproj-fallback-g4-host-plan.tsv"
HOST_ATTEMPTS_TSV="$OUTPUT_DIR/oproj-fallback-g4-host-attempts.tsv"

cat > "$HOST_PLAN_TSV" <<'EOF'
order	label	zone	machine	accelerator	gpu_count	role
1	g4-8g-b	us-central1-b	g4-standard-384	nvidia-rtx-pro-6000	8	preferred_oproj_fallback_probe
2	g4-4g-b	us-central1-b	g4-standard-192	nvidia-rtx-pro-6000	4	fallback_oproj_fallback_probe
EOF

cat > "$HOST_ATTEMPTS_TSV" <<'EOF'
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

find_existing_wide_g4() {
  [[ "$DRY_RUN" = "1" ]] && return 1
  local row
  row="$(gcloud compute instances list \
    --project "$PROJECT_ID" \
    --filter='name~hydralisk-deepseek-v4.*g4-8g AND status=RUNNING' \
    --format='value(name,zone.basename())' \
    --limit=1 2>/dev/null || true)"
  [[ -z "$row" ]] && return 1
  TARGET_INSTANCE="$(awk '{print $1}' <<< "$row")"
  TARGET_ZONE="$(awk '{print $2}' <<< "$row")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\treused_existing\t\n' \
    0 "$TARGET_INSTANCE" "$TARGET_ZONE" "g4-standard-384" "nvidia-rtx-pro-6000" 8 \
    >> "$HOST_ATTEMPTS_TSV"
  return 0
}

attempt_create() {
  local order="$1" label="$2" zone="$3" machine="$4" accelerator="$5" count="$6"
  local instance="hydralisk-deepseek-v4-oproj-${label}-${TS}"
  local log="$OUTPUT_DIR/create-$label.log"

  if [[ "$DRY_RUN" = "1" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tdry_run\t\n' \
      "$order" "$instance" "$zone" "$machine" "$accelerator" "$count" >> "$HOST_ATTEMPTS_TSV"
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
    --tags hydralisk-probe,deepseek-v4,nvfp4,oproj-fallback \
    --labels lane=hydralisk,workload=deepseek-v4-oproj,model=deepseek-v4,probe="$label",issue="$ISSUE_NUMBER" \
    --format=json > "$log" 2>&1; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tadmitted\t\n' \
      "$order" "$instance" "$zone" "$machine" "$accelerator" "$count" >> "$HOST_ATTEMPTS_TSV"
    TARGET_INSTANCE="$instance"
    TARGET_ZONE="$zone"
    return 0
  fi

  local blocker
  blocker="$(tail -n 60 "$log" | tr '\n\t' '  ' | sed 's/  */ /g' | cut -c1-1800)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\tblocked\t%s\n' \
    "$order" "$instance" "$zone" "$machine" "$accelerator" "$count" "$blocker" >> "$HOST_ATTEMPTS_TSV"
  return 1
}

wait_for_ssh() {
  for _ in $(seq 1 90); do
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
  local provider_output_dir="$OUTPUT_DIR/provider-stack-oproj-fallback"
  mkdir -p "$provider_output_dir"

  ISSUE_NUMBER="$ISSUE_NUMBER" \
  MODEL_ID="$MODEL_ID" \
  MODEL_REVISION="$MODEL_REVISION" \
  MOE_BACKEND="$MOE_BACKEND" \
  VLLM_ENABLE_EXPERT_PARALLEL="$VLLM_ENABLE_EXPERT_PARALLEL" \
  ALLOW_NVFP4_SM120="$ALLOW_NVFP4_SM120" \
  DOCKER_BUILD_PULL="$DOCKER_BUILD_PULL" \
  VLLM_LINEAR_BACKEND="$VLLM_LINEAR_BACKEND" \
  VLLM_E8M0_TRITON_UPCAST="$VLLM_E8M0_TRITON_UPCAST" \
  HYDRALISK_DEEPSEEK_O_PROJ_PATCH="$HYDRALISK_DEEPSEEK_O_PROJ_PATCH" \
  HYDRALISK_DEEPSEEK_O_PROJ_RECIPE="$HYDRALISK_DEEPSEEK_O_PROJ_RECIPE" \
  HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE="$HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE" \
  HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS="$HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS" \
  HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE="$HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE" \
  HYDRALISK_DEEPSEEK_O_PROJ_BYPASS="$HYDRALISK_DEEPSEEK_O_PROJ_BYPASS" \
  HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK="$HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK" \
  HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
  HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE" \
  HF_XET_NUM_CONCURRENT_RANGE_GETS="$HF_XET_NUM_CONCURRENT_RANGE_GETS" \
  BASE_IMAGE="$BASE_IMAGE" \
  INSTALL_DEEPGEMM="$INSTALL_DEEPGEMM" \
  DERIVED_IMAGE="hydralisk-deepseek-v4-oproj-fallback-g4-vllm" \
  TARGET_INSTANCE="$TARGET_INSTANCE" \
  TARGET_ZONE="$TARGET_ZONE" \
  READY_TIMEOUT_SECONDS="$READY_TIMEOUT_SECONDS" \
  STACK_BUILD_TIMEOUT_SECONDS="$STACK_BUILD_TIMEOUT_SECONDS" \
  RUN_MODEL_SMOKE="$RUN_MODEL_SMOKE" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  TS="$TS" \
  OUTPUT_DIR="$provider_output_dir" \
  "$PWD/scripts/probe-deepseek-v4-provider-stack-gce.sh" \
    < /dev/null \
    > "$OUTPUT_DIR/provider-stack-oproj-fallback.stdout" \
    2> "$OUTPUT_DIR/provider-stack-oproj-fallback.stderr" || true
}

ready_value() {
  local summary="$1"
  if [[ -f "$summary" ]] && grep -q $'^READY\t1' "$summary"; then
    echo 1
  else
    echo 0
  fi
}

summarize_blocker() {
  local outdir="$1"
  local tail_file="$outdir/provider-stack-vllm-tail-public.txt"
  local summary_file="$outdir/provider-stack-smoke-summary.txt"
  if [[ -f "$tail_file" && -s "$tail_file" ]]; then
    local matched
    matched="$(grep -E 'HYDRALISK_O_PROJ_FALLBACK_TRACE|ValueError|NotImplementedError|RuntimeError|Assertion|InternalError|CUDA|OOM|out of memory|failed|Failed|Error occurred' "$tail_file" \
      | tail -n 12 | tr '\n\t' '  ' | sed 's/  */ /g' | cut -c1-1400 || true)"
    if [[ -n "$matched" ]]; then
      echo "$matched"
      return
    fi
  fi
  if [[ -f "$summary_file" && -s "$summary_file" ]]; then
    sed -n '1,20p' "$summary_file" | tr '\n\t' '  ' | sed 's/  */ /g' | cut -c1-800
    return
  fi
  echo "no provider-stack summary copied"
}

render_markdown() {
  local md="$OUTPUT_DIR/deepseek-v4-oproj-fallback-g4-probe.md"
  local provider_output_dir="$OUTPUT_DIR/provider-stack-oproj-fallback"
  local ready="0"
  local blocker="not_run"
  if [[ -d "$provider_output_dir" ]]; then
    ready="$(ready_value "$provider_output_dir/provider-stack-smoke-summary.txt")"
    blocker="$(summarize_blocker "$provider_output_dir")"
  fi
  {
    echo "# DeepSeek-V4-Flash o_proj fallback G4 probe"
    echo
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "- Issue: https://github.com/OpenAgentsInc/hydralisk/issues/$ISSUE_NUMBER"
    echo "- Project: \`$PROJECT_ID\`"
    echo "- Model: \`$MODEL_ID\`"
    echo "- Model revision: \`$MODEL_REVISION\`"
    echo "- MoE backend: \`$MOE_BACKEND\`"
    echo "- vLLM expert parallel: \`$VLLM_ENABLE_EXPERT_PARALLEL\`"
    echo "- vLLM linear backend: \`$VLLM_LINEAR_BACKEND\`"
    echo "- DeepSeek o_proj recipe: \`$HYDRALISK_DEEPSEEK_O_PROJ_RECIPE\`"
    echo "- DeepSeek o_proj fallback: \`$HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK\`"
    echo "- DeepSeek o_proj bypass: \`$HYDRALISK_DEEPSEEK_O_PROJ_BYPASS\`"
    echo "- HF Hub disable Xet: \`$HF_HUB_DISABLE_XET\`"
    echo "- Target instance: \`${TARGET_INSTANCE:-unadmitted}\`"
    echo "- Target zone: \`${TARGET_ZONE:-unadmitted}\`"
    echo "- Ready: \`$ready\`"
    echo "- Blocker: \`$blocker\`"
    echo
    if [[ "$DRY_RUN" = "1" ]]; then
      echo "DRY_RUN=1"
      echo
    fi
    echo "## Host Plan"
    echo
    echo '```tsv'
    cat "$HOST_PLAN_TSV"
    echo '```'
    echo
    echo "## Host Admission"
    echo
    echo '```tsv'
    cat "$HOST_ATTEMPTS_TSV"
    echo '```'
    if [[ -f "$provider_output_dir/provider-stack-probe.md" ]]; then
      echo
      echo "## Provider Stack"
      echo
      echo "Nested provider-stack report:"
      echo "\`$provider_output_dir/provider-stack-probe.md\`"
      echo
      echo "Smoke summary:"
      echo
      echo '```text'
      sed -n '1,100p' "$provider_output_dir/provider-stack-smoke-summary.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "Engine:"
      echo
      echo '```text'
      sed -n '1,160p' "$provider_output_dir/provider-stack-engine.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "vLLM tail, public redacted:"
      echo
      echo '```text'
      sed -n '1,220p' "$provider_output_dir/provider-stack-vllm-tail-public.txt" 2>/dev/null || true
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
  done < "$HOST_PLAN_TSV"
  render_markdown
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  exit 0
fi

if [[ -z "$TARGET_INSTANCE" ]]; then
  find_existing_wide_g4 || true
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
  done < "$HOST_PLAN_TSV"
fi

if [[ -n "$TARGET_INSTANCE" && -n "$TARGET_ZONE" ]] && wait_for_ssh; then
  run_provider_probe
fi

render_markdown
echo "OUTPUT_DIR=$OUTPUT_DIR"
