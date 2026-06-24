#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
ISSUE_NUMBER="${ISSUE_NUMBER:-24}"
MODEL_ID="${MODEL_ID:-nvidia/DeepSeek-V4-Flash-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-e3cd60e7de98e9867116860d522499a728de1cf9}"
BACKEND_MATRIX="${BACKEND_MATRIX:-flashinfer_cutlass,flashinfer_trtllm}"
VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-1}"
ALLOW_NVFP4_SM120="${ALLOW_NVFP4_SM120:-1}"
DOCKER_BUILD_PULL="${DOCKER_BUILD_PULL:-1}"
VLLM_LINEAR_BACKEND="${VLLM_LINEAR_BACKEND:-triton}"
VLLM_E8M0_TRITON_UPCAST="${VLLM_E8M0_TRITON_UPCAST:-1}"
HYDRALISK_DEEPSEEK_O_PROJ_PATCH="${HYDRALISK_DEEPSEEK_O_PROJ_PATCH:-1}"
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE="${HYDRALISK_DEEPSEEK_O_PROJ_RECIPE:-blackwell}"
HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE="${HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE:-1}"
HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS="${HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS:-1}"
HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE="${HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE:-fp32}"
HYDRALISK_DEEPSEEK_O_PROJ_BYPASS="${HYDRALISK_DEEPSEEK_O_PROJ_BYPASS:-off}"
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
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-clamp-backends-g4-$TS}"

mkdir -p "$OUTPUT_DIR"

HOST_PLAN_TSV="$OUTPUT_DIR/clamp-backends-g4-host-plan.tsv"
HOST_ATTEMPTS_TSV="$OUTPUT_DIR/clamp-backends-g4-host-attempts.tsv"
BACKEND_PLAN_TSV="$OUTPUT_DIR/clamp-backends-g4-backend-plan.tsv"
BACKEND_RESULTS_TSV="$OUTPUT_DIR/clamp-backends-g4-backend-results.tsv"

cat > "$HOST_PLAN_TSV" <<'EOF'
order	label	zone	machine	accelerator	gpu_count	role
1	g4-8g-b	us-central1-b	g4-standard-384	nvidia-rtx-pro-6000	8	preferred_clamp_backend_matrix
2	g4-4g-b	us-central1-b	g4-standard-192	nvidia-rtx-pro-6000	4	fallback_clamp_backend_matrix
EOF

cat > "$HOST_ATTEMPTS_TSV" <<'EOF'
order	instance	zone	machine	accelerator	gpu_count	status	blocker
EOF

cat > "$BACKEND_PLAN_TSV" <<'EOF'
order	backend	expert_parallel	role
EOF

backend_order=0
IFS=',' read -ra backend_items <<< "$BACKEND_MATRIX"
for raw_backend in "${backend_items[@]}"; do
  backend="$(echo "$raw_backend" | xargs)"
  [[ -z "$backend" ]] && continue
  backend_order=$((backend_order + 1))
  printf '%s\t%s\t%s\t%s\n' \
    "$backend_order" "$backend" "$VLLM_ENABLE_EXPERT_PARALLEL" "clamp_capable_provider_backend" \
    >> "$BACKEND_PLAN_TSV"
done

cat > "$BACKEND_RESULTS_TSV" <<'EOF'
order	backend	status	ready	blocker
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
  local instance="hydralisk-deepseek-v4-clamp-${label}-${TS}"
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
    --tags hydralisk-probe,deepseek-v4,nvfp4,clamp-backends \
    --labels lane=hydralisk,workload=deepseek-v4-clamp,model=deepseek-v4,probe="$label",issue="$ISSUE_NUMBER" \
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
  local completion_file="$outdir/provider-stack-completion-public.json"
  if [[ -f "$tail_file" && -s "$tail_file" ]]; then
    local matched
    matched="$(grep -E 'ValueError|NotImplementedError|RuntimeError|Assertion|InternalError|CUDA|OOM|out of memory|failed|Failed|Error occurred' "$tail_file" \
      | tail -n 8 | tr '\n\t' '  ' | sed 's/  */ /g' | cut -c1-1200 || true)"
    if [[ -n "$matched" ]]; then
      echo "$matched"
      return
    fi
  fi
  if [[ -f "$summary_file" && -s "$summary_file" ]]; then
    sed -n '1,20p' "$summary_file" | tr '\n\t' '  ' | sed 's/  */ /g' | cut -c1-800
    return
  fi
  if [[ -f "$completion_file" && -s "$completion_file" ]]; then
    tr '\n\t' '  ' < "$completion_file" | sed 's/  */ /g' | cut -c1-800
    return
  fi
  echo "no provider-stack summary copied"
}

run_provider_probe() {
  local order="$1" backend="$2"
  local backend_slug="${backend//[^A-Za-z0-9_]/_}"
  local provider_output_dir="$OUTPUT_DIR/provider-stack-$backend_slug"
  local derived_image="hydralisk-deepseek-v4-clamp-${backend_slug}-g4-vllm"
  mkdir -p "$provider_output_dir"

  ISSUE_NUMBER="$ISSUE_NUMBER" \
  MODEL_ID="$MODEL_ID" \
  MODEL_REVISION="$MODEL_REVISION" \
  MOE_BACKEND="$backend" \
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
  HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" \
  HF_XET_HIGH_PERFORMANCE="$HF_XET_HIGH_PERFORMANCE" \
  HF_XET_NUM_CONCURRENT_RANGE_GETS="$HF_XET_NUM_CONCURRENT_RANGE_GETS" \
  BASE_IMAGE="$BASE_IMAGE" \
  INSTALL_DEEPGEMM="$INSTALL_DEEPGEMM" \
  DERIVED_IMAGE="$derived_image" \
  TARGET_INSTANCE="$TARGET_INSTANCE" \
  TARGET_ZONE="$TARGET_ZONE" \
  READY_TIMEOUT_SECONDS="$READY_TIMEOUT_SECONDS" \
  STACK_BUILD_TIMEOUT_SECONDS="$STACK_BUILD_TIMEOUT_SECONDS" \
  RUN_MODEL_SMOKE="$RUN_MODEL_SMOKE" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" \
  MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  OUTPUT_DIR="$provider_output_dir" \
  "$PWD/scripts/probe-deepseek-v4-provider-stack-gce.sh" \
    < /dev/null \
    > "$OUTPUT_DIR/provider-stack-$backend_slug.stdout" \
    2> "$OUTPUT_DIR/provider-stack-$backend_slug.stderr" || true

  local ready blocker status
  ready="$(ready_value "$provider_output_dir/provider-stack-smoke-summary.txt")"
  blocker="$(summarize_blocker "$provider_output_dir")"
  if [[ "$ready" = "1" ]]; then
    status="ready"
  else
    status="blocked"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$order" "$backend" "$status" "$ready" "$blocker" \
    >> "$BACKEND_RESULTS_TSV"
}

render_markdown() {
  local md="$OUTPUT_DIR/deepseek-v4-clamp-backends-g4-probe.md"
  {
    echo "# DeepSeek-V4-Flash clamp-capable backend G4 probe"
    echo
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "- Issue: https://github.com/OpenAgentsInc/hydralisk/issues/$ISSUE_NUMBER"
    echo "- Project: \`$PROJECT_ID\`"
    echo "- Model: \`$MODEL_ID\`"
    echo "- Model revision: \`$MODEL_REVISION\`"
    echo "- Backend matrix: \`$BACKEND_MATRIX\`"
    echo "- vLLM expert parallel: \`$VLLM_ENABLE_EXPERT_PARALLEL\`"
    echo "- vLLM linear backend: \`$VLLM_LINEAR_BACKEND\`"
    echo "- HF Hub disable Xet: \`$HF_HUB_DISABLE_XET\`"
    echo "- Target instance: \`${TARGET_INSTANCE:-unadmitted}\`"
    echo "- Target zone: \`${TARGET_ZONE:-unadmitted}\`"
    echo "- Create if missing: \`$CREATE_IF_MISSING\`"
    echo "- Run model smoke: \`$RUN_MODEL_SMOKE\`"
    echo "- Max model length: \`$MAX_MODEL_LEN\`"
    echo "- Max batched tokens: \`$MAX_NUM_BATCHED_TOKENS\`"
    echo "- GPU memory utilization: \`$GPU_MEMORY_UTILIZATION\`"
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
    echo
    echo "## Backend Plan"
    echo
    echo '```tsv'
    cat "$BACKEND_PLAN_TSV"
    echo '```'
    echo
    echo "## Backend Results"
    echo
    echo '```tsv'
    cat "$BACKEND_RESULTS_TSV"
    echo '```'
    while IFS=$'\t' read -r order backend expert_parallel role; do
      [[ "$order" == "order" ]] && continue
      local backend_slug="${backend//[^A-Za-z0-9_]/_}"
      local provider_output_dir="$OUTPUT_DIR/provider-stack-$backend_slug"
      [[ ! -f "$provider_output_dir/provider-stack-probe.md" ]] && continue
      echo
      echo "## Backend: $backend"
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
      sed -n '1,140p' "$provider_output_dir/provider-stack-engine.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "Completion, public redacted:"
      echo
      echo '```json'
      sed -n '1,80p' "$provider_output_dir/provider-stack-completion-public.json" 2>/dev/null || true
      echo '```'
      echo
      echo "vLLM tail, public redacted:"
      echo
      echo '```text'
      sed -n '1,220p' "$provider_output_dir/provider-stack-vllm-tail-public.txt" 2>/dev/null || true
      echo '```'
    done < "$BACKEND_PLAN_TSV"
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

if [[ -z "$TARGET_INSTANCE" || -z "$TARGET_ZONE" ]]; then
  render_markdown
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  exit 0
fi

if wait_for_ssh; then
  while IFS=$'\t' read -r order backend expert_parallel role; do
    [[ "$order" == "order" ]] && continue
    run_provider_probe "$order" "$backend"
    if tail -n 1 "$BACKEND_RESULTS_TSV" | grep -q $'\tready\t1\t'; then
      break
    fi
  done < "$BACKEND_PLAN_TSV"
else
  echo "ssh_not_ready" > "$OUTPUT_DIR/blocker.txt"
fi

render_markdown
echo "OUTPUT_DIR=$OUTPUT_DIR"
