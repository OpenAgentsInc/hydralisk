#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-hyperdisk-balanced}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-600GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
MAX_RUN_DURATION="${MAX_RUN_DURATION:-7200s}"
KEEP_INSTANCE="${KEEP_INSTANCE:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_MODEL_SMOKE="${RUN_MODEL_SMOKE:-1}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-2400}"
SETUP_TIMEOUT_SECONDS="${SETUP_TIMEOUT_SECONDS:-1800}"
DOCKER_SETUP_TIMEOUT_SECONDS="${DOCKER_SETUP_TIMEOUT_SECONDS:-120}"
ISSUE_NUMBER="${ISSUE_NUMBER:-6}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash}"
VLLM_PACKAGE="${VLLM_PACKAGE:-vllm>=0.20.0}"
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-1}"
VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-1}"
VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES="${VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES:-1}"
VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER="${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-1}"
VLLM_LINEAR_BACKEND="${VLLM_LINEAR_BACKEND:-auto}"
VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-0}"
SYSTEM_CUDA_HOME="${SYSTEM_CUDA_HOME:-/usr/local/cuda-12.9}"
TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"
TARGET_LABEL="${TARGET_LABEL:-manual-target}"
TARGET_GPU_COUNT="${TARGET_GPU_COUNT:-}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-smoke-$TS}"

mkdir -p "$OUTPUT_DIR"

ATTEMPTS_TSV="$OUTPUT_DIR/attempts.tsv"
cat > "$ATTEMPTS_TSV" <<'EOF'
order	name	zone	machine	gpu	gpu_count	status	blocker
EOF

PLAN_TSV="$OUTPUT_DIR/attempt-plan.tsv"
cat > "$PLAN_TSV" <<'EOF'
order	label	zone	machine	gpu	gpu_count	role
EOF

ADMITTED_INSTANCE=""
ADMITTED_ZONE=""
ADMITTED_LABEL=""
ADMITTED_GPU_COUNT=""

write_plan_row() {
  local order="$1" label="$2" zone="$3" machine="$4" gpu="$5" count="$6" role="$7"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$order" "$label" "$zone" "$machine" "$gpu" "$count" "$role" >> "$PLAN_TSV"
}

write_plan_row 1 g4-2g-b us-central1-b g4-standard-96 nvidia-rtx-pro-6000 2 all_gpu_low_context_blackwell_first
write_plan_row 2 g4-2g-f us-central1-f g4-standard-96 nvidia-rtx-pro-6000 2 all_gpu_low_context_blackwell_fallback
write_plan_row 3 h100-2g-b us-central1-b a3-highgpu-2g nvidia-h100-80gb 2 all_gpu_low_context_hopper_fallback
write_plan_row 4 h100-2g-a us-central1-a a3-highgpu-2g nvidia-h100-80gb 2 all_gpu_low_context_hopper_fallback
write_plan_row 5 g4-1g-b us-central1-b g4-standard-48 nvidia-rtx-pro-6000 1 offload_prefetch_dev_only
write_plan_row 6 g4-1g-f us-central1-f g4-standard-48 nvidia-rtx-pro-6000 1 offload_prefetch_dev_only

cleanup() {
  if [[ -n "$ADMITTED_INSTANCE" && "$KEEP_INSTANCE" != "1" ]]; then
    gcloud compute instances delete "$ADMITTED_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$ADMITTED_ZONE" \
      --quiet > "$OUTPUT_DIR/delete.log" 2>&1 || true
  fi
}
trap cleanup EXIT

refuse_existing_product_hosts() {
  local running
  running="$(gcloud compute instances list \
    --project "$PROJECT_ID" \
    --filter='guestAccelerators:*' \
    --format='value(name)' || true)"
  if grep -Eq 'hydralisk-gptoss|khala' <<<"$running"; then
    echo "Existing product GPU hosts detected; this script will not target or reuse them." \
      > "$OUTPUT_DIR/product-host-safety.txt"
  fi
}

attempt() {
  local order="$1" label="$2" zone="$3" machine="$4" gpu="$5" count="$6"
  local instance="hydralisk-deepseek-v4-${label}-${TS}"
  local log="$OUTPUT_DIR/${order}-${label}.create.log"
  echo "Attempt $order: $label $zone $machine $gpu x$count"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\tdry_run\t\n' \
      "$order" "$instance" "$zone" "$machine" "$gpu" "$count" >> "$ATTEMPTS_TSV"
    return 1
  fi

  set +e
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
    --accelerator "type=$gpu,count=$count" \
    --metadata enable-oslogin=TRUE \
    --tags hydralisk-probe,deepseek-v4 \
    --labels lane=hydralisk,workload=deepseek-v4-smoke,model=deepseek-v4,probe="$label",issue="$ISSUE_NUMBER" \
    --format=json > "$log" 2>&1
  local rc=$?
  set -e

  if [[ "$rc" -ne 0 ]]; then
    local summary
    summary="$(tail -n 40 "$log" | tr '\n\t' '  ' | sed 's/  */ /g' | cut -c1-1500)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\tblocked\t%s\n' \
      "$order" "$instance" "$zone" "$machine" "$gpu" "$count" "$summary" \
      >> "$ATTEMPTS_TSV"
    echo "blocked: $summary"
    return 1
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\tadmitted\t\n' \
    "$order" "$instance" "$zone" "$machine" "$gpu" "$count" >> "$ATTEMPTS_TSV"
  ADMITTED_INSTANCE="$instance"
  ADMITTED_ZONE="$zone"
  ADMITTED_LABEL="$label"
  ADMITTED_GPU_COUNT="$count"
  echo "$instance" > "$OUTPUT_DIR/admitted_instance"
  echo "$zone" > "$OUTPUT_DIR/admitted_zone"
  echo "$label" > "$OUTPUT_DIR/admitted_label"
  echo "$count" > "$OUTPUT_DIR/admitted_gpu_count"
  return 0
}

wait_for_ssh() {
  for _ in $(seq 1 48); do
    if gcloud compute ssh "$ADMITTED_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$ADMITTED_ZONE" \
      --quiet \
      --command='true' > "$OUTPUT_DIR/ssh-ready.log" 2>&1; then
      return 0
    fi
    sleep 10
  done
  return 1
}

capture_instance() {
  gcloud compute instances describe "$ADMITTED_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$ADMITTED_ZONE" \
    --format=json > "$OUTPUT_DIR/instance-describe.json"
}

run_remote_evidence() {
  local remote_script
  remote_script="$(
    printf 'DOCKER_SETUP_TIMEOUT_SECONDS=%q\n' "$DOCKER_SETUP_TIMEOUT_SECONDS"
    cat <<'REMOTE'
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo install -d -m 0755 /var/lib/hydralisk /var/log/hydralisk

{
  printf "HOSTNAME\t%s\n" "$(hostname)"
  printf "KERNEL\t%s\n" "$(uname -r)"
  if [ -r /etc/os-release ]; then . /etc/os-release; printf "OS\t%s\n" "${PRETTY_NAME:-unknown}"; fi
  printf "DISK_BEGIN\n"
  df -h /
  printf "DISK_END\n"
  printf "NVIDIA_SMI_PATH\t%s\n" "$(command -v nvidia-smi || true)"
  printf "NVIDIA_SMI_HEADER_BEGIN\n"
  nvidia-smi | sed -n "1,8p" || true
  printf "NVIDIA_SMI_HEADER_END\n"
  printf "GPU_QUERY_BEGIN\n"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,pci.bus_id --format=csv,noheader,nounits || true
  printf "GPU_QUERY_END\n"
  printf "TOPOLOGY_BEGIN\n"
  nvidia-smi topo -m || true
  printf "TOPOLOGY_END\n"
  printf "DOCKER_PATH\t%s\n" "$(command -v docker || true)"
  docker --version || true
} | sudo tee /var/log/hydralisk/deepseek-hardware-evidence.txt >/dev/null

if ! command -v docker >/dev/null 2>&1; then
  if timeout "$DOCKER_SETUP_TIMEOUT_SECONDS"s sudo apt-get update -y && timeout "$DOCKER_SETUP_TIMEOUT_SECONDS"s sudo apt-get install -y ca-certificates curl jq docker.io; then
    sudo systemctl enable --now docker
  else
    printf "DOCKER_SETUP_BLOCKER\tapt_setup_failed_or_timed_out\n" | sudo tee /var/log/hydralisk/deepseek-docker-gpu-check.txt >/dev/null
    exit 0
  fi
else
  sudo systemctl start docker || true
fi

{
  printf "DOCKER_GPU_CHECK_BEGIN\n"
  sudo docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
  printf "DOCKER_GPU_CHECK_EXIT\t%s\n" "$?"
} | sudo tee /var/log/hydralisk/deepseek-docker-gpu-check.txt >/dev/null
REMOTE
  )"

  gcloud compute ssh "$ADMITTED_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$ADMITTED_ZONE" \
    --quiet \
    --command "$remote_script" \
    > "$OUTPUT_DIR/runtime-evidence.stdout" \
    2> "$OUTPUT_DIR/runtime-evidence.stderr" || true

  gcloud compute scp \
    "$ADMITTED_INSTANCE:/var/log/hydralisk/deepseek-hardware-evidence.txt" \
    "$OUTPUT_DIR/hardware-evidence.txt" \
    --project "$PROJECT_ID" \
    --zone "$ADMITTED_ZONE" \
    --quiet > "$OUTPUT_DIR/scp-hardware.log" 2>&1 || true

  gcloud compute scp \
    "$ADMITTED_INSTANCE:/var/log/hydralisk/deepseek-docker-gpu-check.txt" \
    "$OUTPUT_DIR/docker-gpu-check.txt" \
    --project "$PROJECT_ID" \
    --zone "$ADMITTED_ZONE" \
    --quiet > "$OUTPUT_DIR/scp-docker-gpu.log" 2>&1 || true
}

run_model_smoke() {
  if [[ "$RUN_MODEL_SMOKE" != "1" ]]; then
    echo "model smoke skipped by RUN_MODEL_SMOKE=$RUN_MODEL_SMOKE" > "$OUTPUT_DIR/model-smoke-skipped.txt"
    return 0
  fi

  local tp="$ADMITTED_GPU_COUNT"
  local model_script
  model_script="$(cat <<REMOTE
set -Eeuo pipefail
sudo install -d -m 0755 /var/lib/hydralisk/huggingface /var/log/hydralisk
sudo chmod 0777 /var/lib/hydralisk/huggingface
sudo chmod 0777 /var/log/hydralisk
if command -v docker >/dev/null 2>&1; then
  printf "BACKEND\tdocker\n" | sudo tee /var/log/hydralisk/deepseek-engine-evidence.txt >/dev/null
  {
    printf "VLLM_USE_DEEP_GEMM\t%s\n" "$VLLM_USE_DEEP_GEMM"
    printf "VLLM_USE_DEEP_GEMM_E8M0\t%s\n" "$VLLM_USE_DEEP_GEMM_E8M0"
    printf "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES\t%s\n" "$VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES"
    printf "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER\t%s\n" "$VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"
    printf "VLLM_LINEAR_BACKEND\t%s\n" "$VLLM_LINEAR_BACKEND"
    printf "VLLM_ENABLE_EXPERT_PARALLEL\t%s\n" "$VLLM_ENABLE_EXPERT_PARALLEL"
  } | sudo tee -a /var/log/hydralisk/deepseek-engine-evidence.txt >/dev/null
  container_name="hydralisk-deepseek-v4-smoke"
  expert_parallel_args=()
  if [[ "$VLLM_ENABLE_EXPERT_PARALLEL" = "1" ]]; then
    expert_parallel_args+=(--enable-expert-parallel)
  fi
  sudo docker rm -f "\$container_name" >/dev/null 2>&1 || true
  sudo docker run --rm --gpus all --ipc=host --network host \\
    --name "\$container_name" \\
    -v /var/lib/hydralisk/huggingface:/root/.cache/huggingface \\
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \\
    -e VLLM_RPC_TIMEOUT=600000 \\
    -e VLLM_LOG_STATS_INTERVAL=1 \\
    -e VLLM_USE_DEEP_GEMM="$VLLM_USE_DEEP_GEMM" \\
    -e VLLM_USE_DEEP_GEMM_E8M0="$VLLM_USE_DEEP_GEMM_E8M0" \\
    -e VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES="$VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES" \\
    -e VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER="$VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER" \\
    vllm/vllm-openai:latest \\
    "$MODEL_ID" \\
    --host 127.0.0.1 \\
    --port 8000 \\
    --trust-remote-code \\
    --kv-cache-dtype fp8 \\
    --block-size 256 \\
    --tensor-parallel-size "$tp" \\
    --gpu-memory-utilization 0.90 \\
    --max-model-len 4096 \\
    --max-num-seqs 1 \\
    --max-num-batched-tokens 1024 \\
    --tokenizer-mode deepseek_v4 \\
    --reasoning-parser deepseek_v4 \\
    --tool-call-parser deepseek_v4 \\
    --linear-backend "$VLLM_LINEAR_BACKEND" \\
    "\${expert_parallel_args[@]}" \\
    --enable-auto-tool-choice \\
    > /var/log/hydralisk/deepseek-vllm.log 2>&1 &
else
  printf "BACKEND\tpython_vllm\n" | sudo tee /var/log/hydralisk/deepseek-engine-evidence.txt >/dev/null
  sudo install -d -m 0777 /opt/hydralisk-deepseek-v4
  cd /opt/hydralisk-deepseek-v4
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > /var/log/hydralisk/deepseek-uv-install.log 2>&1
    export PATH="\$HOME/.local/bin:\$PATH"
  fi
  export UV_PYTHON_INSTALL_DIR=/opt/hydralisk-deepseek-v4/uv-python
  timeout "$SETUP_TIMEOUT_SECONDS"s uv python install 3.12 >> /var/log/hydralisk/deepseek-pip-install.log 2>&1
  timeout "$SETUP_TIMEOUT_SECONDS"s uv venv --clear --python 3.12 --seed .venv >> /var/log/hydralisk/deepseek-pip-install.log 2>&1
  . .venv/bin/activate
  timeout "$SETUP_TIMEOUT_SECONDS"s uv pip install "$VLLM_PACKAGE" >> /var/log/hydralisk/deepseek-pip-install.log 2>&1
  {
    python --version
    vllm --version
    python - <<'PY'
import importlib.metadata
for name in ("vllm", "torch"):
    try:
        print(f"{name}\t{importlib.metadata.version(name)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{name}\tunavailable")
PY
  } | sudo tee -a /var/log/hydralisk/deepseek-engine-evidence.txt >/dev/null
  if ! test -x "\$(gcc -print-prog-name=cc1plus 2>/dev/null)"; then
    timeout "$DOCKER_SETUP_TIMEOUT_SECONDS"s sudo apt-get install -y g++-12 libstdc++-12-dev >> /var/log/hydralisk/deepseek-toolchain-install.log 2>&1
    sudo ln -sf /usr/bin/g++-12 /usr/local/bin/g++
  fi
  {
    printf "cc1plus\t%s\n" "\$(gcc -print-prog-name=cc1plus 2>/dev/null || true)"
    if command -v g++ >/dev/null 2>&1; then g++ --version | head -1; fi
  } | sudo tee -a /var/log/hydralisk/deepseek-engine-evidence.txt >/dev/null
  if [ -x "$SYSTEM_CUDA_HOME/bin/nvcc" ]; then
    export CUDA_HOME="$SYSTEM_CUDA_HOME"
    export CUDA_PATH="$SYSTEM_CUDA_HOME"
    export PATH="$SYSTEM_CUDA_HOME/bin:\$PATH"
    export LD_LIBRARY_PATH="$SYSTEM_CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
  fi
  {
    printf "CUDA_HOME\t%s\n" "\${CUDA_HOME:-unset}"
    if command -v nvcc >/dev/null 2>&1; then nvcc --version | tail -n 1; fi
    printf "VLLM_USE_DEEP_GEMM\t%s\n" "$VLLM_USE_DEEP_GEMM"
    printf "VLLM_USE_DEEP_GEMM_E8M0\t%s\n" "$VLLM_USE_DEEP_GEMM_E8M0"
    printf "VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES\t%s\n" "$VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES"
    printf "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER\t%s\n" "$VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER"
    printf "VLLM_LINEAR_BACKEND\t%s\n" "$VLLM_LINEAR_BACKEND"
    printf "VLLM_ENABLE_EXPERT_PARALLEL\t%s\n" "$VLLM_ENABLE_EXPERT_PARALLEL"
  } | sudo tee -a /var/log/hydralisk/deepseek-engine-evidence.txt >/dev/null
  expert_parallel_args=()
  if [[ "$VLLM_ENABLE_EXPERT_PARALLEL" = "1" ]]; then
    expert_parallel_args+=(--enable-expert-parallel)
  fi
  HF_HOME=/var/lib/hydralisk/huggingface \\
  VLLM_ENGINE_READY_TIMEOUT_S=3600 \\
  VLLM_RPC_TIMEOUT=600000 \\
  VLLM_LOG_STATS_INTERVAL=1 \\
  VLLM_USE_DEEP_GEMM="$VLLM_USE_DEEP_GEMM" \\
  VLLM_USE_DEEP_GEMM_E8M0="$VLLM_USE_DEEP_GEMM_E8M0" \\
  VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES="$VLLM_USE_DEEP_GEMM_TMA_ALIGNED_SCALES" \\
  VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER="$VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER" \\
  vllm serve "$MODEL_ID" \\
    --host 127.0.0.1 \\
    --port 8000 \\
    --trust-remote-code \\
    --kv-cache-dtype fp8 \\
    --block-size 256 \\
    --tensor-parallel-size "$tp" \\
    --gpu-memory-utilization 0.90 \\
    --max-model-len 4096 \\
    --max-num-seqs 1 \\
    --max-num-batched-tokens 1024 \\
    --tokenizer-mode deepseek_v4 \\
    --reasoning-parser deepseek_v4 \\
    --tool-call-parser deepseek_v4 \\
    --linear-backend "$VLLM_LINEAR_BACKEND" \\
    "\${expert_parallel_args[@]}" \\
    --enable-auto-tool-choice \\
    > /var/log/hydralisk/deepseek-vllm.log 2>&1 &
fi
pid="\$!"
ready=0
deadline=\$((SECONDS + $READY_TIMEOUT_SECONDS))
while [ "\$SECONDS" -lt "\$deadline" ]; do
  if curl -fsS http://127.0.0.1:8000/v1/models >/var/log/hydralisk/deepseek-models.json 2>/tmp/deepseek-models.err; then
    ready=1
    break
  fi
  if ! kill -0 "\$pid" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done
if [ "\$ready" = "1" ]; then
  curl -fsS http://127.0.0.1:8000/v1/chat/completions \\
    -H 'content-type: application/json' \\
    -d '{"model":"$MODEL_ID","messages":[{"role":"user","content":"Reply with READY."}],"max_tokens":8,"temperature":0}' \\
    | jq '{id: .id, model: .model, usage: .usage, finish_reason: .choices[0].finish_reason}' \\
    > /var/log/hydralisk/deepseek-completion-public.json || true
else
  printf '{"ready":false,"status":"server_not_ready_or_exited"}\\n' > /var/log/hydralisk/deepseek-completion-public.json
fi
if command -v docker >/dev/null 2>&1; then
  sudo docker rm -f "\${container_name:-hydralisk-deepseek-v4-smoke}" >/dev/null 2>&1 || true
fi
wait "\$pid" >/tmp/deepseek-vllm-exit.log 2>&1 || true
printf "READY\\t%s\\n" "\$ready" | sudo tee /var/log/hydralisk/deepseek-smoke-summary.txt >/dev/null
tail -n 160 /var/log/hydralisk/deepseek-vllm.log | sed -E 's/(hf_[A-Za-z0-9_\\-]+)/<redacted-hf-token>/g' | sudo tee /var/log/hydralisk/deepseek-vllm-tail-public.txt >/dev/null
REMOTE
)"

  gcloud compute ssh "$ADMITTED_INSTANCE" \
    --project "$PROJECT_ID" \
    --zone "$ADMITTED_ZONE" \
    --quiet \
    --command "$model_script" \
    > "$OUTPUT_DIR/model-smoke.stdout" \
    2> "$OUTPUT_DIR/model-smoke.stderr" || true

  for file in \
    deepseek-engine-evidence.txt \
    deepseek-uv-install.log \
    deepseek-pip-install.log \
    deepseek-toolchain-install.log \
    deepseek-smoke-summary.txt \
    deepseek-vllm-tail-public.txt \
    deepseek-completion-public.json \
    deepseek-models.json; do
    gcloud compute scp \
      "$ADMITTED_INSTANCE:/var/log/hydralisk/$file" \
      "$OUTPUT_DIR/$file" \
      --project "$PROJECT_ID" \
      --zone "$ADMITTED_ZONE" \
      --quiet > "$OUTPUT_DIR/scp-$file.log" 2>&1 || true
  done
}

render_markdown() {
  local md="$OUTPUT_DIR/deepseek-v4-gce-smoke.md"
  {
    echo "# DeepSeek-V4-Flash GCE load-smoke evidence"
    echo
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "Issue: https://github.com/OpenAgentsInc/hydralisk/issues/$ISSUE_NUMBER"
    echo
    echo "## Attempt plan"
    echo
    echo '```tsv'
    cat "$PLAN_TSV"
    echo '```'
    echo
    echo "## Admission attempts"
    echo
    echo '```tsv'
    cat "$ATTEMPTS_TSV"
    echo '```'
    echo
    if [[ -n "$ADMITTED_INSTANCE" ]]; then
      echo "## Admitted host"
      echo
      echo "- Instance: \`$ADMITTED_INSTANCE\`"
      echo "- Zone: \`$ADMITTED_ZONE\`"
      echo "- Label: \`$ADMITTED_LABEL\`"
      echo "- GPUs: \`$ADMITTED_GPU_COUNT\`"
      echo "- Keep instance: \`$KEEP_INSTANCE\`"
      echo "- Max run duration: \`$MAX_RUN_DURATION\`"
      echo
      echo "## Hardware evidence"
      echo
      echo '```text'
      sed -n '1,120p' "$OUTPUT_DIR/hardware-evidence.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "## Docker GPU check"
      echo
      echo '```text'
      sed -n '1,120p' "$OUTPUT_DIR/docker-gpu-check.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "## Model smoke"
      echo
      echo '```text'
      sed -n '1,80p' "$OUTPUT_DIR/deepseek-smoke-summary.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "Public completion receipt, if any:"
      echo
      echo '```json'
      sed -n '1,80p' "$OUTPUT_DIR/deepseek-completion-public.json" 2>/dev/null || true
      echo '```'
      echo
      echo "vLLM tail, public redacted:"
      echo
      echo '```text'
      sed -n '1,180p' "$OUTPUT_DIR/deepseek-vllm-tail-public.txt" 2>/dev/null || true
      echo '```'
    else
      echo "## Result"
      echo
      echo "No fresh DeepSeek host was admitted."
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

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1; planned attempts:"
  cat "$PLAN_TSV"
  while IFS=$'\t' read -r order label zone machine gpu count role; do
    [[ "$order" == "order" ]] && continue
    attempt "$order" "$label" "$zone" "$machine" "$gpu" "$count" || true
  done < "$PLAN_TSV"
  render_markdown
  exit 0
fi

refuse_existing_product_hosts

if [[ -n "$TARGET_INSTANCE" ]]; then
  if [[ "$TARGET_INSTANCE" != hydralisk-deepseek-v4-* ]]; then
    echo "error: TARGET_INSTANCE must be a fresh hydralisk-deepseek-v4-* probe host" >&2
    exit 2
  fi
  if [[ -z "$TARGET_ZONE" || -z "$TARGET_GPU_COUNT" ]]; then
    echo "error: TARGET_ZONE and TARGET_GPU_COUNT are required with TARGET_INSTANCE" >&2
    exit 2
  fi
  ADMITTED_INSTANCE="$TARGET_INSTANCE"
  ADMITTED_ZONE="$TARGET_ZONE"
  ADMITTED_LABEL="$TARGET_LABEL"
  ADMITTED_GPU_COUNT="$TARGET_GPU_COUNT"
  echo "$ADMITTED_INSTANCE" > "$OUTPUT_DIR/admitted_instance"
  echo "$ADMITTED_ZONE" > "$OUTPUT_DIR/admitted_zone"
  echo "$ADMITTED_LABEL" > "$OUTPUT_DIR/admitted_label"
  echo "$ADMITTED_GPU_COUNT" > "$OUTPUT_DIR/admitted_gpu_count"
  printf 'target\t%s\t%s\ttarget-reuse\tunknown\t%s\tadmitted\tmanual target\n' \
    "$ADMITTED_INSTANCE" "$ADMITTED_ZONE" "$ADMITTED_GPU_COUNT" >> "$ATTEMPTS_TSV"
else
  while IFS=$'\t' read -r order label zone machine gpu count role; do
    [[ "$order" == "order" ]] && continue
    if attempt "$order" "$label" "$zone" "$machine" "$gpu" "$count"; then
      break
    fi
  done < "$PLAN_TSV"
fi

if [[ -n "$ADMITTED_INSTANCE" ]]; then
  capture_instance
  if wait_for_ssh; then
    run_remote_evidence
    run_model_smoke
  else
    echo "ssh_not_ready" > "$OUTPUT_DIR/ssh-blocker.txt"
  fi
fi

render_markdown
echo "OUTPUT_DIR=$OUTPUT_DIR"
cat "$ATTEMPTS_TSV"
