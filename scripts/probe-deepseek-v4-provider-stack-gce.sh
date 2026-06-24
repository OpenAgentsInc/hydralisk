#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:latest}"
INSTALL_DEEPGEMM="${INSTALL_DEEPGEMM:-1}"
DERIVED_IMAGE="${DERIVED_IMAGE:-hydralisk-deepseek-v4-provider-vllm}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-2400}"
STACK_BUILD_TIMEOUT_SECONDS="${STACK_BUILD_TIMEOUT_SECONDS:-1800}"
DOCKER_SETUP_TIMEOUT_SECONDS="${DOCKER_SETUP_TIMEOUT_SECONDS:-180}"
RUN_MODEL_SMOKE="${RUN_MODEL_SMOKE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-provider-stack-$TS}"

mkdir -p "$OUTPUT_DIR"

if [[ -z "$TARGET_INSTANCE" || -z "$TARGET_ZONE" ]]; then
  echo "error: TARGET_INSTANCE and TARGET_ZONE are required" >&2
  exit 2
fi

if [[ "$TARGET_INSTANCE" != hydralisk-deepseek-v4-* ]]; then
  echo "error: TARGET_INSTANCE must be a fresh hydralisk-deepseek-v4-* probe host" >&2
  exit 2
fi

render_markdown() {
  local md="$OUTPUT_DIR/provider-stack-probe.md"
  {
    echo "# DeepSeek-V4 provider-guided vLLM/DeepGEMM stack probe"
    echo
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "- Issue: https://github.com/OpenAgentsInc/hydralisk/issues/13"
    echo "- Target instance: \`$TARGET_INSTANCE\`"
    echo "- Target zone: \`$TARGET_ZONE\`"
    echo "- Model: \`$MODEL_ID\`"
    echo "- Base image: \`$BASE_IMAGE\`"
    echo "- Derived image: \`$DERIVED_IMAGE\`"
    echo "- Install DeepGEMM helper: \`$INSTALL_DEEPGEMM\`"
    echo "- Run model smoke: \`$RUN_MODEL_SMOKE\`"
    echo
    if [[ "$DRY_RUN" = "1" ]]; then
      echo "DRY_RUN=1"
      echo
    fi
    echo "## Provider Recipe"
    echo
    echo "This probe follows the provider-note lane for DeepSeek-V4-Flash:"
    echo
    echo "- vLLM \`0.20.0+\` / \`vllm/vllm-openai:latest\`"
    echo "- DeepGEMM installed through vLLM's \`tools/install_deepgemm.sh\` helper"
    echo "- \`--kv-cache-dtype fp8\`"
    echo "- \`--block-size 256\`"
    echo "- \`--enable-expert-parallel\`"
    echo "- tensor parallel size equal to visible GPU count"
    echo "- DeepSeek tokenizer, reasoning, and tool-call parsers"
    echo
    if [[ "$DRY_RUN" != "1" ]]; then
      echo "## Hardware"
      echo
      echo '```text'
      sed -n '1,140p' "$OUTPUT_DIR/provider-stack-hardware.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "## Docker Image Evidence"
      echo
      echo '```text'
      sed -n '1,160p' "$OUTPUT_DIR/provider-stack-image.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "Build log tail:"
      echo
      echo '```text'
      tail -n 120 "$OUTPUT_DIR/provider-stack-build.log" 2>/dev/null || true
      echo '```'
      echo
      echo "## Import Probe"
      echo
      echo '```json'
      sed -n '1,160p' "$OUTPUT_DIR/provider-stack-import.jsonl" 2>/dev/null || true
      echo '```'
      echo
      echo "Import stderr:"
      echo
      echo '```text'
      sed -n '1,80p' "$OUTPUT_DIR/provider-stack-import.stderr" 2>/dev/null || true
      echo '```'
      echo
      echo "## Model Smoke"
      echo
      echo '```text'
      sed -n '1,120p' "$OUTPUT_DIR/provider-stack-engine.txt" 2>/dev/null || true
      echo '```'
      echo
      echo '```text'
      sed -n '1,80p' "$OUTPUT_DIR/provider-stack-smoke-summary.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "Public completion receipt, if any:"
      echo
      echo '```json'
      sed -n '1,80p' "$OUTPUT_DIR/provider-stack-completion-public.json" 2>/dev/null || true
      echo '```'
      echo
      echo "vLLM tail, public redacted:"
      echo
      echo '```text'
      sed -n '1,180p' "$OUTPUT_DIR/provider-stack-vllm-tail-public.txt" 2>/dev/null || true
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

if [[ "$DRY_RUN" = "1" ]]; then
  render_markdown
  exit 0
fi

remote_script="$(
  printf 'MODEL_ID=%q\n' "$MODEL_ID"
  printf 'BASE_IMAGE=%q\n' "$BASE_IMAGE"
  printf 'INSTALL_DEEPGEMM=%q\n' "$INSTALL_DEEPGEMM"
  printf 'DERIVED_IMAGE=%q\n' "$DERIVED_IMAGE:$TS"
  printf 'READY_TIMEOUT_SECONDS=%q\n' "$READY_TIMEOUT_SECONDS"
  printf 'STACK_BUILD_TIMEOUT_SECONDS=%q\n' "$STACK_BUILD_TIMEOUT_SECONDS"
  printf 'DOCKER_SETUP_TIMEOUT_SECONDS=%q\n' "$DOCKER_SETUP_TIMEOUT_SECONDS"
  printf 'RUN_MODEL_SMOKE=%q\n' "$RUN_MODEL_SMOKE"
  printf 'MAX_MODEL_LEN=%q\n' "$MAX_MODEL_LEN"
  printf 'MAX_NUM_SEQS=%q\n' "$MAX_NUM_SEQS"
  printf 'MAX_NUM_BATCHED_TOKENS=%q\n' "$MAX_NUM_BATCHED_TOKENS"
  printf 'GPU_MEMORY_UTILIZATION=%q\n' "$GPU_MEMORY_UTILIZATION"
  printf 'REMOTE_LOG_DIR=%q\n' "/var/log/hydralisk/deepseek-provider-stack-$TS"
  cat <<'REMOTE'
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo install -d -m 0777 "$REMOTE_LOG_DIR"

{
  printf "HOSTNAME\t%s\n" "$(hostname)"
  printf "KERNEL\t%s\n" "$(uname -r)"
  if [ -r /etc/os-release ]; then . /etc/os-release; printf "OS\t%s\n" "${PRETTY_NAME:-unknown}"; fi
  printf "NVIDIA_SMI_HEADER_BEGIN\n"
  nvidia-smi | sed -n "1,10p" || true
  printf "NVIDIA_SMI_HEADER_END\n"
  printf "GPU_QUERY_BEGIN\n"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,pci.bus_id --format=csv,noheader,nounits || true
  printf "GPU_QUERY_END\n"
  printf "TOPOLOGY_BEGIN\n"
  nvidia-smi topo -m || true
  printf "TOPOLOGY_END\n"
  printf "DISK_BEGIN\n"
  df -h /
  printf "DISK_END\n"
} > "$REMOTE_LOG_DIR/provider-stack-hardware.txt"

if ! command -v docker >/dev/null 2>&1; then
  timeout "$DOCKER_SETUP_TIMEOUT_SECONDS"s sudo apt-get update -y >> "$REMOTE_LOG_DIR/provider-stack-docker-setup.log" 2>&1 || true
  timeout "$DOCKER_SETUP_TIMEOUT_SECONDS"s sudo apt-get install -y ca-certificates curl jq docker.io >> "$REMOTE_LOG_DIR/provider-stack-docker-setup.log" 2>&1 || true
fi
sudo systemctl enable --now docker >> "$REMOTE_LOG_DIR/provider-stack-docker-setup.log" 2>&1 || true

build_ctx="$(mktemp -d /tmp/hydralisk-provider-stack.XXXXXX)"
cat > "$build_ctx/Dockerfile" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-lc"]
ARG INSTALL_DEEPGEMM=1
RUN if [[ "$INSTALL_DEEPGEMM" == "1" ]]; then \
      apt-get update && \
      apt-get install -y --no-install-recommends ca-certificates git cuda-libraries-dev-13-0 && \
      rm -rf /var/lib/apt/lists/*; \
    fi
RUN if [[ "$INSTALL_DEEPGEMM" == "1" ]]; then \
      python3 -c 'from urllib.request import urlopen; open("/tmp/install_deepgemm.sh", "wb").write(urlopen("https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh", timeout=120).read())' && \
      UV_SYSTEM_PYTHON=1 bash /tmp/install_deepgemm.sh; \
    fi
DOCKERFILE

build_rc=0
timeout "$STACK_BUILD_TIMEOUT_SECONDS"s sudo docker build \
  --pull \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "INSTALL_DEEPGEMM=$INSTALL_DEEPGEMM" \
  -t "$DERIVED_IMAGE" \
  -f "$build_ctx/Dockerfile" \
  "$build_ctx" > "$REMOTE_LOG_DIR/provider-stack-build.log" 2>&1 || build_rc=$?
rm -rf "$build_ctx"

{
  printf "BASE_IMAGE\t%s\n" "$BASE_IMAGE"
  printf "DERIVED_IMAGE\t%s\n" "$DERIVED_IMAGE"
  printf "INSTALL_DEEPGEMM\t%s\n" "$INSTALL_DEEPGEMM"
  printf "BUILD_RC\t%s\n" "$build_rc"
  printf "BASE_IMAGE_INSPECT_BEGIN\n"
  sudo docker image inspect "$BASE_IMAGE" --format '{{json .RepoDigests}} {{json .Id}}' 2>/dev/null || true
  printf "BASE_IMAGE_INSPECT_END\n"
  printf "DERIVED_IMAGE_INSPECT_BEGIN\n"
  sudo docker image inspect "$DERIVED_IMAGE" --format '{{json .RepoDigests}} {{json .Id}}' 2>/dev/null || true
  printf "DERIVED_IMAGE_INSPECT_END\n"
} > "$REMOTE_LOG_DIR/provider-stack-image.txt"

if [[ "$build_rc" != "0" ]]; then
  printf "READY\t0\nBLOCKER\tprovider_stack_build_failed\n" > "$REMOTE_LOG_DIR/provider-stack-smoke-summary.txt"
  printf '{"ready":false,"status":"provider_stack_build_failed"}\n' > "$REMOTE_LOG_DIR/provider-stack-completion-public.json"
  exit 0
fi

sudo docker run --rm --gpus all --ipc=host --network host \
  --entrypoint bash "$DERIVED_IMAGE" -lc 'python3 - <<'"'"'PY'"'"'
import importlib
import importlib.metadata
import json
import torch

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"

record = {
    "schema": "hydralisk.deepseek-v4.provider-stack-import.v1",
    "vllm": version("vllm"),
    "torch": version("torch"),
    "torchCuda": torch.version.cuda,
    "cudaAvailable": torch.cuda.is_available(),
    "deviceCount": torch.cuda.device_count(),
    "devices": [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": list(torch.cuda.get_device_capability(i)),
        }
        for i in range(torch.cuda.device_count())
    ],
    "publicSafety": {
        "containsSecrets": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
        "containsHiddenReasoning": False,
    },
}
try:
    dg = importlib.import_module("vllm.utils.deep_gemm")
    record["deepGemmImport"] = True
    record["deepGemmHasTransformHelper"] = hasattr(dg, "transform_sf_into_required_layout")
except Exception as exc:
    record["deepGemmImport"] = False
    record["deepGemmImportError"] = f"{type(exc).__name__}: {exc}"[:300]
print(json.dumps(record, sort_keys=True))
PY' > "$REMOTE_LOG_DIR/provider-stack-import.jsonl" 2> "$REMOTE_LOG_DIR/provider-stack-import.stderr" || true

if [[ "$RUN_MODEL_SMOKE" != "1" ]]; then
  printf "READY\t0\nBLOCKER\tmodel_smoke_skipped\n" > "$REMOTE_LOG_DIR/provider-stack-smoke-summary.txt"
  printf '{"ready":false,"status":"model_smoke_skipped"}\n' > "$REMOTE_LOG_DIR/provider-stack-completion-public.json"
  exit 0
fi

gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
container_name="hydralisk-deepseek-v4-provider-stack-$RANDOM"
{
  printf "BACKEND\tdocker_provider_stack\n"
  printf "MODEL_ID\t%s\n" "$MODEL_ID"
  printf "BASE_IMAGE\t%s\n" "$BASE_IMAGE"
  printf "DERIVED_IMAGE\t%s\n" "$DERIVED_IMAGE"
  printf "TENSOR_PARALLEL_SIZE\t%s\n" "$gpu_count"
  printf "MAX_MODEL_LEN\t%s\n" "$MAX_MODEL_LEN"
  printf "MAX_NUM_SEQS\t%s\n" "$MAX_NUM_SEQS"
  printf "MAX_NUM_BATCHED_TOKENS\t%s\n" "$MAX_NUM_BATCHED_TOKENS"
  printf "GPU_MEMORY_UTILIZATION\t%s\n" "$GPU_MEMORY_UTILIZATION"
  printf "LOCAL_SITE_PACKAGES_PATCHES\tfalse\n"
  printf "PROVIDER_FLAGS\t--kv-cache-dtype fp8 --block-size 256 --enable-expert-parallel --tensor-parallel-size %s\n" "$gpu_count"
} > "$REMOTE_LOG_DIR/provider-stack-engine.txt"

sudo docker rm -f "$container_name" >/dev/null 2>&1 || true
sudo docker run --rm --gpus all --ipc=host --network host \
  --name "$container_name" \
  -v /var/lib/hydralisk/huggingface:/root/.cache/huggingface \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_RPC_TIMEOUT=600000 \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  "$DERIVED_IMAGE" \
  "$MODEL_ID" \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size "$gpu_count" \
  --enable-expert-parallel \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  > "$REMOTE_LOG_DIR/provider-stack-vllm.log" 2>&1 &
pid="$!"

ready=0
deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
while [ "$SECONDS" -lt "$deadline" ]; do
  if curl -fsS http://127.0.0.1:8000/v1/models > "$REMOTE_LOG_DIR/provider-stack-models.json" 2> "$REMOTE_LOG_DIR/provider-stack-models.stderr"; then
    ready=1
    break
  fi
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done

if [ "$ready" = "1" ]; then
  curl -fsS http://127.0.0.1:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"deepseek-ai/DeepSeek-V4-Flash","messages":[{"role":"user","content":"Reply with READY."}],"max_tokens":8,"temperature":0}' \
    | jq '{id: .id, model: .model, usage: .usage, finish_reason: .choices[0].finish_reason}' \
    > "$REMOTE_LOG_DIR/provider-stack-completion-public.json" || true
else
  printf '{"ready":false,"status":"server_not_ready_or_exited"}\n' > "$REMOTE_LOG_DIR/provider-stack-completion-public.json"
fi

sudo docker rm -f "$container_name" >/dev/null 2>&1 || true
wait "$pid" >/tmp/hydralisk-provider-stack-vllm-exit.log 2>&1 || true
printf "READY\t%s\n" "$ready" > "$REMOTE_LOG_DIR/provider-stack-smoke-summary.txt"
tail -n 180 "$REMOTE_LOG_DIR/provider-stack-vllm.log" \
  | sed -E 's/(hf_[A-Za-z0-9_\-]+)/<redacted-hf-token>/g' \
  > "$REMOTE_LOG_DIR/provider-stack-vllm-tail-public.txt"
REMOTE
)"

gcloud compute ssh "$TARGET_INSTANCE" \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --quiet \
  --command "$remote_script" \
  > "$OUTPUT_DIR/provider-stack-remote.stdout" \
  2> "$OUTPUT_DIR/provider-stack-remote.stderr" || true

remote_dir="/var/log/hydralisk/deepseek-provider-stack-$TS"
for file in \
  provider-stack-hardware.txt \
  provider-stack-docker-setup.log \
  provider-stack-build.log \
  provider-stack-image.txt \
  provider-stack-import.jsonl \
  provider-stack-import.stderr \
  provider-stack-engine.txt \
  provider-stack-smoke-summary.txt \
  provider-stack-completion-public.json \
  provider-stack-models.json \
  provider-stack-models.stderr \
  provider-stack-vllm-tail-public.txt; do
  gcloud compute scp \
    "$TARGET_INSTANCE:$remote_dir/$file" \
    "$OUTPUT_DIR/$file" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet > "$OUTPUT_DIR/scp-$file.log" 2>&1 || true
done

render_markdown
echo "OUTPUT_DIR=$OUTPUT_DIR"
