#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-${CLOUDSDK_CORE_ACCOUNT:-}}"
TARGET_INSTANCE="${TARGET_INSTANCE:-hydralisk-glm52-reap-504b-g4-8g-b-20260624214500}"
TARGET_ZONE="${TARGET_ZONE:-us-central1-b}"
ACTION="${ACTION:-start}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d%H%M%S)}"
MODEL_DIR="${MODEL_DIR:-/opt/hydralisk/models/glm-5.2-504b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.2-reap-504b-g4}"
CONTAINER_NAME="${CONTAINER_NAME:-hydralisk-glm52-reap-504b}"
IMAGE_TAG="${IMAGE_TAG:-voipmonitor/vllm:black-benediction-b12xpr11-vllmbb6c5b7-b12xd90d89c-fi3395b41aa8d-dg324aced12c-cu132-20260608}"
IMAGE_DIGEST="${IMAGE_DIGEST:-sha256:ce23a9b075bd7138ce3b12ee29609b98606e5050e2def4a29bbb917ad96e5997}"
IMAGE_REF="${IMAGE_REF:-voipmonitor/vllm@${IMAGE_DIGEST}}"
DOCKER_RESTART_POLICY="${DOCKER_RESTART_POLICY:-unless-stopped}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/var/log/hydralisk/glm52-reap-504b-b12x-$RUN_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/glm52-reap-504b-b12x-$RUN_ID}"

GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
DCP_SIZE="${DCP_SIZE:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-250000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MTP="${MTP:-0}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
INDEX_TOPK_PATTERN="${INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-7200}"

if [[ -z "$TARGET_INSTANCE" || -z "$TARGET_ZONE" ]]; then
  echo "error: TARGET_INSTANCE and TARGET_ZONE are required" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

run_gcloud() {
  if [[ -n "$GCLOUD_ACCOUNT" ]]; then
    CLOUDSDK_CORE_ACCOUNT="$GCLOUD_ACCOUNT" gcloud "$@"
  else
    gcloud "$@"
  fi
}

remote_script="$(mktemp "${TMPDIR:-/tmp}/hydralisk-glm52-launch.XXXXXX.sh")"
trap 'rm -f "$remote_script"' EXIT

cat > "$remote_script" <<'REMOTE'
#!/usr/bin/env bash
set -Eeuo pipefail

NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"

sudo install -d -m 0755 "$REMOTE_LOG_DIR" /var/log/hydralisk /var/lib/hydralisk/huggingface
sudo chown "$(whoami):$(id -gn)" "$REMOTE_LOG_DIR"

prepare_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io jq ca-certificates curl
  fi
  sudo systemctl enable --now docker
  if command -v nvidia-ctk >/dev/null 2>&1; then
    sudo nvidia-ctk runtime configure --runtime=docker >/tmp/hydralisk-nvidia-ctk.log 2>&1 || true
    sudo systemctl restart docker
  fi
}

write_launch_env() {
  cat > "$REMOTE_LOG_DIR/launch.env.public" <<EOF
IMAGE_TAG=$IMAGE_TAG
IMAGE_DIGEST=$IMAGE_DIGEST
IMAGE_REF=$IMAGE_REF
DOCKER_RESTART_POLICY=$DOCKER_RESTART_POLICY
MODEL_DIR=$MODEL_DIR
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
GPU_DEVICES=$GPU_DEVICES
TP_SIZE=$TP_SIZE
DCP_SIZE=$DCP_SIZE
MAX_MODEL_LEN=$MAX_MODEL_LEN
MAX_NUM_SEQS=$MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
HOST=$HOST
PORT=$PORT
MTP=$MTP
NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS
NCCL_DEBUG=$NCCL_DEBUG
NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE
INDEX_TOPK_PATTERN=$INDEX_TOPK_PATTERN
EOF
}

write_launch_command() {
  python3 - "$REMOTE_LOG_DIR/launch-command.json" <<'PY'
import json
import os

base = [
    "/opt/venv/bin/vllm",
    "serve",
    "/model",
    "--served-model-name",
    os.environ["SERVED_MODEL_NAME"],
    "--host",
    os.environ["HOST"],
    "--port",
    os.environ["PORT"],
    "--trust-remote-code",
    "--tensor-parallel-size",
    os.environ["TP_SIZE"],
    "--decode-context-parallel-size",
    os.environ["DCP_SIZE"],
    "--quantization",
    "modelopt_fp4",
    "--kv-cache-dtype",
    "fp8",
    "--attention-backend",
    "B12X_MLA_SPARSE",
    "--moe-backend",
    "b12x",
    "--tool-call-parser",
    "glm47",
    "--reasoning-parser",
    "glm45",
    "--max-model-len",
    os.environ["MAX_MODEL_LEN"],
    "--max-num-seqs",
    os.environ["MAX_NUM_SEQS"],
    "--max-num-batched-tokens",
    os.environ["MAX_NUM_BATCHED_TOKENS"],
    "--gpu-memory-utilization",
    os.environ["GPU_MEMORY_UTILIZATION"],
    "--hf-overrides",
    json.dumps({"index_topk_pattern": os.environ["INDEX_TOPK_PATTERN"]}),
]

if os.environ.get("MTP") == "1":
    base.extend(
        [
            "--speculative-config",
            json.dumps(
                {
                    "model": "/model",
                    "method": "mtp",
                    "num_speculative_tokens": int(os.environ["NUM_SPECULATIVE_TOKENS"]),
                }
            ),
        ]
    )

payload = {
    "argv": base,
    "publicSafety": {
        "containsSecrets": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
        "containsHiddenReasoning": False,
    },
}
open(os.environ["REMOTE_LOG_DIR"] + "/launch-command.json", "w").write(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
}

prepare_action() {
  prepare_docker
  write_launch_env
  write_launch_command
  sudo docker pull "$IMAGE_REF"
  {
    printf "preparedAt=%s\n" "$(date -u +%FT%TZ)"
    printf "dockerVersion="
    sudo docker --version
    printf "imageRef=%s\n" "$IMAGE_REF"
    printf "imageTag=%s\n" "$IMAGE_TAG"
    printf "imageDigest=%s\n" "$IMAGE_DIGEST"
    printf "modelDirExists=%s\n" "$(test -d "$MODEL_DIR" && echo true || echo false)"
    printf "modelIndexExists=%s\n" "$(test -f "$MODEL_DIR/model.safetensors.index.json" && echo true || echo false)"
    printf "gpuDevices=%s\n" "$GPU_DEVICES"
    printf "gpuInventoryBegin\n"
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits || true
    printf "gpuInventoryEnd\n"
    printf "containerCheckBegin\n"
    sudo docker run --rm --gpus all "$IMAGE_REF" bash -lc 'set +e; /opt/venv/bin/python - <<PY
import importlib.metadata as md
for package in ["vllm", "torch", "flashinfer-python", "flashinfer", "modelopt"]:
    try:
        print(f"{package}={md.version(package)}")
    except Exception as exc:
        print(f"{package}=unavailable:{type(exc).__name__}")
PY
/opt/venv/bin/vllm --help | sed -n "1,12p"'
    printf "containerCheckExit=%s\n" "$?"
    printf "containerCheckEnd\n"
  } > "$REMOTE_LOG_DIR/prepare-public.txt" 2>&1
}

start_action() {
  prepare_docker
  write_launch_env
  write_launch_command
  sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  mapfile -t VLLM_ARGV < <(python3 - "$REMOTE_LOG_DIR/launch-command.json" <<'PY'
import json
import shlex
import sys
payload = json.load(open(sys.argv[1]))
for arg in payload["argv"]:
    print(shlex.quote(str(arg)))
PY
)
  launch_command="${VLLM_ARGV[*]}"
  # The b12x image can carry an empty NCCL_GRAPH_FILE; unset it so NCCL
  # does not try to open a blank XML graph path during communicator init.
  sudo docker run --restart "$DOCKER_RESTART_POLICY" --gpus all --ipc=host --network host \
    --name "$CONTAINER_NAME" \
    --cap-add SYS_NICE \
    --shm-size 64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$MODEL_DIR:/model:ro" \
    -v /var/lib/hydralisk/huggingface:/root/.cache/huggingface \
    -e CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
    -e CUTE_DSL_ARCH=sm_120a \
    -e NCCL_P2P_DISABLE=1 \
    -e NCCL_P2P_LEVEL=SYS \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_DEBUG="$NCCL_DEBUG" \
    -e NCCL_SHM_DISABLE="$NCCL_SHM_DISABLE" \
    -e SAFETENSORS_FAST_GPU=1 \
    -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
    -e VLLM_USE_B12X_MOE=1 \
    -e VLLM_USE_V2_MODEL_RUNNER=1 \
    -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -e VLLM_USE_B12X_FP8_GEMM=1 \
    -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
    -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
    -e B12X_DENSE_SPLITK_TURBO=1 \
    -e B12X_W4A16_TC_DECODE=1 \
    -e B12X_MOE_FORCE_A16=1 \
    "$IMAGE_REF" \
    bash -lc "unset NCCL_GRAPH_FILE; exec $launch_command" \
    > "$REMOTE_LOG_DIR/vllm.log" 2>&1 &
  echo "$!" > "$REMOTE_LOG_DIR/container-launch.pid"
  printf 'status=starting\nstartedAt=%s\ncontainer=%s\n' \
    "$(date -u +%FT%TZ)" "$CONTAINER_NAME" > "$REMOTE_LOG_DIR/server.status"
}

status_action() {
  {
    printf "checkedAt=%s\n" "$(date -u +%FT%TZ)"
    printf "containerStatus="
    sudo docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || true
    printf "restartPolicy="
    sudo docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' "$CONTAINER_NAME" 2>/dev/null || true
    printf "modelsEndpoint="
    if curl -fsS "http://$HOST:$PORT/v1/models" >/tmp/hydralisk-glm52-models.json 2>/tmp/hydralisk-glm52-models.stderr; then
      printf "ready\n"
      cp /tmp/hydralisk-glm52-models.json "$REMOTE_LOG_DIR/models.json"
    else
      printf "not_ready\n"
    fi
    printf "gpuMemoryBegin\n"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits || true
    printf "gpuMemoryEnd\n"
    printf "diskLifecycleBegin\n"
    printf "modelDir=%s\n" "$MODEL_DIR"
    printf "hfCacheDir=/var/lib/hydralisk/huggingface\n"
    printf "logDir=%s\n" "$REMOTE_LOG_DIR"
    df -h "$MODEL_DIR" /var/lib/hydralisk/huggingface "$REMOTE_LOG_DIR" 2>/dev/null || true
    printf "diskLifecycleEnd\n"
  } > "$REMOTE_LOG_DIR/status-public.txt" 2>&1
}

apply_restart_policy_action() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker is required to apply restart policy" >&2
    exit 2
  fi
  write_launch_env
  write_launch_command
  sudo docker update --restart "$DOCKER_RESTART_POLICY" "$CONTAINER_NAME" >/dev/null
  printf 'status=restart_policy_applied\nappliedAt=%s\ncontainer=%s\nrestartPolicy=%s\n' \
    "$(date -u +%FT%TZ)" "$CONTAINER_NAME" "$DOCKER_RESTART_POLICY" \
    > "$REMOTE_LOG_DIR/server.status"
  status_action
}

stop_action() {
  sudo docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  printf 'status=stopped\nstoppedAt=%s\n' "$(date -u +%FT%TZ)" > "$REMOTE_LOG_DIR/server.status"
}

case "$ACTION" in
  prepare) prepare_action ;;
  start) start_action ;;
  apply-restart-policy) apply_restart_policy_action ;;
  status) status_action ;;
  stop) stop_action ;;
  *) echo "bad ACTION: $ACTION" >&2; exit 2 ;;
esac
REMOTE

copy_artifacts() {
  for name in \
    prepare-public.txt \
    launch.env.public \
    launch-command.json \
    status-public.txt \
    server.status \
    models.json; do
    run_gcloud compute scp \
      "$TARGET_INSTANCE:$REMOTE_LOG_DIR/$name" \
      "$OUTPUT_DIR/$name" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --quiet >/dev/null 2>&1 || true
  done
}

run_gcloud compute scp "$remote_script" \
  "$TARGET_INSTANCE:/tmp/hydralisk-glm52-launch-$RUN_ID.sh" \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --quiet

run_gcloud compute ssh "$TARGET_INSTANCE" \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --quiet \
  --command="ACTION='$ACTION' REMOTE_LOG_DIR='$REMOTE_LOG_DIR' MODEL_DIR='$MODEL_DIR' SERVED_MODEL_NAME='$SERVED_MODEL_NAME' CONTAINER_NAME='$CONTAINER_NAME' IMAGE_TAG='$IMAGE_TAG' IMAGE_DIGEST='$IMAGE_DIGEST' IMAGE_REF='$IMAGE_REF' DOCKER_RESTART_POLICY='$DOCKER_RESTART_POLICY' GPU_DEVICES='$GPU_DEVICES' TP_SIZE='$TP_SIZE' DCP_SIZE='$DCP_SIZE' MAX_MODEL_LEN='$MAX_MODEL_LEN' MAX_NUM_SEQS='$MAX_NUM_SEQS' MAX_NUM_BATCHED_TOKENS='$MAX_NUM_BATCHED_TOKENS' GPU_MEMORY_UTILIZATION='$GPU_MEMORY_UTILIZATION' HOST='$HOST' PORT='$PORT' MTP='$MTP' NUM_SPECULATIVE_TOKENS='$NUM_SPECULATIVE_TOKENS' NCCL_DEBUG='$NCCL_DEBUG' NCCL_SHM_DISABLE='$NCCL_SHM_DISABLE' INDEX_TOPK_PATTERN='$INDEX_TOPK_PATTERN' READY_TIMEOUT_SECONDS='$READY_TIMEOUT_SECONDS' bash /tmp/hydralisk-glm52-launch-$RUN_ID.sh"

copy_artifacts
echo "OUTPUT_DIR=$OUTPUT_DIR"
