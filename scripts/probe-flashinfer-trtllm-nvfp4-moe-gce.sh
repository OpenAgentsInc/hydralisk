#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"
ISSUE_NUMBER="${ISSUE_NUMBER:-21}"
IMAGE="${IMAGE:-hydralisk-deepseek-v4-nvfp4-sm120-oproj-bypass-vllm:20260624085429}"
SEQ_LEN="${SEQ_LEN:-1024}"
HIDDEN_SIZE="${HIDDEN_SIZE:-4096}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"
NUM_EXPERTS="${NUM_EXPERTS:-256}"
LOCAL_NUM_EXPERTS="${LOCAL_NUM_EXPERTS:-128}"
TOP_K="${TOP_K:-6}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/flashinfer-trtllm-nvfp4-moe-$TS}"

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
  local md="$OUTPUT_DIR/flashinfer-trtllm-nvfp4-moe-probe.md"
  {
    echo "# FlashInfer TRTLLM NVFP4 MoE G4 probe"
    echo
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "- Issue: https://github.com/OpenAgentsInc/hydralisk/issues/$ISSUE_NUMBER"
    echo "- Target instance: \`$TARGET_INSTANCE\`"
    echo "- Target zone: \`$TARGET_ZONE\`"
    echo "- Image: \`$IMAGE\`"
    echo "- Sequence length: \`$SEQ_LEN\`"
    echo "- Hidden size: \`$HIDDEN_SIZE\`"
    echo "- Intermediate size: \`$INTERMEDIATE_SIZE\`"
    echo "- Experts: \`$NUM_EXPERTS\`"
    echo "- Local experts: \`$LOCAL_NUM_EXPERTS\`"
    echo "- Top-k: \`$TOP_K\`"
    echo
    if [[ "$DRY_RUN" = "1" ]]; then
      echo "DRY_RUN=1"
      echo
    else
      echo "## Hardware"
      echo
      echo '```text'
      sed -n '1,160p' "$OUTPUT_DIR/flashinfer-moe-hardware.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "## Result"
      echo
      echo '```json'
      sed -n '1,80p' "$OUTPUT_DIR/flashinfer-moe-result.jsonl" 2>/dev/null || true
      echo '```'
      echo
      echo "Stderr:"
      echo
      echo '```text'
      sed -n '1,160p' "$OUTPUT_DIR/flashinfer-moe-stderr.txt" 2>/dev/null || true
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
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  exit 0
fi

remote_script="$(
  printf 'IMAGE=%q\n' "$IMAGE"
  printf 'SEQ_LEN=%q\n' "$SEQ_LEN"
  printf 'HIDDEN_SIZE=%q\n' "$HIDDEN_SIZE"
  printf 'INTERMEDIATE_SIZE=%q\n' "$INTERMEDIATE_SIZE"
  printf 'NUM_EXPERTS=%q\n' "$NUM_EXPERTS"
  printf 'LOCAL_NUM_EXPERTS=%q\n' "$LOCAL_NUM_EXPERTS"
  printf 'TOP_K=%q\n' "$TOP_K"
  printf 'REMOTE_LOG_DIR=%q\n' "/var/log/hydralisk/flashinfer-trtllm-nvfp4-moe-$TS"
  cat <<'REMOTE'
set -Eeuo pipefail
sudo install -d -m 0777 "$REMOTE_LOG_DIR"

{
  printf "HOSTNAME\t%s\n" "$(hostname)"
  printf "KERNEL\t%s\n" "$(uname -r)"
  printf "GPU_QUERY_BEGIN\n"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,pci.bus_id --format=csv,noheader,nounits || true
  printf "GPU_QUERY_END\n"
  printf "IMAGE\t%s\n" "$IMAGE"
} > "$REMOTE_LOG_DIR/flashinfer-moe-hardware.txt"

cat > "$REMOTE_LOG_DIR/repro.py" <<'PY'
import json
import os
import traceback

import torch
import flashinfer
import flashinfer.fused_moe as fm


def env_int(name: str) -> int:
    return int(os.environ[name])


torch.cuda.set_device(0)
seq_len = env_int("SEQ_LEN")
hidden_size = env_int("HIDDEN_SIZE")
intermediate_size = env_int("INTERMEDIATE_SIZE")
num_experts = env_int("NUM_EXPERTS")
local_num_experts = env_int("LOCAL_NUM_EXPERTS")
top_k = env_int("TOP_K")

record = {
    "schema": "hydralisk.flashinfer.trtllm-nvfp4-moe.synthetic.v1",
    "flashinfer": getattr(flashinfer, "__version__", "unknown"),
    "torch": torch.__version__,
    "torchCuda": torch.version.cuda,
    "device": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "seqLen": seq_len,
    "hiddenSize": hidden_size,
    "intermediateSize": intermediate_size,
    "numExperts": num_experts,
    "localNumExperts": local_num_experts,
    "topK": top_k,
    "publicSafety": {
        "containsSecrets": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
        "containsHiddenReasoning": False,
    },
}

try:
    topk_ids = (
        torch.arange(seq_len * top_k, device="cuda", dtype=torch.int32)
        .reshape(seq_len, top_k)
        % local_num_experts
    )
    topk_weights = torch.full(
        (seq_len, top_k), 1.0 / top_k, device="cuda", dtype=torch.bfloat16
    )
    hidden_states = torch.zeros(
        (seq_len, hidden_size // 2), device="cuda", dtype=torch.uint8
    )
    hidden_states_scale = torch.ones(
        (seq_len, hidden_size // 16), device="cuda", dtype=torch.float8_e4m3fn
    )
    gemm1_weights = torch.zeros(
        (local_num_experts, 2 * intermediate_size, hidden_size // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    gemm1_weights_scale = torch.ones(
        (local_num_experts, 2 * intermediate_size, hidden_size // 16),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    gemm2_weights = torch.zeros(
        (local_num_experts, hidden_size, intermediate_size // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    gemm2_weights_scale = torch.ones(
        (local_num_experts, hidden_size, intermediate_size // 16),
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    clamp = torch.full((local_num_experts,), 10.0, device="cuda", dtype=torch.float32)
    output1_scale = torch.ones(
        (local_num_experts,), device="cuda", dtype=torch.float32
    )
    output1_gate_scale = torch.ones(
        (local_num_experts,), device="cuda", dtype=torch.float32
    )
    output2_scale = torch.ones(
        (local_num_experts,), device="cuda", dtype=torch.float32
    )
    outputs = fm.trtllm_fp4_block_scale_routed_moe(
        (topk_ids, topk_weights),
        None,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        None,
        None,
        None,
        clamp,
        gemm2_weights,
        gemm2_weights_scale,
        None,
        output1_scale,
        output1_gate_scale,
        output2_scale,
        num_experts,
        top_k,
        None,
        None,
        intermediate_size,
        0,
        local_num_experts,
        None,
        routing_method_type=0,
        do_finalize=True,
        activation_type=3,
        per_token_scale=None,
        output=None,
        tune_max_num_tokens=8192,
    )
    record.update({"ok": True, "outputs": [list(item.shape) for item in outputs]})
except Exception as exc:
    record.update(
        {
            "ok": False,
            "type": type(exc).__name__,
            "message": str(exc)[:1200],
        }
    )
    print(json.dumps(record, sort_keys=True), flush=True)
    traceback.print_exc(limit=24)
else:
    print(json.dumps(record, sort_keys=True), flush=True)
PY

sudo docker run --rm --gpus all --ipc=host --network host \
  -e "SEQ_LEN=$SEQ_LEN" \
  -e "HIDDEN_SIZE=$HIDDEN_SIZE" \
  -e "INTERMEDIATE_SIZE=$INTERMEDIATE_SIZE" \
  -e "NUM_EXPERTS=$NUM_EXPERTS" \
  -e "LOCAL_NUM_EXPERTS=$LOCAL_NUM_EXPERTS" \
  -e "TOP_K=$TOP_K" \
  -v "$REMOTE_LOG_DIR/repro.py:/tmp/repro.py:ro" \
  --entrypoint python3 "$IMAGE" /tmp/repro.py \
  > "$REMOTE_LOG_DIR/flashinfer-moe-result.jsonl" \
  2> "$REMOTE_LOG_DIR/flashinfer-moe-stderr.txt" || true
REMOTE
)"

gcloud compute ssh "$TARGET_INSTANCE" \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --quiet \
  --command "$remote_script" \
  > "$OUTPUT_DIR/remote.stdout" \
  2> "$OUTPUT_DIR/remote.stderr" || true

remote_dir="/var/log/hydralisk/flashinfer-trtllm-nvfp4-moe-$TS"
for file in \
  flashinfer-moe-hardware.txt \
  flashinfer-moe-result.jsonl \
  flashinfer-moe-stderr.txt; do
  gcloud compute scp \
    "$TARGET_INSTANCE:$remote_dir/$file" \
    "$OUTPUT_DIR/$file" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet > "$OUTPUT_DIR/scp-$file.log" 2>&1 || true
done

render_markdown
echo "OUTPUT_DIR=$OUTPUT_DIR"
