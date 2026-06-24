#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-${CLOUDSDK_CORE_ACCOUNT:-}}"
TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"
MODEL_ID="${MODEL_ID:-nvidia/DeepSeek-V4-Flash-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-e3cd60e7de98e9867116860d522499a728de1cf9}"
DERIVED_IMAGE="${DERIVED_IMAGE:-hydralisk-deepseek-v4-b12x-g4-vllm-issue60-vector-v3:20260624v3vector2}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-khala-readiness-$RUN_ID}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/var/log/hydralisk/deepseek-khala-readiness-$RUN_ID}"
BENCH_PROMPT_FILE="${BENCH_PROMPT_FILE:-}"
BENCH_PROMPT_B64="${BENCH_PROMPT_B64:-}"
STREAM_REQUESTS="${STREAM_REQUESTS:-5}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-1}"
MAX_TOKENS="${MAX_TOKENS:-32}"
WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-7200}"
CONTAINER_START_TIMEOUT_SECONDS="${CONTAINER_START_TIMEOUT_SECONDS:-240}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-360}"
MAX_READY_SECONDS="${MAX_READY_SECONDS:-180}"
MAX_WARMUP_SECONDS="${MAX_WARMUP_SECONDS:-45}"
MAX_TTFT_P95_SECONDS="${MAX_TTFT_P95_SECONDS:-2.0}"
MIN_DECODE_TPS_P50="${MIN_DECODE_TPS_P50:-8.0}"
MIN_E2E_TPS_P50="${MIN_E2E_TPS_P50:-8.0}"

if [[ -z "$TARGET_INSTANCE" || -z "$TARGET_ZONE" ]]; then
  echo "error: TARGET_INSTANCE and TARGET_ZONE are required" >&2
  exit 2
fi

if [[ -z "$BENCH_PROMPT_B64" ]]; then
  if [[ -z "$BENCH_PROMPT_FILE" ]]; then
    echo "error: set BENCH_PROMPT_FILE or BENCH_PROMPT_B64; raw prompts are runtime input, not committed repo data" >&2
    exit 2
  fi
  BENCH_PROMPT_B64="$(base64 < "$BENCH_PROMPT_FILE" | tr -d '\n')"
fi

run_gcloud() {
  if [[ -n "$GCLOUD_ACCOUNT" ]]; then
    CLOUDSDK_CORE_ACCOUNT="$GCLOUD_ACCOUNT" gcloud "$@"
  else
    gcloud "$@"
  fi
}

mkdir -p "$OUTPUT_DIR"
remote_script="$(mktemp "${TMPDIR:-/tmp}/hydralisk-khala-readiness.XXXXXX.sh")"
trap 'rm -f "$remote_script"' EXIT

cat > "$remote_script" <<'REMOTE'
#!/usr/bin/env bash
set -Eeuo pipefail

sudo install -d -m 0777 "$REMOTE_LOG_DIR"
container_name="hydralisk-deepseek-khala-readiness-$RUN_ID"
sudo docker rm -f "$container_name" >/dev/null 2>&1 || true

start_epoch="$(date +%s)"
sudo docker run --rm --gpus all --ipc=host --network host \
  --name "$container_name" \
  -v /var/lib/hydralisk/huggingface:/root/.cache/huggingface \
  -e HF_HUB_DISABLE_XET=1 \
  -e HF_XET_HIGH_PERFORMANCE=0 \
  -e HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=hopper \
  -e HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE=0 \
  -e HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS=1 \
  -e HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE=fp32 \
  -e HYDRALISK_DEEPSEEK_O_PROJ_BYPASS=off \
  -e HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=bf16_einsum \
  -e HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK=1 \
  -e HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_RPC_TIMEOUT=600000 \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  -e CUDA_LAUNCH_BLOCKING=0 \
  "$DERIVED_IMAGE" \
  "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --tokenizer-revision "$MODEL_REVISION" \
  --moe-backend flashinfer_b12x \
  --linear-backend triton \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 8 \
  --enforce-eager \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV4"}' \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  > "$REMOTE_LOG_DIR/vllm.log" 2>&1 &
pid="$!"

ready=0
deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
container_start_deadline=$((SECONDS + CONTAINER_START_TIMEOUT_SECONDS))
container_seen=0
container_status=""
while [[ "$SECONDS" -lt "$deadline" ]]; do
  if curl -fsS http://127.0.0.1:8000/v1/models \
    > "$REMOTE_LOG_DIR/models.json" \
    2> "$REMOTE_LOG_DIR/models.stderr"; then
    ready=1
    break
  fi
  container_status="$(sudo docker inspect -f '{{.State.Status}}' "$container_name" 2>/dev/null || true)"
  case "$container_status" in
    running|created|restarting|paused)
      container_seen=1
      ;;
    exited|dead)
      break
      ;;
    "")
      if [[ "$container_seen" = "1" || "$SECONDS" -ge "$container_start_deadline" ]]; then
        break
      fi
      ;;
    *)
      break
      ;;
  esac
  sleep 5
done
ready_epoch="$(date +%s)"
export start_epoch ready_epoch container_status

if [[ "$ready" != "1" ]]; then
  python3 - "$REMOTE_LOG_DIR/readiness-public.json" <<'PY'
import json
import os
import sys

path = sys.argv[1]
payload = {
    "ready": False,
    "status": "server_not_ready",
    "containerStatus": os.environ.get("container_status", ""),
    "publicSafety": {
        "containsPromptText": False,
        "containsResponseText": False,
        "containsSecrets": False,
    },
}
with open(path, "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
PY
else
  python3 - <<'PY' > "$REMOTE_LOG_DIR/readiness-public.json"
import base64
import hashlib
import json
import os
import statistics
import time
import urllib.request

model = os.environ["MODEL_ID"]
prompt = base64.b64decode(os.environ["BENCH_PROMPT_B64"]).decode("utf-8")
url = "http://127.0.0.1:8000/v1/chat/completions"
request_timeout = float(os.environ["REQUEST_TIMEOUT_SECONDS"])
start_epoch = int(os.environ["start_epoch"])
ready_epoch = int(os.environ["ready_epoch"])
stream_requests = int(os.environ["STREAM_REQUESTS"])
warmup_requests = int(os.environ["WARMUP_REQUESTS"])
max_tokens = int(os.environ["MAX_TOKENS"])
warmup_max_tokens = int(os.environ["WARMUP_MAX_TOKENS"])


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    weight = pos - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def payload(max_output_tokens, stream):
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_output_tokens,
        "temperature": 0,
    }
    if stream:
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}
    return request


def post_json(request):
    req = urllib.request.Request(
        url,
        data=json.dumps(request).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=request_timeout) as response:
        return response.read()


def warmup(index):
    t0 = time.perf_counter()
    body = post_json(payload(warmup_max_tokens, False))
    t1 = time.perf_counter()
    data = json.loads(body)
    usage = data.get("usage") or {}
    return {
        "index": index,
        "elapsedSeconds": round(t1 - t0, 6),
        "promptTokens": usage.get("prompt_tokens"),
        "completionTokens": usage.get("completion_tokens"),
        "finishReason": ((data.get("choices") or [{}])[0] or {}).get("finish_reason"),
    }


def stream_once(index):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload(max_tokens, True)).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    first_sse = None
    first_delta = None
    last_delta = None
    chunks_with_delta = 0
    usage = None
    finish_reason = None
    with urllib.request.urlopen(req, timeout=request_timeout) as response:
        for raw in response:
            line = raw.strip()
            if not line or not line.startswith(b"data: "):
                continue
            now = time.perf_counter()
            data = line[6:]
            if data == b"[DONE]":
                break
            if first_sse is None:
                first_sse = now
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                finish_reason = choices[0].get("finish_reason") or finish_reason
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    chunks_with_delta += 1
                    if first_delta is None:
                        first_delta = now
                    last_delta = now
    t1 = time.perf_counter()
    usage = usage or {}
    completion_tokens = usage.get("completion_tokens")
    decode_window = (
        last_delta - first_delta
        if first_delta and last_delta and last_delta > first_delta
        else None
    )
    return {
        "index": index,
        "elapsedSeconds": round(t1 - t0, 6),
        "timeToFirstSseSeconds": round(first_sse - t0, 6) if first_sse else None,
        "timeToFirstDeltaSeconds": round(first_delta - t0, 6) if first_delta else None,
        "decodeWindowSeconds": round(decode_window, 6) if decode_window else None,
        "promptTokens": usage.get("prompt_tokens"),
        "completionTokens": completion_tokens,
        "totalTokens": usage.get("total_tokens"),
        "chunksWithDelta": chunks_with_delta,
        "finishReason": finish_reason,
        "outputTokensPerSecondFullElapsed": (
            round(completion_tokens / (t1 - t0), 6) if completion_tokens else None
        ),
        "outputTokensPerSecondAfterFirstDelta": (
            round(max(completion_tokens - 1, 0) / decode_window, 6)
            if completion_tokens and decode_window
            else None
        ),
    }


warmups = [warmup(i) for i in range(warmup_requests)]
streams = [stream_once(i) for i in range(stream_requests)]
successful = [
    run
    for run in streams
    if run.get("completionTokens")
    and run.get("timeToFirstDeltaSeconds") is not None
    and run.get("outputTokensPerSecondAfterFirstDelta") is not None
]
ttft = [run["timeToFirstDeltaSeconds"] for run in successful]
decode_tps = [run["outputTokensPerSecondAfterFirstDelta"] for run in successful]
e2e_tps = [run["outputTokensPerSecondFullElapsed"] for run in successful]

thresholds = {
    "maxReadySeconds": float(os.environ["MAX_READY_SECONDS"]),
    "maxWarmupSeconds": float(os.environ["MAX_WARMUP_SECONDS"]),
    "maxTtftP95Seconds": float(os.environ["MAX_TTFT_P95_SECONDS"]),
    "minDecodeTokensPerSecondP50": float(os.environ["MIN_DECODE_TPS_P50"]),
    "minEndToEndTokensPerSecondP50": float(os.environ["MIN_E2E_TPS_P50"]),
    "minSuccessfulRequests": stream_requests,
}
summary = {
    "serverStartToReadyWallSeconds": ready_epoch - start_epoch,
    "successfulRequests": len(successful),
    "ttftP50Seconds": percentile(ttft, 0.5),
    "ttftP95Seconds": percentile(ttft, 0.95),
    "decodeTokensPerSecondP50": percentile(decode_tps, 0.5),
    "decodeTokensPerSecondP95": percentile(decode_tps, 0.95),
    "endToEndTokensPerSecondP50": percentile(e2e_tps, 0.5),
    "endToEndTokensPerSecondP95": percentile(e2e_tps, 0.95),
    "warmupMaxSeconds": max((run["elapsedSeconds"] for run in warmups), default=None),
}
passed = (
    summary["serverStartToReadyWallSeconds"] <= thresholds["maxReadySeconds"]
    and summary["successfulRequests"] >= thresholds["minSuccessfulRequests"]
    and summary["warmupMaxSeconds"] is not None
    and summary["warmupMaxSeconds"] <= thresholds["maxWarmupSeconds"]
    and summary["ttftP95Seconds"] is not None
    and summary["ttftP95Seconds"] <= thresholds["maxTtftP95Seconds"]
    and summary["decodeTokensPerSecondP50"] is not None
    and summary["decodeTokensPerSecondP50"] >= thresholds["minDecodeTokensPerSecondP50"]
    and summary["endToEndTokensPerSecondP50"] is not None
    and summary["endToEndTokensPerSecondP50"] >= thresholds["minEndToEndTokensPerSecondP50"]
)

print(
    json.dumps(
        {
            "ready": True,
            "khalaReadinessGatePassed": passed,
            "image": os.environ["DERIVED_IMAGE"],
            "model": model,
            "promptSha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "promptBytes": len(prompt.encode("utf-8")),
            "maxTokens": max_tokens,
            "warmupMaxTokens": warmup_max_tokens,
            "thresholds": thresholds,
            "summary": summary,
            "warmups": warmups,
            "streams": streams,
            "publicSafety": {
                "containsPromptText": False,
                "containsResponseText": False,
                "containsSecrets": False,
            },
        },
        indent=2,
        sort_keys=True,
    )
)
PY
fi

tail -n 260 "$REMOTE_LOG_DIR/vllm.log" \
  | sed -E 's/(hf_[A-Za-z0-9_\-]+)/<redacted-hf-token>/g' \
  > "$REMOTE_LOG_DIR/vllm-tail-public.txt" || true

sudo docker rm -f "$container_name" >/dev/null 2>&1 || true
wait "$pid" >/tmp/hydralisk-khala-readiness-vllm-exit.log 2>&1 || true
cat "$REMOTE_LOG_DIR/readiness-public.json"
REMOTE

chmod +x "$remote_script"
remote_path="/tmp/hydralisk-khala-readiness-$RUN_ID.sh"
run_gcloud compute scp \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --tunnel-through-iap \
  "$remote_script" \
  "$TARGET_INSTANCE:$remote_path" \
  > "$OUTPUT_DIR/scp-remote-script.log" 2>&1

remote_env=(
  "MODEL_ID=$MODEL_ID"
  "MODEL_REVISION=$MODEL_REVISION"
  "DERIVED_IMAGE=$DERIVED_IMAGE"
  "RUN_ID=$RUN_ID"
  "REMOTE_LOG_DIR=$REMOTE_LOG_DIR"
  "BENCH_PROMPT_B64=$BENCH_PROMPT_B64"
  "STREAM_REQUESTS=$STREAM_REQUESTS"
  "WARMUP_REQUESTS=$WARMUP_REQUESTS"
  "MAX_TOKENS=$MAX_TOKENS"
  "WARMUP_MAX_TOKENS=$WARMUP_MAX_TOKENS"
  "MAX_MODEL_LEN=$MAX_MODEL_LEN"
  "MAX_NUM_SEQS=$MAX_NUM_SEQS"
  "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
  "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
  "READY_TIMEOUT_SECONDS=$READY_TIMEOUT_SECONDS"
  "CONTAINER_START_TIMEOUT_SECONDS=$CONTAINER_START_TIMEOUT_SECONDS"
  "REQUEST_TIMEOUT_SECONDS=$REQUEST_TIMEOUT_SECONDS"
  "MAX_READY_SECONDS=$MAX_READY_SECONDS"
  "MAX_WARMUP_SECONDS=$MAX_WARMUP_SECONDS"
  "MAX_TTFT_P95_SECONDS=$MAX_TTFT_P95_SECONDS"
  "MIN_DECODE_TPS_P50=$MIN_DECODE_TPS_P50"
  "MIN_E2E_TPS_P50=$MIN_E2E_TPS_P50"
)
remote_prefix=""
for item in "${remote_env[@]}"; do
  remote_prefix+="$(printf '%q' "$item") "
done

run_gcloud compute ssh \
  "$TARGET_INSTANCE" \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --tunnel-through-iap \
  --command "${remote_prefix}bash $(printf '%q' "$remote_path")" \
  | tee "$OUTPUT_DIR/readiness-public.json"

run_gcloud compute scp \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --tunnel-through-iap \
  --recurse \
  "$TARGET_INSTANCE:$REMOTE_LOG_DIR" \
  "$OUTPUT_DIR/remote" \
  > "$OUTPUT_DIR/scp-remote-logs.log" 2>&1 || true

python3 - "$OUTPUT_DIR/readiness-public.json" "$OUTPUT_DIR/readiness-report.md" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
data = json.loads(source.read_text())
summary = data.get("summary", {})
thresholds = data.get("thresholds", {})

def value(name):
    item = summary.get(name)
    return "n/a" if item is None else str(round(item, 6) if isinstance(item, float) else item)

target.write_text(
    "\n".join(
        [
            "# DeepSeek V4 Khala readiness probe",
            "",
            f"- Ready: `{data.get('ready')}`",
            f"- Gate passed: `{data.get('khalaReadinessGatePassed')}`",
            f"- Image: `{data.get('image', 'unknown')}`",
            f"- Model: `{data.get('model', 'unknown')}`",
            f"- Prompt SHA-256: `{data.get('promptSha256', 'unavailable')}`",
            f"- Prompt bytes: `{data.get('promptBytes', 'unavailable')}`",
            "",
            "## Summary",
            "",
            f"- Start to ready seconds: `{value('serverStartToReadyWallSeconds')}`",
            f"- Successful requests: `{value('successfulRequests')}`",
            f"- Warmup max seconds: `{value('warmupMaxSeconds')}`",
            f"- TTFT p50 seconds: `{value('ttftP50Seconds')}`",
            f"- TTFT p95 seconds: `{value('ttftP95Seconds')}`",
            f"- Decode tok/s p50: `{value('decodeTokensPerSecondP50')}`",
            f"- Decode tok/s p95: `{value('decodeTokensPerSecondP95')}`",
            f"- End-to-end tok/s p50: `{value('endToEndTokensPerSecondP50')}`",
            f"- End-to-end tok/s p95: `{value('endToEndTokensPerSecondP95')}`",
            "",
            "## Thresholds",
            "",
            *[f"- {key}: `{val}`" for key, val in sorted(thresholds.items())],
            "",
            "## Public safety",
            "",
            "- Contains prompt text: false",
            "- Contains response text: false",
            "- Contains secrets: false",
            "",
        ]
    )
)
PY

echo "Wrote $OUTPUT_DIR/readiness-report.md"
