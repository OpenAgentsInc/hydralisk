#!/usr/bin/env bash
set -Eeuo pipefail

origin="${HYDRALISK_ORIGIN:-http://127.0.0.1:8080}"
token="${HYDRALISK_BEARER_TOKEN:-}"
if [[ -z "${token}" ]]; then
  echo "HYDRALISK_BEARER_TOKEN is required" >&2
  exit 2
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

curl_json() {
  local url="$1"
  local output="$2"
  shift 2
  curl -fsS "$url" "$@" -o "$output"
}

assert_json() {
  local path="$1"
  local script="$2"
  python3 - "$path" "$script" <<'PY'
import json
import sys

path, script = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
namespace = {"data": data}
if not eval(script, {}, namespace):
    raise SystemExit(f"assertion failed for {path}: {script}")
PY
}

extract_header() {
  local headers="$1"
  local name="$2"
  awk -F': ' -v name="$(tr '[:upper:]' '[:lower:]' <<<"${name}")" '
    tolower($1) == name { value=$2 }
    END { gsub(/\r/, "", value); print value }
  ' "$headers"
}

health="${tmp_dir}/health.json"
curl_json "${origin}/health" "$health"
assert_json "$health" "data.get('servedModel') == 'openai/gpt-oss-20b' and 'token' not in str(data).lower()"

capabilities="${tmp_dir}/capabilities.json"
curl_json "${origin}/hydralisk/v1/capabilities" "$capabilities"
assert_json "$capabilities" "data.get('schema') == 'hydralisk.serve.capabilities.v1' and data.get('servedModel') == 'openai/gpt-oss-20b' and data.get('quantization', {}).get('weights') == 'MXFP4' and data.get('chatCompletions') is True"

nonstream_headers="${tmp_dir}/nonstream.headers"
nonstream="${tmp_dir}/nonstream.json"
curl -fsS "${origin}/v1/chat/completions" \
  -D "$nonstream_headers" \
  -H "authorization: Bearer ${token}" \
  -H "content-type: application/json" \
  -d '{
    "model": "openagents/khala",
    "messages": [
      { "role": "user", "content": "Say READY in one word." }
    ],
    "max_tokens": 8
  }' \
  -o "$nonstream"
assert_json "$nonstream" "isinstance(data.get('usage'), dict) and data['usage'].get('total_tokens', 0) > 0"

run_ref="$(extract_header "$nonstream_headers" "x-hydralisk-run-ref")"
receipt="${tmp_dir}/receipt.json"
curl_json "${origin}/hydralisk/v1/receipts/${run_ref}" "$receipt"
assert_json "$receipt" "data.get('schema') == 'hydralisk.serve.run_receipt.v1' and data.get('usage', {}).get('totalTokens', 0) > 0 and data.get('publicSafe') is True"

stream_headers="${tmp_dir}/stream.headers"
stream="${tmp_dir}/stream.txt"
curl -fsS -N "${origin}/v1/chat/completions" \
  -D "$stream_headers" \
  -H "authorization: Bearer ${token}" \
  -H "content-type: application/json" \
  -d '{
    "model": "openagents/khala",
    "stream": true,
    "messages": [
      { "role": "user", "content": "Write a two sentence service status report." }
    ],
    "max_tokens": 120
  }' \
  -o "$stream"

if ! grep -q '^data:' "$stream"; then
  echo "streaming response did not contain SSE data frames" >&2
  exit 1
fi

stream_run_ref="$(extract_header "$stream_headers" "x-hydralisk-run-ref")"
stream_receipt="${tmp_dir}/stream-receipt.json"
curl_json "${origin}/hydralisk/v1/receipts/${stream_run_ref}" "$stream_receipt"
assert_json "$stream_receipt" "data.get('schema') == 'hydralisk.serve.run_receipt.v1' and data.get('usage', {}).get('totalTokens', 0) > 0 and data.get('latency', {}).get('ttftMs') is not None"

echo "hydralisk GPT-OSS 20B smoke passed"
