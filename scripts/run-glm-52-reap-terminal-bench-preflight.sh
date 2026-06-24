#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${HYDRALISK_TB_BASE_URL:-http://127.0.0.1:8080}"
MODEL="${HYDRALISK_TB_MODEL:-glm-5.2-reap-504b-g4}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/glm52-reap-terminal-bench-preflight-$(date -u +%Y%m%d%H%M%S)}"
REQUIRE_HARBOR="${REQUIRE_HARBOR:-0}"
REQUIRE_DOCKER="${REQUIRE_DOCKER:-0}"

mkdir -p "$OUTPUT_DIR"

if [[ -z "${HYDRALISK_BEARER_TOKEN:-}" ]]; then
  echo "error: HYDRALISK_BEARER_TOKEN is required and will not be printed" >&2
  exit 2
fi

harbor_status="not_checked"
harbor_version="unavailable"
if command -v harbor >/dev/null 2>&1; then
  harbor_status="available"
  harbor_version="$(harbor --version 2>/dev/null | head -n 1 || true)"
elif [[ "$REQUIRE_HARBOR" == "1" ]]; then
  echo "error: harbor is required; install with: uv tool install harbor" >&2
  exit 2
fi

docker_status="not_checked"
if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    docker_status="ready"
  else
    docker_status="not_ready"
  fi
elif [[ "$REQUIRE_DOCKER" == "1" ]]; then
  echo "error: docker is required for Terminal-Bench/Harbor preflight" >&2
  exit 2
fi

uv run hydralisk-smoke \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  > "$OUTPUT_DIR/hydralisk-proxy-smoke.json"

python3 - "$OUTPUT_DIR/preflight-public.json" <<'PY'
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
smoke_path = output.with_name("hydralisk-proxy-smoke.json")
smoke = json.loads(smoke_path.read_text())
payload = {
    "schema": "hydralisk.evals.terminal_bench.preflight.v1",
    "baseUrlConfigured": bool(os.environ.get("HYDRALISK_TB_BASE_URL")),
    "model": os.environ.get("HYDRALISK_TB_MODEL", "glm-5.2-reap-504b-g4"),
    "harbor": {
        "status": os.environ.get("HYDRALISK_TB_HARBOR_STATUS", "unknown"),
        "version": os.environ.get("HYDRALISK_TB_HARBOR_VERSION", "unknown"),
    },
    "docker": {
        "status": os.environ.get("HYDRALISK_TB_DOCKER_STATUS", "unknown"),
    },
    "proxySmoke": {
        "passed": bool(smoke.get("passed")),
        "healthStatus": smoke.get("healthStatus"),
        "completionStatus": smoke.get("completionStatus"),
        "completionHasUsage": smoke.get("completionHasUsage"),
        "completionHasFinalContent": smoke.get("completionHasFinalContent"),
        "streamChunksObserved": smoke.get("streamChunksObserved"),
        "completionRunRef": smoke.get("completionRunRef"),
    },
    "publicSafety": {
        "containsSecrets": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsHiddenReasoning": False,
        "containsRawBenchmarkLogs": False,
    },
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

python3 - "$OUTPUT_DIR/preflight-public.json" "$harbor_status" "$harbor_version" "$docker_status" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["harbor"] = {"status": sys.argv[2], "version": sys.argv[3]}
payload["docker"] = {"status": sys.argv[4]}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "OUTPUT_DIR=$OUTPUT_DIR"
