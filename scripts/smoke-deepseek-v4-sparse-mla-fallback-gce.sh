#!/usr/bin/env bash
set -euo pipefail

TARGET_INSTANCE="${TARGET_INSTANCE:-hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036}"
TARGET_ZONE="${TARGET_ZONE:-us-central1-b}"
IMAGE="${IMAGE:-hydralisk-deepseek-v4-b12x-g4-vllm:20260624150453}"
OUTPUT_DIR="${OUTPUT_DIR:-.hydralisk/deepseek-v4-sparse-mla-fallback-container-$(date -u +%Y%m%d%H%M%S)}"

mkdir -p "$OUTPUT_DIR"

json_path="$OUTPUT_DIR/deepseek-v4-sparse-mla-fallback-container.json"
stderr_path="$OUTPUT_DIR/deepseek-v4-sparse-mla-fallback-container.stderr"
report_path="$OUTPUT_DIR/deepseek-v4-sparse-mla-fallback-container.md"
archive_path="$OUTPUT_DIR/hydralisk-sparse-mla-smoke.tgz"

if ! instance_status="$(gcloud compute instances describe "$TARGET_INSTANCE" \
  --zone "$TARGET_ZONE" \
  --format='value(status)' 2>"$stderr_path")"; then
  cat >"$json_path" <<EOF
{
  "schema": "hydralisk.deepseek-v4.sparse-mla-fallback-container-smoke.v1",
  "status": "target_missing",
  "target": {
    "instance": "$TARGET_INSTANCE",
    "zone": "$TARGET_ZONE",
    "image": "$IMAGE"
  },
  "result": {
    "ranContainer": false,
    "reason": "target instance was not found or could not be described"
  },
  "publicSafety": {
    "containsSecrets": false,
    "containsPrompts": false,
    "containsResponses": false,
    "containsWeights": false,
    "containsHiddenReasoning": false
  }
}
EOF
  cat >"$report_path" <<EOF
# DeepSeek V4 sparse MLA fallback container smoke

Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)

- Target instance: \`$TARGET_INSTANCE\`
- Target zone: \`$TARGET_ZONE\`
- Docker image: \`$IMAGE\`
- Status: \`target_missing\`

The target instance was not found or could not be described, so the fallback
container smoke did not run. The local Hydralisk smoke remains reproducible
with:

\`\`\`bash
uv run hydralisk-deepseek-v4-sparse-mla-smoke
\`\`\`

Raw public-safe JSON:

\`\`\`json
$(jq -c . "$json_path")
\`\`\`

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
EOF
  echo "Wrote $report_path"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  exit 0
fi

tar \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='.hydralisk' \
  --exclude='__pycache__' \
  -czf "$archive_path" \
  hydralisk pyproject.toml

remote_dir="/tmp/hydralisk-sparse-mla-smoke-$(date -u +%Y%m%d%H%M%S)-$$"
remote_archive="$remote_dir/hydralisk.tgz"

gcloud compute ssh "$TARGET_INSTANCE" \
  --zone "$TARGET_ZONE" \
  --command "rm -rf '$remote_dir' && mkdir -p '$remote_dir'"

gcloud compute scp "$archive_path" "$TARGET_INSTANCE:$remote_archive" \
  --zone "$TARGET_ZONE" \
  2>>"$stderr_path"

remote_command=$(cat <<EOF
set -euo pipefail
cd "$remote_dir"
mkdir repo
tar -xzf hydralisk.tgz -C repo
sudo docker run -i --rm \
  -v "$remote_dir/repo:/workspace/hydralisk:ro" \
  -w /workspace/hydralisk \
  --entrypoint python3 \
  "$IMAGE" \
  -m hydralisk.admission.deepseek_v4_sparse_mla_smoke \
  --stdout-json
rm -rf "$remote_dir"
EOF
)

gcloud compute ssh "$TARGET_INSTANCE" \
  --zone "$TARGET_ZONE" \
  --command "$remote_command" \
  >"$json_path" \
  2>>"$stderr_path"

tmp_json="$OUTPUT_DIR/container-smoke-inner.json"
mv "$json_path" "$tmp_json"
jq \
  --arg instance "$TARGET_INSTANCE" \
  --arg zone "$TARGET_ZONE" \
  --arg image "$IMAGE" \
  --arg status "$instance_status" \
  '{
    schema: "hydralisk.deepseek-v4.sparse-mla-fallback-container-smoke.v1",
    status: "ok",
    target: {
      instance: $instance,
      zone: $zone,
      image: $image,
      instanceStatus: $status
    },
    smoke: .,
    publicSafety: .publicSafety
  }' \
  "$tmp_json" >"$json_path"

cat >"$report_path" <<EOF
# DeepSeek V4 sparse MLA fallback container smoke

Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)

- Target instance: \`$TARGET_INSTANCE\`
- Target zone: \`$TARGET_ZONE\`
- Docker image: \`$IMAGE\`
- Instance status: \`$instance_status\`
- Status: \`$(jq -r '.status' "$json_path")\`

## Result

\`\`\`json
$(jq -c '.smoke.result' "$json_path")
\`\`\`

## Decision

\`\`\`json
$(jq -c '.smoke.decision' "$json_path")
\`\`\`

Raw public-safe JSON:

\`\`\`json
$(jq -c . "$json_path")
\`\`\`

## Public safety

- Contains secrets: false
- Contains private prompts: false
- Contains private responses: false
- Contains weights: false
- Contains hidden reasoning: false
EOF

echo "Wrote $report_path"
echo "OUTPUT_DIR=$OUTPUT_DIR"
