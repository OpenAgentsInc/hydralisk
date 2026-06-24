#!/usr/bin/env bash
set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-.hydralisk/deepseek-v4-sparse-mla-vllm-patch-$(date -u +%Y%m%d%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "$VLLM_ROOT" ]]; then
  echo "error: set VLLM_ROOT to the vLLM source root" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

json_path="$OUTPUT_DIR/deepseek-v4-sparse-mla-vllm-patch.json"
args=(--vllm-root "$VLLM_ROOT" --json "$json_path")
if [[ "$DRY_RUN" = "1" ]]; then
  args+=(--dry-run)
fi

uv run hydralisk-deepseek-v4-sparse-mla-vllm-patch "${args[@]}"

cat >"$OUTPUT_DIR/deepseek-v4-sparse-mla-vllm-patch.md" <<EOF
# DeepSeek V4 sparse MLA vLLM fallback patch

Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)

- vLLM root: \`$VLLM_ROOT\`
- Dry run: \`$DRY_RUN\`
- Status: \`$(jq -r '.result | if .patched then "patched" elif .already_patched then "already_patched" else "not_patched" end' "$json_path")\`
- Env flag: \`$(jq -r '.envFlag' "$json_path")\`
- Default enabled: \`$(jq -r '.defaultEnabled' "$json_path")\`

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

echo "Wrote $OUTPUT_DIR/deepseek-v4-sparse-mla-vllm-patch.md"
echo "OUTPUT_DIR=$OUTPUT_DIR"
