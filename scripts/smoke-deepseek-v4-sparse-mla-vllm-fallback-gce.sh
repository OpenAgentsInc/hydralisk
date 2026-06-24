#!/usr/bin/env bash
set -euo pipefail

TARGET_INSTANCE="${TARGET_INSTANCE:-hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036}"
TARGET_ZONE="${TARGET_ZONE:-us-central1-b}"
IMAGE="${IMAGE:-hydralisk-deepseek-v4-b12x-g4-vllm:20260624150453}"
OUTPUT_DIR="${OUTPUT_DIR:-.hydralisk/deepseek-v4-sparse-mla-vllm-fallback-$(date -u +%Y%m%d%H%M%S)}"

mkdir -p "$OUTPUT_DIR"

json_path="$OUTPUT_DIR/deepseek-v4-sparse-mla-vllm-fallback.json"
stderr_path="$OUTPUT_DIR/deepseek-v4-sparse-mla-vllm-fallback.stderr"
report_path="$OUTPUT_DIR/deepseek-v4-sparse-mla-vllm-fallback.md"
archive_path="$OUTPUT_DIR/hydralisk-vllm-fallback-smoke.tgz"

write_target_missing() {
  cat >"$json_path" <<EOF
{
  "schema": "hydralisk.deepseek-v4.sparse-mla-vllm-fallback-smoke.v1",
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
# DeepSeek V4 sparse MLA patched-vLLM fallback smoke

Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)

- Target instance: \`$TARGET_INSTANCE\`
- Target zone: \`$TARGET_ZONE\`
- Docker image: \`$IMAGE\`
- Status: \`target_missing\`

The target instance was not found or could not be described, so the patched
vLLM fallback smoke did not run.

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
}

if ! instance_status="$(gcloud compute instances describe "$TARGET_INSTANCE" \
  --zone "$TARGET_ZONE" \
  --format='value(status)' 2>"$stderr_path")"; then
  write_target_missing
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

remote_dir="/tmp/hydralisk-vllm-fallback-smoke-$(date -u +%Y%m%d%H%M%S)-$$"
remote_archive="$remote_dir/hydralisk.tgz"

gcloud compute ssh "$TARGET_INSTANCE" \
  --zone "$TARGET_ZONE" \
  --command "rm -rf '$remote_dir' && mkdir -p '$remote_dir'"

gcloud compute scp "$archive_path" "$TARGET_INSTANCE:$remote_archive" \
  --zone "$TARGET_ZONE" \
  2>>"$stderr_path"

remote_command=$(cat <<'EOF'
set -euo pipefail
cd "$REMOTE_DIR"
mkdir repo
tar -xzf hydralisk.tgz -C repo
sudo docker run -i --rm --gpus all \
  -e HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK=1 \
  -v "$REMOTE_DIR/repo:/workspace/hydralisk:ro" \
  -w /workspace/hydralisk \
  --entrypoint python3 \
  "$IMAGE" - <<'PY'
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, "/workspace/hydralisk")

result = {
    "schema": "hydralisk.deepseek-v4.sparse-mla-vllm-fallback-inner.v1",
    "status": "unknown",
    "publicSafety": {
        "containsSecrets": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
        "containsHiddenReasoning": False,
    },
}

try:
    import torch
    import vllm

    from hydralisk.admission.deepseek_v4_sparse_mla_vllm_patch import (
        TARGET_RELATIVE_PATH,
        patch_file,
    )

    vllm_root = Path(vllm.__file__).resolve().parents[1]
    patch_target = vllm_root / TARGET_RELATIVE_PATH
    patch_result = patch_file(patch_target)

    module = importlib.import_module("vllm.models.deepseek_v4.nvidia.flashinfer_sparse")
    helper = getattr(module, "_hydralisk_sparse_mla_fallback")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    query = torch.randn((1, 64, 512), device=device, dtype=torch.bfloat16)
    swa_kv_cache = torch.randn((1, 1, 256, 512), device=device, dtype=torch.bfloat16)
    compressed_kv_cache = torch.randn(
        (1, 1, 256, 512),
        device=device,
        dtype=torch.bfloat16,
    )
    sparse_indices = torch.arange(128, device=device, dtype=torch.int32).reshape(1, 128)
    sparse_topk_lens = torch.full((1,), 128, device=device, dtype=torch.int32)
    seq_lens = torch.full((1,), 128, device=device, dtype=torch.int32)
    out = torch.empty((1, 64, 512), device=device, dtype=torch.bfloat16)

    helper(
        query=query,
        swa_kv_cache=swa_kv_cache,
        compressed_kv_cache=compressed_kv_cache,
        sparse_indices=sparse_indices,
        sparse_topk_lens=sparse_topk_lens,
        seq_lens=seq_lens,
        out=out,
        window_size=128,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    out_float = out.float().cpu()
    result.update(
        {
            "status": "ok",
            "envFlag": os.getenv("HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK"),
            "vllmVersion": getattr(vllm, "__version__", None),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "deviceName": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
            "patch": patch_result.__dict__,
            "inputs": {
                "query": list(query.shape),
                "swaKvCache": list(swa_kv_cache.shape),
                "compressedKvCache": list(compressed_kv_cache.shape),
                "sparseIndices": list(sparse_indices.shape),
                "sparseTopkLens": list(sparse_topk_lens.shape),
                "seqLens": list(seq_lens.shape),
                "out": list(out.shape),
                "dtype": "bf16",
                "kvLayout": "HND",
                "synthetic": True,
                "loadsModelWeights": False,
                "containsPrompts": False,
                "containsResponses": False,
            },
            "output": {
                "finite": bool(torch.isfinite(out_float).all().item()),
                "nonzero": bool(torch.count_nonzero(out_float).item() > 0),
                "shape": list(out.shape),
                "checksum": {
                    "sum": round(float(out_float.sum().item()), 6),
                    "l1": round(float(out_float.abs().sum().item()), 6),
                    "maxAbs": round(float(out_float.abs().max().item()), 6),
                    "count": int(out_float.numel()),
                },
            },
        }
    )
except Exception as exc:
    result.update(
        {
            "status": "error",
            "errorType": type(exc).__name__,
            "error": str(exc),
            "traceTail": traceback.format_exc().splitlines()[-12:],
        }
    )

print(json.dumps(result, sort_keys=True))
PY
rm -rf "$REMOTE_DIR"
EOF
)

remote_command="${remote_command//\$REMOTE_DIR/$remote_dir}"
remote_command="${remote_command//\$IMAGE/$IMAGE}"

gcloud compute ssh "$TARGET_INSTANCE" \
  --zone "$TARGET_ZONE" \
  --command "$remote_command" \
  >"$json_path" \
  2>>"$stderr_path"

tmp_json="$OUTPUT_DIR/vllm-fallback-inner.json"
mv "$json_path" "$tmp_json"
jq \
  --arg instance "$TARGET_INSTANCE" \
  --arg zone "$TARGET_ZONE" \
  --arg image "$IMAGE" \
  --arg status "$instance_status" \
  '{
    schema: "hydralisk.deepseek-v4.sparse-mla-vllm-fallback-smoke.v1",
    status: .status,
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
# DeepSeek V4 sparse MLA patched-vLLM fallback smoke

Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)

- Target instance: \`$TARGET_INSTANCE\`
- Target zone: \`$TARGET_ZONE\`
- Docker image: \`$IMAGE\`
- Instance status: \`$instance_status\`
- Status: \`$(jq -r '.status' "$json_path")\`

## Result

\`\`\`json
$(jq -c '.smoke.output // .smoke' "$json_path")
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
