#!/usr/bin/env bash
set -euo pipefail

TARGET_INSTANCE="${TARGET_INSTANCE:-hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036}"
TARGET_ZONE="${TARGET_ZONE:-us-central1-b}"
IMAGE="${IMAGE:-hydralisk-deepseek-v4-b12x-g4-vllm:20260624150453}"
OUTPUT_DIR="${OUTPUT_DIR:-.hydralisk/flashinfer-dsv4-fmha-$(date -u +%Y%m%d%H%M%S)}"

mkdir -p "$OUTPUT_DIR"

json_path="$OUTPUT_DIR/flashinfer-dsv4-fmha.json"
stderr_path="$OUTPUT_DIR/flashinfer-dsv4-fmha.stderr"
report_path="$OUTPUT_DIR/flashinfer-dsv4-fmha.md"

remote_command=$(cat <<EOF
sudo docker run -i --rm --gpus all --entrypoint python3 "$IMAGE" - <<'PY'
import json
import traceback

import torch
import flashinfer
from flashinfer import mla

result = {"schema": "hydralisk.flashinfer-dsv4-fmha-repro.v1"}
try:
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    result.update(
        {
            "flashinfer": getattr(flashinfer, "__version__", None),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "deviceName": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    )

    sum_q = 1
    heads = 64
    dim = 512
    page_size = 256
    pages = 1
    sparse_capacity = 128

    query = torch.randn((sum_q, heads, dim), device=device, dtype=torch.bfloat16)
    swa_kv_cache = torch.randn(
        (pages, 1, page_size, dim), device=device, dtype=torch.bfloat16
    )
    compressed_kv_cache = torch.randn(
        (pages, 1, page_size, dim), device=device, dtype=torch.bfloat16
    )
    workspace = torch.zeros((128 * 1024 * 1024,), device=device, dtype=torch.uint8)
    sparse_indices = torch.arange(
        sparse_capacity, device=device, dtype=torch.int32
    ).reshape(1, sparse_capacity)
    sparse_topk_lens = torch.full((sum_q,), 128, device=device, dtype=torch.int32)
    seq_lens = torch.full((1,), 128, device=device, dtype=torch.int32)
    out = torch.empty((sum_q, heads, dim), device=device, dtype=torch.bfloat16)
    cum_seq_lens_q = torch.tensor([0, 1], device=device, dtype=torch.int32)

    result["inputs"] = {
        "query": list(query.shape),
        "swaKvCache": list(swa_kv_cache.shape),
        "compressedKvCache": list(compressed_kv_cache.shape),
        "workspaceBytes": int(workspace.numel()),
        "sparseIndices": list(sparse_indices.shape),
        "sparseTopkLens": list(sparse_topk_lens.shape),
        "seqLens": list(seq_lens.shape),
        "out": list(out.shape),
        "dtype": "bf16",
        "kvLayout": "HND",
        "synthetic": True,
        "loadsModelWeights": False,
        "containsPrompts": False,
    }

    mla.trtllm_batch_decode_sparse_mla_dsv4(
        query=query,
        swa_kv_cache=swa_kv_cache,
        workspace_buffer=workspace,
        sparse_indices=sparse_indices,
        compressed_kv_cache=compressed_kv_cache,
        sparse_topk_lens=sparse_topk_lens,
        seq_lens=seq_lens,
        out=out,
        bmm1_scale=1.0,
        bmm2_scale=1.0,
        sinks=None,
        kv_layout="HND",
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=1,
    )
    torch.cuda.synchronize()
    result.update(
        {
            "status": "ok",
            "outFinite": bool(torch.isfinite(out).all().item()),
            "outShape": list(out.shape),
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
EOF
)

gcloud compute ssh "$TARGET_INSTANCE" \
  --zone "$TARGET_ZONE" \
  --command "$remote_command" \
  >"$json_path" \
  2>"$stderr_path"

status="$(jq -r '.status' "$json_path")"
error="$(jq -r '.error // ""' "$json_path")"
error_type="$(jq -r '.errorType // ""' "$json_path")"
device_name="$(jq -r '.deviceName // ""' "$json_path")"
capability="$(jq -c '.capability // []' "$json_path")"
flashinfer_version="$(jq -r '.flashinfer // ""' "$json_path")"
torch_version="$(jq -r '.torch // ""' "$json_path")"
cuda_version="$(jq -r '.cuda // ""' "$json_path")"
inputs="$(jq -c '.inputs // {}' "$json_path")"

cat >"$report_path" <<EOF
# FlashInfer DSV4 FMHA synthetic G4 repro

Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)

- Target instance: \`$TARGET_INSTANCE\`
- Target zone: \`$TARGET_ZONE\`
- Docker image: \`$IMAGE\`
- FlashInfer: \`$flashinfer_version\`
- Torch: \`$torch_version\`
- CUDA: \`$cuda_version\`
- Device: \`$device_name\`
- Capability: \`$capability\`

## Inputs

\`\`\`json
$inputs
\`\`\`

## Result

- Status: \`$status\`
- Error type: \`$error_type\`

\`\`\`text
$error
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
