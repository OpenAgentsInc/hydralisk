#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"
PYTHON_BIN="${PYTHON_BIN:-/opt/hydralisk-deepseek-v4/.venv/bin/python}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-scaled-mm-probe-$TS}"

mkdir -p "$OUTPUT_DIR"

if [[ -z "$TARGET_INSTANCE" || -z "$TARGET_ZONE" ]]; then
  echo "error: TARGET_INSTANCE and TARGET_ZONE are required" >&2
  exit 2
fi

if [[ "$TARGET_INSTANCE" != hydralisk-deepseek-v4-* ]]; then
  echo "error: TARGET_INSTANCE must be a fresh hydralisk-deepseek-v4-* probe host" >&2
  exit 2
fi

remote_probe="$(cat <<'PY'
import json
import sys

import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    w8a8_triton_block_scaled_mm,
)


def emit(record):
    print(json.dumps(record, sort_keys=True), flush=True)


def run_case(name, fn):
    torch.cuda.empty_cache()
    try:
        out = fn()
        torch.cuda.synchronize()
        emit(
            {
                "case": name,
                "ok": True,
                "shape": list(out.shape),
                "dtype": str(out.dtype),
                "mean": float(out.float().mean().cpu()),
            }
        )
    except Exception as exc:
        emit(
            {
                "case": name,
                "ok": False,
                "errorType": type(exc).__name__,
                "message": str(exc)[:600],
            }
        )


def fp8_tensor_case(m, k, n):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scale_a = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    return ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype=torch.bfloat16)


def triton_block_case(m, k, n):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    a = a.contiguous()
    b = b.contiguous()
    scale_a = torch.ones((m, k // 128), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((n // 128, k // 128), device="cuda", dtype=torch.float32)
    return w8a8_triton_block_scaled_mm(
        a, b, scale_a, scale_b, [128, 128], output_dtype=torch.bfloat16
    )


def triton_block_e8m0_case(m, k, n):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    a = a.contiguous()
    b = b.contiguous()
    scale_a = torch.ones((m, k // 128), device="cuda", dtype=torch.float32).to(
        torch.float8_e8m0fnu
    )
    scale_b = torch.ones((n // 128, k // 128), device="cuda", dtype=torch.float32).to(
        torch.float8_e8m0fnu
    )
    return w8a8_triton_block_scaled_mm(
        a, b, scale_a, scale_b, [128, 128], output_dtype=torch.bfloat16
    )


def main():
    if not torch.cuda.is_available():
        emit({"fatal": "cuda_unavailable"})
        return 1

    emit(
        {
            "schema": "hydralisk.deepseek-v4.scaled-mm-probe.v1",
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "deviceCount": torch.cuda.device_count(),
            "publicSafety": {
                "containsSecrets": False,
                "containsPrompts": False,
                "containsResponses": False,
                "containsWeights": False,
                "containsHiddenReasoning": False,
            },
        }
    )

    for cap in [80, 90, 100, 120]:
        row = {"capabilityInt": cap}
        for name in [
            "cutlass_scaled_mm_supports_fp8",
            "cutlass_scaled_mm_supports_block_fp8",
            "cutlass_scaled_mm_supports_fp4",
        ]:
            try:
                row[name] = bool(getattr(ops, name)(cap))
            except Exception as exc:
                row[name] = f"{type(exc).__name__}: {exc}"[:200]
        emit(row)

    run_case("cutlass_fp8_16", lambda: fp8_tensor_case(16, 16, 16))
    run_case("cutlass_fp8_m1_k4096_n4096", lambda: fp8_tensor_case(1, 4096, 4096))
    run_case("cutlass_fp8_m16_k4096_n4096", lambda: fp8_tensor_case(16, 4096, 4096))
    run_case("triton_block_fp8_m1_k128_n128", lambda: triton_block_case(1, 128, 128))
    run_case(
        "triton_block_fp8_m16_k4096_n4096",
        lambda: triton_block_case(16, 4096, 4096),
    )
    run_case(
        "triton_block_e8m0_m1_k128_n128",
        lambda: triton_block_e8m0_case(1, 128, 128),
    )
    return 0


sys.exit(main())
PY
)"

if [[ "${DRY_RUN:-0}" = "1" ]]; then
  {
    echo "# DeepSeek-V4-Flash scaled-mm probe"
    echo
    echo "DRY_RUN=1"
    echo
    echo "- Target instance: \`$TARGET_INSTANCE\`"
    echo "- Target zone: \`$TARGET_ZONE\`"
    echo "- Python: \`$PYTHON_BIN\`"
    echo
    echo "## Public safety"
    echo
    echo "- Contains secrets: false"
    echo "- Contains private prompts: false"
    echo "- Contains private responses: false"
    echo "- Contains weights: false"
    echo "- Contains hidden reasoning: false"
  } > "$OUTPUT_DIR/scaled-mm-probe.md"
  echo "Wrote $OUTPUT_DIR/scaled-mm-probe.md"
  exit 0
fi

gcloud compute ssh "$TARGET_INSTANCE" \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --quiet \
  --command "$PYTHON_BIN - <<'PY'
$remote_probe
PY" \
  > "$OUTPUT_DIR/scaled-mm-probe.jsonl" \
  2> "$OUTPUT_DIR/scaled-mm-probe.stderr" || true

{
  echo "# DeepSeek-V4-Flash scaled-mm probe"
  echo
  echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "- Target instance: \`$TARGET_INSTANCE\`"
  echo "- Target zone: \`$TARGET_ZONE\`"
  echo "- Python: \`$PYTHON_BIN\`"
  echo
  echo "## JSONL evidence"
  echo
  echo '```json'
  sed -n '1,160p' "$OUTPUT_DIR/scaled-mm-probe.jsonl"
  echo '```'
  echo
  echo "## Stderr"
  echo
  echo '```text'
  sed -n '1,120p' "$OUTPUT_DIR/scaled-mm-probe.stderr"
  echo '```'
  echo
  echo "## Public safety"
  echo
  echo "- Contains secrets: false"
  echo "- Contains private prompts: false"
  echo "- Contains private responses: false"
  echo "- Contains weights: false"
  echo "- Contains hidden reasoning: false"
} > "$OUTPUT_DIR/scaled-mm-probe.md"

echo "OUTPUT_DIR=$OUTPUT_DIR"
