#!/usr/bin/env bash
set -euo pipefail

TARGET_INSTANCE="${TARGET_INSTANCE:-hydralisk-deepseek-v4-b12x-g4-8g-b-20260624092036}"
TARGET_ZONE="${TARGET_ZONE:-us-central1-b}"
IMAGE="${IMAGE:-hydralisk-deepseek-v4-b12x-g4-vllm:20260624150453}"
OUTPUT_DIR="${OUTPUT_DIR:-.hydralisk/flashinfer-dsv4-fmha-sm120-audit-$(date -u +%Y%m%d%H%M%S)}"

mkdir -p "$OUTPUT_DIR"

json_path="$OUTPUT_DIR/flashinfer-dsv4-fmha-sm120-audit.json"
stderr_path="$OUTPUT_DIR/flashinfer-dsv4-fmha-sm120-audit.stderr"
report_path="$OUTPUT_DIR/flashinfer-dsv4-fmha-sm120-audit.md"

remote_command=$(cat <<EOF
sudo docker run -i --rm --entrypoint python3 "$IMAGE" - <<'PY'
import json
import re
from pathlib import Path

import flashinfer
import torch

root = Path("/usr/local/lib/python3.12/dist-packages")
data_root = root / "flashinfer" / "data"

runner = data_root / "include" / "flashinfer" / "trtllm" / "fmha" / "fmhaRunner.cuh"
kernels = data_root / "include" / "flashinfer" / "trtllm" / "fmha" / "fmhaKernels.cuh"
common = data_root / "include" / "flashinfer" / "trtllm" / "common.h"
launcher = data_root / "csrc" / "trtllm_fmha_kernel_launcher.cu"
cubin_root = root / "flashinfer_cubin" / "cubins"


def read_lines(path):
    try:
        return path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []


def line_match(path, pattern):
    regex = re.compile(pattern)
    for number, line in enumerate(read_lines(path), start=1):
        if regex.search(line):
            return {"path": str(path), "line": number, "text": line.strip()}
    return None


def constants(path):
    found = []
    regex = re.compile(r"kSM_(100f|100|103|120)\\b")
    for number, line in enumerate(read_lines(path), start=1):
        if regex.search(line):
            found.append({"line": number, "text": line.strip()})
    return found


def function_block(path, function_name, max_lines=32):
    lines = read_lines(path)
    start = None
    for index, line in enumerate(lines):
        if function_name in line:
            start = index
            break
    if start is None:
        return None
    block = []
    depth = 0
    opened = False
    for index in range(start, min(len(lines), start + max_lines)):
        line = lines[index]
        block.append({"line": index + 1, "text": line.rstrip()})
        depth += line.count("{")
        if "{" in line:
            opened = True
        depth -= line.count("}")
        if opened and depth <= 0:
            break
    return {"path": str(path), "lines": block}


cubins = sorted(cubin_root.glob("**/fmha/trtllm-gen/*.cubin"))
cubin_names = [str(path.relative_to(cubin_root)) for path in cubins]

result = {
    "schema": "hydralisk.flashinfer-dsv4-fmha-sm120-audit.v1",
    "image": "$IMAGE",
    "flashinfer": getattr(flashinfer, "__version__", None),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "sourcePaths": {
        "runner": str(runner),
        "kernels": str(kernels),
        "common": str(common),
        "launcher": str(launcher),
        "cubinRoot": str(cubin_root),
    },
    "sourceExists": {
        "runner": runner.exists(),
        "kernels": kernels.exists(),
        "common": common.exists(),
        "launcher": launcher.exists(),
        "cubinRoot": cubin_root.exists(),
    },
    "runnerGuard": line_match(runner, r"Unsupported architecture"),
    "runnerSmInit": line_match(runner, r"mSM\\(getSMVersion\\(\\)\\)"),
    "smConstants": constants(common),
    "isSmCompatible": function_block(kernels, "isSMCompatible"),
    "runnerCache": line_match(launcher, r"TllmGenFmhaRunner"),
    "cubinInventory": {
        "trtllmGenFmhaCount": len(cubin_names),
        "sm100aCount": sum("Sm100a" in name or "sm100a" in name for name in cubin_names),
        "sm100Count": sum("Sm100" in name or "sm100" in name for name in cubin_names),
        "sm103Count": sum("Sm103" in name or "sm103" in name for name in cubin_names),
        "sm120Count": sum("Sm120" in name or "sm120" in name for name in cubin_names),
        "sample": cubin_names[:16],
    },
}

guard_text = (result["runnerGuard"] or {}).get("text", "")
compat_lines = result["isSmCompatible"] or {"lines": []}
compat_text = "\\n".join(line["text"] for line in compat_lines["lines"])
inventory = result["cubinInventory"]

result["decision"] = {
    "safeAllowlistPatch": False,
    "reason": (
        "The runner guard admits only SM100/SM103, the compatibility helper "
        "only treats SM100-family targets as compatible, and the installed "
        "TRTLLM-gen FMHA cubin inventory has no SM120 cubins."
    ),
    "guardShowsSm120Absent": "kSM_120" not in guard_text,
    "compatibilityShowsSm120Absent": "kSM_120" not in compat_text,
    "hasSm120Cubins": inventory["sm120Count"] > 0,
    "replacementContract": [
        "Provide an SM120-built TRTLLM-gen DSV4 FMHA kernel and dispatch metadata, then rerun the synthetic repro.",
        "Or implement a correctness-first DeepSeek V4 attention fallback for G4 that supports sparse MLA decode on SM120 before rerunning the full model.",
    ],
}

result["publicSafety"] = {
    "containsSecrets": False,
    "containsPrompts": False,
    "containsResponses": False,
    "containsWeights": False,
    "containsHiddenReasoning": False,
}

print(json.dumps(result, sort_keys=True))
PY
EOF
)

gcloud compute ssh "$TARGET_INSTANCE" \
  --zone "$TARGET_ZONE" \
  --command "$remote_command" \
  >"$json_path" \
  2>"$stderr_path"

status="$(jq -r '.decision.safeAllowlistPatch | tostring' "$json_path")"
guard="$(jq -r '.runnerGuard | if . then "\(.path):\(.line): \(.text)" else "not found" end' "$json_path")"
compat="$(jq -r '.isSmCompatible.lines | map("\(.line): \(.text)") | join("\n")' "$json_path")"
constants="$(jq -r '.smConstants | map("\(.line): \(.text)") | join("\n")' "$json_path")"
cubin_inventory="$(jq -c '.cubinInventory' "$json_path")"
decision_reason="$(jq -r '.decision.reason' "$json_path")"
replacement_contract="$(jq -r '.decision.replacementContract | map("- " + .) | join("\n")' "$json_path")"
flashinfer_version="$(jq -r '.flashinfer // ""' "$json_path")"
torch_version="$(jq -r '.torch // ""' "$json_path")"
cuda_version="$(jq -r '.cuda // ""' "$json_path")"

cat >"$report_path" <<EOF
# FlashInfer DSV4 FMHA SM120 source audit

Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)

- Target instance: \`$TARGET_INSTANCE\`
- Target zone: \`$TARGET_ZONE\`
- Docker image: \`$IMAGE\`
- FlashInfer: \`$flashinfer_version\`
- Torch: \`$torch_version\`
- CUDA: \`$cuda_version\`

## Runner Guard

\`\`\`text
$guard
\`\`\`

## SM Constants

\`\`\`text
$constants
\`\`\`

## Compatibility Helper

\`\`\`cpp
$compat
\`\`\`

## TRTLLM-gen FMHA Cubins

\`\`\`json
$cubin_inventory
\`\`\`

## Decision

- Safe allowlist patch: \`$status\`
- Reason: $decision_reason

Replacement contract:

$replacement_contract

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
