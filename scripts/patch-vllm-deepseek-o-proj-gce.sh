#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"
PYTHON_BIN="${PYTHON_BIN:-/opt/hydralisk-deepseek-v4/.venv/bin/python}"
ACTION="${ACTION:-apply}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-o-proj-patch-$TS}"

mkdir -p "$OUTPUT_DIR"

if [[ -z "$TARGET_INSTANCE" || -z "$TARGET_ZONE" ]]; then
  echo "error: TARGET_INSTANCE and TARGET_ZONE are required" >&2
  exit 2
fi

if [[ "$TARGET_INSTANCE" != hydralisk-deepseek-v4-* ]]; then
  echo "error: TARGET_INSTANCE must be a fresh hydralisk-deepseek-v4-* probe host" >&2
  exit 2
fi

case "$ACTION" in
  apply|rollback|status) ;;
  *)
    echo "error: ACTION must be apply, rollback, or status" >&2
    exit 2
    ;;
esac

render_evidence() {
  local md="$OUTPUT_DIR/o-proj-patch.md"
  {
    echo "# DeepSeek-V4 o_proj recipe patch"
    echo
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "- Target instance: \`$TARGET_INSTANCE\`"
    echo "- Target zone: \`$TARGET_ZONE\`"
    echo "- Python: \`$PYTHON_BIN\`"
    echo "- Action: \`$ACTION\`"
    echo
    if [[ "$DRY_RUN" = "1" ]]; then
      echo "DRY_RUN=1"
    else
      echo "## Patch result"
      echo
      echo '```json'
      sed -n '1,120p' "$OUTPUT_DIR/o-proj-patch.jsonl" 2>/dev/null || true
      echo '```'
      echo
      echo "## Stderr"
      echo
      echo '```text'
      sed -n '1,120p' "$OUTPUT_DIR/o-proj-patch.stderr" 2>/dev/null || true
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
  render_evidence
  exit 0
fi

remote_patch="$(cat <<'PY'
import hashlib
import json
import os
import py_compile
import shutil
import sys
from pathlib import Path


MARKER = "Hydralisk issue #10 DeepSeek o_proj recipe probe"

OLD_IMPORTS = '''import torch
import torch.nn as nn
'''

NEW_IMPORTS = '''import json
import os

import torch
import torch.nn as nn
'''

OLD_RECIPE = '''    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales
'''

NEW_RECIPE = '''    forced_recipe = os.environ.get("HYDRALISK_DEEPSEEK_O_PROJ_RECIPE", "auto")
    if forced_recipe == "hopper":
        return (1, 128, 128), False
    if forced_recipe == "blackwell":
        return (1, 1, 128), True
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    # Hydralisk issue #10 DeepSeek o_proj recipe probe.
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales
'''

OLD_AFTER_QUANT = '''    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
'''

NEW_AFTER_QUANT = '''    if os.environ.get("HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE", "0") == "1":
        def _tensor_meta(value):
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }

        print(
            "HYDRALISK_O_PROJ_SHAPE_TRACE\\t"
            + json.dumps(
                {
                    "o": _tensor_meta(o),
                    "o_fp8": _tensor_meta(o_fp8),
                    "o_scale": _tensor_meta(o_scale),
                    "wo_a_weight": _tensor_meta(wo_a.weight),
                    "wo_a_weight_scale_inv": _tensor_meta(wo_a.weight_scale_inv),
                    "einsum_recipe": list(einsum_recipe),
                    "tma_aligned_scales": bool(tma_aligned_scales),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
'''


def emit(record):
    print(json.dumps(record, sort_keys=True), flush=True)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_target():
    import vllm.models.deepseek_v4.nvidia.ops.o_proj as o_proj

    path = Path(o_proj.__file__)
    backup = path.with_name(path.name + ".hydralisk-o-proj-recipe.bak")
    return path, backup


def main():
    action = os.environ.get("HYDRALISK_PATCH_ACTION", "apply")
    path, backup = load_target()
    text = path.read_text()
    patched = MARKER in text

    base = {
        "schema": "hydralisk.deepseek-v4.o-proj-recipe-patch.v1",
        "action": action,
        "path": str(path),
        "backup": str(backup),
        "backupExists": backup.exists(),
        "patched": patched,
        "sha256Before": digest(text),
        "publicSafety": {
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
            "containsHiddenReasoning": False,
        },
    }

    if action == "status":
        emit(base)
        return 0

    if action == "rollback":
        if not backup.exists():
            base.update({"ok": False, "error": "backup_missing"})
            emit(base)
            return 3
        path.write_text(backup.read_text())
        py_compile.compile(str(path), doraise=True)
        base.update({"ok": True, "rolledBack": True, "sha256After": digest(path.read_text())})
        emit(base)
        return 0

    if action != "apply":
        base.update({"ok": False, "error": "unsupported_action"})
        emit(base)
        return 2

    if patched:
        base.update({"ok": True, "changed": False, "reason": "already_patched"})
        emit(base)
        return 0

    missing = [
        name
        for name, block in [
            ("imports", OLD_IMPORTS),
            ("recipe", OLD_RECIPE),
            ("shape_trace_anchor", OLD_AFTER_QUANT),
        ]
        if block not in text
    ]
    if missing:
        base.update({"ok": False, "error": "expected_block_not_found", "missing": missing})
        emit(base)
        return 3

    if not backup.exists():
        shutil.copy2(path, backup)

    new_text = text.replace(OLD_IMPORTS, NEW_IMPORTS, 1)
    new_text = new_text.replace(OLD_RECIPE, NEW_RECIPE, 1)
    new_text = new_text.replace(OLD_AFTER_QUANT, NEW_AFTER_QUANT, 1)
    path.write_text(new_text)
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception:
        path.write_text(text)
        raise

    base.update({"ok": True, "changed": True, "sha256After": digest(path.read_text())})
    emit(base)
    return 0


sys.exit(main())
PY
)"

gcloud compute ssh "$TARGET_INSTANCE" \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --quiet \
  --command "HYDRALISK_PATCH_ACTION=$ACTION $PYTHON_BIN - <<'PY'
$remote_patch
PY" \
  > "$OUTPUT_DIR/o-proj-patch.jsonl" \
  2> "$OUTPUT_DIR/o-proj-patch.stderr"

render_evidence
echo "OUTPUT_DIR=$OUTPUT_DIR"
