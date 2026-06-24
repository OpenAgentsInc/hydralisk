#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"
PYTHON_BIN="${PYTHON_BIN:-/opt/hydralisk-deepseek-v4/.venv/bin/python}"
ACTION="${ACTION:-apply}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-o-proj-rhs-scale-patch-$TS}"

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
  local md="$OUTPUT_DIR/o-proj-rhs-scale-patch.md"
  {
    echo "# DeepSeek-V4 o_proj grouped RHS scale patch"
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
      sed -n '1,120p' "$OUTPUT_DIR/o-proj-rhs-scale-patch.jsonl" 2>/dev/null || true
      echo '```'
      echo
      echo "## Stderr"
      echo
      echo '```text'
      sed -n '1,120p' "$OUTPUT_DIR/o-proj-rhs-scale-patch.stderr" 2>/dev/null || true
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


REQUIRED_RHS_MARKER = "Hydralisk issue #11 DeepSeek o_proj grouped RHS probe"
MARKER = "Hydralisk issue #12 DeepSeek o_proj grouped RHS scale probe"

OLD_BLOCK = '''        rhs_scale = rhs_scale.view(n_groups, o_lora_rank // 128, -1)
        if os.environ.get("HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE", "0") == "1":
'''

NEW_BLOCK = '''        rhs_scale = rhs_scale.view(n_groups, o_lora_rank // 128, -1)
        rhs_scale_mode = os.environ.get(
            "HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE", "raw_e8m0"
        )
        # Hydralisk issue #12 DeepSeek o_proj grouped RHS scale probe.
        if rhs_scale_mode == "fp32":
            if rhs_scale.dtype == torch.float8_e8m0fnu:
                from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                    _upcast_e8m0_to_fp32,
                )

                rhs_scale = _upcast_e8m0_to_fp32(rhs_scale).contiguous()
            else:
                rhs_scale = rhs_scale.to(torch.float32).contiguous()
        elif rhs_scale_mode == "deepgemm_transform":
            from vllm.utils.deep_gemm import transform_sf_into_required_layout

            rhs_scale = transform_sf_into_required_layout(
                sf=rhs_scale,
                mn=o_lora_rank,
                k=rhs_weight.shape[-1],
                recipe=(1, 128, 128),
                num_groups=n_groups,
                is_sfa=False,
            )
        elif rhs_scale_mode == "deepgemm_transform_fp32":
            from vllm.model_executor.layers.quantization.utils.fp8_utils import (
                _upcast_e8m0_to_fp32,
            )
            from vllm.utils.deep_gemm import transform_sf_into_required_layout

            if rhs_scale.dtype == torch.float8_e8m0fnu:
                rhs_scale = _upcast_e8m0_to_fp32(rhs_scale).contiguous()
            else:
                rhs_scale = rhs_scale.to(torch.float32).contiguous()
            rhs_scale = transform_sf_into_required_layout(
                sf=rhs_scale,
                mn=o_lora_rank,
                k=rhs_weight.shape[-1],
                recipe=(1, 128, 128),
                num_groups=n_groups,
                is_sfa=False,
            )
        elif rhs_scale_mode != "raw_e8m0":
            raise RuntimeError(f"unsupported RHS scale mode: {rhs_scale_mode}")
        if os.environ.get("HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE", "0") == "1":
'''

OLD_TRACE_ENTRY = '''                        "rhs_scale": _tensor_meta(rhs_scale),
                        "group_rhs": True,
'''

NEW_TRACE_ENTRY = '''                        "rhs_scale": _tensor_meta(rhs_scale),
                        "rhs_scale_mode": rhs_scale_mode,
                        "group_rhs": True,
'''


def emit(record):
    print(json.dumps(record, sort_keys=True), flush=True)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_target():
    import vllm.models.deepseek_v4.nvidia.ops.o_proj as o_proj

    path = Path(o_proj.__file__)
    backup = path.with_name(path.name + ".hydralisk-o-proj-rhs-scale.bak")
    return path, backup


def main():
    action = os.environ.get("HYDRALISK_PATCH_ACTION", "apply")
    path, backup = load_target()
    text = path.read_text()
    patched = MARKER in text
    rhs_patch_present = REQUIRED_RHS_MARKER in text

    base = {
        "schema": "hydralisk.deepseek-v4.o-proj-rhs-scale-patch.v1",
        "action": action,
        "path": str(path),
        "backup": str(backup),
        "backupExists": backup.exists(),
        "patched": patched,
        "rhsPatchPresent": rhs_patch_present,
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

    if not rhs_patch_present:
        base.update({"ok": False, "error": "o_proj_rhs_patch_required"})
        emit(base)
        return 3

    missing = []
    if OLD_BLOCK not in text:
        missing.append("scale_mode_anchor")
    if OLD_TRACE_ENTRY not in text:
        missing.append("trace_entry")
    if missing:
        base.update({"ok": False, "error": "expected_block_not_found", "missing": missing})
        emit(base)
        return 3

    if not backup.exists():
        shutil.copy2(path, backup)

    new_text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    new_text = new_text.replace(OLD_TRACE_ENTRY, NEW_TRACE_ENTRY, 1)
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
  > "$OUTPUT_DIR/o-proj-rhs-scale-patch.jsonl" \
  2> "$OUTPUT_DIR/o-proj-rhs-scale-patch.stderr"

render_evidence
echo "OUTPUT_DIR=$OUTPUT_DIR"
