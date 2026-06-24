#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"
PYTHON_BIN="${PYTHON_BIN:-/opt/hydralisk-deepseek-v4/.venv/bin/python}"
ACTION="${ACTION:-apply}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-e8m0-upcast-patch-$TS}"

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
  local md="$OUTPUT_DIR/e8m0-upcast-patch.md"
  {
    echo "# DeepSeek-V4-Flash E8M0 Triton upcast patch"
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
      sed -n '1,120p' "$OUTPUT_DIR/e8m0-upcast-patch.jsonl" 2>/dev/null || true
      echo '```'
      echo
      echo "## Stderr"
      echo
      echo '```text'
      sed -n '1,120p' "$OUTPUT_DIR/e8m0-upcast-patch.stderr" 2>/dev/null || true
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


MARKER = "Hydralisk issue #9 E8M0 CUDA Triton upcast"

OLD_BLOCK = '''    # Triton cannot currently bind E8M0 scale tensors directly. On ROCm,
    # DeepSeek-V4 checkpoints store block scales in exponent-only E8M0 format,
    # so decode them to fp32 before launching the kernel.
    if current_platform.is_rocm() or current_platform.is_xpu():
        if As.dtype == torch.float8_e8m0fnu:
            As = _upcast_e8m0_to_fp32(As).contiguous()
        if Bs.dtype == torch.float8_e8m0fnu:
            Bs = _upcast_e8m0_to_fp32(Bs).contiguous()
'''

NEW_BLOCK = '''    # Triton cannot currently bind E8M0 scale tensors directly. DeepSeek-V4
    # checkpoints can store block scales in exponent-only E8M0 format.
    # Hydralisk issue #9 E8M0 CUDA Triton upcast: decode these scales to fp32
    # before launching Triton on CUDA as well as ROCm/XPU. Host-local hotpatch.
    if As.dtype == torch.float8_e8m0fnu:
        As = _upcast_e8m0_to_fp32(As).contiguous()
    if Bs.dtype == torch.float8_e8m0fnu:
        Bs = _upcast_e8m0_to_fp32(Bs).contiguous()
'''


def emit(record):
    print(json.dumps(record, sort_keys=True), flush=True)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_target():
    import vllm.model_executor.layers.quantization.utils.fp8_utils as fp8_utils

    path = Path(fp8_utils.__file__)
    backup = path.with_name(path.name + ".hydralisk-e8m0-upcast.bak")
    return path, backup


def main():
    action = os.environ.get("HYDRALISK_PATCH_ACTION", "apply")
    path, backup = load_target()
    text = path.read_text()
    patched = MARKER in text

    base = {
        "schema": "hydralisk.deepseek-v4.e8m0-triton-upcast-patch.v1",
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
        restored = backup.read_text()
        path.write_text(restored)
        py_compile.compile(str(path), doraise=True)
        base.update(
            {
                "ok": True,
                "rolledBack": True,
                "sha256After": digest(path.read_text()),
            }
        )
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

    if OLD_BLOCK not in text:
        base.update({"ok": False, "error": "expected_block_not_found"})
        emit(base)
        return 3

    if not backup.exists():
        shutil.copy2(path, backup)

    new_text = text.replace(OLD_BLOCK, NEW_BLOCK, 1)
    path.write_text(new_text)
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception:
        path.write_text(text)
        raise

    base.update(
        {
            "ok": True,
            "changed": True,
            "sha256After": digest(path.read_text()),
        }
    )
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
  > "$OUTPUT_DIR/e8m0-upcast-patch.jsonl" \
  2> "$OUTPUT_DIR/e8m0-upcast-patch.stderr"

render_evidence
echo "OUTPUT_DIR=$OUTPUT_DIR"
