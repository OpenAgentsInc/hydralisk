#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-${CLOUDSDK_CORE_ACCOUNT:-}}"
TARGET_INSTANCE="${TARGET_INSTANCE:-hydralisk-glm52-reap-504b-g4-8g-b-20260624214500}"
TARGET_ZONE="${TARGET_ZONE:-us-central1-b}"
MODEL_ID="${MODEL_ID:-0xSero/GLM-5.2-504B}"
MODEL_REVISION="${MODEL_REVISION:-cb6b1e0451b9d560cda864f84187869c9a679712}"
MODEL_DIR="${MODEL_DIR:-/opt/hydralisk/models/glm-5.2-504b}"
HF_HOME_DIR="${HF_HOME_DIR:-/var/lib/hydralisk/huggingface}"
VENV_DIR="${VENV_DIR:-/opt/hydralisk/venvs/hf-staging}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d%H%M%S)}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-/var/log/hydralisk/glm52-reap-504b-staging-$RUN_ID}"
EXPECTED_SHARD_COUNT="${EXPECTED_SHARD_COUNT:-63}"
EXPECTED_INDEX_TOTAL_SIZE_BYTES="${EXPECTED_INDEX_TOTAL_SIZE_BYTES:-318247808128}"
ACTION="${ACTION:-run}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/glm52-reap-504b-staging-$RUN_ID}"

if [[ -z "$TARGET_INSTANCE" || -z "$TARGET_ZONE" ]]; then
  echo "error: TARGET_INSTANCE and TARGET_ZONE are required" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

run_gcloud() {
  if [[ -n "$GCLOUD_ACCOUNT" ]]; then
    CLOUDSDK_CORE_ACCOUNT="$GCLOUD_ACCOUNT" gcloud "$@"
  else
    gcloud "$@"
  fi
}

remote_script="$(mktemp "${TMPDIR:-/tmp}/hydralisk-glm52-stage.XXXXXX.sh")"
trap 'rm -f "$remote_script"' EXIT

cat > "$remote_script" <<'REMOTE'
#!/usr/bin/env bash
set -Eeuo pipefail

sudo install -d -m 0755 "$REMOTE_LOG_DIR"
sudo chown "$(whoami):$(id -gn)" "$REMOTE_LOG_DIR"
LOG_FILE="$REMOTE_LOG_DIR/stage.log"
EXIT_FILE="$REMOTE_LOG_DIR/stage.exit"
STATUS_FILE="$REMOTE_LOG_DIR/stage.status"
MANIFEST_FILE="$REMOTE_LOG_DIR/staging-public-manifest.json"
SUMMARY_FILE="$REMOTE_LOG_DIR/staging-public-summary.md"

exec >>"$LOG_FILE" 2>&1

finish() {
  local rc="$?"
  printf 'exitCode=%s\nfinishedAt=%s\n' "$rc" "$(date -u +%FT%TZ)" > "$EXIT_FILE"
  exit "$rc"
}
trap finish EXIT

printf 'status=starting\nstartedAt=%s\nmodel=%s\nrevision=%s\nmodelDir=%s\n' \
  "$(date -u +%FT%TZ)" "$MODEL_ID" "$MODEL_REVISION" "$MODEL_DIR" > "$STATUS_FILE"

echo "== host =="
hostname
df -h / "$MODEL_DIR" 2>/dev/null || df -h /

echo "== install staging tooling =="
if ! python3 -m pip --version >/dev/null 2>&1; then
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-venv
fi

sudo install -d -m 0755 /opt/hydralisk /opt/hydralisk/models /opt/hydralisk/venvs /var/lib/hydralisk
sudo install -d -m 0775 "$MODEL_DIR" "$HF_HOME_DIR"
sudo chown -R "$(whoami):$(id -gn)" /opt/hydralisk/models /opt/hydralisk/venvs "$HF_HOME_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install --upgrade \
  'huggingface_hub>=0.34,<1' \
  'hf_transfer>=0.1.9,<1' \
  'hf_xet>=1.1,<2'

echo "== tool versions =="
"$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata as md
for package in ["huggingface_hub", "hf_transfer", "hf_xet"]:
    try:
        print(f"{package}={md.version(package)}")
    except Exception as exc:
        print(f"{package}=unavailable:{type(exc).__name__}")
PY

printf 'status=downloading\ndownloadStartedAt=%s\nmodel=%s\nrevision=%s\nmodelDir=%s\n' \
  "$(date -u +%FT%TZ)" "$MODEL_ID" "$MODEL_REVISION" "$MODEL_DIR" > "$STATUS_FILE"

echo "== hf download =="
export HF_HOME="$HF_HOME_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
"$VENV_DIR/bin/hf" download "$MODEL_ID" \
  --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_DIR"

printf 'status=manifesting\nmanifestStartedAt=%s\nmodel=%s\nrevision=%s\nmodelDir=%s\n' \
  "$(date -u +%FT%TZ)" "$MODEL_ID" "$MODEL_REVISION" "$MODEL_DIR" > "$STATUS_FILE"

"$VENV_DIR/bin/python" - "$MODEL_DIR" "$MANIFEST_FILE" "$SUMMARY_FILE" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

model_dir = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
expected_shard_count = int(os.environ["EXPECTED_SHARD_COUNT"])
expected_index_total = int(os.environ["EXPECTED_INDEX_TOTAL_SIZE_BYTES"])

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

index_path = model_dir / "model.safetensors.index.json"
index = json.loads(index_path.read_text()) if index_path.exists() else {}
weight_map = index.get("weight_map") or {}
metadata = index.get("metadata") or {}
expected_shards = sorted({name for name in weight_map.values() if name.endswith(".safetensors")})
local_shards = sorted(path.name for path in model_dir.glob("*.safetensors"))
missing_shards = [name for name in expected_shards if not (model_dir / name).exists()]
unexpected_shards = [name for name in local_shards if name not in set(expected_shards)]
shard_sizes = {name: (model_dir / name).stat().st_size for name in local_shards}

all_files = [path for path in model_dir.rglob("*") if path.is_file()]
all_file_bytes = sum(path.stat().st_size for path in all_files)
local_shard_bytes = sum(shard_sizes.values())

small_hashes = {}
for name in [
    "README.md",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
]:
    path = model_dir / name
    if path.exists() and path.stat().st_size <= 100 * 1024 * 1024:
        small_hashes[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

config = {}
config_path = model_dir / "config.json"
if config_path.exists():
    config = json.loads(config_path.read_text())

payload = {
    "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "model": os.environ["MODEL_ID"],
    "revision": os.environ["MODEL_REVISION"],
    "modelDir": str(model_dir),
    "expected": {
        "shardCount": expected_shard_count,
        "indexMetadataTotalSizeBytes": expected_index_total,
    },
    "observed": {
        "indexExists": index_path.exists(),
        "indexMetadataTotalSizeBytes": metadata.get("total_size"),
        "weightMapEntries": len(weight_map),
        "expectedShardNamesFromIndex": len(expected_shards),
        "localSafetensorShardCount": len(local_shards),
        "localSafetensorShardBytes": local_shard_bytes,
        "allFileCount": len(all_files),
        "allFileBytes": all_file_bytes,
        "missingShardCount": len(missing_shards),
        "unexpectedShardCount": len(unexpected_shards),
        "missingShards": missing_shards,
        "unexpectedShards": unexpected_shards,
        "metadataHashes": small_hashes,
    },
    "config": {
        "architectures": config.get("architectures"),
        "modelType": config.get("model_type"),
        "maxPositionEmbeddings": config.get("max_position_embeddings"),
        "numHiddenLayers": config.get("num_hidden_layers"),
        "nRoutedExperts": config.get("n_routed_experts"),
        "numExpertsPerTok": config.get("num_experts_per_tok"),
        "quantizationConfig": config.get("quantization_config"),
    },
    "complete": bool(
        index_path.exists()
        and len(local_shards) == expected_shard_count
        and len(expected_shards) == expected_shard_count
        and not missing_shards
        and metadata.get("total_size") == expected_index_total
    ),
    "publicSafety": {
        "containsSecrets": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
        "containsHiddenReasoning": False,
        "containsRawTransferLogs": False,
    },
}

manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

summary = [
    "# GLM-5.2 504B REAP staging summary",
    "",
    f"- generatedAt: `{payload['generatedAt']}`",
    f"- model: `{payload['model']}`",
    f"- revision: `{payload['revision']}`",
    f"- modelDir: `{payload['modelDir']}`",
    f"- complete: `{str(payload['complete']).lower()}`",
    f"- localSafetensorShardCount: `{len(local_shards)}`",
    f"- localSafetensorShardBytes: `{local_shard_bytes}`",
    f"- indexMetadataTotalSizeBytes: `{metadata.get('total_size')}`",
    f"- weightMapEntries: `{len(weight_map)}`",
    f"- allFileCount: `{len(all_files)}`",
    f"- allFileBytes: `{all_file_bytes}`",
    f"- missingShardCount: `{len(missing_shards)}`",
    f"- unexpectedShardCount: `{len(unexpected_shards)}`",
    "",
    "Public-safety boundary: this summary contains reduced metadata only; it",
    "does not contain secrets, prompts, responses, hidden reasoning, weights,",
    "or raw transfer logs.",
    "",
]
summary_path.write_text("\n".join(summary))
PY

printf 'status=complete\ncompletedAt=%s\nmodel=%s\nrevision=%s\nmodelDir=%s\n' \
  "$(date -u +%FT%TZ)" "$MODEL_ID" "$MODEL_REVISION" "$MODEL_DIR" > "$STATUS_FILE"

df -h / "$MODEL_DIR" 2>/dev/null || df -h /
REMOTE

copy_remote_artifacts() {
  run_gcloud compute scp \
    "$TARGET_INSTANCE:$REMOTE_LOG_DIR/stage.status" \
    "$OUTPUT_DIR/stage.status" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet >/dev/null 2>&1 || true
  run_gcloud compute scp \
    "$TARGET_INSTANCE:$REMOTE_LOG_DIR/stage.exit" \
    "$OUTPUT_DIR/stage.exit" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet >/dev/null 2>&1 || true
  run_gcloud compute scp \
    "$TARGET_INSTANCE:$REMOTE_LOG_DIR/staging-public-manifest.json" \
    "$OUTPUT_DIR/staging-public-manifest.json" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet >/dev/null 2>&1 || true
  run_gcloud compute scp \
    "$TARGET_INSTANCE:$REMOTE_LOG_DIR/staging-public-summary.md" \
    "$OUTPUT_DIR/staging-public-summary.md" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet >/dev/null 2>&1 || true
}

case "$ACTION" in
  start)
    run_gcloud compute scp "$remote_script" \
      "$TARGET_INSTANCE:/tmp/hydralisk-glm52-stage-$RUN_ID.sh" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --quiet
    run_gcloud compute ssh "$TARGET_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --quiet \
      --command="MODEL_ID='$MODEL_ID' MODEL_REVISION='$MODEL_REVISION' MODEL_DIR='$MODEL_DIR' HF_HOME_DIR='$HF_HOME_DIR' VENV_DIR='$VENV_DIR' REMOTE_LOG_DIR='$REMOTE_LOG_DIR' EXPECTED_SHARD_COUNT='$EXPECTED_SHARD_COUNT' EXPECTED_INDEX_TOTAL_SIZE_BYTES='$EXPECTED_INDEX_TOTAL_SIZE_BYTES' nohup bash /tmp/hydralisk-glm52-stage-$RUN_ID.sh >/tmp/hydralisk-glm52-stage-$RUN_ID.nohup 2>&1 &"
    echo "started remote staging"
    echo "REMOTE_LOG_DIR=$REMOTE_LOG_DIR"
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    ;;
  status)
    copy_remote_artifacts
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    if [[ -f "$OUTPUT_DIR/stage.status" ]]; then
      cat "$OUTPUT_DIR/stage.status"
    else
      echo "status=missing"
    fi
    if [[ -f "$OUTPUT_DIR/stage.exit" ]]; then
      cat "$OUTPUT_DIR/stage.exit"
    fi
    if [[ -f "$OUTPUT_DIR/staging-public-summary.md" ]]; then
      cat "$OUTPUT_DIR/staging-public-summary.md"
    fi
    ;;
  run)
    run_gcloud compute scp "$remote_script" \
      "$TARGET_INSTANCE:/tmp/hydralisk-glm52-stage-$RUN_ID.sh" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --quiet
    run_gcloud compute ssh "$TARGET_INSTANCE" \
      --project "$PROJECT_ID" \
      --zone "$TARGET_ZONE" \
      --quiet \
      --command="MODEL_ID='$MODEL_ID' MODEL_REVISION='$MODEL_REVISION' MODEL_DIR='$MODEL_DIR' HF_HOME_DIR='$HF_HOME_DIR' VENV_DIR='$VENV_DIR' REMOTE_LOG_DIR='$REMOTE_LOG_DIR' EXPECTED_SHARD_COUNT='$EXPECTED_SHARD_COUNT' EXPECTED_INDEX_TOTAL_SIZE_BYTES='$EXPECTED_INDEX_TOTAL_SIZE_BYTES' bash /tmp/hydralisk-glm52-stage-$RUN_ID.sh"
    copy_remote_artifacts
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    ;;
  *)
    echo "error: ACTION must be run, start, or status" >&2
    exit 2
    ;;
esac
