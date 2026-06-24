#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"
ISSUE_NUMBER="${ISSUE_NUMBER:-31}"
IMAGE="${IMAGE:-hydralisk-deepseek-v4-oproj-fallback-g4-vllm:20260624095206}"
FLASHINFER_INSTALL_MODE="${FLASHINFER_INSTALL_MODE:-none}"
FLASHINFER_PIP_PACKAGE="${FLASHINFER_PIP_PACKAGE:-flashinfer-python}"
FLASHINFER_PIP_PACKAGES="${FLASHINFER_PIP_PACKAGES:-$FLASHINFER_PIP_PACKAGE}"
FLASHINFER_PIP_INDEX_URL="${FLASHINFER_PIP_INDEX_URL:-}"
FLASHINFER_PIP_EXTRA_INDEX_URL="${FLASHINFER_PIP_EXTRA_INDEX_URL:-}"
FLASHINFER_PIP_PRE="${FLASHINFER_PIP_PRE:-1}"
FLASHINFER_PIP_NO_DEPS="${FLASHINFER_PIP_NO_DEPS:-1}"
FLASHINFER_INSTALL_TIMEOUT_SECONDS="${FLASHINFER_INSTALL_TIMEOUT_SECONDS:-600}"
SEQ_LEN="${SEQ_LEN:-512}"
HIDDEN_SIZE="${HIDDEN_SIZE:-4096}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"
NUM_EXPERTS="${NUM_EXPERTS:-256}"
LOCAL_NUM_EXPERTS="${LOCAL_NUM_EXPERTS:-32}"
TOP_K="${TOP_K:-6}"
SWIGLU_LIMIT="${SWIGLU_LIMIT:-10.0}"
RUN_LOCAL_SHARD_REMAP_CASE="${RUN_LOCAL_SHARD_REMAP_CASE:-1}"
RUN_NO_EP_CASE="${RUN_NO_EP_CASE:-1}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/flashinfer-b12x-moe-$TS}"

mkdir -p "$OUTPUT_DIR"

if [[ -z "$TARGET_INSTANCE" || -z "$TARGET_ZONE" ]]; then
  echo "error: TARGET_INSTANCE and TARGET_ZONE are required" >&2
  exit 2
fi

if [[ "$TARGET_INSTANCE" != hydralisk-deepseek-v4-* ]]; then
  echo "error: TARGET_INSTANCE must be a fresh hydralisk-deepseek-v4-* probe host" >&2
  exit 2
fi

render_markdown() {
  local md="$OUTPUT_DIR/flashinfer-b12x-moe-probe.md"
  {
    echo "# FlashInfer B12x SM12x MoE G4 probe"
    echo
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "- Issue: https://github.com/OpenAgentsInc/hydralisk/issues/$ISSUE_NUMBER"
    echo "- Target instance: \`$TARGET_INSTANCE\`"
    echo "- Target zone: \`$TARGET_ZONE\`"
    echo "- Image: \`$IMAGE\`"
    echo "- FlashInfer install mode: \`$FLASHINFER_INSTALL_MODE\`"
    echo "- FlashInfer pip package: \`$FLASHINFER_PIP_PACKAGE\`"
    echo "- FlashInfer pip packages: \`$FLASHINFER_PIP_PACKAGES\`"
    echo "- FlashInfer pip index URL: \`${FLASHINFER_PIP_INDEX_URL:-default}\`"
    echo "- FlashInfer pip extra index URL: \`${FLASHINFER_PIP_EXTRA_INDEX_URL:-none}\`"
    echo "- FlashInfer pip pre-release: \`$FLASHINFER_PIP_PRE\`"
    echo "- FlashInfer pip no-deps: \`$FLASHINFER_PIP_NO_DEPS\`"
    echo "- Sequence length: \`$SEQ_LEN\`"
    echo "- Hidden size: \`$HIDDEN_SIZE\`"
    echo "- Intermediate size: \`$INTERMEDIATE_SIZE\`"
    echo "- Experts: \`$NUM_EXPERTS\`"
    echo "- Local experts: \`$LOCAL_NUM_EXPERTS\`"
    echo "- Top-k: \`$TOP_K\`"
    echo "- SwiGLU limit: \`$SWIGLU_LIMIT\`"
    echo "- Run local-shard remap case: \`$RUN_LOCAL_SHARD_REMAP_CASE\`"
    echo "- Run no-EP case: \`$RUN_NO_EP_CASE\`"
    echo
    if [[ "$DRY_RUN" = "1" ]]; then
      echo "DRY_RUN=1"
      echo
    else
      echo "## Hardware"
      echo
      echo '```text'
      sed -n '1,160p' "$OUTPUT_DIR/flashinfer-b12x-hardware.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "## Results"
      echo
      echo '```json'
      sed -n '1,180p' "$OUTPUT_DIR/flashinfer-b12x-result.jsonl" 2>/dev/null || true
      echo '```'
      echo
      echo "Stderr:"
      echo
      echo '```text'
      sed -n '1,220p' "$OUTPUT_DIR/flashinfer-b12x-stderr.txt" 2>/dev/null || true
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
  render_markdown
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  exit 0
fi

remote_script="$(
  printf 'IMAGE=%q\n' "$IMAGE"
  printf 'FLASHINFER_INSTALL_MODE=%q\n' "$FLASHINFER_INSTALL_MODE"
  printf 'FLASHINFER_PIP_PACKAGE=%q\n' "$FLASHINFER_PIP_PACKAGE"
  printf 'FLASHINFER_PIP_PACKAGES=%q\n' "$FLASHINFER_PIP_PACKAGES"
  printf 'FLASHINFER_PIP_INDEX_URL=%q\n' "$FLASHINFER_PIP_INDEX_URL"
  printf 'FLASHINFER_PIP_EXTRA_INDEX_URL=%q\n' "$FLASHINFER_PIP_EXTRA_INDEX_URL"
  printf 'FLASHINFER_PIP_PRE=%q\n' "$FLASHINFER_PIP_PRE"
  printf 'FLASHINFER_PIP_NO_DEPS=%q\n' "$FLASHINFER_PIP_NO_DEPS"
  printf 'FLASHINFER_INSTALL_TIMEOUT_SECONDS=%q\n' "$FLASHINFER_INSTALL_TIMEOUT_SECONDS"
  printf 'SEQ_LEN=%q\n' "$SEQ_LEN"
  printf 'HIDDEN_SIZE=%q\n' "$HIDDEN_SIZE"
  printf 'INTERMEDIATE_SIZE=%q\n' "$INTERMEDIATE_SIZE"
  printf 'NUM_EXPERTS=%q\n' "$NUM_EXPERTS"
  printf 'LOCAL_NUM_EXPERTS=%q\n' "$LOCAL_NUM_EXPERTS"
  printf 'TOP_K=%q\n' "$TOP_K"
  printf 'SWIGLU_LIMIT=%q\n' "$SWIGLU_LIMIT"
  printf 'RUN_LOCAL_SHARD_REMAP_CASE=%q\n' "$RUN_LOCAL_SHARD_REMAP_CASE"
  printf 'RUN_NO_EP_CASE=%q\n' "$RUN_NO_EP_CASE"
  printf 'REMOTE_LOG_DIR=%q\n' "/var/log/hydralisk/flashinfer-b12x-moe-$TS"
  cat <<'REMOTE'
set -Eeuo pipefail
sudo install -d -m 0777 "$REMOTE_LOG_DIR"

{
  printf "HOSTNAME\t%s\n" "$(hostname)"
  printf "KERNEL\t%s\n" "$(uname -r)"
  printf "GPU_QUERY_BEGIN\n"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,pci.bus_id --format=csv,noheader,nounits || true
  printf "GPU_QUERY_END\n"
  printf "IMAGE\t%s\n" "$IMAGE"
} > "$REMOTE_LOG_DIR/flashinfer-b12x-hardware.txt"

cat > "$REMOTE_LOG_DIR/repro.py" <<'PY'
import importlib
import inspect
import json
import os
import traceback

import torch


def env_int(name: str) -> int:
    return int(os.environ[name])


def emit(record: dict) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def module_record() -> dict:
    modules = [
        "flashinfer",
        "flashinfer.fused_moe",
        "flashinfer.cute_dsl.utils",
        "flashinfer.gemm",
    ]
    required_attrs = [
        "B12xMoEWrapper",
        "b12x_fused_moe",
        "convert_sf_to_mma_layout",
        "Sm120B12xBlockScaledDenseGemmKernel",
        "Sm121B12xBlockScaledDenseGemmKernel",
        "Sm120BlockScaledDenseGemmKernel",
    ]
    record = {
        "schema": "hydralisk.flashinfer.b12x-moe.availability.v1",
        "case": "availability",
        "publicSafety": {
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
            "containsHiddenReasoning": False,
        },
    }
    for name in modules:
        try:
            mod = importlib.import_module(name)
            record[name] = {
                "ok": True,
                "file": getattr(mod, "__file__", None),
                "attrs": [attr for attr in required_attrs if hasattr(mod, attr)],
            }
            if name == "flashinfer.fused_moe":
                record[name]["b12xAttrs"] = sorted(
                    attr for attr in dir(mod) if "b12x" in attr.lower()
                )
        except Exception as exc:
            record[name] = {
                "ok": False,
                "type": type(exc).__name__,
                "message": str(exc),
            }
    try:
        import flashinfer
        import flashinfer.fused_moe as fm

        fn = getattr(fm, "b12x_fused_moe", None)
        signature = inspect.signature(fn) if fn else None
        source = ""
        if fn:
            try:
                source = inspect.getsource(fn)
            except Exception:
                source = ""
        record["runtime"] = {
            "flashinfer": getattr(flashinfer, "__version__", "unknown"),
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
        record["b12xSignature"] = str(signature) if signature else None
        record["b12xInterface"] = {
            "supportsSwigluLimitKwarg": (
                bool(signature) and "swiglu_limit" in signature.parameters
            ),
            "supportsNumLocalExpertsKwarg": (
                bool(signature) and "num_local_experts" in signature.parameters
            ),
            "supportsActivationKwarg": (
                bool(signature) and "activation" in signature.parameters
            ),
        }
        record["b12xSourceSignals"] = {
            "mentionsSwigluLimit": "swiglu_limit" in source,
            "mentionsSwigluLimitValue": "swiglu_limit_value" in source,
            "mentionsExpertParallelRejection": (
                "does not yet support Expert Parallelism" in source
            ),
        }
        wrapper = getattr(fm, "B12xMoEWrapper", None)
        if wrapper is not None:
            wrapper_signature = inspect.signature(wrapper)
            record["b12xWrapper"] = {
                "name": "B12xMoEWrapper",
                "signature": str(wrapper_signature),
                "supportsNumLocalExpertsKwarg": (
                    "num_local_experts" in wrapper_signature.parameters
                ),
                "supportsLocalExpertOffsetKwarg": (
                    "local_expert_offset" in wrapper_signature.parameters
                ),
                "supportsSwigluLimitKwarg": (
                    "swiglu_limit" in wrapper_signature.parameters
                ),
                "supportsActivationKwarg": (
                    "activation" in wrapper_signature.parameters
                ),
            }
        else:
            record["b12xWrapper"] = {
                "name": "B12xMoEWrapper",
                "available": False,
            }
    except Exception as exc:
        record["runtimeError"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    return record


def run_b12x_case(
    case: str,
    *,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    local_num_experts: int,
    top_k: int,
    global_num_experts=None,
    routing_domain="global",
    swiglu_limit=None,
) -> dict:
    import flashinfer
    import flashinfer.fused_moe as fm
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    record = {
        "schema": "hydralisk.flashinfer.b12x-moe.synthetic.v1",
        "case": case,
        "flashinfer": getattr(flashinfer, "__version__", "unknown"),
        "torch": torch.__version__,
        "torchCuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "seqLen": seq_len,
        "hiddenSize": hidden_size,
        "intermediateSize": intermediate_size,
        "numExperts": num_experts,
        "kernelNumExperts": num_experts,
        "globalNumExperts": (
            global_num_experts if global_num_experts is not None else num_experts
        ),
        "localNumExperts": local_num_experts,
        "topK": top_k,
        "routingDomain": routing_domain,
        "swigluLimitKwarg": swiglu_limit,
        "loadsModelWeights": False,
        "publicSafety": {
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
            "containsHiddenReasoning": False,
        },
    }
    try:
        x = torch.zeros((seq_len, hidden_size), device="cuda", dtype=torch.bfloat16)
        w1 = torch.zeros(
            (local_num_experts, 2 * intermediate_size, hidden_size // 2),
            device="cuda",
            dtype=torch.uint8,
        )
        w2 = torch.zeros(
            (local_num_experts, hidden_size, intermediate_size // 2),
            device="cuda",
            dtype=torch.uint8,
        )
        w1_scale = torch.ones(
            (local_num_experts, 2 * intermediate_size, hidden_size // 16),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        w2_scale = torch.ones(
            (local_num_experts, hidden_size, intermediate_size // 16),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        w1_scale_mma = convert_sf_to_mma_layout(
            w1_scale.reshape(
                local_num_experts * 2 * intermediate_size, hidden_size // 16
            ),
            m=2 * intermediate_size,
            k=hidden_size,
            num_groups=local_num_experts,
        )
        w2_scale_mma = convert_sf_to_mma_layout(
            w2_scale.reshape(
                local_num_experts * hidden_size, intermediate_size // 16
            ),
            m=hidden_size,
            k=intermediate_size,
            num_groups=local_num_experts,
        )
        token_selected_experts = (
            torch.arange(seq_len * top_k, device="cuda", dtype=torch.int32)
            .reshape(seq_len, top_k)
            % local_num_experts
        )
        token_final_scales = torch.full(
            (seq_len, top_k), 1.0 / top_k, device="cuda", dtype=torch.float32
        )
        w1_alpha = torch.ones(
            (local_num_experts,), device="cuda", dtype=torch.float32
        )
        w2_alpha = torch.ones(
            (local_num_experts,), device="cuda", dtype=torch.float32
        )
        fc2_input_scale = torch.ones(
            (local_num_experts,), device="cuda", dtype=torch.float32
        )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call_kwargs = {
            "w1_alpha": w1_alpha,
            "w2_alpha": w2_alpha,
            "fc2_input_scale": fc2_input_scale,
            "num_local_experts": local_num_experts,
            "quant_mode": "nvfp4",
        }
        if swiglu_limit is not None:
            call_kwargs["swiglu_limit"] = swiglu_limit
        out = fm.b12x_fused_moe(
            x,
            w1,
            w1_scale_mma,
            w2,
            w2_scale_mma,
            token_selected_experts,
            token_final_scales,
            num_experts,
            top_k,
            **call_kwargs,
        )
        end.record()
        torch.cuda.synchronize()
        record.update(
            {
                "ok": True,
                "outShape": list(out.shape),
                "outDtype": str(out.dtype),
                "outSum": float(out.float().sum().item()),
                "elapsedMs": start.elapsed_time(end),
                "maxMemoryAllocatedBytes": int(torch.cuda.max_memory_allocated()),
            }
        )
    except Exception as exc:
        record.update(
            {
                "ok": False,
                "type": type(exc).__name__,
                "message": str(exc)[:1600],
            }
        )
        traceback.print_exc(limit=24)
    return record


torch.cuda.set_device(0)
emit(module_record())
emit(
    run_b12x_case(
        "small_no_ep",
        seq_len=8,
        hidden_size=256,
        intermediate_size=256,
        num_experts=8,
        local_num_experts=8,
        top_k=2,
    )
)
emit(
    run_b12x_case(
        "b12x_swiglu_limit_kwarg_probe",
        seq_len=8,
        hidden_size=256,
        intermediate_size=256,
        num_experts=8,
        local_num_experts=8,
        top_k=2,
        swiglu_limit=float(os.environ["SWIGLU_LIMIT"]),
    )
)
seq_len = env_int("SEQ_LEN")
hidden_size = env_int("HIDDEN_SIZE")
intermediate_size = env_int("INTERMEDIATE_SIZE")
num_experts = env_int("NUM_EXPERTS")
local_num_experts = env_int("LOCAL_NUM_EXPERTS")
top_k = env_int("TOP_K")
emit(
    run_b12x_case(
        "deepseek_shape_ep",
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        local_num_experts=local_num_experts,
        top_k=top_k,
    )
)
if os.environ.get("RUN_LOCAL_SHARD_REMAP_CASE", "1") == "1":
    emit(
        run_b12x_case(
            "deepseek_shape_local_shard_remap",
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=local_num_experts,
            local_num_experts=local_num_experts,
            top_k=top_k,
            global_num_experts=num_experts,
            routing_domain="local_shard_remapped",
        )
    )
if os.environ.get("RUN_NO_EP_CASE", "1") == "1":
    emit(
        run_b12x_case(
            "deepseek_shape_no_ep",
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            local_num_experts=num_experts,
            top_k=top_k,
        )
    )
PY

cat > "$REMOTE_LOG_DIR/run-probe.py" <<'PY'
import json
import os
import subprocess
import sys
import time


def emit(record: dict) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in {"0", "false", "False", "no", "NO"}


mode = os.environ.get("FLASHINFER_INSTALL_MODE", "none")
package = os.environ.get("FLASHINFER_PIP_PACKAGE", "flashinfer-python")
packages = [
    item
    for item in os.environ.get("FLASHINFER_PIP_PACKAGES", package).split()
    if item
]
index_url = os.environ.get("FLASHINFER_PIP_INDEX_URL", "")
if mode == "nightly" and not index_url:
    index_url = "https://flashinfer.ai/whl/nightly/"
extra_index_url = os.environ.get("FLASHINFER_PIP_EXTRA_INDEX_URL", "")

install_record = {
    "schema": "hydralisk.flashinfer.b12x-moe.install.v1",
    "case": "flashinfer_install",
    "mode": mode,
    "package": package,
    "packages": packages,
    "indexUrl": index_url or None,
    "extraIndexUrl": extra_index_url or None,
    "preRelease": env_bool("FLASHINFER_PIP_PRE", True),
    "noDeps": env_bool("FLASHINFER_PIP_NO_DEPS", True),
    "timeoutSeconds": int(os.environ.get("FLASHINFER_INSTALL_TIMEOUT_SECONDS", "600")),
    "publicSafety": {
        "containsSecrets": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
        "containsHiddenReasoning": False,
    },
}

if mode != "none":
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--disable-pip-version-check",
        "--no-input",
    ]
    if install_record["preRelease"]:
        cmd.append("--pre")
    if index_url:
        cmd.extend(["--index-url", index_url])
    if extra_index_url:
        cmd.extend(["--extra-index-url", extra_index_url])
    if install_record["noDeps"]:
        cmd.append("--no-deps")
    cmd.extend(packages)
    install_record["argv"] = cmd[:]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=install_record["timeoutSeconds"],
        )
        install_record.update(
            {
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "elapsedSeconds": round(time.monotonic() - start, 3),
                "stdoutLineCount": len(completed.stdout.splitlines()),
                "stderrLineCount": len(completed.stderr.splitlines()),
            }
        )
    except subprocess.TimeoutExpired as exc:
        install_record.update(
            {
                "ok": False,
                "timedOut": True,
                "elapsedSeconds": round(time.monotonic() - start, 3),
                "stdoutLineCount": len((exc.stdout or "").splitlines()),
                "stderrLineCount": len((exc.stderr or "").splitlines()),
            }
        )
else:
    install_record["ok"] = True

emit(install_record)
subprocess.run([sys.executable, "/tmp/repro.py"], check=False)
PY

sudo docker run --rm --gpus all --ipc=host --network host \
  -e "FLASHINFER_INSTALL_MODE=$FLASHINFER_INSTALL_MODE" \
  -e "FLASHINFER_PIP_PACKAGE=$FLASHINFER_PIP_PACKAGE" \
  -e "FLASHINFER_PIP_PACKAGES=$FLASHINFER_PIP_PACKAGES" \
  -e "FLASHINFER_PIP_INDEX_URL=$FLASHINFER_PIP_INDEX_URL" \
  -e "FLASHINFER_PIP_EXTRA_INDEX_URL=$FLASHINFER_PIP_EXTRA_INDEX_URL" \
  -e "FLASHINFER_PIP_PRE=$FLASHINFER_PIP_PRE" \
  -e "FLASHINFER_PIP_NO_DEPS=$FLASHINFER_PIP_NO_DEPS" \
  -e "FLASHINFER_INSTALL_TIMEOUT_SECONDS=$FLASHINFER_INSTALL_TIMEOUT_SECONDS" \
  -e "SEQ_LEN=$SEQ_LEN" \
  -e "HIDDEN_SIZE=$HIDDEN_SIZE" \
  -e "INTERMEDIATE_SIZE=$INTERMEDIATE_SIZE" \
  -e "NUM_EXPERTS=$NUM_EXPERTS" \
  -e "LOCAL_NUM_EXPERTS=$LOCAL_NUM_EXPERTS" \
  -e "TOP_K=$TOP_K" \
  -e "SWIGLU_LIMIT=$SWIGLU_LIMIT" \
  -e "RUN_LOCAL_SHARD_REMAP_CASE=$RUN_LOCAL_SHARD_REMAP_CASE" \
  -e "RUN_NO_EP_CASE=$RUN_NO_EP_CASE" \
  -v "$REMOTE_LOG_DIR/repro.py:/tmp/repro.py:ro" \
  -v "$REMOTE_LOG_DIR/run-probe.py:/tmp/run-probe.py:ro" \
  --entrypoint python3 "$IMAGE" /tmp/run-probe.py \
  > "$REMOTE_LOG_DIR/flashinfer-b12x-result.jsonl" \
  2> "$REMOTE_LOG_DIR/flashinfer-b12x-stderr.txt" || true
REMOTE
)"

gcloud compute ssh "$TARGET_INSTANCE" \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --quiet \
  --command "$remote_script" \
  > "$OUTPUT_DIR/remote.stdout" \
  2> "$OUTPUT_DIR/remote.stderr" || true

remote_dir="/var/log/hydralisk/flashinfer-b12x-moe-$TS"
for file in \
  flashinfer-b12x-hardware.txt \
  flashinfer-b12x-result.jsonl \
  flashinfer-b12x-stderr.txt; do
  gcloud compute scp \
    "$TARGET_INSTANCE:$remote_dir/$file" \
    "$OUTPUT_DIR/$file" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet > "$OUTPUT_DIR/scp-$file.log" 2>&1 || true
done

render_markdown
echo "OUTPUT_DIR=$OUTPUT_DIR"
