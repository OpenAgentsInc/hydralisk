#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-openagentsgemini}"
TARGET_INSTANCE="${TARGET_INSTANCE:-}"
TARGET_ZONE="${TARGET_ZONE:-}"
ISSUE_NUMBER="${ISSUE_NUMBER:-13}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-V4-Flash}"
MODEL_REVISION="${MODEL_REVISION:-}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
ALLOW_NVFP4_SM120="${ALLOW_NVFP4_SM120:-0}"
DOCKER_BUILD_PULL="${DOCKER_BUILD_PULL:-1}"
VLLM_LINEAR_BACKEND="${VLLM_LINEAR_BACKEND:-auto}"
VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-1}"
VLLM_E8M0_TRITON_UPCAST="${VLLM_E8M0_TRITON_UPCAST:-0}"
HYDRALISK_DEEPSEEK_O_PROJ_PATCH="${HYDRALISK_DEEPSEEK_O_PROJ_PATCH:-0}"
HYDRALISK_DEEPSEEK_O_PROJ_RECIPE="${HYDRALISK_DEEPSEEK_O_PROJ_RECIPE:-auto}"
HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE="${HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE:-0}"
HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS="${HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS:-0}"
HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE="${HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE:-raw_e8m0}"
HYDRALISK_DEEPSEEK_O_PROJ_BYPASS="${HYDRALISK_DEEPSEEK_O_PROJ_BYPASS:-off}"
HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK="${HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK:-off}"
HYDRALISK_B12X_CLAMP_PATCH="${HYDRALISK_B12X_CLAMP_PATCH:-0}"
HYDRALISK_B12X_CLAMP_LIMIT="${HYDRALISK_B12X_CLAMP_LIMIT:-10.0}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-0}"
HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-}"
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:latest}"
INSTALL_DEEPGEMM="${INSTALL_DEEPGEMM:-1}"
DERIVED_IMAGE="${DERIVED_IMAGE:-hydralisk-deepseek-v4-provider-vllm}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-2400}"
STACK_BUILD_TIMEOUT_SECONDS="${STACK_BUILD_TIMEOUT_SECONDS:-1800}"
DOCKER_SETUP_TIMEOUT_SECONDS="${DOCKER_SETUP_TIMEOUT_SECONDS:-180}"
RUN_MODEL_SMOKE="${RUN_MODEL_SMOKE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DRY_RUN="${DRY_RUN:-0}"
TS="${TS:-$(date -u +%Y%m%d%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/.hydralisk/deepseek-v4-provider-stack-$TS}"

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
  local md="$OUTPUT_DIR/provider-stack-probe.md"
  {
    echo "# DeepSeek-V4 provider-guided vLLM/DeepGEMM stack probe"
    echo
    echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "- Issue: https://github.com/OpenAgentsInc/hydralisk/issues/$ISSUE_NUMBER"
    echo "- Target instance: \`$TARGET_INSTANCE\`"
    echo "- Target zone: \`$TARGET_ZONE\`"
    echo "- Model: \`$MODEL_ID\`"
    if [[ -n "$MODEL_REVISION" ]]; then
      echo "- Model revision: \`$MODEL_REVISION\`"
    fi
    echo "- MoE backend: \`$MOE_BACKEND\`"
    echo "- Allow NVFP4 SM120 guard patch: \`$ALLOW_NVFP4_SM120\`"
    echo "- Docker build pull: \`$DOCKER_BUILD_PULL\`"
    echo "- vLLM linear backend: \`$VLLM_LINEAR_BACKEND\`"
    echo "- vLLM expert parallel: \`$VLLM_ENABLE_EXPERT_PARALLEL\`"
    echo "- vLLM E8M0 Triton upcast patch: \`$VLLM_E8M0_TRITON_UPCAST\`"
    echo "- DeepSeek o_proj provider patch: \`$HYDRALISK_DEEPSEEK_O_PROJ_PATCH\`"
    echo "- DeepSeek o_proj recipe: \`$HYDRALISK_DEEPSEEK_O_PROJ_RECIPE\`"
    echo "- DeepSeek o_proj shape trace: \`$HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE\`"
    echo "- DeepSeek o_proj grouped RHS: \`$HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS\`"
    echo "- DeepSeek o_proj RHS scale mode: \`$HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE\`"
    echo "- DeepSeek o_proj bypass: \`$HYDRALISK_DEEPSEEK_O_PROJ_BYPASS\`"
    echo "- DeepSeek o_proj fallback: \`$HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK\`"
    echo "- B12x clamp patch: \`$HYDRALISK_B12X_CLAMP_PATCH\`"
    echo "- B12x clamp limit: \`$HYDRALISK_B12X_CLAMP_LIMIT\`"
    echo "- HF Hub disable Xet: \`$HF_HUB_DISABLE_XET\`"
    echo "- HF Xet high performance: \`$HF_XET_HIGH_PERFORMANCE\`"
    echo "- HF Xet concurrent range gets: \`${HF_XET_NUM_CONCURRENT_RANGE_GETS:-default}\`"
    echo "- Base image: \`$BASE_IMAGE\`"
    echo "- Derived image: \`$DERIVED_IMAGE\`"
    echo "- Install DeepGEMM helper: \`$INSTALL_DEEPGEMM\`"
    echo "- Run model smoke: \`$RUN_MODEL_SMOKE\`"
    echo
    if [[ "$DRY_RUN" = "1" ]]; then
      echo "DRY_RUN=1"
      echo
    fi
    echo "## Provider Recipe"
    echo
    echo "This probe follows the provider-note lane for DeepSeek-V4-Flash:"
    echo
    echo "- vLLM \`0.20.0+\` / \`vllm/vllm-openai:latest\`"
    echo "- DeepGEMM installed through vLLM's \`tools/install_deepgemm.sh\` helper"
    echo "- \`--kv-cache-dtype fp8\`"
    echo "- \`--block-size 256\`"
    echo "- \`--enable-expert-parallel\`"
    echo "- tensor parallel size equal to visible GPU count"
    echo "- DeepSeek tokenizer, reasoning, and tool-call parsers"
    echo
    if [[ "$DRY_RUN" != "1" ]]; then
      echo "## Hardware"
      echo
      echo '```text'
      sed -n '1,140p' "$OUTPUT_DIR/provider-stack-hardware.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "## Docker Image Evidence"
      echo
      echo '```text'
      sed -n '1,160p' "$OUTPUT_DIR/provider-stack-image.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "Build log tail:"
      echo
      echo '```text'
      tail -n 120 "$OUTPUT_DIR/provider-stack-build.log" 2>/dev/null || true
      echo '```'
      echo
      echo "## Import Probe"
      echo
      echo '```json'
      sed -n '1,160p' "$OUTPUT_DIR/provider-stack-import.jsonl" 2>/dev/null || true
      echo '```'
      echo
      echo "Import stderr:"
      echo
      echo '```text'
      sed -n '1,80p' "$OUTPUT_DIR/provider-stack-import.stderr" 2>/dev/null || true
      echo '```'
      echo
      echo "## Network Artifact Fetch Probe"
      echo
      echo '```text'
      sed -n '1,160p' "$OUTPUT_DIR/provider-stack-network.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "## Model Smoke"
      echo
      echo '```text'
      sed -n '1,120p' "$OUTPUT_DIR/provider-stack-engine.txt" 2>/dev/null || true
      echo '```'
      echo
      echo '```text'
      sed -n '1,80p' "$OUTPUT_DIR/provider-stack-smoke-summary.txt" 2>/dev/null || true
      echo '```'
      echo
      echo "Public completion receipt, if any:"
      echo
      echo '```json'
      sed -n '1,80p' "$OUTPUT_DIR/provider-stack-completion-public.json" 2>/dev/null || true
      echo '```'
      echo
      echo "vLLM tail, public redacted:"
      echo
      echo '```text'
      sed -n '1,180p' "$OUTPUT_DIR/provider-stack-vllm-tail-public.txt" 2>/dev/null || true
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
  exit 0
fi

remote_script="$(
  printf 'MODEL_ID=%q\n' "$MODEL_ID"
  printf 'MODEL_REVISION=%q\n' "$MODEL_REVISION"
  printf 'MOE_BACKEND=%q\n' "$MOE_BACKEND"
  printf 'ALLOW_NVFP4_SM120=%q\n' "$ALLOW_NVFP4_SM120"
  printf 'DOCKER_BUILD_PULL=%q\n' "$DOCKER_BUILD_PULL"
  printf 'VLLM_LINEAR_BACKEND=%q\n' "$VLLM_LINEAR_BACKEND"
  printf 'VLLM_ENABLE_EXPERT_PARALLEL=%q\n' "$VLLM_ENABLE_EXPERT_PARALLEL"
  printf 'VLLM_E8M0_TRITON_UPCAST=%q\n' "$VLLM_E8M0_TRITON_UPCAST"
  printf 'HYDRALISK_DEEPSEEK_O_PROJ_PATCH=%q\n' "$HYDRALISK_DEEPSEEK_O_PROJ_PATCH"
  printf 'HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=%q\n' "$HYDRALISK_DEEPSEEK_O_PROJ_RECIPE"
  printf 'HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE=%q\n' "$HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE"
  printf 'HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS=%q\n' "$HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS"
  printf 'HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE=%q\n' "$HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE"
  printf 'HYDRALISK_DEEPSEEK_O_PROJ_BYPASS=%q\n' "$HYDRALISK_DEEPSEEK_O_PROJ_BYPASS"
  printf 'HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=%q\n' "$HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK"
  printf 'HYDRALISK_B12X_CLAMP_PATCH=%q\n' "$HYDRALISK_B12X_CLAMP_PATCH"
  printf 'HYDRALISK_B12X_CLAMP_LIMIT=%q\n' "$HYDRALISK_B12X_CLAMP_LIMIT"
  printf 'HF_HUB_DISABLE_XET=%q\n' "$HF_HUB_DISABLE_XET"
  printf 'HF_XET_HIGH_PERFORMANCE=%q\n' "$HF_XET_HIGH_PERFORMANCE"
  printf 'HF_XET_NUM_CONCURRENT_RANGE_GETS=%q\n' "$HF_XET_NUM_CONCURRENT_RANGE_GETS"
  printf 'BASE_IMAGE=%q\n' "$BASE_IMAGE"
  printf 'INSTALL_DEEPGEMM=%q\n' "$INSTALL_DEEPGEMM"
  printf 'DERIVED_IMAGE=%q\n' "$DERIVED_IMAGE:$TS"
  printf 'READY_TIMEOUT_SECONDS=%q\n' "$READY_TIMEOUT_SECONDS"
  printf 'STACK_BUILD_TIMEOUT_SECONDS=%q\n' "$STACK_BUILD_TIMEOUT_SECONDS"
  printf 'DOCKER_SETUP_TIMEOUT_SECONDS=%q\n' "$DOCKER_SETUP_TIMEOUT_SECONDS"
  printf 'RUN_MODEL_SMOKE=%q\n' "$RUN_MODEL_SMOKE"
  printf 'MAX_MODEL_LEN=%q\n' "$MAX_MODEL_LEN"
  printf 'MAX_NUM_SEQS=%q\n' "$MAX_NUM_SEQS"
  printf 'MAX_NUM_BATCHED_TOKENS=%q\n' "$MAX_NUM_BATCHED_TOKENS"
  printf 'GPU_MEMORY_UTILIZATION=%q\n' "$GPU_MEMORY_UTILIZATION"
  printf 'REMOTE_LOG_DIR=%q\n' "/var/log/hydralisk/deepseek-provider-stack-$TS"
  cat <<'REMOTE'
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo install -d -m 0777 "$REMOTE_LOG_DIR"

{
  printf "HOSTNAME\t%s\n" "$(hostname)"
  printf "KERNEL\t%s\n" "$(uname -r)"
  if [ -r /etc/os-release ]; then . /etc/os-release; printf "OS\t%s\n" "${PRETTY_NAME:-unknown}"; fi
  printf "NVIDIA_SMI_HEADER_BEGIN\n"
  nvidia-smi | sed -n "1,10p" || true
  printf "NVIDIA_SMI_HEADER_END\n"
  printf "GPU_QUERY_BEGIN\n"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,pci.bus_id --format=csv,noheader,nounits || true
  printf "GPU_QUERY_END\n"
  printf "TOPOLOGY_BEGIN\n"
  nvidia-smi topo -m || true
  printf "TOPOLOGY_END\n"
  printf "DISK_BEGIN\n"
  df -h /
  printf "DISK_END\n"
} > "$REMOTE_LOG_DIR/provider-stack-hardware.txt"

if ! command -v docker >/dev/null 2>&1; then
  timeout "$DOCKER_SETUP_TIMEOUT_SECONDS"s sudo apt-get update -y >> "$REMOTE_LOG_DIR/provider-stack-docker-setup.log" 2>&1 || true
  timeout "$DOCKER_SETUP_TIMEOUT_SECONDS"s sudo apt-get install -y ca-certificates curl jq docker.io >> "$REMOTE_LOG_DIR/provider-stack-docker-setup.log" 2>&1 || true
fi
sudo systemctl enable --now docker >> "$REMOTE_LOG_DIR/provider-stack-docker-setup.log" 2>&1 || true

build_ctx="$(mktemp -d /tmp/hydralisk-provider-stack.XXXXXX)"
cat > "$build_ctx/patch_o_proj.py" <<'PY'
import os
import py_compile
from pathlib import Path

import vllm.models.deepseek_v4.nvidia.ops.o_proj as o_proj


path = Path(o_proj.__file__)
text = path.read_text()
marker = "Hydralisk issue #20 DeepSeek NVFP4 o_proj provider patch"
if marker in text:
    print(f"{path} already patched")
    raise SystemExit(0)

old_imports = """import torch
import torch.nn as nn
"""
new_imports = """import json
import os

import torch
import torch.nn as nn
"""

old_recipe = """    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales
"""
new_recipe = """    forced_recipe = os.environ.get("HYDRALISK_DEEPSEEK_O_PROJ_RECIPE", "auto")
    if forced_recipe == "hopper":
        return (1, 128, 128), False
    if forced_recipe == "blackwell":
        return (1, 1, 128), True
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    # Hydralisk issue #20 DeepSeek NVFP4 o_proj provider patch.
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales
"""

old_quant = """    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
"""
new_quant = """    if os.environ.get("HYDRALISK_DEEPSEEK_O_PROJ_BYPASS", "off") == "zero":
        if os.environ.get("HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE", "0") == "1":
            print(
                "HYDRALISK_O_PROJ_BYPASS_TRACE\\t"
                + json.dumps(
                    {
                        "bypass": "zero",
                        "o": {
                            "shape": list(o.shape),
                            "dtype": str(o.dtype),
                            "device": str(o.device),
                        },
                        "n_groups": int(n_groups),
                        "o_lora_rank": int(o_lora_rank),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        z = torch.zeros(
            (o.shape[0], n_groups, o_lora_rank),
            device=o.device,
            dtype=torch.bfloat16,
        )
        return wo_b(z.flatten(1))

    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    if os.environ.get("HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE", "0") == "1":
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
"""

old_call = """    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (wo_a.weight, wo_a.weight_scale_inv),
        z,
        recipe=einsum_recipe,
    )
"""
new_call = """    rhs_weight = wo_a.weight
    rhs_scale = wo_a.weight_scale_inv
    if os.environ.get("HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS", "0") == "1":
        rhs_weight = rhs_weight.view(n_groups, o_lora_rank, -1)
        rhs_scale = rhs_scale.view(n_groups, o_lora_rank // 128, -1)
        rhs_scale_mode = os.environ.get(
            "HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE", "raw_e8m0"
        )
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
            def _tensor_meta(value):
                return {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }

            print(
                "HYDRALISK_O_PROJ_RHS_TRACE\\t"
                + json.dumps(
                    {
                        "rhs_weight": _tensor_meta(rhs_weight),
                        "rhs_scale": _tensor_meta(rhs_scale),
                        "rhs_scale_mode": rhs_scale_mode,
                        "group_rhs": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    fallback_mode = os.environ.get("HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK", "off")
    if fallback_mode == "bf16_einsum":
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            _upcast_e8m0_to_fp32,
        )

        def _scale_to_fp32(value):
            if value.dtype == torch.float8_e8m0fnu:
                return _upcast_e8m0_to_fp32(value).contiguous()
            return value.to(torch.float32).contiguous()

        if o_scale.dtype == torch.int32:
            raise RuntimeError(
                "HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=bf16_einsum requires "
                "non-TMA activation scales; set "
                "HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=hopper for this probe"
            )
        fallback_rhs_weight = rhs_weight
        fallback_rhs_scale = rhs_scale
        if fallback_rhs_weight.ndim == 2:
            fallback_rhs_weight = fallback_rhs_weight.view(n_groups, o_lora_rank, -1)
        if fallback_rhs_scale.ndim == 2:
            fallback_rhs_scale = fallback_rhs_scale.view(n_groups, o_lora_rank // 128, -1)
        lhs_scale = _scale_to_fp32(o_scale)
        lhs_scale = lhs_scale.repeat_interleave(128, dim=-1)[..., : o_fp8.shape[-1]]
        rhs_scale_for_dequant = _scale_to_fp32(fallback_rhs_scale)
        rhs_scale_for_dequant = rhs_scale_for_dequant.repeat_interleave(128, dim=1)
        rhs_scale_for_dequant = rhs_scale_for_dequant.repeat_interleave(128, dim=2)
        rhs_scale_for_dequant = rhs_scale_for_dequant[
            :, : fallback_rhs_weight.shape[1], : fallback_rhs_weight.shape[2]
        ]
        lhs = o_fp8.to(torch.float32) * lhs_scale
        rhs = fallback_rhs_weight.to(torch.float32) * rhs_scale_for_dequant
        if os.environ.get("HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE", "0") == "1":
            print(
                "HYDRALISK_O_PROJ_FALLBACK_TRACE\\t"
                + json.dumps(
                    {
                        "fallback": fallback_mode,
                        "lhs": {
                            "shape": list(lhs.shape),
                            "dtype": str(lhs.dtype),
                            "device": str(lhs.device),
                        },
                        "rhs": {
                            "shape": list(rhs.shape),
                            "dtype": str(rhs.dtype),
                            "device": str(rhs.device),
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        z.copy_(torch.einsum("bhr,hdr->bhd", lhs, rhs).to(z.dtype))
        return wo_b(z.flatten(1))
    if fallback_mode != "off":
        raise RuntimeError(f"unsupported o_proj fallback mode: {fallback_mode}")

    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (rhs_weight, rhs_scale),
        z,
        recipe=einsum_recipe,
    )
"""

missing = [
    name
    for name, block in [
        ("imports", old_imports),
        ("recipe", old_recipe),
        ("quant_anchor", old_quant),
        ("call", old_call),
    ]
    if block not in text
]
if missing:
    raise RuntimeError(f"o_proj patch target blocks missing: {missing} in {path}")

text = text.replace(old_imports, new_imports, 1)
text = text.replace(old_recipe, new_recipe, 1)
text = text.replace(old_quant, new_quant, 1)
text = text.replace(old_call, new_call, 1)
path.write_text(text)
py_compile.compile(str(path), doraise=True)
print(f"patched {path} for Hydralisk issue #20 o_proj provider probe")
PY

cat > "$build_ctx/patch_b12x_clamp.py" <<'PY'
import os
import py_compile
from pathlib import Path

import flashinfer
import vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe as b12x_vllm
import vllm.model_executor.layers.fused_moe.oracle.nvfp4 as nvfp4_oracle


LIMIT = os.environ.get("HYDRALISK_B12X_CLAMP_LIMIT", "10.0")
flashinfer_root = Path(flashinfer.__file__).resolve().parent


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if new in text:
        print(f"{label}: already patched")
        return
    if old not in text:
        raise RuntimeError(f"{label}: patch target missing in {path}")
    path.write_text(text.replace(old, new, 1))
    py_compile.compile(str(path), doraise=True)
    print(f"{label}: patched {path}")


api_path = flashinfer_root / "fused_moe/cute_dsl/b12x_moe.py"
dispatch_path = (
    flashinfer_root / "fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py"
)
static_path = (
    flashinfer_root / "fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py"
)
dynamic_path = (
    flashinfer_root / "fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py"
)
micro_path = (
    flashinfer_root / "fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py"
)

replace_once(
    api_path,
    '    quant_mode: Optional[str] = None,\n'
    '    source_format: str = "modelopt",\n'
    ") -> torch.Tensor:\n",
    '    quant_mode: Optional[str] = None,\n'
    '    source_format: str = "modelopt",\n'
    "    swiglu_limit: Optional[float] = None,\n"
    ") -> torch.Tensor:\n",
    "b12x api swiglu_limit parameter",
)
replace_once(
    api_path,
    "        quant_mode=quant_mode,\n"
    "        source_format=source_format,\n"
    "    )\n",
    "        quant_mode=quant_mode,\n"
    "        source_format=source_format,\n"
    "        swiglu_limit=swiglu_limit,\n"
    "    )\n",
    "b12x api forwards swiglu_limit",
)
replace_once(
    api_path,
    '        quant_mode: Optional[str] = None,\n'
    '        source_format: str = "modelopt",\n'
    "    ):\n",
    '        quant_mode: Optional[str] = None,\n'
    '        source_format: str = "modelopt",\n'
    "        swiglu_limit: Optional[float] = None,\n"
    "    ):\n",
    "b12x wrapper swiglu_limit parameter",
)
replace_once(
    api_path,
    "        self.source_format = source_format\n",
    "        self.source_format = source_format\n"
    "        self.swiglu_limit = swiglu_limit\n",
    "b12x wrapper stores swiglu_limit",
)
replace_once(
    api_path,
    "            quant_mode=self.quant_mode,\n"
    "            source_format=self.source_format,\n"
    "            _workspace=workspace,\n",
    "            quant_mode=self.quant_mode,\n"
    "            source_format=self.source_format,\n"
    "            swiglu_limit=self.swiglu_limit,\n"
    "            _workspace=workspace,\n",
    "b12x wrapper forwards swiglu_limit",
)
replace_once(
    dispatch_path,
    '    quant_mode: str | None = None,\n'
    '    source_format: str = "modelopt",\n'
    "    _workspace=None,\n",
    '    quant_mode: str | None = None,\n'
    '    source_format: str = "modelopt",\n'
    "    swiglu_limit: float | None = None,\n"
    "    _workspace=None,\n",
    "b12x dispatch swiglu_limit parameter",
)
replace_once(
    dispatch_path,
    "    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)\n"
    "    source_format = _normalize_source_format_for_quant_mode(source_format, quant_mode)\n"
    "    activation_precision = _activation_precision_from_quant_mode(quant_mode)\n",
    "    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)\n"
    "    source_format = _normalize_source_format_for_quant_mode(source_format, quant_mode)\n"
    "    activation_precision = _activation_precision_from_quant_mode(quant_mode)\n"
    "    swiglu_limit = float(swiglu_limit or 0.0)\n",
    "b12x dispatch normalizes swiglu_limit",
)


def patch_kernel(path: Path, label: str) -> None:
    text = path.read_text()
    if "fmin_f32" not in text.split("from flashinfer.cute_dsl.fp4_common import", 1)[1].split(")", 1)[0]:
        text = text.replace("fmax_f32,\n", "fmax_f32,\n    fmin_f32,\n", 1)
    lines = text.splitlines(keepends=True)
    for index in range(len(lines) - 2):
        if (
            lines[index].lstrip().startswith(
                "g = alpha_value * gate_slice[elem_idx]"
            )
            and lines[index + 1].lstrip().startswith(
                "u = alpha_value * up_slice[elem_idx]"
            )
            and lines[index + 2].lstrip().startswith("sigmoid_g =")
        ):
            window = "".join(lines[index : index + 5])
            if "fmin_f32(g, cutlass.Float32(" in window:
                print(f"{label}: already patched")
                return
            indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
            lines[index + 2 : index + 2] = [
                indent
                + "# HYDRALISK_B12X_SWIGLU_CLAMP_PATCH_POINT: runtime clamp for "
                + f"DeepSeek swiglu_limit={LIMIT}.\n",
                indent + f"g = fmin_f32(g, cutlass.Float32({LIMIT}))\n",
                indent
                + f"u = fmin_f32(fmax_f32(u, cutlass.Float32(-{LIMIT})), "
                + f"cutlass.Float32({LIMIT}))\n",
            ]
            path.write_text("".join(lines))
            py_compile.compile(str(path), doraise=True)
            print(f"{label}: patched {path}")
            return
    raise RuntimeError(f"{label}: clamp activation block missing in {path}")


patch_kernel(static_path, "b12x static clamp")
patch_kernel(dynamic_path, "b12x dynamic clamp")
patch_kernel(micro_path, "b12x micro clamp")

vllm_b12x_path = Path(b12x_vllm.__file__)
replace_once(
    vllm_b12x_path,
    "        self._fc2_input_scale: torch.Tensor | None = None\n",
    "        self._fc2_input_scale: torch.Tensor | None = None\n"
    "        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit\n",
    "vllm b12x stores gemm1_clamp_limit",
)
replace_once(
    vllm_b12x_path,
    "            output_dtype=self.out_dtype,\n"
    "            output=output,\n",
    "            output_dtype=self.out_dtype,\n"
    "            output=output,\n"
    "            swiglu_limit=self.gemm1_clamp_limit,\n",
    "vllm b12x forwards swiglu_limit",
)

oracle_path = Path(nvfp4_oracle.__file__)
replace_once(
    oracle_path,
    "    NVFP4_BACKENDS_WITH_CLAMP = {\n"
    "        NvFp4MoeBackend.FLASHINFER_TRTLLM,\n"
    "    }\n",
    "    NVFP4_BACKENDS_WITH_CLAMP = {\n"
    "        NvFp4MoeBackend.FLASHINFER_TRTLLM,\n"
    "        NvFp4MoeBackend.FLASHINFER_B12X,\n"
    "    }\n",
    "vllm nvfp4 oracle marks b12x clamp-capable",
)
PY

cat > "$build_ctx/patch_nvfp4_sm120.py" <<'PY'
import py_compile
from pathlib import Path

import vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe as mod


path = Path(mod.__file__)
text = path.read_text()
old = (
    "p.is_device_capability_family(100)\n"
    "            and has_flashinfer_trtllm_fused_moe()"
)
new = (
    "(p.is_device_capability_family(100)\n"
    "             or p.is_device_capability_family(120))\n"
    "            and has_flashinfer_trtllm_fused_moe()"
)
if new in text:
    print(f"{path} already permits NVFP4 SM120")
elif "is_device_capability_family(120)" in text and "has_flashinfer_trtllm_fused_moe()" in text:
    print(f"{path} appears to permit NVFP4 SM120")
elif old in text:
    path.write_text(text.replace(old, new, 1))
    py_compile.compile(str(path), doraise=True)
    print(f"patched {path} for NVFP4 SM120 probe")
else:
    raise RuntimeError(f"SM120 guard patch target not found in {path}")
PY

cat > "$build_ctx/Dockerfile" <<'DOCKERFILE'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
SHELL ["/bin/bash", "-lc"]
ARG INSTALL_DEEPGEMM=1
ARG ALLOW_NVFP4_SM120=0
ARG VLLM_E8M0_TRITON_UPCAST=0
ARG HYDRALISK_DEEPSEEK_O_PROJ_PATCH=0
ARG HYDRALISK_B12X_CLAMP_PATCH=0
ARG HYDRALISK_B12X_CLAMP_LIMIT=10.0
COPY patch_o_proj.py /tmp/patch_o_proj.py
COPY patch_b12x_clamp.py /tmp/patch_b12x_clamp.py
COPY patch_nvfp4_sm120.py /tmp/patch_nvfp4_sm120.py
RUN if [[ "$INSTALL_DEEPGEMM" == "1" ]]; then \
      apt-get update && \
      apt-get install -y --no-install-recommends ca-certificates git cuda-libraries-dev-13-0 && \
      rm -rf /var/lib/apt/lists/*; \
    fi
RUN if [[ "$INSTALL_DEEPGEMM" == "1" ]]; then \
      python3 -c 'from urllib.request import urlopen; open("/tmp/install_deepgemm.sh", "wb").write(urlopen("https://raw.githubusercontent.com/vllm-project/vllm/main/tools/install_deepgemm.sh", timeout=120).read())' && \
      UV_SYSTEM_PYTHON=1 bash /tmp/install_deepgemm.sh; \
    fi
RUN if [[ "$ALLOW_NVFP4_SM120" == "1" ]]; then \
      python3 /tmp/patch_nvfp4_sm120.py; \
    fi
RUN if [[ "$VLLM_E8M0_TRITON_UPCAST" == "1" ]]; then \
      python3 -c "from pathlib import Path; import py_compile; import vllm.model_executor.layers.quantization.utils.fp8_utils as mod; path = Path(mod.__file__); text = path.read_text(); marker = 'Hydralisk issue #19 E8M0 CUDA Triton upcast'; old = '''    # Triton cannot currently bind E8M0 scale tensors directly. On ROCm,\n    # DeepSeek-V4 checkpoints store block scales in exponent-only E8M0 format,\n    # so decode them to fp32 before launching the kernel.\n    if current_platform.is_rocm() or current_platform.is_xpu():\n        if As.dtype == torch.float8_e8m0fnu:\n            As = _upcast_e8m0_to_fp32(As).contiguous()\n        if Bs.dtype == torch.float8_e8m0fnu:\n            Bs = _upcast_e8m0_to_fp32(Bs).contiguous()\n'''; new = '''    # Triton cannot currently bind E8M0 scale tensors directly. DeepSeek-V4\n    # checkpoints can store block scales in exponent-only E8M0 format.\n    # Hydralisk issue #19 E8M0 CUDA Triton upcast: decode these scales to fp32\n    # before launching Triton on CUDA as well as ROCm/XPU. Derived-image patch.\n    if As.dtype == torch.float8_e8m0fnu:\n        As = _upcast_e8m0_to_fp32(As).contiguous()\n    if Bs.dtype == torch.float8_e8m0fnu:\n        Bs = _upcast_e8m0_to_fp32(Bs).contiguous()\n'''; assert marker in text or old in text, f'E8M0 upcast patch target not found in {path}'; path.write_text(text if marker in text else text.replace(old, new)); py_compile.compile(str(path), doraise=True); print(f'patched {path} for CUDA Triton E8M0 upcast probe')"; \
    fi
RUN if [[ "$HYDRALISK_DEEPSEEK_O_PROJ_PATCH" == "1" ]]; then \
      python3 /tmp/patch_o_proj.py; \
    fi
RUN if [[ "$HYDRALISK_B12X_CLAMP_PATCH" == "1" ]]; then \
      HYDRALISK_B12X_CLAMP_LIMIT="$HYDRALISK_B12X_CLAMP_LIMIT" python3 /tmp/patch_b12x_clamp.py; \
    fi
DOCKERFILE

build_rc=0
pull_args=()
if [[ "$DOCKER_BUILD_PULL" == "1" ]]; then
  pull_args+=(--pull)
fi
timeout "$STACK_BUILD_TIMEOUT_SECONDS"s sudo docker build \
  "${pull_args[@]}" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "INSTALL_DEEPGEMM=$INSTALL_DEEPGEMM" \
  --build-arg "ALLOW_NVFP4_SM120=$ALLOW_NVFP4_SM120" \
  --build-arg "VLLM_E8M0_TRITON_UPCAST=$VLLM_E8M0_TRITON_UPCAST" \
  --build-arg "HYDRALISK_DEEPSEEK_O_PROJ_PATCH=$HYDRALISK_DEEPSEEK_O_PROJ_PATCH" \
  --build-arg "HYDRALISK_B12X_CLAMP_PATCH=$HYDRALISK_B12X_CLAMP_PATCH" \
  --build-arg "HYDRALISK_B12X_CLAMP_LIMIT=$HYDRALISK_B12X_CLAMP_LIMIT" \
  -t "$DERIVED_IMAGE" \
  -f "$build_ctx/Dockerfile" \
  "$build_ctx" > "$REMOTE_LOG_DIR/provider-stack-build.log" 2>&1 || build_rc=$?
rm -rf "$build_ctx"

{
  printf "BASE_IMAGE\t%s\n" "$BASE_IMAGE"
  printf "DERIVED_IMAGE\t%s\n" "$DERIVED_IMAGE"
  printf "INSTALL_DEEPGEMM\t%s\n" "$INSTALL_DEEPGEMM"
  printf "VLLM_E8M0_TRITON_UPCAST\t%s\n" "$VLLM_E8M0_TRITON_UPCAST"
  printf "HYDRALISK_DEEPSEEK_O_PROJ_PATCH\t%s\n" "$HYDRALISK_DEEPSEEK_O_PROJ_PATCH"
  printf "HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK\t%s\n" "$HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK"
  printf "HYDRALISK_B12X_CLAMP_PATCH\t%s\n" "$HYDRALISK_B12X_CLAMP_PATCH"
  printf "HYDRALISK_B12X_CLAMP_LIMIT\t%s\n" "$HYDRALISK_B12X_CLAMP_LIMIT"
  printf "DOCKER_BUILD_PULL\t%s\n" "$DOCKER_BUILD_PULL"
  printf "BUILD_RC\t%s\n" "$build_rc"
  printf "BASE_IMAGE_INSPECT_BEGIN\n"
  sudo docker image inspect "$BASE_IMAGE" --format '{{json .RepoDigests}} {{json .Id}}' 2>/dev/null || true
  printf "BASE_IMAGE_INSPECT_END\n"
  printf "DERIVED_IMAGE_INSPECT_BEGIN\n"
  sudo docker image inspect "$DERIVED_IMAGE" --format '{{json .RepoDigests}} {{json .Id}}' 2>/dev/null || true
  printf "DERIVED_IMAGE_INSPECT_END\n"
} > "$REMOTE_LOG_DIR/provider-stack-image.txt"

if [[ "$build_rc" != "0" ]]; then
  printf "READY\t0\nBLOCKER\tprovider_stack_build_failed\n" > "$REMOTE_LOG_DIR/provider-stack-smoke-summary.txt"
  printf '{"ready":false,"status":"provider_stack_build_failed"}\n' > "$REMOTE_LOG_DIR/provider-stack-completion-public.json"
  exit 0
fi

hf_env_args=(
  -e "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
  -e "HF_XET_HIGH_PERFORMANCE=$HF_XET_HIGH_PERFORMANCE"
  -e "HYDRALISK_DEEPSEEK_O_PROJ_RECIPE=$HYDRALISK_DEEPSEEK_O_PROJ_RECIPE"
  -e "HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE=$HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE"
  -e "HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS=$HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS"
  -e "HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE=$HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE"
  -e "HYDRALISK_DEEPSEEK_O_PROJ_BYPASS=$HYDRALISK_DEEPSEEK_O_PROJ_BYPASS"
  -e "HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=$HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK"
)
if [[ -n "$HF_XET_NUM_CONCURRENT_RANGE_GETS" ]]; then
  hf_env_args+=(-e "HF_XET_NUM_CONCURRENT_RANGE_GETS=$HF_XET_NUM_CONCURRENT_RANGE_GETS")
fi

sudo docker run --rm --gpus all --ipc=host --network host \
  "${hf_env_args[@]}" \
  --entrypoint bash "$DERIVED_IMAGE" -lc 'python3 - <<'"'"'PY'"'"'
import importlib
import importlib.metadata
import json
import torch

def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"

record = {
    "schema": "hydralisk.deepseek-v4.provider-stack-import.v1",
    "vllm": version("vllm"),
    "torch": version("torch"),
    "torchCuda": torch.version.cuda,
    "cudaAvailable": torch.cuda.is_available(),
    "deviceCount": torch.cuda.device_count(),
    "devices": [
        {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "capability": list(torch.cuda.get_device_capability(i)),
        }
        for i in range(torch.cuda.device_count())
    ],
    "publicSafety": {
        "containsSecrets": False,
        "containsPrompts": False,
        "containsResponses": False,
        "containsWeights": False,
        "containsHiddenReasoning": False,
    },
}
try:
    dg = importlib.import_module("vllm.utils.deep_gemm")
    record["deepGemmImport"] = True
    record["deepGemmHasTransformHelper"] = hasattr(dg, "transform_sf_into_required_layout")
except Exception as exc:
    record["deepGemmImport"] = False
    record["deepGemmImportError"] = f"{type(exc).__name__}: {exc}"[:300]
print(json.dumps(record, sort_keys=True))
PY' > "$REMOTE_LOG_DIR/provider-stack-import.jsonl" 2> "$REMOTE_LOG_DIR/provider-stack-import.stderr" || true

network_rc=0
timeout 45s sudo docker run --rm --network host \
  "${hf_env_args[@]}" \
  --entrypoint python3 "$DERIVED_IMAGE" -c '
import socket
import urllib.request

hosts = ["huggingface.co", "cdn-lfs.huggingface.co", "google.com"]
for host in hosts:
    try:
        print("DNS\t%s\t%s" % (host, socket.getaddrinfo(host, 443)[0][4][0]))
    except Exception as exc:
        print("DNS_ERROR\t%s\t%s: %s" % (host, type(exc).__name__, exc))

url = "https://huggingface.co/%s/resolve/%s/config.json" % (
    "'"$MODEL_ID"'",
    "'"${MODEL_REVISION:-main}"'",
)
try:
    with urllib.request.urlopen(url, timeout=10) as response:
        print("FETCH\t%s\t%s\t%s" % (url, response.status, response.headers.get("content-length")))
except Exception as exc:
    print("FETCH_ERROR\t%s\t%s: %s" % (url, type(exc).__name__, exc))
' > "$REMOTE_LOG_DIR/provider-stack-network.txt" 2>&1 || network_rc=$?
printf "NETWORK_RC\t%s\n" "$network_rc" >> "$REMOTE_LOG_DIR/provider-stack-network.txt"

if [[ "$RUN_MODEL_SMOKE" != "1" ]]; then
  printf "READY\t0\nBLOCKER\tmodel_smoke_skipped\n" > "$REMOTE_LOG_DIR/provider-stack-smoke-summary.txt"
  printf '{"ready":false,"status":"model_smoke_skipped"}\n' > "$REMOTE_LOG_DIR/provider-stack-completion-public.json"
  exit 0
fi

gpu_count="$(nvidia-smi -L | wc -l | tr -d ' ')"
container_name="hydralisk-deepseek-v4-provider-stack-$RANDOM"
{
  printf "BACKEND\tdocker_provider_stack\n"
  printf "MODEL_ID\t%s\n" "$MODEL_ID"
  printf "MODEL_REVISION\t%s\n" "${MODEL_REVISION:-unconfigured}"
  printf "MOE_BACKEND\t%s\n" "$MOE_BACKEND"
  printf "ALLOW_NVFP4_SM120\t%s\n" "$ALLOW_NVFP4_SM120"
  printf "VLLM_LINEAR_BACKEND\t%s\n" "$VLLM_LINEAR_BACKEND"
  printf "VLLM_ENABLE_EXPERT_PARALLEL\t%s\n" "$VLLM_ENABLE_EXPERT_PARALLEL"
  printf "VLLM_E8M0_TRITON_UPCAST\t%s\n" "$VLLM_E8M0_TRITON_UPCAST"
  printf "HYDRALISK_DEEPSEEK_O_PROJ_PATCH\t%s\n" "$HYDRALISK_DEEPSEEK_O_PROJ_PATCH"
  printf "HYDRALISK_DEEPSEEK_O_PROJ_RECIPE\t%s\n" "$HYDRALISK_DEEPSEEK_O_PROJ_RECIPE"
  printf "HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE\t%s\n" "$HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE"
  printf "HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS\t%s\n" "$HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS"
  printf "HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE\t%s\n" "$HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE"
  printf "HYDRALISK_DEEPSEEK_O_PROJ_BYPASS\t%s\n" "$HYDRALISK_DEEPSEEK_O_PROJ_BYPASS"
  printf "HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK\t%s\n" "$HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK"
  printf "HYDRALISK_B12X_CLAMP_PATCH\t%s\n" "$HYDRALISK_B12X_CLAMP_PATCH"
  printf "HYDRALISK_B12X_CLAMP_LIMIT\t%s\n" "$HYDRALISK_B12X_CLAMP_LIMIT"
  printf "HF_HUB_DISABLE_XET\t%s\n" "$HF_HUB_DISABLE_XET"
  printf "HF_XET_HIGH_PERFORMANCE\t%s\n" "$HF_XET_HIGH_PERFORMANCE"
  printf "HF_XET_NUM_CONCURRENT_RANGE_GETS\t%s\n" "${HF_XET_NUM_CONCURRENT_RANGE_GETS:-default}"
  printf "BASE_IMAGE\t%s\n" "$BASE_IMAGE"
  printf "DERIVED_IMAGE\t%s\n" "$DERIVED_IMAGE"
  printf "TENSOR_PARALLEL_SIZE\t%s\n" "$gpu_count"
  printf "MAX_MODEL_LEN\t%s\n" "$MAX_MODEL_LEN"
  printf "MAX_NUM_SEQS\t%s\n" "$MAX_NUM_SEQS"
  printf "MAX_NUM_BATCHED_TOKENS\t%s\n" "$MAX_NUM_BATCHED_TOKENS"
  printf "GPU_MEMORY_UTILIZATION\t%s\n" "$GPU_MEMORY_UTILIZATION"
  if [[ "$HYDRALISK_B12X_CLAMP_PATCH" = "1" ]]; then
    printf "LOCAL_SITE_PACKAGES_PATCHES\tb12x_clamp\n"
  else
    printf "LOCAL_SITE_PACKAGES_PATCHES\tfalse\n"
  fi
  expert_parallel_flags=()
  if [[ "$VLLM_ENABLE_EXPERT_PARALLEL" = "1" ]]; then
    expert_parallel_flags+=(--enable-expert-parallel)
  fi
  printf "PROVIDER_FLAGS\t--kv-cache-dtype fp8 --block-size 256 %s --tensor-parallel-size %s --linear-backend %s\n" "${expert_parallel_flags[*]:-}" "$gpu_count" "$VLLM_LINEAR_BACKEND"
} > "$REMOTE_LOG_DIR/provider-stack-engine.txt"

sudo docker rm -f "$container_name" >/dev/null 2>&1 || true
revision_args=()
if [[ -n "$MODEL_REVISION" ]]; then
  revision_args+=(--revision "$MODEL_REVISION" --tokenizer-revision "$MODEL_REVISION")
fi
moe_backend_args=()
if [[ "$MOE_BACKEND" != "auto" ]]; then
  moe_backend_args+=(--moe-backend "$MOE_BACKEND")
fi
linear_backend_args=()
if [[ "$VLLM_LINEAR_BACKEND" != "auto" ]]; then
  linear_backend_args+=(--linear-backend "$VLLM_LINEAR_BACKEND")
fi
sudo docker run --rm --gpus all --ipc=host --network host \
  --name "$container_name" \
  -v /var/lib/hydralisk/huggingface:/root/.cache/huggingface \
  "${hf_env_args[@]}" \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e VLLM_RPC_TIMEOUT=600000 \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  "$DERIVED_IMAGE" \
  "$MODEL_ID" \
  "${revision_args[@]}" \
  "${moe_backend_args[@]}" \
  "${linear_backend_args[@]}" \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size "$gpu_count" \
  "${expert_parallel_flags[@]}" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  > "$REMOTE_LOG_DIR/provider-stack-vllm.log" 2>&1 &
pid="$!"

ready=0
deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
while [ "$SECONDS" -lt "$deadline" ]; do
  if curl -fsS http://127.0.0.1:8000/v1/models > "$REMOTE_LOG_DIR/provider-stack-models.json" 2> "$REMOTE_LOG_DIR/provider-stack-models.stderr"; then
    ready=1
    break
  fi
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done

if [ "$ready" = "1" ]; then
  completion_payload="$(python3 - "$MODEL_ID" <<'PY'
import json
import sys

print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role": "user", "content": "Reply with READY."}],
    "max_tokens": 8,
    "temperature": 0,
}))
PY
)"
  curl -fsS http://127.0.0.1:8000/v1/chat/completions \
    -H 'content-type: application/json' \
    -d "$completion_payload" \
    | jq '{id: .id, model: .model, usage: .usage, finish_reason: .choices[0].finish_reason}' \
    > "$REMOTE_LOG_DIR/provider-stack-completion-public.json" || true
else
  printf '{"ready":false,"status":"server_not_ready_or_exited"}\n' > "$REMOTE_LOG_DIR/provider-stack-completion-public.json"
fi

sudo docker rm -f "$container_name" >/dev/null 2>&1 || true
wait "$pid" >/tmp/hydralisk-provider-stack-vllm-exit.log 2>&1 || true
printf "READY\t%s\n" "$ready" > "$REMOTE_LOG_DIR/provider-stack-smoke-summary.txt"
tail -n 180 "$REMOTE_LOG_DIR/provider-stack-vllm.log" \
  | sed -E 's/(hf_[A-Za-z0-9_\-]+)/<redacted-hf-token>/g' \
  > "$REMOTE_LOG_DIR/provider-stack-vllm-tail-public.txt"
REMOTE
)"

gcloud compute ssh "$TARGET_INSTANCE" \
  --project "$PROJECT_ID" \
  --zone "$TARGET_ZONE" \
  --quiet \
  --command "$remote_script" \
  > "$OUTPUT_DIR/provider-stack-remote.stdout" \
  2> "$OUTPUT_DIR/provider-stack-remote.stderr" || true

remote_dir="/var/log/hydralisk/deepseek-provider-stack-$TS"
for file in \
  provider-stack-hardware.txt \
  provider-stack-docker-setup.log \
  provider-stack-build.log \
  provider-stack-image.txt \
  provider-stack-import.jsonl \
  provider-stack-import.stderr \
  provider-stack-network.txt \
  provider-stack-engine.txt \
  provider-stack-smoke-summary.txt \
  provider-stack-completion-public.json \
  provider-stack-models.json \
  provider-stack-models.stderr \
  provider-stack-vllm-tail-public.txt; do
  gcloud compute scp \
    "$TARGET_INSTANCE:$remote_dir/$file" \
    "$OUTPUT_DIR/$file" \
    --project "$PROJECT_ID" \
    --zone "$TARGET_ZONE" \
    --quiet > "$OUTPUT_DIR/scp-$file.log" 2>&1 || true
done

render_markdown
echo "OUTPUT_DIR=$OUTPUT_DIR"
