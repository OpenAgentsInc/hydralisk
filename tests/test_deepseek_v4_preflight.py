from __future__ import annotations

from datetime import UTC, datetime
import math
import os
from pathlib import Path
import subprocess
import struct

from hydralisk.admission.deepseek_v4_flash import (
    MIB,
    build_preflight_report,
    classify_lanes,
    parse_gguf_metadata,
    render_markdown,
)


def test_parse_gguf_metadata_reads_selected_keys_without_tensors(tmp_path: Path) -> None:
    gguf = tmp_path / "tiny.gguf"
    _write_tiny_gguf(gguf)

    metadata = parse_gguf_metadata(gguf)

    assert metadata.schema == "hydralisk.gguf.metadata.v1"
    assert metadata.version == 3
    assert metadata.tensorCount == 0
    assert metadata.metadataCount == 5
    assert metadata.values["general.architecture"] == "deepseek4"
    assert metadata.values["general.name"] == "DeepSeek V4 Flash"
    assert metadata.values["deepseek4.context_length"] == 1_048_576
    assert metadata.values["deepseek4.attention.compress_ratios"] == [0, 4, 128]
    assert metadata.parser["loadsWeights"] is False
    assert metadata.parser["readsTensorData"] is False


def test_classify_lanes_rejects_live_h100_and_finds_g4_candidates() -> None:
    model_mib = math.ceil(86_720_111_200 / MIB)
    admissions = {item.laneId: item for item in classify_lanes(model_file_mib=model_mib)}

    assert (
        admissions["live-a3-highgpu-1g-h100-gptoss120b"].admissionClass
        == "blocked_reserved_live_host"
    )
    assert (
        admissions["g4-standard-48-rtxpro6000-1g"].admissionClass
        == "candidate_offload_prefetch_smoke"
    )
    assert admissions["g4-standard-48-rtxpro6000-1g"].marginMiB < 0
    assert (
        admissions["g4-standard-96-rtxpro6000-2g"].admissionClass
        == "candidate_all_gpu_low_context_smoke"
    )
    assert (
        admissions["a3-highgpu-2g-h100"].admissionClass
        == "candidate_all_gpu_low_context_smoke"
    )


def test_preflight_report_and_markdown_are_public_safe(tmp_path: Path) -> None:
    gguf = tmp_path / "tiny.gguf"
    _write_tiny_gguf(gguf)
    metadata = parse_gguf_metadata(gguf)
    admissions = classify_lanes(model_file_mib=math.ceil(86_720_111_200 / MIB))

    report = build_preflight_report(
        metadata=metadata,
        admissions=admissions,
        created_at=datetime(2026, 6, 24, tzinfo=UTC),
    )
    rendered = render_markdown(report)

    assert report["schema"] == "hydralisk.deepseek-v4-flash.gce-preflight.v1"
    assert report["publicSafety"]["containsSecrets"] is False
    assert report["recommendation"]["nextStep"].startswith("try_g4_standard_96")
    assert "Do not disturb the live single-H100" in rendered
    assert "Contains weights: false" in rendered


def test_gce_smoke_script_dry_run_plans_fresh_probe_hosts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "smoke-deepseek-v4-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
            "TS": "20260624000000",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    plan = (tmp_path / "attempt-plan.tsv").read_text()
    attempts = (tmp_path / "attempts.tsv").read_text()
    evidence = (tmp_path / "deepseek-v4-gce-smoke.md").read_text()

    assert "hydralisk-deepseek-v4" in attempts
    assert "DRY_RUN=1" in result.stdout
    assert "g4-standard-96" in plan
    assert "a3-highgpu-2g" in plan
    assert "hydralisk-gptoss" not in plan
    assert "khala" not in plan.lower()
    assert "Contains weights: false" in evidence


def test_gce_smoke_script_exposes_backend_and_issue_knobs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts" / "smoke-deepseek-v4-gce.sh").read_text()

    assert 'ISSUE_NUMBER="${ISSUE_NUMBER:-6}"' in script
    assert 'VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-1}"' in script
    assert 'VLLM_USE_DEEP_GEMM_E8M0="${VLLM_USE_DEEP_GEMM_E8M0:-1}"' in script
    assert (
        'VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER="${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-1}"'
        in script
    )
    assert 'VLLM_LINEAR_BACKEND="${VLLM_LINEAR_BACKEND:-auto}"' in script
    assert 'VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-0}"' in script
    assert 'FORCE_PYTHON_VLLM="${FORCE_PYTHON_VLLM:-0}"' in script
    assert 'REUSE_PYTHON_VENV="${REUSE_PYTHON_VENV:-0}"' in script
    assert (
        'HYDRALISK_DEEPSEEK_O_PROJ_RECIPE="${HYDRALISK_DEEPSEEK_O_PROJ_RECIPE:-auto}"'
        in script
    )
    assert (
        'HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE="${HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE:-0}"'
        in script
    )
    assert (
        'HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS="${HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS:-0}"'
        in script
    )
    assert (
        'HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE="${HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE:-raw_e8m0}"'
        in script
    )
    assert 'printf "VLLM_USE_DEEP_GEMM\\t%s\\n"' in script
    assert 'printf "VLLM_LINEAR_BACKEND\\t%s\\n"' in script
    assert 'printf "VLLM_ENABLE_EXPERT_PARALLEL\\t%s\\n"' in script
    assert 'printf "FORCE_PYTHON_VLLM\\t%s\\n"' in script
    assert 'printf "REUSE_PYTHON_VENV\\t%s\\n"' in script
    assert 'printf "HYDRALISK_DEEPSEEK_O_PROJ_RECIPE\\t%s\\n"' in script
    assert 'printf "HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE\\t%s\\n"' in script
    assert 'printf "HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS\\t%s\\n"' in script
    assert 'printf "HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE\\t%s\\n"' in script
    assert '[[ "$FORCE_PYTHON_VLLM" != "1" ]] && command -v docker' in script
    assert '[[ "$REUSE_PYTHON_VENV" != "1" || ! -x .venv/bin/python ]]' in script
    assert "--no-address" in script
    assert '--linear-backend "$VLLM_LINEAR_BACKEND"' in script
    assert 'HYDRALISK_DEEPSEEK_O_PROJ_RECIPE="$HYDRALISK_DEEPSEEK_O_PROJ_RECIPE"' in script
    assert (
        'HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE="$HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE"'
        in script
    )
    assert 'HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS="$HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS"' in script
    assert (
        'HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE="$HYDRALISK_DEEPSEEK_O_PROJ_RHS_SCALE_MODE"'
        in script
    )
    assert "expert_parallel_args+=(--enable-expert-parallel)" in script
    assert 'issue="$ISSUE_NUMBER"' in script


def test_scaled_mm_probe_script_is_public_safe_and_target_scoped(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-scaled-mm-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
            "TARGET_INSTANCE": "hydralisk-deepseek-v4-g4-2g-b-test",
            "TARGET_ZONE": "us-central1-b",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "scaled-mm-probe.md").read_text()

    assert "Wrote" in result.stdout
    assert "hydralisk-deepseek-v4-g4-2g-b-test" in evidence
    assert "Contains weights: false" in evidence

    rejected = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path / "rejected"),
            "TARGET_INSTANCE": "hydralisk-gptoss20b-l4-prod",
            "TARGET_ZONE": "us-central1-a",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode == 2
    assert "fresh hydralisk-deepseek-v4" in rejected.stderr


def test_provider_stack_probe_script_is_public_safe_and_target_scoped(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-provider-stack-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
            "TARGET_INSTANCE": "hydralisk-deepseek-v4-g4-2g-b-test",
            "TARGET_ZONE": "us-central1-b",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "provider-stack-probe.md").read_text()

    assert "Wrote" in result.stdout
    assert "hydralisk-deepseek-v4-g4-2g-b-test" in evidence
    assert "vllm/vllm-openai:latest" in evidence
    assert "install_deepgemm.sh" in evidence
    assert "DeepSeek sparse MLA fallback patch: `0`" in evidence
    assert "DeepSeek sparse MLA fallback runtime: `0`" in evidence
    assert "CUDA launch blocking: `0`" in evidence
    assert "--enable-expert-parallel" in evidence
    assert "Contains weights: false" in evidence

    rejected = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path / "rejected"),
            "TARGET_INSTANCE": "hydralisk-gptoss20b-l4-prod",
            "TARGET_ZONE": "us-central1-a",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode == 2
    assert "fresh hydralisk-deepseek-v4" in rejected.stderr


def test_provider_stack_probe_uses_clean_container_lane() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts" / "probe-deepseek-v4-provider-stack-gce.sh").read_text()

    assert 'ISSUE_NUMBER="${ISSUE_NUMBER:-13}"' in script
    assert 'GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-${CLOUDSDK_CORE_ACCOUNT:-}}"' in script
    assert 'CLOUDSDK_CORE_ACCOUNT="$GCLOUD_ACCOUNT" gcloud "$@"' in script
    assert 'MODEL_REVISION="${MODEL_REVISION:-}"' in script
    assert 'MOE_BACKEND="${MOE_BACKEND:-auto}"' in script
    assert 'ALLOW_NVFP4_SM120="${ALLOW_NVFP4_SM120:-0}"' in script
    assert 'DOCKER_BUILD_PULL="${DOCKER_BUILD_PULL:-1}"' in script
    assert 'VLLM_LINEAR_BACKEND="${VLLM_LINEAR_BACKEND:-auto}"' in script
    assert 'VLLM_ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-1}"' in script
    assert 'VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"' in script
    assert 'VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-auto}"' in script
    assert 'VLLM_E8M0_TRITON_UPCAST="${VLLM_E8M0_TRITON_UPCAST:-0}"' in script
    assert 'HYDRALISK_DEEPSEEK_O_PROJ_PATCH="${HYDRALISK_DEEPSEEK_O_PROJ_PATCH:-0}"' in script
    assert (
        'HYDRALISK_DEEPSEEK_O_PROJ_RECIPE="${HYDRALISK_DEEPSEEK_O_PROJ_RECIPE:-auto}"'
        in script
    )
    assert (
        'HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS="${HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS:-0}"'
        in script
    )
    assert (
        'HYDRALISK_DEEPSEEK_O_PROJ_BYPASS="${HYDRALISK_DEEPSEEK_O_PROJ_BYPASS:-off}"'
        in script
    )
    assert (
        'HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK="${HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK:-off}"'
        in script
    )
    assert (
        'HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK="${HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK:-0}"'
        in script
    )
    assert (
        'HYDRALISK_DEEPSEEK_SPARSE_MLA_PATCH="${HYDRALISK_DEEPSEEK_SPARSE_MLA_PATCH:-$HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK}"'
        in script
    )
    assert 'HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"' in script
    assert 'HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-0}"' in script
    assert 'HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-}"' in script
    assert 'BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:latest}"' in script
    assert "tools/install_deepgemm.sh" in script
    assert "cuda-libraries-dev-13-0" in script
    assert "UV_SYSTEM_PYTHON=1 bash /tmp/install_deepgemm.sh" in script
    assert "LOCAL_SITE_PACKAGES_PATCHES\\tfalse" in script
    assert '--revision "$MODEL_REVISION"' in script
    assert '--tokenizer-revision "$MODEL_REVISION"' in script
    assert '--moe-backend "$MOE_BACKEND"' in script
    assert "--build-arg \"ALLOW_NVFP4_SM120=$ALLOW_NVFP4_SM120\"" in script
    assert "--build-arg \"VLLM_E8M0_TRITON_UPCAST=$VLLM_E8M0_TRITON_UPCAST\"" in script
    assert (
        "--build-arg \"HYDRALISK_DEEPSEEK_O_PROJ_PATCH=$HYDRALISK_DEEPSEEK_O_PROJ_PATCH\""
        in script
    )
    assert (
        "--build-arg \"HYDRALISK_DEEPSEEK_SPARSE_MLA_PATCH=$HYDRALISK_DEEPSEEK_SPARSE_MLA_PATCH\""
        in script
    )
    assert (
        "--build-arg \"HYDRALISK_DEEPSEEK_INDEXER_PATCH=$HYDRALISK_DEEPSEEK_INDEXER_PATCH\""
        in script
    )
    assert "pull_args+=(--pull)" in script
    assert "p.is_device_capability_family(120)" in script
    assert "Hydralisk issue #19 E8M0 CUDA Triton upcast" in script
    assert "Hydralisk issue #20 DeepSeek NVFP4 o_proj provider patch" in script
    assert "patch_sparse_mla.py" in script
    assert "patch_sparse_indexer.py" in script
    assert "patched {path} for Hydralisk sparse MLA fallback" in script
    assert "patched {path} for Hydralisk SWA-only indexer fallback" in script
    assert "Hydralisk SWA-only indexer metadata fallback" in script
    assert "HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY_METADATA" in script
    assert "self.scheduler_metadata_buffer.zero_()" in script
    assert "HYDRALISK_O_PROJ_SHAPE_TRACE" in script
    assert "HYDRALISK_O_PROJ_RHS_TRACE" in script
    assert "HYDRALISK_O_PROJ_BYPASS_TRACE" in script
    assert "HYDRALISK_O_PROJ_FALLBACK_TRACE" in script
    assert 'fallback_mode == "bf16_einsum"' in script
    assert "HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=bf16_einsum requires" in script
    assert "provider-stack-network.txt" in script
    assert "cdn-lfs.huggingface.co" in script
    assert "NETWORK_RC" in script
    assert 'HF_HUB_DISABLE_XET\\t%s' in script
    assert 'HF_XET_NUM_CONCURRENT_RANGE_GETS\\t%s' in script
    assert 'VLLM_LINEAR_BACKEND\\t%s' in script
    assert 'VLLM_ENABLE_EXPERT_PARALLEL\\t%s' in script
    assert 'VLLM_ENFORCE_EAGER\\t%s' in script
    assert 'VLLM_ATTENTION_BACKEND\\t%s' in script
    assert 'COMPLETION_TIMEOUT_SECONDS\\t%s' in script
    assert 'CONTAINER_START_TIMEOUT_SECONDS\\t%s' in script
    assert 'VLLM_E8M0_TRITON_UPCAST\\t%s' in script
    assert 'CUDA_LAUNCH_BLOCKING\\t%s' in script
    assert 'HYDRALISK_DEEPSEEK_O_PROJ_PATCH\\t%s' in script
    assert 'HYDRALISK_DEEPSEEK_O_PROJ_BYPASS\\t%s' in script
    assert 'HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK\\t%s' in script
    assert 'HYDRALISK_DEEPSEEK_SPARSE_MLA_PATCH\\t%s' in script
    assert 'HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK\\t%s' in script
    assert 'HYDRALISK_DEEPSEEK_INDEXER_PATCH\\t%s' in script
    assert 'HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY\\t%s' in script
    assert 'record["sparseMlaFallbackPatched"]' in script
    assert 'record["indexerSwaOnlyEnv"]' in script
    assert "unsupported query dtype" in script
    assert "_hydralisk_sparse_mla_floatable_dtype" in script
    assert "HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK_CACHE_LAYOUT_V2" in script
    assert "upgraded {path} for Hydralisk sparse MLA fallback" in script
    assert (
        '-e "HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK=$HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK"'
        in script
    )
    assert (
        '-e "HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY=$HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY"'
        in script
    )
    assert '-e "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING"' in script
    assert '"${hf_env_args[@]}"' in script
    assert 'linear_backend_args+=(--linear-backend "$VLLM_LINEAR_BACKEND")' in script
    assert 'expert_parallel_flags+=(--enable-expert-parallel)' in script
    assert 'eager_args+=(--enforce-eager)' in script
    assert 'attention_config_args+=(--attention-config "{\\"backend\\":\\"$VLLM_ATTENTION_BACKEND\\"}")' in script
    assert "run_gcloud compute ssh" in script
    assert "run_gcloud compute scp" in script
    assert "sudo docker inspect -f '{{.State.Status}}'" in script
    assert "container_start_deadline" in script
    assert "container_seen=0" in script
    assert '--max-time "$COMPLETION_TIMEOUT_SECONDS"' in script
    assert "completion_failed_or_timed_out" in script
    assert "--kv-cache-dtype fp8" in script
    assert "--block-size 256" in script
    assert '"${expert_parallel_flags[@]}"' in script
    assert '"${eager_args[@]}"' in script
    assert '"${attention_config_args[@]}"' in script
    assert "--tensor-parallel-size \"$gpu_count\"" in script


def test_nvfp4_g4_probe_script_is_public_safe_in_dry_run(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-nvfp4-g4-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "nvfp4-g4-probe.md").read_text()
    plan = (tmp_path / "nvfp4-g4-plan.tsv").read_text()

    assert "Wrote" in result.stdout
    assert "nvidia/DeepSeek-V4-Flash-NVFP4" in evidence
    assert "e3cd60e7de98e9867116860d522499a728de1cf9" in evidence
    assert "MoE backend: `auto`" in evidence
    assert "Allow NVFP4 SM120 guard patch: `0`" in evidence
    assert "Docker build pull: `1`" in evidence
    assert "vLLM linear backend: `auto`" in evidence
    assert "vLLM E8M0 Triton upcast patch: `0`" in evidence
    assert "DeepSeek o_proj provider patch: `0`" in evidence
    assert "DeepSeek o_proj recipe: `auto`" in evidence
    assert "DeepSeek o_proj shape trace: `0`" in evidence
    assert "DeepSeek o_proj grouped RHS: `0`" in evidence
    assert "DeepSeek o_proj RHS scale mode: `raw_e8m0`" in evidence
    assert "DeepSeek o_proj bypass: `off`" in evidence
    assert "HF Hub disable Xet: `0`" in evidence
    assert "HF Xet high performance: `0`" in evidence
    assert "HF Xet concurrent range gets: `default`" in evidence
    assert "Install DeepGEMM helper: `1`" in evidence
    assert "g4-standard-96" in plan
    assert "nvidia-rtx-pro-6000" in plan
    assert "--no-address" in script.read_text()
    assert "hydralisk-gptoss" not in plan
    assert "khala" not in plan.lower()
    assert "Contains weights: false" in evidence


def test_b12x_g4_probe_script_is_public_safe_in_dry_run(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-b12x-g4-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
            "TS": "20260624000000",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "deepseek-v4-b12x-g4-probe.md").read_text()
    plan = (tmp_path / "b12x-g4-plan.tsv").read_text()
    script_text = script.read_text()

    assert "Wrote" in result.stdout
    assert "g4-standard-384" in plan
    assert "g4-standard-192" in plan
    assert "flashinfer_b12x" in evidence
    assert "gcloud account override: `default`" in evidence
    assert "gcloud IAM preflight: `1`" in evidence
    assert "vLLM expert parallel: `0`" in evidence
    assert "vLLM enforce eager: `0`" in evidence
    assert "vLLM attention backend: `auto`" in evidence
    assert "DeepSeek o_proj shape trace: `0`" in evidence
    assert "DeepSeek sparse MLA fallback patch: `0`" in evidence
    assert "DeepSeek sparse MLA fallback runtime: `0`" in evidence
    assert "DeepSeek indexer patch: `0`" in evidence
    assert "DeepSeek indexer SWA-only runtime: `0`" in evidence
    assert "gcloud auth preflight: `1`" in evidence
    assert "Completion timeout seconds: `180`" in evidence
    assert "Container start timeout seconds: `180`" in evidence
    assert "Contains weights: false" in evidence
    assert 'GCLOUD_AUTH_PREFLIGHT="${GCLOUD_AUTH_PREFLIGHT:-1}"' in script_text
    assert 'GCLOUD_IAM_PREFLIGHT="${GCLOUD_IAM_PREFLIGHT:-1}"' in script_text
    assert (
        'HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE="${HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE:-0}"'
        in script_text
    )
    assert "gcloud auth print-access-token > /dev/null" in script_text
    assert "testIamPermissions" in script_text
    assert "blocked_auth" in script_text
    assert "blocked_iam" in script_text
    assert "VLLM_ENABLE_EXPERT_PARALLEL=\"$VLLM_ENABLE_EXPERT_PARALLEL\"" in script_text
    assert "VLLM_ENFORCE_EAGER=\"$VLLM_ENFORCE_EAGER\"" in script_text
    assert "VLLM_ATTENTION_BACKEND=\"$VLLM_ATTENTION_BACKEND\"" in script_text
    assert (
        "HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK=\"$HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK\""
        in script_text
    )
    assert (
        "HYDRALISK_DEEPSEEK_SPARSE_MLA_PATCH=\"$HYDRALISK_DEEPSEEK_SPARSE_MLA_PATCH\""
        in script_text
    )
    assert (
        "HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY=\"$HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY\""
        in script_text
    )
    assert (
        "HYDRALISK_DEEPSEEK_INDEXER_PATCH=\"$HYDRALISK_DEEPSEEK_INDEXER_PATCH\""
        in script_text
    )
    assert "COMPLETION_TIMEOUT_SECONDS=\"$COMPLETION_TIMEOUT_SECONDS\"" in script_text
    assert "CONTAINER_START_TIMEOUT_SECONDS=\"$CONTAINER_START_TIMEOUT_SECONDS\"" in script_text
    assert "hydralisk-gptoss" not in plan
    assert "khala" not in plan.lower()

    rejected = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path / "rejected"),
            "TARGET_INSTANCE": "hydralisk-gptoss20b-l4-prod",
            "TARGET_ZONE": "us-central1-a",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode == 2
    assert "fresh hydralisk-deepseek-v4" in rejected.stderr


def test_b12x_g4_probe_blocks_before_create_when_gcloud_auth_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-b12x-g4-gce.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    compute_called = tmp_path / "compute-called"
    account_seen = tmp_path / "account-seen"
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s' "${{CLOUDSDK_CORE_ACCOUNT:-}}" > {account_seen}
if [[ "$1 $2" == "auth print-access-token" ]]; then
  echo "ERROR: reauth required" >&2
  exit 1
fi
if [[ "$1 $2 $3" == "compute instances create" ]]; then
  echo called > {compute_called}
  exit 0
fi
echo "unexpected gcloud command: $*" >&2
exit 99
""",
    )
    gcloud.chmod(0o755)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
            "OUTPUT_DIR": str(tmp_path / "out"),
            "TS": "20260624000000",
            "ISSUE_NUMBER": "42",
            "GCLOUD_ACCOUNT": "oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com",
            "VLLM_ATTENTION_BACKEND": "FLASHINFER_MLA_SPARSE_DSV4",
            "VLLM_ENFORCE_EAGER": "1",
            "HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK": "bf16_einsum",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "out" / "deepseek-v4-b12x-g4-probe.md").read_text()
    attempts = (tmp_path / "out" / "b12x-g4-attempts.tsv").read_text()

    assert "Wrote" in result.stdout
    assert not compute_called.exists()
    assert account_seen.read_text() == "oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com"
    assert "Issue: https://github.com/OpenAgentsInc/hydralisk/issues/42" in evidence
    assert (
        "gcloud account override: `oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com`"
        in evidence
    )
    assert "gcloud Auth Preflight" in evidence
    assert "Status: `blocked_auth`" in evidence
    assert "gcloud auth login" in evidence
    assert "FLASHINFER_MLA_SPARSE_DSV4" in evidence
    assert "bf16_einsum" in evidence
    assert "blocked_auth" in attempts
    assert "reauth required" in attempts


def test_b12x_g4_probe_blocks_before_create_when_iam_preflight_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-b12x-g4-gce.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    compute_called = tmp_path / "compute-called"
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2" == "auth print-access-token" ]]; then
  echo "fake-token"
  exit 0
fi
if [[ "$1 $2 $3" == "compute instances create" ]]; then
  echo called > {compute_called}
  exit 0
fi
echo "unexpected gcloud command: $*" >&2
exit 99
""",
    )
    gcloud.chmod(0o755)
    curl = fake_bin / "curl"
    curl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '{"permissions":[]}'
""",
    )
    curl.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "OUTPUT_DIR": str(tmp_path / "out"),
        "TS": "20260624000000",
        "ISSUE_NUMBER": "46",
        "GCLOUD_ACCOUNT": "oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com",
    }
    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "out" / "deepseek-v4-b12x-g4-probe.md").read_text()
    attempts = (tmp_path / "out" / "b12x-g4-attempts.tsv").read_text()

    assert "Wrote" in result.stdout
    assert not compute_called.exists()
    assert "Issue: https://github.com/OpenAgentsInc/hydralisk/issues/46" in evidence
    assert "Status: `ok`" in evidence
    assert "Status: `blocked_iam`" in evidence
    assert "compute.instances.create" in evidence
    assert "blocked_iam" in attempts
    assert "missing permissions:" in attempts


def test_deepseek_g4_iam_grant_helper_is_plan_only_by_default(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "plan-deepseek-v4-g4-iam-grant.sh"
    role = repo_root / "deploy" / "gce" / "deepseek-v4-g4-runner-role.yaml"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gcloud_called = tmp_path / "gcloud-called"
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
echo called > {gcloud_called}
exit 99
""",
    )
    gcloud.chmod(0o755)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    output = result.stdout
    role_text = role.read_text()

    assert not gcloud_called.exists()
    assert "Hydralisk DeepSeek-V4-Flash G4 IAM grant plan" in output
    assert "gcloud account override: default" in output
    assert "Grant authority preflight: 1" in output
    assert "resourcemanager.projects.setIamPolicy" in output
    assert "iam.serviceAccounts.setIamPolicy" in output
    assert "gcloud iam roles create" in output
    assert "roles/compute.osAdminLogin" in output
    assert "roles/iam.serviceAccountUser" in output
    assert "probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh" in output
    assert "compute.instances.create" in role_text
    assert "compute.instances.get" in role_text
    assert "compute.disks.create" in role_text
    assert "compute.subnetworks.use" in role_text


def test_deepseek_g4_iam_grant_helper_blocks_apply_without_authority(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "plan-deepseek-v4-g4-iam-grant.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    mutation_called = tmp_path / "mutation-called"
    account_seen = tmp_path / "account-seen"
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s' "${{CLOUDSDK_CORE_ACCOUNT:-}}" > {account_seen}
if [[ "$1 $2" == "auth print-access-token" ]]; then
  echo "fake-token"
  exit 0
fi
if [[ "$1 $2" == "projects describe" ]]; then
  echo "123456789"
  exit 0
fi
if [[ "$1 $2 $3" == "iam roles create" || "$1 $2 $3" == "iam roles update" || "$1 $2" == "projects add-iam-policy-binding" || "$1 $2 $3" == "iam service-accounts add-iam-policy-binding" ]]; then
  echo called > {mutation_called}
  exit 0
fi
echo "unexpected gcloud command: $*" >&2
exit 99
""",
    )
    gcloud.chmod(0o755)
    curl = fake_bin / "curl"
    curl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '{"permissions":[]}'
""",
    )
    curl.chmod(0o755)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "APPLY": "1",
            "GCLOUD_ACCOUNT": "oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    assert not mutation_called.exists()
    assert account_seen.read_text() == "oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com"
    assert "blocked_grant_iam" in result.stderr
    assert "resourcemanager.projects.setIamPolicy" in result.stderr
    assert "iam.serviceAccounts.setIamPolicy" in result.stderr


def test_deepseek_g4_iam_grant_helper_blocks_apply_when_auth_cannot_resolve_project(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "plan-deepseek-v4-g4-iam-grant.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    mutation_called = tmp_path / "mutation-called"
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2" == "projects describe" ]]; then
  echo "reauth required" >&2
  exit 1
fi
if [[ "$1 $2 $3" == "iam roles create" || "$1 $2" == "projects add-iam-policy-binding" ]]; then
  echo called > {mutation_called}
  exit 0
fi
echo "unexpected gcloud command: $*" >&2
exit 99
""",
    )
    gcloud.chmod(0o755)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "APPLY": "1",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode != 0
    assert not mutation_called.exists()
    assert "blocked_grant_auth" in result.stderr
    assert "reauth required" in result.stderr


def test_deepseek_gcloud_credentials_probe_reports_auth_and_iam_without_tokens(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-gcloud-credentials.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2" == "auth print-access-token" ]]; then
  if [[ "${CLOUDSDK_CORE_ACCOUNT:-}" == "chris@openagents.com" ]]; then
    echo "reauth required" >&2
    exit 1
  fi
  echo "fake-token"
  exit 0
fi
if [[ "$1 $2" == "projects describe" ]]; then
  echo "123456789"
  exit 0
fi
echo "unexpected gcloud command: $*" >&2
exit 99
""",
    )
    gcloud.chmod(0o755)
    curl = fake_bin / "curl"
    curl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '{"permissions":[]}'
""",
    )
    curl.chmod(0o755)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "OUTPUT_DIR": str(tmp_path / "out"),
            "ACCOUNTS": "chris@openagents.com,oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "out" / "gcloud-credentials.md").read_text()
    tsv = (tmp_path / "out" / "gcloud-credentials.tsv").read_text()

    assert "Wrote" in result.stdout
    assert "Tokens written to evidence: `false`" in evidence
    assert "chris@openagents.com\tauth failed" not in tsv
    assert "chris@openagents.com\tfailed" in tsv
    assert "oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com\tok\tmissing_permissions" in tsv
    assert "compute.instances.create" in evidence
    assert "resourcemanager.projects.setIamPolicy" in evidence
    assert not list((tmp_path / "out").glob("token-*"))


def test_deepseek_google_alt_credentials_probe_covers_adc_and_impersonation(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-google-alt-credentials.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2 $3" == "config get-value auth/impersonate_service_account" ]]; then
  echo "configured-runner@openagentsgemini.iam.gserviceaccount.com"
  exit 0
fi
if [[ "$1 $2 $3" == "auth application-default print-access-token" ]]; then
  echo "fake-adc-token"
  exit 0
fi
if [[ "$1 $2" == "auth print-access-token" ]]; then
  if [[ "${3:-}" == "--impersonate-service-account=configured-runner@openagentsgemini.iam.gserviceaccount.com" ]]; then
    echo "fake-impersonated-token"
    exit 0
  fi
  echo "impersonation denied" >&2
  exit 1
fi
echo "unexpected gcloud command: $*" >&2
exit 99
""",
    )
    gcloud.chmod(0o755)
    curl = fake_bin / "curl"
    curl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
token=""
payload=""
for arg in "$@"; do
  case "$arg" in
    Authorization:*) token="${arg#Authorization: Bearer }" ;;
    *compute.instances.create*) payload="$arg" ;;
    *iam.roles.create*) payload="$arg" ;;
  esac
done
if [[ "$*" == *"https://cloudresourcemanager.googleapis.com/v1/projects/openagentsgemini"* && "$*" != *":testIamPermissions"* ]]; then
  printf '{"projectNumber":"123456789"}'
  exit 0
fi
if [[ "$payload" == *"compute.instances.create"* && "$token" == "fake-adc-token" ]]; then
  printf '{"permissions":["compute.instances.create","compute.instances.get","compute.instances.setLabels","compute.instances.setMetadata","compute.instances.setTags","compute.disks.create","compute.subnetworks.use"]}'
  exit 0
fi
printf '{"permissions":[]}'
""",
    )
    curl.chmod(0o755)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "OUTPUT_DIR": str(tmp_path / "out"),
            "IMPERSONATE_ACCOUNTS": "denied@openagentsgemini.iam.gserviceaccount.com",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "out" / "google-alt-credentials.md").read_text()
    tsv = (tmp_path / "out" / "google-alt-credentials.tsv").read_text()

    assert "Wrote" in result.stdout
    assert "Tokens written to evidence: `false`" in evidence
    assert "adc\tapplication-default\tok\tok" in tsv
    assert (
        "configured_impersonation\tconfigured-runner@openagentsgemini.iam.gserviceaccount.com\tok\tmissing_permissions"
        in tsv
    )
    assert "explicit_impersonation\tdenied@openagentsgemini.iam.gserviceaccount.com\tfailed" in tsv
    assert not list((tmp_path / "out").glob("token-*"))


def test_deepseek_service_account_key_probe_is_public_safe(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-service-account-keys.sh"
    key_ok = tmp_path / "runner-key.json"
    key_limited = tmp_path / "limited-key.json"
    key_ok.write_text(
        """{
  "type": "service_account",
  "project_id": "openagentsgemini",
  "private_key": "PRIVATE_KEY_FOR_TEST_ONLY",
  "client_email": "runner@openagentsgemini.iam.gserviceaccount.com"
}
"""
    )
    key_limited.write_text(
        """{
  "type": "service_account",
  "project_id": "openagentsgemini",
  "private_key": "PRIVATE_KEY_FOR_TEST_ONLY",
  "client_email": "limited@openagentsgemini.iam.gserviceaccount.com"
}
"""
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1 $2" == "auth activate-service-account" ]]; then
  key_file=""
  for arg in "$@"; do
    case "$arg" in
      --key-file=*) key_file="${arg#--key-file=}" ;;
    esac
  done
  if [[ "$key_file" == *"runner-key.json" ]]; then
    echo "runner" > "$CLOUDSDK_CONFIG/account"
  else
    echo "limited" > "$CLOUDSDK_CONFIG/account"
  fi
  exit 0
fi
if [[ "$1 $2" == "auth print-access-token" ]]; then
  account="$(cat "$CLOUDSDK_CONFIG/account")"
  echo "fake-${account}-token"
  exit 0
fi
echo "unexpected gcloud command: $*" >&2
exit 99
""",
    )
    gcloud.chmod(0o755)
    curl = fake_bin / "curl"
    curl.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
token=""
payload=""
for arg in "$@"; do
  case "$arg" in
    Authorization:*) token="${arg#Authorization: Bearer }" ;;
    *compute.instances.create*) payload="$arg" ;;
    *iam.roles.create*) payload="$arg" ;;
  esac
done
if [[ "$*" == *"https://cloudresourcemanager.googleapis.com/v1/projects/openagentsgemini"* && "$*" != *":testIamPermissions"* ]]; then
  printf '{"projectNumber":"123456789"}'
  exit 0
fi
if [[ "$payload" == *"compute.instances.create"* && "$token" == "fake-runner-token" ]]; then
  printf '{"permissions":["compute.instances.create","compute.instances.get","compute.instances.setLabels","compute.instances.setMetadata","compute.instances.setTags","compute.disks.create","compute.subnetworks.use"]}'
  exit 0
fi
printf '{"permissions":[]}'
""",
    )
    curl.chmod(0o755)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "OUTPUT_DIR": str(tmp_path / "out"),
            "KEY_FILES": f"{key_ok},{key_limited}",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "out" / "service-account-keys.md").read_text()
    tsv = (tmp_path / "out" / "service-account-keys.tsv").read_text()

    assert "Wrote" in result.stdout
    assert "Private key material written to evidence: `false`" in evidence
    assert "Key file paths written to evidence: `false`" in evidence
    assert "Tokens written to evidence: `false`" in evidence
    assert "PRIVATE_KEY_FOR_TEST_ONLY" not in evidence
    assert str(key_ok) not in evidence
    assert str(key_limited) not in evidence
    assert "runner@openagentsgemini.iam.gserviceaccount.com\topenagentsgemini\tok\tok\tok" in tsv
    assert (
        "limited@openagentsgemini.iam.gserviceaccount.com\topenagentsgemini\tok\tok\tmissing_permissions"
        in tsv
    )
    assert not list((tmp_path / "out").glob("token-*"))


def test_flashinfer_dsv4_g4_wrapper_sets_issue_60_defaults(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
            "TS": "20260624000000",
            "GCLOUD_ACCOUNT": "oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "deepseek-v4-b12x-g4-probe.md").read_text()
    script_text = script.read_text()

    assert "Wrote" in result.stdout
    assert "Issue: https://github.com/OpenAgentsInc/hydralisk/issues/60" in evidence
    assert (
        "gcloud account override: `oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com`"
        in evidence
    )
    assert "gcloud IAM preflight: `1`" in evidence
    assert "vLLM enforce eager: `1`" in evidence
    assert "vLLM attention backend: `FLASHINFER_MLA_SPARSE_DSV4`" in evidence
    assert "DeepSeek o_proj shape trace: `0`" in evidence
    assert "DeepSeek o_proj fallback: `bf16_einsum`" in evidence
    assert "DeepSeek sparse MLA fallback patch: `1`" in evidence
    assert "DeepSeek sparse MLA fallback runtime: `1`" in evidence
    assert "DeepSeek indexer patch: `1`" in evidence
    assert "DeepSeek indexer SWA-only runtime: `1`" in evidence
    assert "B12x clamp patch: `1`" in evidence
    assert "B12x clamp limit: `10.0`" in evidence
    assert "Max model length: `2048`" in evidence
    assert "Max batched tokens: `512`" in evidence
    assert "GPU memory utilization: `0.95`" in evidence
    assert "ISSUE_NUMBER=\"${ISSUE_NUMBER:-60}\"" in script_text
    assert 'GCLOUD_ACCOUNT="${GCLOUD_ACCOUNT:-${CLOUDSDK_CORE_ACCOUNT:-}}"' in script_text
    assert "FLASHINFER_MLA_SPARSE_DSV4" in script_text
    assert (
        'HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK="${HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK:-1}"'
        in script_text
    )
    assert (
        'HYDRALISK_DEEPSEEK_SPARSE_MLA_PATCH="${HYDRALISK_DEEPSEEK_SPARSE_MLA_PATCH:-1}"'
        in script_text
    )
    assert (
        'HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE="${HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE:-0}"'
        in script_text
    )
    assert (
        'HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY="${HYDRALISK_DEEPSEEK_INDEXER_SWA_ONLY:-1}"'
        in script_text
    )
    assert (
        'HYDRALISK_DEEPSEEK_INDEXER_PATCH="${HYDRALISK_DEEPSEEK_INDEXER_PATCH:-1}"'
        in script_text
    )
    assert "probe-deepseek-v4-b12x-g4-gce.sh" in script_text


def test_flashinfer_dsv4_g4_wrapper_keeps_auth_preflight(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-flashinfer-dsv4-g4-gce.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    compute_called = tmp_path / "compute-called"
    account_seen = tmp_path / "account-seen"
    gcloud = fake_bin / "gcloud"
    gcloud.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s' "${{CLOUDSDK_CORE_ACCOUNT:-}}" > {account_seen}
if [[ "$1 $2" == "auth print-access-token" ]]; then
  echo "ERROR: reauth required" >&2
  exit 1
fi
if [[ "$1 $2 $3" == "compute instances create" ]]; then
  echo called > {compute_called}
  exit 0
fi
echo "unexpected gcloud command: $*" >&2
exit 99
""",
    )
    gcloud.chmod(0o755)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
            "OUTPUT_DIR": str(tmp_path / "out"),
            "TS": "20260624000000",
            "GCLOUD_ACCOUNT": "oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "out" / "deepseek-v4-b12x-g4-probe.md").read_text()

    assert "Wrote" in result.stdout
    assert not compute_called.exists()
    assert account_seen.read_text() == "oa-vertex-inference@openagentsgemini.iam.gserviceaccount.com"
    assert "Issue: https://github.com/OpenAgentsInc/hydralisk/issues/60" in evidence
    assert "Status: `blocked_auth`" in evidence
    assert "FLASHINFER_MLA_SPARSE_DSV4" in evidence
    assert "bf16_einsum" in evidence
    assert "DeepSeek sparse MLA fallback runtime: `1`" in evidence


def test_clamp_backends_g4_probe_script_is_public_safe_in_dry_run(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-clamp-backends-g4-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
            "TS": "20260624000000",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "deepseek-v4-clamp-backends-g4-probe.md").read_text()
    host_plan = (tmp_path / "clamp-backends-g4-host-plan.tsv").read_text()
    backend_plan = (tmp_path / "clamp-backends-g4-backend-plan.tsv").read_text()
    script_text = script.read_text()

    assert "Wrote" in result.stdout
    assert "g4-standard-384" in host_plan
    assert "g4-standard-192" in host_plan
    assert "flashinfer_cutlass" in backend_plan
    assert "flashinfer_trtllm" in backend_plan
    assert "vLLM expert parallel: `1`" in evidence
    assert "Contains weights: false" in evidence
    assert "VLLM_ENABLE_EXPERT_PARALLEL=\"$VLLM_ENABLE_EXPERT_PARALLEL\"" in script_text
    assert "hydralisk-gptoss" not in host_plan
    assert "khala" not in host_plan.lower()

    rejected = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path / "rejected"),
            "TARGET_INSTANCE": "hydralisk-gptoss20b-l4-prod",
            "TARGET_ZONE": "us-central1-a",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode == 2
    assert "fresh hydralisk-deepseek-v4" in rejected.stderr


def test_oproj_fallback_g4_probe_script_is_public_safe_in_dry_run(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-oproj-fallback-g4-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
            "TS": "20260624000000",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "deepseek-v4-oproj-fallback-g4-probe.md").read_text()
    host_plan = (tmp_path / "oproj-fallback-g4-host-plan.tsv").read_text()
    script_text = script.read_text()

    assert "Wrote" in result.stdout
    assert "g4-standard-384" in host_plan
    assert "g4-standard-192" in host_plan
    assert "MoE backend: `flashinfer_trtllm`" in evidence
    assert "DeepSeek o_proj recipe: `hopper`" in evidence
    assert "DeepSeek o_proj fallback: `bf16_einsum`" in evidence
    assert "DeepSeek o_proj bypass: `off`" in evidence
    assert "Contains weights: false" in evidence
    assert "HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK=\"$HYDRALISK_DEEPSEEK_O_PROJ_FALLBACK\"" in script_text
    assert "hydralisk-gptoss" not in host_plan
    assert "khala" not in host_plan.lower()

    rejected = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path / "rejected"),
            "TARGET_INSTANCE": "hydralisk-gptoss20b-l4-prod",
            "TARGET_ZONE": "us-central1-a",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode == 2
    assert "fresh hydralisk-deepseek-v4" in rejected.stderr


def test_nvfp4_g4_probe_refuses_non_probe_targets(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-nvfp4-g4-gce.sh"

    rejected = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path / "rejected"),
            "TARGET_INSTANCE": "hydralisk-gptoss20b-l4-prod",
            "TARGET_ZONE": "us-central1-a",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode == 2
    assert "fresh hydralisk-deepseek-v4" in rejected.stderr


def test_flashinfer_trtllm_nvfp4_moe_probe_is_public_safe_and_target_scoped(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-flashinfer-trtllm-nvfp4-moe-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
            "TARGET_INSTANCE": "hydralisk-deepseek-v4-nvfp4-g4-test",
            "TARGET_ZONE": "us-central1-b",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "flashinfer-trtllm-nvfp4-moe-probe.md").read_text()
    script_text = script.read_text()

    assert "Wrote" in result.stdout
    assert "hydralisk-deepseek-v4-nvfp4-g4-test" in evidence
    assert "Sequence length: `1024`" in evidence
    assert "Contains weights: false" in evidence
    assert "trtllm_fp4_block_scale_routed_moe" in script_text
    assert "hydralisk.flashinfer.trtllm-nvfp4-moe.synthetic.v1" in script_text

    rejected = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path / "rejected"),
            "TARGET_INSTANCE": "hydralisk-gptoss20b-l4-prod",
            "TARGET_ZONE": "us-central1-a",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode == 2
    assert "fresh hydralisk-deepseek-v4" in rejected.stderr


def test_flashinfer_b12x_moe_probe_is_public_safe_and_target_scoped(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-flashinfer-b12x-moe-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
            "TARGET_INSTANCE": "hydralisk-deepseek-v4-nvfp4-g4-test",
            "TARGET_ZONE": "us-central1-b",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "flashinfer-b12x-moe-probe.md").read_text()
    script_text = script.read_text()

    assert "Wrote" in result.stdout
    assert "hydralisk-deepseek-v4-nvfp4-g4-test" in evidence
    assert "Sequence length: `512`" in evidence
    assert "Local experts: `32`" in evidence
    assert "SwiGLU limit: `10.0`" in evidence
    assert "Run local-shard remap case: `1`" in evidence
    assert "Run no-EP case: `1`" in evidence
    assert "Contains weights: false" in evidence
    assert "b12x_fused_moe" in script_text
    assert "hydralisk.flashinfer.b12x-moe.synthetic.v1" in script_text
    assert "hydralisk.flashinfer.b12x-moe.install.v1" in script_text
    assert "FLASHINFER_INSTALL_MODE" in script_text
    assert "FLASHINFER_PIP_PACKAGES" in script_text
    assert "FLASHINFER_PIP_EXTRA_INDEX_URL" in script_text
    assert "https://flashinfer.ai/whl/nightly/" in script_text
    assert "B12xMoEWrapper" in script_text
    assert "supportsLocalExpertOffsetKwarg" in script_text
    assert "supportsSwigluLimitKwarg" in script_text
    assert "b12x_swiglu_limit_kwarg_probe" in script_text
    assert "deepseek_shape_local_shard_remap" in script_text
    assert "RUN_MASKED_LOCAL_SHARD_CASE" in script_text
    assert "deepseek_shape_local_shard_masked_dispatch" in script_text
    assert "local_shard_masked_zero_scale" in script_text
    assert '-e "SWIGLU_LIMIT=$SWIGLU_LIMIT"' in script_text
    assert '-e "RUN_LOCAL_SHARD_REMAP_CASE=$RUN_LOCAL_SHARD_REMAP_CASE"' in script_text
    assert '-e "RUN_MASKED_LOCAL_SHARD_CASE=$RUN_MASKED_LOCAL_SHARD_CASE"' in script_text
    assert "deepseek_shape_ep" in script_text
    assert "deepseek_shape_no_ep" in script_text

    rejected = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path / "rejected"),
            "TARGET_INSTANCE": "hydralisk-gptoss20b-l4-prod",
            "TARGET_ZONE": "us-central1-a",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode == 2
    assert "fresh hydralisk-deepseek-v4" in rejected.stderr


def test_published_recipe_probe_script_is_public_safe_in_dry_run(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "probe-deepseek-v4-published-recipe-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "published-recipe-gce-probe.md").read_text()
    plan = (tmp_path / "published-recipe-candidates.tsv").read_text()

    assert "Wrote" in result.stdout
    assert "a3-ultragpu-8g" in plan
    assert "a4-highgpu-8g" in plan
    assert "nvidia-h200-141gb" in plan
    assert "nvidia-b200" in plan
    assert "nvidia-gb200" in plan
    assert "hydralisk-gptoss" not in plan
    assert "khala" not in plan.lower()
    assert "Contains weights: false" in evidence


def test_published_recipe_probe_requires_explicit_create() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (
        repo_root / "scripts" / "probe-deepseek-v4-published-recipe-gce.sh"
    ).read_text()

    assert 'ATTEMPT_CREATE="${ATTEMPT_CREATE:-0}"' in script
    assert "--provisioning-model \"$PROVISIONING_MODEL\"" in script
    assert "--instance-termination-action DELETE" in script
    assert "--max-run-duration \"$MAX_RUN_DURATION\"" in script
    assert "--no-address" in script
    assert "hydralisk-probe,deepseek-v4,published-recipe" in script
    assert 'quota_${quota%%:*}' in script
    assert "ATTEMPT_CREATE=0" in script


def test_gce_probe_create_paths_are_private_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    for name in [
        "scripts/probe-deepseek-v4-nvfp4-g4-gce.sh",
        "scripts/probe-deepseek-v4-published-recipe-gce.sh",
        "scripts/smoke-deepseek-v4-gce.sh",
        "scripts/probe-glm-52-gce-admission.sh",
    ]:
        script = (repo_root / name).read_text()
        assert "--no-address" in script


def test_e8m0_upcast_patch_script_is_public_safe_and_target_scoped(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "patch-vllm-e8m0-triton-upcast-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
            "TARGET_INSTANCE": "hydralisk-deepseek-v4-g4-2g-b-test",
            "TARGET_ZONE": "us-central1-b",
            "ACTION": "apply",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "e8m0-upcast-patch.md").read_text()

    assert "Wrote" in result.stdout
    assert "hydralisk-deepseek-v4-g4-2g-b-test" in evidence
    assert "Action: `apply`" in evidence
    assert "Contains weights: false" in evidence

    rejected = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path / "rejected"),
            "TARGET_INSTANCE": "hydralisk-gptoss20b-l4-prod",
            "TARGET_ZONE": "us-central1-a",
            "ACTION": "apply",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode == 2
    assert "fresh hydralisk-deepseek-v4" in rejected.stderr


def test_o_proj_patch_script_is_public_safe_and_target_scoped(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "patch-vllm-deepseek-o-proj-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
            "TARGET_INSTANCE": "hydralisk-deepseek-v4-g4-2g-b-test",
            "TARGET_ZONE": "us-central1-b",
            "ACTION": "apply",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "o-proj-patch.md").read_text()

    assert "Wrote" in result.stdout
    assert "hydralisk-deepseek-v4-g4-2g-b-test" in evidence
    assert "Action: `apply`" in evidence
    assert "Contains weights: false" in evidence

    rejected = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path / "rejected"),
            "TARGET_INSTANCE": "hydralisk-gptoss20b-l4-prod",
            "TARGET_ZONE": "us-central1-a",
            "ACTION": "apply",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode == 2
    assert "fresh hydralisk-deepseek-v4" in rejected.stderr


def test_o_proj_rhs_patch_script_is_public_safe_and_target_scoped(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "patch-vllm-deepseek-o-proj-rhs-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
            "TARGET_INSTANCE": "hydralisk-deepseek-v4-g4-2g-b-test",
            "TARGET_ZONE": "us-central1-b",
            "ACTION": "apply",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "o-proj-rhs-patch.md").read_text()

    assert "Wrote" in result.stdout
    assert "hydralisk-deepseek-v4-g4-2g-b-test" in evidence
    assert "Action: `apply`" in evidence
    assert "Contains weights: false" in evidence

    rejected = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path / "rejected"),
            "TARGET_INSTANCE": "hydralisk-gptoss20b-l4-prod",
            "TARGET_ZONE": "us-central1-a",
            "ACTION": "apply",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode == 2
    assert "fresh hydralisk-deepseek-v4" in rejected.stderr


def test_o_proj_rhs_scale_patch_script_is_public_safe_and_target_scoped(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "patch-vllm-deepseek-o-proj-rhs-scale-gce.sh"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path),
            "TARGET_INSTANCE": "hydralisk-deepseek-v4-g4-2g-b-test",
            "TARGET_ZONE": "us-central1-b",
            "ACTION": "apply",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    evidence = (tmp_path / "o-proj-rhs-scale-patch.md").read_text()

    assert "Wrote" in result.stdout
    assert "hydralisk-deepseek-v4-g4-2g-b-test" in evidence
    assert "Action: `apply`" in evidence
    assert "Contains weights: false" in evidence

    rejected = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "DRY_RUN": "1",
            "OUTPUT_DIR": str(tmp_path / "rejected"),
            "TARGET_INSTANCE": "hydralisk-gptoss20b-l4-prod",
            "TARGET_ZONE": "us-central1-a",
            "ACTION": "apply",
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode == 2
    assert "fresh hydralisk-deepseek-v4" in rejected.stderr


def _write_tiny_gguf(path: Path) -> None:
    entries = [
        ("general.architecture", 8, "deepseek4"),
        ("general.name", 8, "DeepSeek V4 Flash"),
        ("deepseek4.context_length", 4, 1_048_576),
        ("deepseek4.expert_used_count", 4, 6),
        ("deepseek4.attention.compress_ratios", 9, (4, [0, 4, 128])),
    ]
    data = bytearray()
    data.extend(b"GGUF")
    data.extend(struct.pack("<I", 3))
    data.extend(struct.pack("<Q", 0))
    data.extend(struct.pack("<Q", len(entries)))
    for key, value_type, value in entries:
        _write_string(data, key)
        data.extend(struct.pack("<I", value_type))
        _write_value(data, value_type, value)
    path.write_bytes(bytes(data))


def _write_string(data: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    data.extend(struct.pack("<Q", len(encoded)))
    data.extend(encoded)


def _write_value(data: bytearray, value_type: int, value: object) -> None:
    if value_type == 8:
        _write_string(data, str(value))
        return
    if value_type == 4:
        data.extend(struct.pack("<I", int(value)))
        return
    if value_type == 9:
        element_type, items = value
        data.extend(struct.pack("<I", int(element_type)))
        data.extend(struct.pack("<Q", len(items)))
        for item in items:
            _write_value(data, int(element_type), item)
        return
    raise AssertionError(f"unsupported fixture value type {value_type}")
