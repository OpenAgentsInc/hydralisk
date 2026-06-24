from __future__ import annotations

from datetime import UTC, datetime
import math
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
    assert 'MODEL_REVISION="${MODEL_REVISION:-}"' in script
    assert 'MOE_BACKEND="${MOE_BACKEND:-auto}"' in script
    assert 'ALLOW_NVFP4_SM120="${ALLOW_NVFP4_SM120:-0}"' in script
    assert 'DOCKER_BUILD_PULL="${DOCKER_BUILD_PULL:-1}"' in script
    assert 'VLLM_LINEAR_BACKEND="${VLLM_LINEAR_BACKEND:-auto}"' in script
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
    assert "pull_args+=(--pull)" in script
    assert "p.is_device_capability_family(120)" in script
    assert "Hydralisk issue #19 E8M0 CUDA Triton upcast" in script
    assert "Hydralisk issue #20 DeepSeek NVFP4 o_proj provider patch" in script
    assert "HYDRALISK_O_PROJ_SHAPE_TRACE" in script
    assert "HYDRALISK_O_PROJ_RHS_TRACE" in script
    assert "HYDRALISK_O_PROJ_BYPASS_TRACE" in script
    assert "provider-stack-network.txt" in script
    assert "cdn-lfs.huggingface.co" in script
    assert "NETWORK_RC" in script
    assert 'HF_HUB_DISABLE_XET\\t%s' in script
    assert 'HF_XET_NUM_CONCURRENT_RANGE_GETS\\t%s' in script
    assert 'VLLM_LINEAR_BACKEND\\t%s' in script
    assert 'VLLM_E8M0_TRITON_UPCAST\\t%s' in script
    assert 'HYDRALISK_DEEPSEEK_O_PROJ_PATCH\\t%s' in script
    assert 'HYDRALISK_DEEPSEEK_O_PROJ_BYPASS\\t%s' in script
    assert '"${hf_env_args[@]}"' in script
    assert 'linear_backend_args+=(--linear-backend "$VLLM_LINEAR_BACKEND")' in script
    assert "--kv-cache-dtype fp8" in script
    assert "--block-size 256" in script
    assert "--enable-expert-parallel" in script
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
