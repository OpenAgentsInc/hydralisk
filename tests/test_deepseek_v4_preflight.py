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
    assert 'printf "VLLM_USE_DEEP_GEMM\\t%s\\n"' in script
    assert 'printf "VLLM_LINEAR_BACKEND\\t%s\\n"' in script
    assert 'printf "VLLM_ENABLE_EXPERT_PARALLEL\\t%s\\n"' in script
    assert 'printf "FORCE_PYTHON_VLLM\\t%s\\n"' in script
    assert 'printf "REUSE_PYTHON_VENV\\t%s\\n"' in script
    assert 'printf "HYDRALISK_DEEPSEEK_O_PROJ_RECIPE\\t%s\\n"' in script
    assert 'printf "HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE\\t%s\\n"' in script
    assert 'printf "HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS\\t%s\\n"' in script
    assert '[[ "$FORCE_PYTHON_VLLM" != "1" ]] && command -v docker' in script
    assert '[[ "$REUSE_PYTHON_VENV" != "1" || ! -x .venv/bin/python ]]' in script
    assert '--linear-backend "$VLLM_LINEAR_BACKEND"' in script
    assert 'HYDRALISK_DEEPSEEK_O_PROJ_RECIPE="$HYDRALISK_DEEPSEEK_O_PROJ_RECIPE"' in script
    assert (
        'HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE="$HYDRALISK_DEEPSEEK_O_PROJ_SHAPE_TRACE"'
        in script
    )
    assert 'HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS="$HYDRALISK_DEEPSEEK_O_PROJ_GROUP_RHS"' in script
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
