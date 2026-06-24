from __future__ import annotations

from datetime import UTC, datetime
import math
from pathlib import Path
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
