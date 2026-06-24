from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from hydralisk.admission.deepseek_v4_fable import (
    FABLE_SCHEMA,
    FABLE_LOAD_CANARY_SCHEMA,
    FableProbeError,
    build_load_canary_report,
    build_report,
    compare_adapter_targets,
    load_metadata_from_dir,
    load_canary_main,
    load_runtime_module_names,
    main,
    render_load_canary_markdown,
    render_markdown,
    validate_requested_files,
)


def test_validate_requested_files_refuses_merged_shards_by_default() -> None:
    with pytest.raises(FableProbeError, match="refusing merged checkpoint shard"):
        validate_requested_files(("adapter_config.json", "model-00001-of-00047.safetensors"))

    assert validate_requested_files(
        ("model-00001-of-00047.safetensors",),
        allow_merged_shards=True,
    ) == ("model-00001-of-00047.safetensors",)


def test_compare_adapter_targets_matches_only_exact_suffixes() -> None:
    matches = {
        item.target: item
        for item in compare_adapter_targets(
            ("q_proj", "up_proj", "down_proj"),
            (
                "layers.0.attn.fused_wqa_wkv",
                "layers.0.mlp.gate_up_proj",
                "layers.0.mlp.down_proj",
            ),
        )
    }

    assert matches["q_proj"].status == "missing"
    assert matches["up_proj"].status == "missing"
    assert matches["down_proj"].status == "matched"
    assert matches["down_proj"].matchedRuntimeModules == ("layers.0.mlp.down_proj",)


def test_build_report_rejects_incompatible_fable_targets(tmp_path: Path) -> None:
    metadata_dir = _write_fable_metadata(tmp_path / "metadata")
    metadata, files = load_metadata_from_dir(metadata_dir)
    runtime_modules = (
        "layers.0.attn.fused_wqa_wkv",
        "layers.0.mlp.gate_up_proj",
        "layers.0.mlp.down_proj",
    )

    report = build_report(
        metadata=metadata,
        files=files,
        runtime_modules=runtime_modules,
        runtime_source="test-runtime-modules.txt",
        created_at=datetime(2026, 6, 24, tzinfo=UTC),
    )
    rendered = render_markdown(report)

    assert report["schema"] == FABLE_SCHEMA
    assert report["status"] == "rejected_adapter_incompatible"
    assert report["decision"]["canAttemptPrivateAdapterLoad"] is False
    assert report["decision"]["canAttemptMergedCheckpointLoad"] is False
    assert report["decision"]["khalaGeneralRouteAllowed"] is False
    assert report["mergedCheckpoint"]["shardsFetchedByDefault"] is False
    assert report["targetCompatibility"]["missingTargets"] == [
        "v_proj",
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "gate_proj",
    ]
    assert "Contains weights: false" in rendered
    assert "model-00001-of-00047.safetensors" not in rendered


def test_runtime_module_loader_accepts_json_and_text(tmp_path: Path) -> None:
    text_path = tmp_path / "modules.txt"
    text_path.write_text("# comment\nlayers.0.mlp.down_proj\n\n")
    json_path = tmp_path / "modules.json"
    json_path.write_text(json.dumps(["layers.0.attn.fused_wqa_wkv"]))

    assert load_runtime_module_names(text_path) == ("layers.0.mlp.down_proj",)
    assert load_runtime_module_names(json_path) == ("layers.0.attn.fused_wqa_wkv",)


def test_cli_writes_public_safe_rejection_report(tmp_path: Path) -> None:
    metadata_dir = _write_fable_metadata(tmp_path / "metadata")
    runtime_modules = tmp_path / "runtime-modules.txt"
    runtime_modules.write_text(
        "\n".join(
            (
                "layers.0.attn.fused_wqa_wkv",
                "layers.0.mlp.gate_up_proj",
                "layers.0.mlp.down_proj",
            )
        )
    )
    output_dir = tmp_path / "out"

    status = main(
        [
            "--metadata-dir",
            str(metadata_dir),
            "--runtime-modules-file",
            str(runtime_modules),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 2
    evidence = (output_dir / "deepseek-v4-fable-adapter-compatibility.md").read_text()
    assert "Status: `rejected_adapter_incompatible`" in evidence
    assert "Contains weights: false" in evidence
    assert "Khala general route allowed: `false`" in evidence


def test_load_canary_blocks_when_adapter_compatibility_rejected(tmp_path: Path) -> None:
    compatibility = _rejected_compatibility_report(tmp_path)

    report = build_load_canary_report(
        compatibility_report=compatibility,
        created_at=datetime(2026, 6, 24, tzinfo=UTC),
    )
    rendered = render_load_canary_markdown(report)

    assert report["schema"] == FABLE_LOAD_CANARY_SCHEMA
    assert report["status"] == "blocked_adapter_incompatible"
    assert report["loadCanary"]["attempted"] is False
    assert report["decision"]["canAttemptPrivateAdapterLoad"] is False
    assert report["decision"]["canRouteKhalaGeneralTraffic"] is False
    assert report["blockers"][0]["code"] == "adapter_runtime_targets_missing"
    assert "Timing metrics are intentionally empty" in rendered
    assert "Contains weights: false" in rendered


def test_load_canary_cli_writes_public_safe_blocked_report(tmp_path: Path) -> None:
    compatibility = _rejected_compatibility_report(tmp_path)
    compatibility_path = tmp_path / "compatibility.json"
    compatibility_path.write_text(json.dumps(compatibility))
    output_dir = tmp_path / "out"

    status = load_canary_main(
        [
            "--compatibility-report",
            str(compatibility_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 2
    evidence = (output_dir / "deepseek-v4-fable-load-canary.md").read_text()
    assert "Status: `blocked_adapter_incompatible`" in evidence
    assert "Attempted: `false`" in evidence
    assert "Contains weights: false" in evidence


def _write_fable_metadata(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "deepseek-ai/DeepSeek-V4-Flash",
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 16,
                "lora_alpha": 32,
                "target_modules": [
                    "v_proj",
                    "q_proj",
                    "up_proj",
                    "o_proj",
                    "k_proj",
                    "gate_proj",
                    "down_proj",
                ],
            }
        )
    )
    (directory / "generation_config.json").write_text(json.dumps({"temperature": 1.0}))
    (directory / "merge_info.json").write_text(
        json.dumps(
            {
                "lora_r": 8,
                "lora_alpha": 16.0,
                "output_dtype": "torch.bfloat16",
                "num_shards": 47,
            }
        )
    )
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepseekV4ForCausalLM"],
                "model_type": "deepseek_v4",
                "quantization_config": {"quant_method": "fp8"},
                "max_position_embeddings": 1048576,
                "n_routed_experts": 256,
                "num_experts_per_tok": 6,
            }
        )
    )
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 298425334924},
                "weight_map": {
                    "embed.weight": "model-00001-of-00047.safetensors",
                    "head.weight": "model-00047-of-00047.safetensors",
                },
            }
        )
    )
    (directory / "adapter_model.safetensors").write_bytes(b"fake-adapter")
    return directory


def _rejected_compatibility_report(tmp_path: Path) -> dict:
    metadata_dir = _write_fable_metadata(tmp_path / "metadata-for-canary")
    metadata, files = load_metadata_from_dir(metadata_dir)
    return build_report(
        metadata=metadata,
        files=files,
        runtime_modules=(
            "layers.0.attn.fused_wqa_wkv",
            "layers.0.mlp.gate_up_proj",
            "layers.0.mlp.down_proj",
        ),
        runtime_source="test-runtime-modules.txt",
        created_at=datetime(2026, 6, 24, tzinfo=UTC),
    )
