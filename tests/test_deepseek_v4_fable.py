from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import struct

import pytest

from hydralisk.admission.deepseek_v4_fable import (
    FABLE_LAB_EVAL_SCHEMA,
    FABLE_OPROJ_OWNERSHIP_SCHEMA,
    FABLE_RETARGET_SCHEMA,
    FABLE_SCHEMA,
    FABLE_LOAD_CANARY_SCHEMA,
    FABLE_TRANSFORM_SMOKE_SCHEMA,
    FABLE_CONTEXT_MAP_SCHEMA,
    FableProbeError,
    build_context_map_report,
    build_lab_eval_report,
    build_load_canary_report,
    build_o_proj_ownership_report,
    build_retarget_plan_report,
    build_report,
    build_transform_smoke_report,
    context_map_main,
    compare_adapter_targets,
    lab_eval_main,
    load_metadata_from_dir,
    load_canary_main,
    load_runtime_module_names,
    main,
    o_proj_ownership_main,
    render_load_canary_markdown,
    render_lab_eval_markdown,
    render_markdown,
    render_o_proj_ownership_markdown,
    render_retarget_plan_markdown,
    render_transform_smoke_markdown,
    render_context_map_markdown,
    retarget_plan_main,
    transform_smoke_main,
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


def test_lab_eval_rejects_when_load_canary_blocked(tmp_path: Path) -> None:
    compatibility = _rejected_compatibility_report(tmp_path)
    load_report = build_load_canary_report(
        compatibility_report=compatibility,
        created_at=datetime(2026, 6, 24, tzinfo=UTC),
    )

    report = build_lab_eval_report(
        load_canary_report=load_report,
        created_at=datetime(2026, 6, 24, tzinfo=UTC),
    )
    rendered = render_lab_eval_markdown(report)

    assert report["schema"] == FABLE_LAB_EVAL_SCHEMA
    assert report["status"] == "rejected_runtime_unstable"
    assert report["labEval"]["attempted"] is False
    assert report["decision"]["admittedPrivateAuthorizedSecurityLabCanary"] is False
    assert report["decision"]["canRouteKhalaGeneralTraffic"] is False
    assert report["prerequisites"]["authorizedSecurityUnscopedRequestsBlocked"] is True
    assert report["blockers"][0]["code"] == "adapter_runtime_targets_missing"
    assert "No lab eval traffic was run" in rendered
    assert "Contains prompts: false" in rendered


def test_lab_eval_cli_writes_public_safe_rejection_report(tmp_path: Path) -> None:
    compatibility = _rejected_compatibility_report(tmp_path)
    load_report = build_load_canary_report(
        compatibility_report=compatibility,
        created_at=datetime(2026, 6, 24, tzinfo=UTC),
    )
    load_path = tmp_path / "load-canary.json"
    load_path.write_text(json.dumps(load_report))
    output_dir = tmp_path / "out"

    status = lab_eval_main(
        [
            "--load-canary-report",
            str(load_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 2
    evidence = (output_dir / "deepseek-v4-fable-lab-eval-decision.md").read_text()
    assert "Status: `rejected_runtime_unstable`" in evidence
    assert "Attempted: `false`" in evidence
    assert "Contains target details: false" in evidence


def test_retarget_plan_classifies_packed_runtime_blockers(tmp_path: Path) -> None:
    compatibility = _rejected_compatibility_report(tmp_path)

    report = build_retarget_plan_report(
        compatibility_report=compatibility,
        created_at=datetime(2026, 6, 24, tzinfo=UTC),
    )
    rendered = render_retarget_plan_markdown(report)
    targets = {
        item["target"]: item
        for item in report["retargetPlan"]["targetClassifications"]
    }

    assert report["schema"] == FABLE_RETARGET_SCHEMA
    assert report["status"] == "blocked_source_inventory_required"
    assert report["decision"]["canAttemptPackedRetargetSmoke"] is False
    assert report["decision"]["canAttemptCanonicalRuntimeProbe"] is True
    assert targets["q_proj"]["status"] == "packed_transform_required"
    assert targets["q_proj"]["packedFamily"] == "attention_fused_wqa_wkv"
    assert targets["gate_proj"]["status"] == "packed_transform_required"
    assert targets["gate_proj"]["packedFamily"] == "swiglu_gate_up_proj"
    assert targets["down_proj"]["status"] == "direct_attachable"
    assert targets["o_proj"]["status"] == "source_inventory_required"
    assert report["retargetPlan"]["sourceInventoryRequiredTargets"] == ["o_proj"]
    assert "canonical DeepSeek-V4-Flash base runtime feasibility probe" in rendered
    assert "Contains weights: false" in rendered


def test_retarget_plan_cli_writes_public_safe_report(tmp_path: Path) -> None:
    compatibility = _rejected_compatibility_report(tmp_path)
    compatibility_path = tmp_path / "compatibility.json"
    compatibility_path.write_text(json.dumps(compatibility))
    output_dir = tmp_path / "out"

    status = retarget_plan_main(
        [
            "--compatibility-report",
            str(compatibility_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 2
    evidence = (output_dir / "deepseek-v4-fable-retarget-plan.md").read_text()
    assert "Status: `blocked_source_inventory_required`" in evidence
    assert "Packed retarget smoke can be attempted: `false`" in evidence
    assert "Contains prompts: false" in evidence


def test_o_proj_ownership_proves_kernel_provider_path() -> None:
    report = build_o_proj_ownership_report(
        source_inventory=_o_proj_source_inventory(),
        created_at=datetime(2026, 6, 24, tzinfo=UTC),
    )
    rendered = render_o_proj_ownership_markdown(report)

    assert report["schema"] == FABLE_OPROJ_OWNERSHIP_SCHEMA
    assert report["status"] == "o_proj_owner_proven_kernel_provider"
    assert report["ownership"]["adapterAddressableModule"] is False
    assert report["ownership"]["kernelProviderOwned"] is True
    assert report["decision"]["canUseVanillaPeftOProj"] is False
    assert report["decision"]["canProceedToPackedLoraTransformSmoke"] is True
    assert report["decision"]["nextStep"] == (
        "implement_offline_packed_lora_delta_transform_smoke"
    )
    assert "Contains full third-party source: false" in rendered


def test_o_proj_ownership_cli_writes_public_safe_report(tmp_path: Path) -> None:
    inventory_path = tmp_path / "source-inventory.json"
    inventory_path.write_text(json.dumps(_o_proj_source_inventory()))
    output_dir = tmp_path / "out"

    status = o_proj_ownership_main(
        [
            "--source-inventory",
            str(inventory_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    evidence = (output_dir / "deepseek-v4-fable-o-proj-ownership.md").read_text()
    assert "Status: `o_proj_owner_proven_kernel_provider`" in evidence
    assert "Packed-LoRA transform smoke can proceed: `true`" in evidence
    assert "Contains weights: false" in evidence


def test_transform_smoke_accepts_complete_fake_adapter(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter_model.safetensors"
    _write_fake_fable_adapter(adapter_path, layers=(0, 1))

    report = build_transform_smoke_report(
        adapter_path=adapter_path,
        created_at=datetime(2026, 6, 24, tzinfo=UTC),
    )
    rendered = render_transform_smoke_markdown(report)

    assert report["schema"] == FABLE_TRANSFORM_SMOKE_SCHEMA
    assert report["status"] == "shape_manifest_ready_for_transform_writer"
    assert report["adapter"]["tensorValuesRead"] is False
    assert report["targets"]["q_proj"]["completePairCount"] == 2
    assert report["targets"]["q_proj"]["ranks"] == [4]
    assert report["packedFamilies"]["attention_fused_wqa_wkv"]["complete"] is True
    assert report["packedFamilies"]["swiglu_gate_up_proj"]["complete"] is True
    assert report["packedFamilies"]["attention_output_o_proj"]["complete"] is True
    assert report["decision"]["canWritePackedDeltaNow"] is False
    assert report["decision"]["canImplementPackedDeltaWriter"] is True
    assert "Contains tensor values: false" in rendered


def test_transform_smoke_blocks_missing_target(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter_model.safetensors"
    _write_fake_fable_adapter(adapter_path, layers=(0,), omit_targets=("o_proj",))

    report = build_transform_smoke_report(
        adapter_path=adapter_path,
        created_at=datetime(2026, 6, 24, tzinfo=UTC),
    )
    rendered = render_transform_smoke_markdown(report)

    assert report["status"] == "blocked_adapter_config_payload_mismatch"
    assert report["decision"]["canImplementPackedDeltaWriter"] is False
    assert "does not match the module-family assumptions" in rendered
    assert any(
        blocker["target"] == "o_proj" and blocker["code"] == "missing_lora_pairs"
        for blocker in report["blockers"]
    )


def test_transform_smoke_cli_writes_public_safe_report(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter_model.safetensors"
    _write_fake_fable_adapter(adapter_path, layers=(0,))
    output_dir = tmp_path / "out"

    status = transform_smoke_main(
        [
            "--adapter-path",
            str(adapter_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    evidence = (output_dir / "deepseek-v4-fable-transform-smoke.md").read_text()
    assert "Status: `shape_manifest_ready_for_transform_writer`" in evidence
    assert "Packed delta writer can be implemented from this manifest: `true`" in evidence
    assert "Contains weights: false" in evidence


def test_context_map_identifies_indexer_loader_blocker(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter_model.safetensors"
    _write_fake_fable_context_adapter(adapter_path)

    report = build_context_map_report(
        adapter_path=adapter_path,
        created_at=datetime(2026, 6, 24, tzinfo=UTC),
    )
    rendered = render_context_map_markdown(report)

    assert report["schema"] == FABLE_CONTEXT_MAP_SCHEMA
    assert report["status"] == "blocked_indexer_loader_mapping_required"
    assert report["decision"]["nextStep"] == (
        "prove_indexer_compressor_loader_mapping_then_write_context_transform"
    )
    shared_gate = _context_entry(report, "mlp_shared_experts", "gate_proj")
    assert shared_gate["runtimeCandidate"]["candidate"] == (
        "layers.*.mlp.shared_experts.gate_up_proj"
    )
    assert shared_gate["status"] == "candidate_transform_ready"
    indexer_gate = _context_entry(report, "attention_compressor_indexer", "gate_proj")
    assert indexer_gate["runtimeCandidate"]["candidate"].endswith(
        "indexer.compressor.fused_wkv_wgate"
    )
    assert indexer_gate["blockers"][0]["code"] == "loader_path_proof_required"
    assert "nested indexer compressor family" in rendered


def test_context_map_cli_writes_public_safe_report(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter_model.safetensors"
    _write_fake_fable_context_adapter(adapter_path)
    output_dir = tmp_path / "out"

    status = context_map_main(
        [
            "--adapter-path",
            str(adapter_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 2
    evidence = (output_dir / "deepseek-v4-fable-context-map.md").read_text()
    assert "Status: `blocked_indexer_loader_mapping_required`" in evidence
    assert "Contains tensor values: false" in evidence


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


def _o_proj_source_inventory() -> dict:
    return {
        "image": (
            "hydralisk-deepseek-v4-b12x-g4-vllm-issue60-vector-v3:"
            "20260624v3vector2"
        ),
        "inspection": "gce_docker_ast_summary",
        "modules": [
            {
                "module": "vllm.models.deepseek_v4.nvidia.model",
                "file": (
                    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
                    "deepseek_v4/nvidia/model.py"
                ),
                "classes": [
                    "DeepseekV4MLP",
                    "DeepseekV4Model",
                    "DeepseekV4ForCausalLM",
                ],
                "functions": ["forward"],
                "names": [],
                "attrs": ["gate_up_proj"],
                "calls": ["gate_up_proj"],
            },
            {
                "module": "vllm.models.deepseek_v4.nvidia.flashmla",
                "file": (
                    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
                    "deepseek_v4/nvidia/flashmla.py"
                ),
                "classes": ["DeepseekV4FlashMLAAttention"],
                "functions": ["__init__", "_o_proj", "forward_mqa"],
                "names": ["deep_gemm_fp8_o_proj"],
                "attrs": [],
                "calls": ["deep_gemm_fp8_o_proj"],
            },
            {
                "module": "vllm.models.deepseek_v4.nvidia.flashinfer_sparse",
                "file": (
                    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
                    "deepseek_v4/nvidia/flashinfer_sparse.py"
                ),
                "classes": [
                    "DeepseekV4FlashInferMLASparseBackend",
                    "DeepseekV4FlashInferMLAAttention",
                ],
                "functions": ["_o_proj", "__init__", "forward_mqa"],
                "names": ["deep_gemm_fp8_o_proj"],
                "attrs": [],
                "calls": ["deep_gemm_fp8_o_proj"],
            },
            {
                "module": "vllm.models.deepseek_v4.nvidia.ops.o_proj",
                "file": (
                    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
                    "deepseek_v4/nvidia/ops/o_proj.py"
                ),
                "classes": [],
                "functions": [
                    "compute_fp8_einsum_recipe",
                    "deep_gemm_fp8_o_proj",
                    "_tensor_meta",
                    "_scale_to_fp32",
                ],
                "names": [],
                "attrs": [],
                "calls": [],
            },
        ],
    }


def _write_fake_fable_adapter(
    path: Path,
    *,
    layers: tuple[int, ...],
    omit_targets: tuple[str, ...] = (),
) -> None:
    entries: dict[str, tuple[str, list[int]]] = {}
    for layer in layers:
        for target in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            if target in omit_targets:
                continue
            module = f"base_model.model.model.layers.{layer}.self_attn.{target}"
            if target in {"gate_proj", "up_proj", "down_proj"}:
                module = f"base_model.model.model.layers.{layer}.mlp.{target}"
            entries[f"{module}.lora_A.weight"] = ("F32", [4, 8])
            entries[f"{module}.lora_B.weight"] = ("F32", [16, 4])
    _write_fake_safetensors(path, entries)


def _write_fake_fable_context_adapter(path: Path) -> None:
    entries: dict[str, tuple[str, list[int]]] = {}
    for layer in (2, 4):
        for target, b_rows in (
            ("gate_proj", 2048),
            ("up_proj", 2048),
            ("down_proj", 4096),
        ):
            a_cols = 2048 if target == "down_proj" else 4096
            module = f"base_model.model.model.layers.{layer}.mlp.shared_experts.{target}"
            entries[f"{module}.lora_A.weight"] = ("F32", [16, a_cols])
            entries[f"{module}.lora_B.weight"] = ("F32", [b_rows, 16])
        compressor = (
            f"base_model.model.model.layers.{layer}.self_attn.compressor.gate_proj"
        )
        entries[f"{compressor}.lora_A.weight"] = ("F32", [16, 4096])
        entries[f"{compressor}.lora_B.weight"] = ("F32", [512, 16])
        indexer = (
            "base_model.model.model.layers."
            f"{layer}.self_attn.compressor.indexer.gate_proj"
        )
        entries[f"{indexer}.lora_A.weight"] = ("F32", [16, 4096])
        entries[f"{indexer}.lora_B.weight"] = ("F32", [256, 16])
    _write_fake_safetensors(path, entries)


def _context_entry(report: dict, context: str, target: str) -> dict:
    for entry in report["contextMap"]:
        if entry["adapterContext"] == context and entry["target"] == target:
            return entry
    raise AssertionError(f"missing context map entry {context}.{target}")


def _write_fake_safetensors(
    path: Path,
    entries: dict[str, tuple[str, list[int]]],
) -> None:
    offset = 0
    header: dict[str, dict] = {}
    payload_parts: list[bytes] = []
    for key, (dtype, shape) in sorted(entries.items()):
        size = _fake_tensor_size(dtype, shape)
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        payload_parts.append(b"\0" * size)
        offset += size
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(
        struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(payload_parts)
    )


def _fake_tensor_size(dtype: str, shape: list[int]) -> int:
    dtype_bytes = {"F32": 4, "BF16": 2, "F16": 2}
    size = dtype_bytes[dtype]
    for dimension in shape:
        size *= dimension
    return size
