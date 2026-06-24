from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


B12X_API = Path("flashinfer/fused_moe/cute_dsl/b12x_moe.py")
B12X_DISPATCH = Path("flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py")
B12X_KERNELS = [
    Path("flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py"),
    Path("flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_micro_kernel.py"),
    Path("flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py"),
    Path("flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py"),
]
PATCH_MARKER = "HYDRALISK_B12X_SWIGLU_CLAMP_PATCH_POINT"
PATCH_COMMENT = (
    "                                        # HYDRALISK_B12X_SWIGLU_CLAMP_PATCH_POINT: "
    "apply vLLM swiglu_limit before sigmoid; gate=min(gate, limit), "
    "up=clamp(up, -limit, limit).\n"
)
DYNAMIC_PATCH_COMMENT = (
    "                                            # HYDRALISK_B12X_SWIGLU_CLAMP_PATCH_POINT: "
    "apply vLLM swiglu_limit before sigmoid; gate=min(gate, limit), "
    "up=clamp(up, -limit, limit).\n"
)
W4A16_PATCH_COMMENT = (
    "                # HYDRALISK_B12X_SWIGLU_CLAMP_PATCH_POINT: apply vLLM "
    "swiglu_limit before sigmoid; gate=min(gate, limit), up=clamp(up, -limit, limit).\n"
)


@dataclass(frozen=True)
class SourceEdit:
    path: Path
    old: str
    new: str
    description: str
    count: int = 1


def apply_overlay(flashinfer_root: Path, *, dry_run: bool = False) -> dict[str, Any]:
    edits = _source_edits()
    results: list[dict[str, Any]] = []
    patched_texts: dict[Path, str] = {}
    for edit in edits:
        target = flashinfer_root / edit.path
        original = patched_texts.get(edit.path, target.read_text())
        changed, status = _apply_edit_text(original, edit)
        if status == "missing":
            raise RuntimeError(f"Patch point missing for {edit.path}: {edit.description}")
        if changed:
            patched_texts[edit.path] = changed
            if not dry_run:
                target.write_text(changed)
        results.append(
            {
                "path": str(edit.path),
                "description": edit.description,
                "status": status,
                "dryRun": dry_run,
            }
        )

    validation = validate_overlay(
        flashinfer_root,
        text_overrides=patched_texts if dry_run else None,
    )
    return {
        "schema": "hydralisk.deepseek-v4.b12x-clamp-patch-overlay.v1",
        "dryRun": dry_run,
        "edits": results,
        "validation": validation,
    }


def validate_overlay(
    flashinfer_root: Path,
    *,
    text_overrides: dict[Path, str] | None = None,
) -> dict[str, Any]:
    text_overrides = text_overrides or {}

    def read(path: Path) -> str:
        if path in text_overrides:
            return text_overrides[path]
        return (flashinfer_root / path).read_text()

    api_text = read(B12X_API)
    dispatch_text = read(B12X_DISPATCH)
    kernel_texts = {path: read(path) for path in B12X_KERNELS}

    b12x_fused_params = _function_parameters(api_text, "b12x_fused_moe")
    wrapper_params = _class_init_parameters(api_text, "B12xMoEWrapper")
    launch_params = _function_parameters(dispatch_text, "launch_sm120_moe")
    kernel_markers = {
        str(path): {
            "hasMarker": PATCH_MARKER in text,
            "hasClampExpression": "gate=min(gate, limit)" in text
            and "up=clamp(up, -limit, limit)" in text,
        }
        for path, text in kernel_texts.items()
    }
    validation = {
        "b12xFusedMoeHasSwigluLimit": "swiglu_limit" in b12x_fused_params,
        "b12xWrapperHasSwigluLimit": "swiglu_limit" in wrapper_params,
        "launchSm120MoeHasSwigluLimit": "swiglu_limit" in launch_params,
        "apiForwardsSwigluLimit": "swiglu_limit=swiglu_limit" in api_text,
        "wrapperForwardsSwigluLimit": "swiglu_limit=self.swiglu_limit" in api_text,
        "dispatchNormalizesSwigluLimit": "swiglu_limit = float(swiglu_limit or 0.0)"
        in dispatch_text,
        "kernelMarkers": kernel_markers,
    }
    validation["ok"] = (
        validation["b12xFusedMoeHasSwigluLimit"]
        and validation["b12xWrapperHasSwigluLimit"]
        and validation["launchSm120MoeHasSwigluLimit"]
        and validation["apiForwardsSwigluLimit"]
        and validation["wrapperForwardsSwigluLimit"]
        and validation["dispatchNormalizesSwigluLimit"]
        and all(
            marker["hasMarker"] and marker["hasClampExpression"]
            for marker in kernel_markers.values()
        )
    )
    return validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply or validate the Hydralisk FlashInfer B12x swiglu_limit overlay."
    )
    parser.add_argument("command", choices=["apply", "validate"])
    parser.add_argument("--flashinfer-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "apply":
        result = apply_overlay(args.flashinfer_root, dry_run=args.dry_run)
    else:
        result = validate_overlay(args.flashinfer_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("validation", result).get("ok") else 1


def _source_edits() -> list[SourceEdit]:
    return [
        SourceEdit(
            path=B12X_API,
            old=(
                '    quant_mode: Optional[str] = None,\n'
                '    source_format: str = "modelopt",\n'
                ") -> torch.Tensor:\n"
            ),
            new=(
                '    quant_mode: Optional[str] = None,\n'
                '    source_format: str = "modelopt",\n'
                "    swiglu_limit: Optional[float] = None,\n"
                ") -> torch.Tensor:\n"
            ),
            description="add swiglu_limit to b12x_fused_moe",
        ),
        SourceEdit(
            path=B12X_API,
            old=(
                "        quant_mode=quant_mode,\n"
                "        source_format=source_format,\n"
                "    )\n"
            ),
            new=(
                "        quant_mode=quant_mode,\n"
                "        source_format=source_format,\n"
                "        swiglu_limit=swiglu_limit,\n"
                "    )\n"
            ),
            description="forward swiglu_limit from b12x_fused_moe into launch_sm120_moe",
        ),
        SourceEdit(
            path=B12X_API,
            old=(
                '        quant_mode: Optional[str] = None,\n'
                '        source_format: str = "modelopt",\n'
                "    ):\n"
            ),
            new=(
                '        quant_mode: Optional[str] = None,\n'
                '        source_format: str = "modelopt",\n'
                "        swiglu_limit: Optional[float] = None,\n"
                "    ):\n"
            ),
            description="add swiglu_limit to B12xMoEWrapper",
        ),
        SourceEdit(
            path=B12X_API,
            old="        self.source_format = source_format\n",
            new=(
                "        self.source_format = source_format\n"
                "        self.swiglu_limit = swiglu_limit\n"
            ),
            description="store wrapper swiglu_limit",
        ),
        SourceEdit(
            path=B12X_API,
            old=(
                "            quant_mode=self.quant_mode,\n"
                "            source_format=self.source_format,\n"
                "            _workspace=workspace,\n"
            ),
            new=(
                "            quant_mode=self.quant_mode,\n"
                "            source_format=self.source_format,\n"
                "            swiglu_limit=self.swiglu_limit,\n"
                "            _workspace=workspace,\n"
            ),
            description="forward wrapper swiglu_limit into launch_sm120_moe",
        ),
        SourceEdit(
            path=B12X_DISPATCH,
            old=(
                '    quant_mode: str | None = None,\n'
                '    source_format: str = "modelopt",\n'
                "    _workspace=None,\n"
            ),
            new=(
                '    quant_mode: str | None = None,\n'
                '    source_format: str = "modelopt",\n'
                "    swiglu_limit: float | None = None,\n"
                "    _workspace=None,\n"
            ),
            description="add swiglu_limit to launch_sm120_moe",
        ),
        SourceEdit(
            path=B12X_DISPATCH,
            old=(
                "    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)\n"
                "    source_format = _normalize_source_format_for_quant_mode(source_format, quant_mode)\n"
                "    activation_precision = _activation_precision_from_quant_mode(quant_mode)\n"
            ),
            new=(
                "    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)\n"
                "    source_format = _normalize_source_format_for_quant_mode(source_format, quant_mode)\n"
                "    activation_precision = _activation_precision_from_quant_mode(quant_mode)\n"
                "    swiglu_limit = float(swiglu_limit or 0.0)\n"
                "    # HYDRALISK_B12X_SWIGLU_CLAMP_PATCH_POINT: launch accepts the\n"
                "    # DeepSeek/vLLM clamp value; the kernel overlay below marks the\n"
                "    # fused activation sites that must consume it in the G4 compile step.\n"
            ),
            description="normalize swiglu_limit in launch_sm120_moe",
        ),
        *[
            SourceEdit(
                path=kernel,
                old=(
                    "                                        g = alpha_value * gate_slice[elem_idx]\n"
                    "                                        u = alpha_value * up_slice[elem_idx]\n"
                    "                                        sigmoid_g = cute.arch.rcp_approx(\n"
                ),
                new=(
                    "                                        g = alpha_value * gate_slice[elem_idx]\n"
                    "                                        u = alpha_value * up_slice[elem_idx]\n"
                    + PATCH_COMMENT
                    + "                                        sigmoid_g = cute.arch.rcp_approx(\n"
                ),
                description=f"mark NVFP4 gated activation clamp site in {kernel.name}",
                count=0,
            )
            for kernel in B12X_KERNELS[:2]
        ],
        SourceEdit(
            path=B12X_KERNELS[2],
            old=(
                "                                            g = alpha_value * gate_slice[elem_idx]\n"
                "                                            u = alpha_value * up_slice[elem_idx]\n"
                "                                            sigmoid_g = cute.arch.rcp_approx(\n"
            ),
            new=(
                "                                            g = alpha_value * gate_slice[elem_idx]\n"
                "                                            u = alpha_value * up_slice[elem_idx]\n"
                + DYNAMIC_PATCH_COMMENT
                + "                                            sigmoid_g = cute.arch.rcp_approx(\n"
            ),
            description="mark NVFP4 gated activation clamp site in moe_dynamic_kernel.py",
            count=0,
        ),
        SourceEdit(
            path=B12X_KERNELS[3],
            old=(
                "                up = fc1_bf16_flat[base + Int32(self.intermediate_size) + col].to(\n"
                "                    cutlass.Float32\n"
                "                )\n"
                "                silu = gate / (\n"
            ),
            new=(
                "                up = fc1_bf16_flat[base + Int32(self.intermediate_size) + col].to(\n"
                "                    cutlass.Float32\n"
                "                )\n"
                + W4A16_PATCH_COMMENT
                + "                silu = gate / (\n"
            ),
            description="mark W4A16 fused MoE gated activation clamp site",
        ),
        SourceEdit(
            path=B12X_KERNELS[3],
            old=(
                "                up = fc1_flat[base + Int32(self.intermediate_size) + col].to(\n"
                "                    cutlass.Float32\n"
                "                )\n"
                "                silu = gate / (\n"
            ),
            new=(
                "                up = fc1_flat[base + Int32(self.intermediate_size) + col].to(\n"
                "                    cutlass.Float32\n"
                "                )\n"
                + W4A16_PATCH_COMMENT
                + "                silu = gate / (\n"
            ),
            description="mark W4A16 standalone activation clamp site",
        ),
    ]


def _apply_edit_text(source: str, edit: SourceEdit) -> tuple[str | None, str]:
    if edit.new in source:
        return None, "already_applied"
    if edit.old not in source:
        return None, "missing"
    count = edit.count if edit.count > 0 else source.count(edit.old)
    return source.replace(edit.old, edit.new, count), "would_change"


def _function_parameters(source: str, function_name: str) -> list[str]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                return _argument_names(node.args)
    return []


def _class_init_parameters(source: str, class_name: str) -> list[str]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name == "__init__":
                        return _argument_names(child.args)
    return []


def _argument_names(arguments: ast.arguments) -> list[str]:
    positional = [arg.arg for arg in arguments.posonlyargs + arguments.args]
    keyword_only = [arg.arg for arg in arguments.kwonlyargs]
    return positional + keyword_only


if __name__ == "__main__":
    raise SystemExit(main())
