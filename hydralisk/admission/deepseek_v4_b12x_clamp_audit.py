from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable
from datetime import UTC, datetime
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
VLLM_ACTIVATION = Path("vllm/model_executor/layers/activation.py")
VLLM_FUSED_MOE_UTILS = Path("vllm/model_executor/layers/fused_moe/utils.py")
VLLM_FUSED_BATCHED_MOE = Path(
    "vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py"
)
VLLM_FP8_UTILS = Path("vllm/model_executor/layers/quantization/utils/fp8_utils.py")


def build_audit(
    *,
    flashinfer_root: Path,
    vllm_root: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat()

    b12x_api = _read_source(flashinfer_root / B12X_API)
    b12x_dispatch = _read_source(flashinfer_root / B12X_DISPATCH)
    kernel_sources = [
        _read_source(flashinfer_root / kernel_path) for kernel_path in B12X_KERNELS
    ]

    vllm_activation = _read_source(vllm_root / VLLM_ACTIVATION)
    vllm_utils = _read_source(vllm_root / VLLM_FUSED_MOE_UTILS)
    vllm_batched = _read_source(vllm_root / VLLM_FUSED_BATCHED_MOE)
    vllm_fp8 = _read_source(vllm_root / VLLM_FP8_UTILS)

    b12x_fused_params = _function_parameters(b12x_api["text"], "b12x_fused_moe")
    b12x_wrapper_params = _class_init_parameters(
        b12x_api["text"],
        "B12xMoEWrapper",
    )
    launch_params = _function_parameters(b12x_dispatch["text"], "launch_sm120_moe")

    api_terms = _term_report(
        b12x_api["text"],
        [
            "b12x_fused_moe",
            "B12xMoEWrapper",
            "num_local_experts",
            "activation",
            "activation_precision",
            "quant_mode",
            "swiglu_limit",
            "gemm1_clamp_limit",
            "does not yet support Expert Parallelism",
        ],
    )
    dispatch_terms = _term_report(
        b12x_dispatch["text"],
        [
            "launch_sm120_moe",
            "launch_sm120_static_moe",
            "launch_sm120_micro_moe",
            "launch_sm120_dynamic_moe",
            "activation",
            "activation_precision",
            "quant_mode",
            "is_gated = activation == \"silu\"",
            "swiglu_limit",
            "gemm1_clamp_limit",
        ],
    )
    kernel_reports = [
        {
            "path": source["path"],
            "present": source["present"],
            "activationSignals": _term_report(
                source["text"],
                [
                    "SiLU(gate) * up",
                    "activation=\"silu\"",
                    "self.is_gated",
                    "gate =",
                    "up =",
                    "silu =",
                    "cute.math.exp(-gate",
                ],
            ),
            "clampSignals": _term_report(
                source["text"],
                ["swiglu_limit", "gemm1_clamp_limit", "clamp_limit"],
            ),
        }
        for source in kernel_sources
    ]

    vllm_terms = {
        "activation": _term_report(
            vllm_activation["text"],
            [
                "SiluAndMulWithClamp",
                "self.swiglu_limit",
                "torch.clamp(x[..., :d], max=self.swiglu_limit)",
                "torch.clamp(x[..., d:], min=-self.swiglu_limit, max=self.swiglu_limit)",
            ],
        ),
        "fusedMoeUtils": _term_report(
            vllm_utils["text"],
            [
                "def swiglu_limit_func",
                "torch.clamp(gate, max=swiglu_limit)",
                "torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)",
            ],
        ),
        "fusedBatchedMoe": _term_report(
            vllm_batched["text"],
            [
                "gemm1_clamp_limit = self.quant_config.gemm1_clamp_limit",
                "swiglu_limit_func(output, input, float(gemm1_clamp_limit))",
            ],
        ),
        "fp8Utils": _term_report(
            vllm_fp8["text"],
            [
                "clamp_limit",
                "tl.minimum(act_f32, clamp_limit)",
                "tl.clamp(mul_f32, -clamp_limit, clamp_limit)",
            ],
        ),
    }

    b12x_lacks_clamp_surface = not any(
        _has_parameter(params, ("swiglu_limit", "gemm1_clamp_limit"))
        for params in (b12x_fused_params, b12x_wrapper_params, launch_params)
    )
    b12x_kernel_lacks_clamp_terms = not any(
        report["clampSignals"]["presentTerms"] for report in kernel_reports
    )
    vllm_has_clamp_contract = all(
        section["requiredTermsPresent"] for section in vllm_terms.values()
    )

    return {
        "schema": "hydralisk.deepseek-v4.b12x-clamp-audit.v1",
        "generatedAt": generated_at,
        "publicSafety": {
            "loadsModelWeights": False,
            "requiresGpu": False,
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
        },
        "inputs": {
            "flashinferRoot": str(flashinfer_root),
            "vllmRoot": str(vllm_root),
        },
        "flashinferB12xApi": {
            "path": b12x_api["path"],
            "present": b12x_api["present"],
            "b12xFusedMoeParameters": b12x_fused_params,
            "b12xWrapperParameters": b12x_wrapper_params,
            "terms": api_terms,
            "hasSwiGluLimitParameter": _has_parameter(
                b12x_fused_params,
                ("swiglu_limit",),
            )
            or _has_parameter(b12x_wrapper_params, ("swiglu_limit",)),
            "hasGemm1ClampLimitParameter": _has_parameter(
                b12x_fused_params,
                ("gemm1_clamp_limit",),
            )
            or _has_parameter(b12x_wrapper_params, ("gemm1_clamp_limit",)),
            "hasNumLocalExpertsParameter": _has_parameter(
                b12x_fused_params,
                ("num_local_experts",),
            )
            and _has_parameter(b12x_wrapper_params, ("num_local_experts",)),
            "hasExpertParallelismRejection": api_terms["termPresent"][
                "does not yet support Expert Parallelism"
            ],
        },
        "flashinferB12xDispatch": {
            "path": b12x_dispatch["path"],
            "present": b12x_dispatch["present"],
            "launchSm120MoeParameters": launch_params,
            "terms": dispatch_terms,
            "hasSwiGluLimitParameter": _has_parameter(launch_params, ("swiglu_limit",)),
            "hasGemm1ClampLimitParameter": _has_parameter(
                launch_params,
                ("gemm1_clamp_limit",),
            ),
        },
        "flashinferB12xKernels": kernel_reports,
        "vllmClampContract": {
            "paths": {
                "activation": vllm_activation["path"],
                "fusedMoeUtils": vllm_utils["path"],
                "fusedBatchedMoe": vllm_batched["path"],
                "fp8Utils": vllm_fp8["path"],
            },
            "terms": vllm_terms,
            "hasClampContract": vllm_has_clamp_contract,
            "semantics": [
                "gate branch is clamped only above +limit",
                "up branch is clamped into [-limit, +limit]",
                "SwiGLU output is silu(gate) * up after clamping",
                "DeepSeek vLLM MoE path wires gemm1_clamp_limit into that contract",
            ],
        },
        "decision": {
            "status": (
                "b12x_clamp_missing_in_api_launch_and_kernel_terms"
                if b12x_lacks_clamp_surface and b12x_kernel_lacks_clamp_terms
                else "b12x_clamp_surface_needs_manual_review"
            ),
            "b12xLacksClampSurface": b12x_lacks_clamp_surface,
            "b12xKernelLacksClampTerms": b12x_kernel_lacks_clamp_terms,
            "vllmClampContractPresent": vllm_has_clamp_contract,
            "nextStep": "patch FlashInfer B12x SM120 API, dispatch, and kernel activation paths for DeepSeek swiglu_limit=10.0 before another full-model G4 retry",
        },
        "patchPlan": [
            {
                "step": "api_surface",
                "path": str(B12X_API),
                "work": "add a swiglu_limit or gemm1_clamp_limit keyword to b12x_fused_moe and B12xMoEWrapper, defaulting to None/0 for compatibility",
            },
            {
                "step": "dispatch_threading",
                "path": str(B12X_DISPATCH),
                "work": "thread the clamp value through launch_sm120_moe and backend launch helpers without changing activation='silu' selection",
            },
            {
                "step": "nvfp4_activation",
                "paths": [str(path) for path in B12X_KERNELS[:3]],
                "work": "apply vLLM-compatible clamp at the fused SwiGLU gate/up activation point before FP4 re-quant and FC2",
            },
            {
                "step": "w4a16_activation",
                "path": str(B12X_KERNELS[3]),
                "work": "apply the same clamp in W4A16 gated activation helpers if that backend remains in the fallback matrix",
            },
            {
                "step": "correctness_gate",
                "path": "hydralisk/admission/deepseek_v4_moe.py",
                "work": "compare a tiny nonzero B12x local-shard GPU output against Hydralisk's reference clamp/remap fixture before any model-weight retry",
            },
        ],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    decision = audit["decision"]
    api = audit["flashinferB12xApi"]
    dispatch = audit["flashinferB12xDispatch"]
    vllm = audit["vllmClampContract"]
    kernel_rows = "\n".join(
        "- `{path}`: activation signals={activation}; clamp signals={clamp}".format(
            path=kernel["path"],
            activation=", ".join(kernel["activationSignals"]["presentTerms"]) or "none",
            clamp=", ".join(kernel["clampSignals"]["presentTerms"]) or "none",
        )
        for kernel in audit["flashinferB12xKernels"]
    )
    patch_rows = "\n".join(
        "- `{step}`: {work}".format(step=item["step"], work=item["work"])
        for item in audit["patchPlan"]
    )
    provider_note = (
        "The pasted provider inventory helps by confirming the expected stock "
        "recipe shape for DeepSeek-V4-Flash: vLLM 0.20+, DeepGEMM, FP8 KV, "
        "block size 256, tensor parallel set to local GPU count, expert "
        "parallel enabled, and H200/B200/GB-class hardware as verified targets. "
        "It does not remove the G4/SM120 B12x clamp blocker."
    )

    lines = [
        "# DeepSeek B12x clamp patch-point audit",
        "",
        f"Generated: `{audit['generatedAt']}`",
        "",
        "This source audit does not load model weights, prompts, responses, or GPU kernels.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- B12x lacks API/launch clamp surface: `{decision['b12xLacksClampSurface']}`",
        f"- B12x kernel files lack clamp terms: `{decision['b12xKernelLacksClampTerms']}`",
        f"- vLLM clamp contract present: `{decision['vllmClampContractPresent']}`",
        f"- Next step: {decision['nextStep']}",
        "",
        "## Provider Inventory Signal",
        "",
        provider_note,
        "",
        "## FlashInfer B12x Surface",
        "",
        f"- API file: `{api['path']}`",
        f"- `b12x_fused_moe` params: `{', '.join(api['b12xFusedMoeParameters'])}`",
        f"- `B12xMoEWrapper.__init__` params: `{', '.join(api['b12xWrapperParameters'])}`",
        f"- Has `num_local_experts`: `{api['hasNumLocalExpertsParameter']}`",
        f"- Has `swiglu_limit`: `{api['hasSwiGluLimitParameter']}`",
        f"- Has `gemm1_clamp_limit`: `{api['hasGemm1ClampLimitParameter']}`",
        f"- Has current direct EP rejection: `{api['hasExpertParallelismRejection']}`",
        "",
        "## SM120 Dispatch",
        "",
        f"- Dispatch file: `{dispatch['path']}`",
        f"- `launch_sm120_moe` params: `{', '.join(dispatch['launchSm120MoeParameters'])}`",
        f"- Has `swiglu_limit`: `{dispatch['hasSwiGluLimitParameter']}`",
        f"- Has `gemm1_clamp_limit`: `{dispatch['hasGemm1ClampLimitParameter']}`",
        "",
        "## Kernel Activation Patch Points",
        "",
        kernel_rows,
        "",
        "## vLLM Clamp Contract",
        "",
        f"- Contract present: `{vllm['hasClampContract']}`",
        "- Semantics: gate clamp is one-sided at `+limit`; up clamp is symmetric; output is `silu(gate) * up`.",
        "- vLLM MoE wiring uses `gemm1_clamp_limit` to call `swiglu_limit_func` for SILU activation.",
        "",
        "## Patch Plan",
        "",
        patch_rows,
        "",
        "## Public Safety",
        "",
        "- No secrets.",
        "- No model weights.",
        "- No prompts or model responses.",
        "- No private benchmark/profiler output.",
        "",
    ]
    return "\n".join(lines)


def write_audit(audit: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "b12x-clamp-audit.json"
    markdown_path = output_dir / "b12x-clamp-audit.md"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(audit))
    return json_path, markdown_path


def find_workspace_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "projects" / "repos").exists():
            return candidate
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit FlashInfer B12x DeepSeek SwiGLU clamp patch points."
    )
    workspace_root = find_workspace_root()
    parser.add_argument(
        "--flashinfer-root",
        type=Path,
        default=workspace_root / "projects" / "repos" / "flashinfer",
    )
    parser.add_argument(
        "--vllm-root",
        type=Path,
        default=workspace_root / "projects" / "repos" / "vllm",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".hydralisk")
        / f"b12x-clamp-audit-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
    )
    args = parser.parse_args(argv)

    audit = build_audit(
        flashinfer_root=args.flashinfer_root,
        vllm_root=args.vllm_root,
    )
    json_path, markdown_path = write_audit(audit, args.output_dir)
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}, indent=2))
    return 0


def _read_source(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "present": False, "text": ""}
    return {"path": str(path), "present": True, "text": path.read_text()}


def _function_parameters(source: str, function_name: str) -> list[str]:
    for node in _parse_nodes(source):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                return _argument_names(node.args)
    return []


def _class_init_parameters(source: str, class_name: str) -> list[str]:
    for node in _parse_nodes(source):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name == "__init__":
                        return _argument_names(child.args)
    return []


def _parse_nodes(source: str) -> Iterable[ast.AST]:
    if not source:
        return []
    try:
        return ast.walk(ast.parse(source))
    except SyntaxError:
        return []


def _argument_names(arguments: ast.arguments) -> list[str]:
    positional = [arg.arg for arg in arguments.posonlyargs + arguments.args]
    keyword_only = [arg.arg for arg in arguments.kwonlyargs]
    variadic = []
    if arguments.vararg is not None:
        variadic.append("*" + arguments.vararg.arg)
    if arguments.kwarg is not None:
        variadic.append("**" + arguments.kwarg.arg)
    return positional + variadic + keyword_only


def _has_parameter(parameters: list[str], names: tuple[str, ...]) -> bool:
    return any(name in parameters for name in names)


def _term_report(source: str, terms: list[str]) -> dict[str, Any]:
    term_present = {term: term in source for term in terms}
    present_terms = [term for term, present in term_present.items() if present]
    missing_terms = [term for term, present in term_present.items() if not present]
    return {
        "termPresent": term_present,
        "presentTerms": present_terms,
        "missingTerms": missing_terms,
        "requiredTermsPresent": not missing_terms,
    }


if __name__ == "__main__":
    raise SystemExit(main())
