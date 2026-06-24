from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CUDA_MODEL = Path("vllm/models/deepseek_v4/nvidia/model.py")
FLASHMLA_ATTENTION = Path("vllm/models/deepseek_v4/nvidia/flashmla.py")
FLASHINFER_ATTENTION = Path("vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py")
FLASHMLA_OPS = Path("vllm/v1/attention/ops/flashmla.py")
ATTENTION_CONFIG = Path("vllm/config/attention.py")
ARG_UTILS = Path("vllm/engine/arg_utils.py")
BACKEND_REGISTRY = Path("vllm/v1/attention/backends/registry.py")


def build_audit(*, vllm_root: Path, generated_at: str | None = None) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat()

    cuda_model = _read_source(vllm_root / CUDA_MODEL)
    flashmla = _read_source(vllm_root / FLASHMLA_ATTENTION)
    flashinfer = _read_source(vllm_root / FLASHINFER_ATTENTION)
    flashmla_ops = _read_source(vllm_root / FLASHMLA_OPS)
    attention_config = _read_source(vllm_root / ATTENTION_CONFIG)
    arg_utils = _read_source(vllm_root / ARG_UTILS)
    registry = _read_source(vllm_root / BACKEND_REGISTRY)

    selector_terms = _term_report(
        cuda_model["text"],
        [
            "_select_dsv4_attn_cls",
            "AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4",
            "return DeepseekV4FlashInferMLAAttention",
            "return DeepseekV4FlashMLAAttention",
        ],
    )
    flashmla_terms = _term_report(
        flashmla["text"],
        [
            "class DeepseekV4FlashMLAAttention",
            "class DeepseekV4FlashMLASparseBackend",
            "FLASHMLA_SPARSE_DSV4",
            "use_flashmla_fp8_layout",
            "def _forward_prefill",
            "flash_mla_sparse_fwd(",
            "flash_mla_with_kvcache(",
        ],
    )
    flashinfer_terms = _term_report(
        flashinfer["text"],
        [
            "class DeepseekV4FlashInferMLAAttention",
            "class DeepseekV4FlashInferMLASparseBackend",
            "FLASHINFER_MLA_SPARSE_DSV4",
            "use_flashmla_fp8_layout: ClassVar[bool] = False",
            "flashinfer_trtllm_batch_decode_sparse_mla_dsv4",
            "build_flashinfer_mixed_sparse_indices",
            "def _forward(",
        ],
    )
    ops_terms = _term_report(
        flashmla_ops["text"],
        [
            "def is_flashmla_sparse_supported",
            "is_device_capability_family(90)",
            "is_device_capability_family(100)",
            "is_device_capability_family(120)",
            "flash_mla_sparse_fwd",
        ],
    )
    config_terms = _term_report(
        attention_config["text"],
        [
            "class AttentionConfig",
            "backend: AttentionBackendEnum | None",
            "use_fp4_indexer_cache: bool = False",
            "validate_backend_before",
        ],
    )
    arg_terms = _term_report(
        arg_utils["text"],
        ["--attention-config", "attention_config"],
    )
    registry_terms = _term_report(
        registry["text"],
        [
            "FLASHMLA_SPARSE_DSV4",
            "FLASHINFER_MLA_SPARSE_DSV4",
            "DeepseekV4FlashInferMLASparseBackend",
            "DeepseekV4FlashMLASparseBackend",
        ],
    )

    selector_supports_flashinfer = all(
        selector_terms["termPresent"][term]
        for term in (
            "_select_dsv4_attn_cls",
            "AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4",
            "return DeepseekV4FlashInferMLAAttention",
        )
    )
    flashmla_prefill_calls_sparse = all(
        flashmla_terms["termPresent"][term]
        for term in ("def _forward_prefill", "flash_mla_sparse_fwd(")
    )
    flashmla_python_guard_lacks_sm120 = (
        ops_terms["termPresent"]["def is_flashmla_sparse_supported"]
        and ops_terms["termPresent"]["is_device_capability_family(90)"]
        and ops_terms["termPresent"]["is_device_capability_family(100)"]
        and not ops_terms["termPresent"]["is_device_capability_family(120)"]
    )
    flashinfer_avoids_flashmla_sparse_call = (
        flashinfer_terms["termPresent"][
            "flashinfer_trtllm_batch_decode_sparse_mla_dsv4"
        ]
        and "flash_mla_sparse_fwd(" not in flashinfer["text"]
    )
    cli_can_select_attention_backend = (
        config_terms["termPresent"]["backend: AttentionBackendEnum | None"]
        and config_terms["termPresent"]["validate_backend_before"]
        and arg_terms["termPresent"]["--attention-config"]
        and registry_terms["termPresent"]["FLASHINFER_MLA_SPARSE_DSV4"]
    )

    existing_flashinfer_probe_ready = all(
        [
            selector_supports_flashinfer,
            flashmla_prefill_calls_sparse,
            flashmla_python_guard_lacks_sm120,
            flashinfer_avoids_flashmla_sparse_call,
            cli_can_select_attention_backend,
        ]
    )

    return {
        "schema": "hydralisk.deepseek-v4.flashmla-sparse-audit.v1",
        "generatedAt": generated_at,
        "publicSafety": {
            "loadsModelWeights": False,
            "requiresGpu": False,
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
        },
        "inputs": {"vllmRoot": str(vllm_root)},
        "sourceFiles": {
            "cudaModel": cuda_model["path"],
            "flashmlaAttention": flashmla["path"],
            "flashinferAttention": flashinfer["path"],
            "flashmlaOps": flashmla_ops["path"],
            "attentionConfig": attention_config["path"],
            "argUtils": arg_utils["path"],
            "backendRegistry": registry["path"],
        },
        "flashmlaPath": {
            "terms": flashmla_terms,
            "prefillCallsFlashMlaSparseFwd": flashmla_prefill_calls_sparse,
            "pythonSparseSupportGuardLacksSm120": flashmla_python_guard_lacks_sm120,
            "runtimeBlocker": (
                "flash_mla_sparse_fwd rejects SM120 at first generation even "
                "after eager mode reaches /v1/models"
            ),
        },
        "flashinferPath": {
            "terms": flashinfer_terms,
            "avoidsFlashMlaSparseFwd": flashinfer_avoids_flashmla_sparse_call,
            "usesTrtllmSparseMlaLauncher": flashinfer_terms["termPresent"][
                "flashinfer_trtllm_batch_decode_sparse_mla_dsv4"
            ],
            "usesPlainKvLayout": flashinfer_terms["termPresent"][
                "use_flashmla_fp8_layout: ClassVar[bool] = False"
            ],
        },
        "selector": {
            "terms": selector_terms,
            "supportsExplicitFlashinferBackend": selector_supports_flashinfer,
        },
        "configuration": {
            "attentionConfigTerms": config_terms,
            "argTerms": arg_terms,
            "registryTerms": registry_terms,
            "cliCanSelectAttentionBackend": cli_can_select_attention_backend,
            "recommendedBackend": "FLASHINFER_MLA_SPARSE_DSV4",
            "recommendedWrapperEnv": "VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4",
            "recommendedCliFragment": "--attention-config '{\"backend\":\"FLASHINFER_MLA_SPARSE_DSV4\"}'",
        },
        "decision": {
            "status": (
                "existing_flashinfer_sparse_backend_is_next_sm120_probe"
                if existing_flashinfer_probe_ready
                else "flashmla_sparse_fallback_needs_manual_review"
            ),
            "existingFlashinferProbeReady": existing_flashinfer_probe_ready,
            "nextStep": (
                "rerun the eager B12x/o_proj-fallback G4 smoke with "
                "VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4 before "
                "writing a new FlashMLA sparse-prefill kernel fallback"
            ),
        },
        "fallbackPlan": [
            {
                "step": "explicit_backend_probe",
                "work": "set attention_config.backend to FLASHINFER_MLA_SPARSE_DSV4 through the Hydralisk wrapper and rerun the tiny generation smoke",
            },
            {
                "step": "flashinfer_result_gate",
                "work": "if FlashInfer TRTLLM sparse MLA reaches generation, keep it as the G4 prefill/decode path and collect latency/quality receipts",
            },
            {
                "step": "flashinfer_failure_gate",
                "work": "if FlashInfer fails, capture its exact kernel/runtime blocker before patching any FlashMLA SM120 architecture guard",
            },
            {
                "step": "last_resort_fallback",
                "work": "only implement a correctness-first Python/Triton prefill fallback after proving no existing vLLM DeepSeek V4 backend works on SM120",
            },
        ],
    }


def render_markdown(audit: dict[str, Any]) -> str:
    decision = audit["decision"]
    config = audit["configuration"]
    fallback_rows = "\n".join(
        f"- `{item['step']}`: {item['work']}" for item in audit["fallbackPlan"]
    )
    lines = [
        "# DeepSeek FlashMLA sparse-prefill SM120 audit",
        "",
        f"Generated: `{audit['generatedAt']}`",
        "",
        "This source audit does not load model weights, prompts, responses, or GPU kernels.",
        "",
        "## Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Existing FlashInfer probe ready: `{decision['existingFlashinferProbeReady']}`",
        f"- Next step: {decision['nextStep']}",
        "",
        "## FlashMLA Path",
        "",
        f"- Prefill calls `flash_mla_sparse_fwd`: `{audit['flashmlaPath']['prefillCallsFlashMlaSparseFwd']}`",
        f"- Python sparse support guard lacks SM120: `{audit['flashmlaPath']['pythonSparseSupportGuardLacksSm120']}`",
        f"- Runtime blocker: {audit['flashmlaPath']['runtimeBlocker']}",
        "",
        "## Existing FlashInfer Path",
        "",
        f"- Avoids `flash_mla_sparse_fwd`: `{audit['flashinferPath']['avoidsFlashMlaSparseFwd']}`",
        f"- Uses TRTLLM sparse MLA launcher: `{audit['flashinferPath']['usesTrtllmSparseMlaLauncher']}`",
        f"- Uses plain KV layout instead of `fp8_ds_mla`: `{audit['flashinferPath']['usesPlainKvLayout']}`",
        "",
        "## Configuration",
        "",
        f"- CLI can select attention backend: `{config['cliCanSelectAttentionBackend']}`",
        f"- Recommended backend: `{config['recommendedBackend']}`",
        f"- Recommended wrapper env: `{config['recommendedWrapperEnv']}`",
        f"- Recommended CLI fragment: `{config['recommendedCliFragment']}`",
        "",
        "## Fallback Plan",
        "",
        fallback_rows,
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
    json_path = output_dir / "flashmla-sparse-audit.json"
    markdown_path = output_dir / "flashmla-sparse-audit.md"
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
        description="Audit DeepSeek V4 FlashMLA sparse-prefill SM120 fallback points."
    )
    workspace_root = find_workspace_root()
    parser.add_argument(
        "--vllm-root",
        type=Path,
        default=workspace_root / "projects" / "repos" / "vllm",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".hydralisk")
        / f"flashmla-sparse-audit-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
    )
    args = parser.parse_args(argv)

    audit = build_audit(vllm_root=args.vllm_root)
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
