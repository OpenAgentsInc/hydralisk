from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


TARGET_RELATIVE_PATH = Path("vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py")
PATCH_SENTINEL = "HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK"


@dataclass(frozen=True)
class PatchResult:
    patched: bool
    already_patched: bool
    target: str
    inserted_import: bool
    inserted_helpers: bool
    decode_branch_patched: bool
    prefill_branch_patched: bool


HELPER_BLOCK = r'''

_HYDRALISK_SPARSE_MLA_FALLBACK_ENV = "HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK"


def _hydralisk_sparse_mla_fallback_enabled() -> bool:
    return os.getenv(_HYDRALISK_SPARSE_MLA_FALLBACK_ENV) == "1"


def _hydralisk_sparse_mla_fallback(
    *,
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    window_size: int,
) -> None:
    """Correctness-first SM120 sparse MLA fallback for Hydralisk probes.

    This is intentionally narrow and default-off. It exists to unblock the
    public-safe synthetic DeepSeek V4 DSV4 sparse MLA shape on RTX PRO 6000
    while FlashInfer ships no SM120 TRTLLM-gen FMHA cubins for the fast path.
    """

    if query.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{_HYDRALISK_SPARSE_MLA_FALLBACK_ENV}=1 only supports bf16 query"
        )
    if swa_kv_cache.dtype != torch.bfloat16 or compressed_kv_cache.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{_HYDRALISK_SPARSE_MLA_FALLBACK_ENV}=1 only supports bf16 KV caches"
        )
    if query.dim() != 3:
        raise RuntimeError("Hydralisk sparse MLA fallback expects query [tokens, heads, dim]")
    if swa_kv_cache.dim() != 4 or compressed_kv_cache.dim() != 4:
        raise RuntimeError(
            "Hydralisk sparse MLA fallback expects KV caches [pages, kv_heads, page, dim]"
        )
    if swa_kv_cache.shape != compressed_kv_cache.shape:
        raise RuntimeError("SWA and compressed KV caches must share shape")
    if sparse_indices.dim() != 2 or sparse_topk_lens.dim() != 1 or seq_lens.dim() != 1:
        raise RuntimeError("Hydralisk sparse MLA fallback expects 2D sparse indices and 1D lens")

    num_tokens, num_heads, dim = query.shape
    pages, kv_heads, page_size, kv_dim = swa_kv_cache.shape
    if dim != kv_dim:
        raise RuntimeError("query dim must match KV dim")
    if kv_heads not in (1, num_heads):
        raise RuntimeError("KV heads must broadcast from 1 or match query heads")
    if sparse_indices.shape[0] != num_tokens or sparse_topk_lens.shape[0] != num_tokens:
        raise RuntimeError("sparse metadata must have one row/value per query token")
    if seq_lens.numel() not in (1, num_tokens):
        raise RuntimeError(
            "Hydralisk sparse MLA fallback currently supports one-token decode "
            "or one sequence length per query token"
        )
    if out.shape[0] != num_tokens or out.shape[1] < num_heads or out.shape[-1] != dim:
        raise RuntimeError("output shape is incompatible with query shape")

    total_tokens = pages * page_size
    scale = dim ** -0.5
    out.zero_()

    for token_idx in range(num_tokens):
        seq_idx = 0 if seq_lens.numel() == 1 else token_idx
        seq_len = int(seq_lens[seq_idx].item())
        seq_len = max(0, min(seq_len, total_tokens))
        candidates: list[tuple[torch.Tensor, int]] = []

        if window_size > 0 and seq_len > 0:
            for position in range(max(0, seq_len - window_size), seq_len):
                candidates.append((swa_kv_cache, position))

        topk_len = min(
            max(0, int(sparse_topk_lens[token_idx].item())),
            sparse_indices.shape[1],
        )
        for raw_position in sparse_indices[token_idx, :topk_len]:
            position = int(raw_position.item())
            if 0 <= position < seq_len:
                candidates.append((compressed_kv_cache, position))

        if not candidates:
            continue

        for head_idx in range(num_heads):
            kv_head = 0 if kv_heads == 1 else head_idx
            keys = []
            for cache, position in candidates:
                keys.append(cache[position // page_size, kv_head, position % page_size])
            key_tensor = torch.stack(keys, dim=0).to(dtype=torch.float32)
            query_vec = query[token_idx, head_idx].to(dtype=torch.float32)
            weights = torch.softmax(torch.matmul(key_tensor, query_vec) * scale, dim=0)
            out[token_idx, head_idx] = torch.sum(
                weights[:, None] * key_tensor,
                dim=0,
            ).to(dtype=out.dtype)
'''


DECODE_CALL = '''            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=query[:num_decode_tokens],
                swa_kv_cache=swa_k_cache,
                workspace_buffer=workspace,
                sparse_indices=sparse_indices[:num_decode_tokens],
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens[:num_decode_tokens],
                seq_lens=seq_lens[:num_decodes],
                out=output[:num_decode_tokens],
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=self.attn_sink,
                cum_seq_lens_q=decode_cu,
                max_q_len=int(decode_lens_cpu.max().item()),
            )'''

DECODE_BRANCH = '''            if _hydralisk_sparse_mla_fallback_enabled():
                _hydralisk_sparse_mla_fallback(
                    query=query[:num_decode_tokens],
                    swa_kv_cache=swa_k_cache,
                    compressed_kv_cache=compressed_kv_cache,
                    sparse_indices=sparse_indices[:num_decode_tokens],
                    sparse_topk_lens=sparse_topk_lens[:num_decode_tokens],
                    seq_lens=seq_lens[:num_decodes],
                    out=output[:num_decode_tokens],
                    window_size=self.window_size,
                )
            else:
                flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                    query=query[:num_decode_tokens],
                    swa_kv_cache=swa_k_cache,
                    workspace_buffer=workspace,
                    sparse_indices=sparse_indices[:num_decode_tokens],
                    compressed_kv_cache=compressed_kv_cache,
                    sparse_topk_lens=sparse_topk_lens[:num_decode_tokens],
                    seq_lens=seq_lens[:num_decodes],
                    out=output[:num_decode_tokens],
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    sinks=self.attn_sink,
                    cum_seq_lens_q=decode_cu,
                    max_q_len=int(decode_lens_cpu.max().item()),
                )'''

PREFILL_CALL = '''            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                query=query[num_decode_tokens:num_tokens],
                swa_kv_cache=swa_k_cache,
                workspace_buffer=workspace,
                sparse_indices=sparse_indices[num_decode_tokens:num_tokens],
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens[num_decode_tokens:num_tokens],
                seq_lens=seq_lens[num_decodes:num_reqs],
                out=output[num_decode_tokens:num_tokens],
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=self.attn_sink,
                cum_seq_lens_q=prefill_cu,
                max_q_len=int(prefill_lens_cpu.max().item()),
            )'''

PREFILL_BRANCH = '''            if _hydralisk_sparse_mla_fallback_enabled():
                _hydralisk_sparse_mla_fallback(
                    query=query[num_decode_tokens:num_tokens],
                    swa_kv_cache=swa_k_cache,
                    compressed_kv_cache=compressed_kv_cache,
                    sparse_indices=sparse_indices[num_decode_tokens:num_tokens],
                    sparse_topk_lens=sparse_topk_lens[num_decode_tokens:num_tokens],
                    seq_lens=seq_lens[num_decodes:num_reqs],
                    out=output[num_decode_tokens:num_tokens],
                    window_size=self.window_size,
                )
            else:
                flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
                    query=query[num_decode_tokens:num_tokens],
                    swa_kv_cache=swa_k_cache,
                    workspace_buffer=workspace,
                    sparse_indices=sparse_indices[num_decode_tokens:num_tokens],
                    compressed_kv_cache=compressed_kv_cache,
                    sparse_topk_lens=sparse_topk_lens[num_decode_tokens:num_tokens],
                    seq_lens=seq_lens[num_decodes:num_reqs],
                    out=output[num_decode_tokens:num_tokens],
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                    sinks=self.attn_sink,
                    cum_seq_lens_q=prefill_cu,
                    max_q_len=int(prefill_lens_cpu.max().item()),
                )'''


def patch_source(source: str, *, target: str = str(TARGET_RELATIVE_PATH)) -> tuple[str, PatchResult]:
    if PATCH_SENTINEL in source:
        return source, PatchResult(
            patched=False,
            already_patched=True,
            target=target,
            inserted_import=False,
            inserted_helpers=False,
            decode_branch_patched=False,
            prefill_branch_patched=False,
        )

    patched = source
    inserted_import = False
    if "import os\n" not in patched:
        if "from typing import TYPE_CHECKING, ClassVar, cast\n" not in patched:
            raise ValueError("could not locate typing import anchor")
        patched = patched.replace(
            "from typing import TYPE_CHECKING, ClassVar, cast\n",
            "import os\nfrom typing import TYPE_CHECKING, ClassVar, cast\n",
            1,
        )
        inserted_import = True

    helper_anchor = "_flashinfer_dsv4_workspace_by_device: dict[torch.device, torch.Tensor] = {}\n"
    if helper_anchor not in patched:
        raise ValueError("could not locate FlashInfer workspace helper anchor")
    patched = patched.replace(helper_anchor, helper_anchor + HELPER_BLOCK, 1)

    if DECODE_CALL not in patched:
        raise ValueError("could not locate decode FlashInfer DSV4 call")
    patched = patched.replace(DECODE_CALL, DECODE_BRANCH, 1)

    if PREFILL_CALL not in patched:
        raise ValueError("could not locate prefill FlashInfer DSV4 call")
    patched = patched.replace(PREFILL_CALL, PREFILL_BRANCH, 1)

    return patched, PatchResult(
        patched=True,
        already_patched=False,
        target=target,
        inserted_import=inserted_import,
        inserted_helpers=True,
        decode_branch_patched=True,
        prefill_branch_patched=True,
    )


def patch_file(path: Path) -> PatchResult:
    source = path.read_text()
    patched, result = patch_source(source, target=str(path))
    if result.patched:
        path.write_text(patched)
    return result


def build_report(result: PatchResult, *, generated_at: str | None = None) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    return {
        "schema": "hydralisk.deepseek-v4.sparse-mla-vllm-patch.v1",
        "generatedAt": generated_at,
        "result": asdict(result),
        "envFlag": PATCH_SENTINEL,
        "defaultEnabled": False,
        "publicSafety": {
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
            "containsHiddenReasoning": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Patch vLLM DeepSeek V4 FlashInfer sparse MLA with Hydralisk fallback.",
    )
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    target = args.vllm_root / TARGET_RELATIVE_PATH
    if not target.exists():
        raise FileNotFoundError(target)
    if args.dry_run:
        _, result = patch_source(target.read_text(), target=str(target))
    else:
        result = patch_file(target)

    report = build_report(result)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
