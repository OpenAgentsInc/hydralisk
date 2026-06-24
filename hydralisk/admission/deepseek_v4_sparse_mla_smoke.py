from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

from hydralisk.admission.deepseek_v4_sparse_mla import reference_sparse_mla_decode


SCHEMA = "hydralisk.deepseek-v4.sparse-mla-fallback-smoke.v1"


def build_issue52_inputs(
    *,
    query_count: int = 1,
    heads: int = 64,
    dim: int = 512,
    page_size: int = 256,
    pages: int = 1,
    sparse_capacity: int = 128,
    seq_len: int = 128,
) -> dict[str, Any]:
    if query_count <= 0:
        raise ValueError("query_count must be positive")
    if heads <= 0:
        raise ValueError("heads must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if pages <= 0:
        raise ValueError("pages must be positive")
    if sparse_capacity < 0:
        raise ValueError("sparse_capacity must be non-negative")
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")

    query = [
        [
            _vector(seed=100_000 + query_idx * 1_000 + head_idx, dim=dim, scale=19.0)
            for head_idx in range(heads)
        ]
        for query_idx in range(query_count)
    ]
    swa_kv_cache = [
        [
            [
                _vector(seed=200_000 + page_idx * page_size + token_idx, dim=dim, scale=23.0)
                for token_idx in range(page_size)
            ]
        ]
        for page_idx in range(pages)
    ]
    compressed_kv_cache = [
        [
            [
                _vector(seed=300_000 + page_idx * page_size + token_idx, dim=dim, scale=29.0)
                for token_idx in range(page_size)
            ]
        ]
        for page_idx in range(pages)
    ]
    sparse_indices = [list(range(sparse_capacity)) for _ in range(query_count)]
    sparse_topk_lens = [sparse_capacity for _ in range(query_count)]
    seq_lens = [seq_len for _ in range(query_count)]

    return {
        "query": query,
        "swa_kv_cache": swa_kv_cache,
        "compressed_kv_cache": compressed_kv_cache,
        "sparse_indices": sparse_indices,
        "sparse_topk_lens": sparse_topk_lens,
        "seq_lens": seq_lens,
        "kv_layout": "HND",
        "sliding_window_tokens": min(seq_len, sparse_capacity),
        "include_swa_window": True,
        "shape": {
            "query": [query_count, heads, dim],
            "swaKvCache": [pages, 1, page_size, dim],
            "compressedKvCache": [pages, 1, page_size, dim],
            "sparseIndices": [query_count, sparse_capacity],
            "sparseTopkLens": [query_count],
            "seqLens": [query_count],
            "kvLayout": "HND",
        },
    }


def run_smoke(
    *,
    query_count: int = 1,
    heads: int = 64,
    dim: int = 512,
    page_size: int = 256,
    pages: int = 1,
    sparse_capacity: int = 128,
    seq_len: int = 128,
    generated_at: str | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now(UTC).replace(microsecond=0).isoformat()
    inputs = build_issue52_inputs(
        query_count=query_count,
        heads=heads,
        dim=dim,
        page_size=page_size,
        pages=pages,
        sparse_capacity=sparse_capacity,
        seq_len=seq_len,
    )

    started = perf_counter()
    result = reference_sparse_mla_decode(
        query=inputs["query"],
        swa_kv_cache=inputs["swa_kv_cache"],
        compressed_kv_cache=inputs["compressed_kv_cache"],
        sparse_indices=inputs["sparse_indices"],
        sparse_topk_lens=inputs["sparse_topk_lens"],
        seq_lens=inputs["seq_lens"],
        kv_layout=inputs["kv_layout"],
        sliding_window_tokens=inputs["sliding_window_tokens"],
        include_swa_window=inputs["include_swa_window"],
    )
    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    checksum = _checksum(result.output)
    nonzero = any(value != 0.0 for query in result.output for head in query for value in head)

    return {
        "schema": SCHEMA,
        "generatedAt": generated_at,
        "status": "ok",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "inputs": {
            **inputs["shape"],
            "dtypeContract": "bf16-compatible values represented as CPU floats",
            "synthetic": True,
            "loadsModelWeights": False,
            "containsPrompts": False,
            "containsResponses": False,
        },
        "result": {
            "outputShape": [len(result.output), len(result.output[0]), len(result.output[0][0])],
            "finite": True,
            "nonzero": nonzero,
            "checksum": checksum,
            "elapsedMs": elapsed_ms,
            "stats": asdict(result.stats),
        },
        "decision": {
            "containerFallbackContractReady": True,
            "nextStep": (
                "wire this fallback contract into vLLM's DeepSeek V4 SM120 "
                "attention path and rerun the synthetic container smoke before "
                "another full 8 x G4 model attempt"
            ),
        },
        "publicSafety": {
            "containsSecrets": False,
            "containsPrompts": False,
            "containsResponses": False,
            "containsWeights": False,
            "containsHiddenReasoning": False,
        },
    }


def render_markdown(smoke: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# DeepSeek V4 sparse MLA fallback smoke",
            "",
            f"Generated: `{smoke['generatedAt']}`",
            "",
            "## Summary",
            "",
            "This smoke runs the Hydralisk sparse MLA fallback contract on a "
            "public-safe synthetic tensor family matching the FlashInfer DSV4 "
            "FMHA repro shape. It does not load model weights, prompts, vLLM "
            "scheduling, or GPU kernels.",
            "",
            "## Inputs",
            "",
            "```json",
            json.dumps(smoke["inputs"], sort_keys=True),
            "```",
            "",
            "## Result",
            "",
            "```json",
            json.dumps(smoke["result"], sort_keys=True),
            "```",
            "",
            "## Decision",
            "",
            f"- Container fallback contract ready: `{smoke['decision']['containerFallbackContractReady']}`",
            f"- Next step: {smoke['decision']['nextStep']}",
            "",
            "## Public safety",
            "",
            "- Contains secrets: false",
            "- Contains private prompts: false",
            "- Contains private responses: false",
            "- Contains weights: false",
            "- Contains hidden reasoning: false",
            "",
        ]
    )


def write_smoke(smoke: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "deepseek-v4-sparse-mla-fallback-smoke.json"
    markdown_path = output_dir / "deepseek-v4-sparse-mla-fallback-smoke.md"
    json_path.write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(smoke))
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a public-safe DeepSeek V4 sparse MLA fallback smoke.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(".hydralisk/sparse-mla-smoke"))
    parser.add_argument("--stdout-json", action="store_true")
    parser.add_argument("--query-count", type=int, default=1)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--sparse-capacity", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args(argv)

    smoke = run_smoke(
        query_count=args.query_count,
        heads=args.heads,
        dim=args.dim,
        page_size=args.page_size,
        pages=args.pages,
        sparse_capacity=args.sparse_capacity,
        seq_len=args.seq_len,
    )
    if args.stdout_json:
        print(json.dumps(smoke, sort_keys=True))
    else:
        json_path, markdown_path = write_smoke(smoke, args.output_dir)
        print(f"Wrote {json_path}")
        print(f"Wrote {markdown_path}")
    return 0


def _vector(*, seed: int, dim: int, scale: float) -> list[float]:
    return [(((seed + idx * 37) % 41) - 20) / scale for idx in range(dim)]


def _checksum(output: list[list[list[float]]]) -> dict[str, float]:
    total = 0.0
    l1 = 0.0
    max_abs = 0.0
    count = 0
    for query in output:
        for head in query:
            for value in head:
                total += value
                abs_value = abs(value)
                l1 += abs_value
                max_abs = max(max_abs, abs_value)
                count += 1
    return {
        "sum": round(total, 6),
        "l1": round(l1, 6),
        "maxAbs": round(max_abs, 6),
        "count": count,
    }


if __name__ == "__main__":
    raise SystemExit(main())
