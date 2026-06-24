from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite, sqrt


Vector = Sequence[float]
Tensor3D = Sequence[Sequence[Vector]]
Tensor4D = Sequence[Sequence[Sequence[Vector]]]


@dataclass(frozen=True)
class SparseMlaFallbackStats:
    query_count: int
    head_count: int
    dim: int
    page_size: int
    sliding_window_tokens: int
    swa_route_count: int
    sparse_route_count: int
    masked_sparse_route_count: int
    empty_route_count: int


@dataclass(frozen=True)
class SparseMlaFallbackResult:
    output: list[list[list[float]]]
    stats: SparseMlaFallbackStats


@dataclass(frozen=True)
class _Candidate:
    source: str
    position: int


def reference_sparse_mla_decode(
    *,
    query: Tensor3D,
    swa_kv_cache: Tensor4D,
    compressed_kv_cache: Tensor4D,
    sparse_indices: Sequence[Sequence[int]],
    sparse_topk_lens: Sequence[int],
    seq_lens: Sequence[int],
    kv_layout: str = "HND",
    sliding_window_tokens: int | None = None,
    include_swa_window: bool = True,
) -> SparseMlaFallbackResult:
    """Correctness-first sparse MLA decode reference for the SM120 fallback.

    This intentionally models the public-safe issue #52 tensor family rather
    than FlashInfer internals: one or more decode queries, head-major HND KV
    caches, sparse index/top-k masking, sequence-length truncation, and stable
    softmax attention. The same cached vector is used as key and value because
    the DSV4 launcher receives compressed KV tensors rather than separate K/V
    payloads at this boundary.
    """

    if kv_layout != "HND":
        raise ValueError("only HND KV layout is supported by the fallback contract")

    query_shape = _shape3(query, "query")
    swa_shape = _shape4(swa_kv_cache, "swa_kv_cache")
    compressed_shape = _shape4(compressed_kv_cache, "compressed_kv_cache")
    if swa_shape != compressed_shape:
        raise ValueError("SWA and compressed KV caches must share shape")

    query_count, head_count, dim = query_shape
    pages, kv_heads, page_size, kv_dim = swa_shape
    if dim != kv_dim:
        raise ValueError("query and KV cache dimensions must match")
    if kv_heads not in (1, head_count):
        raise ValueError("KV heads must either broadcast from 1 or match query heads")
    if len(sparse_indices) != query_count:
        raise ValueError("sparse_indices must have one row per query")
    if len(sparse_topk_lens) != query_count:
        raise ValueError("sparse_topk_lens must have one value per query")
    if len(seq_lens) not in (1, query_count):
        raise ValueError("seq_lens must contain either one shared value or one per query")

    sliding_window_tokens = page_size if sliding_window_tokens is None else sliding_window_tokens
    if sliding_window_tokens < 0:
        raise ValueError("sliding_window_tokens must be non-negative")

    output: list[list[list[float]]] = [
        [[0.0 for _ in range(dim)] for _ in range(head_count)]
        for _ in range(query_count)
    ]
    total_tokens = pages * page_size
    swa_route_count = 0
    sparse_route_count = 0
    masked_sparse_route_count = 0
    empty_route_count = 0

    for query_idx in range(query_count):
        seq_len = _seq_len_at(seq_lens, query_idx)
        seq_len = max(0, min(seq_len, total_tokens))
        candidates: list[_Candidate] = []

        if include_swa_window and sliding_window_tokens > 0 and seq_len > 0:
            window_start = max(0, seq_len - sliding_window_tokens)
            for position in range(window_start, seq_len):
                candidates.append(_Candidate(source="swa", position=position))
                swa_route_count += 1

        topk_len = min(max(0, int(sparse_topk_lens[query_idx])), len(sparse_indices[query_idx]))
        for position in sparse_indices[query_idx][:topk_len]:
            if position < 0 or position >= seq_len:
                masked_sparse_route_count += 1
                continue
            candidates.append(_Candidate(source="compressed", position=int(position)))
            sparse_route_count += 1

        for head_idx in range(head_count):
            if not candidates:
                empty_route_count += 1
                continue

            kv_head = 0 if kv_heads == 1 else head_idx
            query_vec = _as_float_vector(query[query_idx][head_idx], dim, "query vector")
            keys = [
                _cache_vector(
                    swa_kv_cache if candidate.source == "swa" else compressed_kv_cache,
                    position=candidate.position,
                    kv_head=kv_head,
                    page_size=page_size,
                    dim=dim,
                )
                for candidate in candidates
            ]
            logits = [_dot(query_vec, key) / sqrt(dim) for key in keys]
            weights = _softmax(logits)
            output[query_idx][head_idx] = _weighted_sum(weights, keys, dim)

    _ensure_finite(output)
    return SparseMlaFallbackResult(
        output=output,
        stats=SparseMlaFallbackStats(
            query_count=query_count,
            head_count=head_count,
            dim=dim,
            page_size=page_size,
            sliding_window_tokens=sliding_window_tokens,
            swa_route_count=swa_route_count,
            sparse_route_count=sparse_route_count,
            masked_sparse_route_count=masked_sparse_route_count,
            empty_route_count=empty_route_count,
        ),
    )


def _shape3(tensor: Tensor3D, name: str) -> tuple[int, int, int]:
    if not tensor:
        raise ValueError(f"{name} must not be empty")
    outer = len(tensor)
    middle = len(tensor[0])
    if middle == 0:
        raise ValueError(f"{name} middle dimension must not be empty")
    inner = len(tensor[0][0])
    if inner == 0:
        raise ValueError(f"{name} inner dimension must not be empty")
    for row in tensor:
        if len(row) != middle:
            raise ValueError(f"{name} must be rectangular")
        for vector in row:
            if len(vector) != inner:
                raise ValueError(f"{name} must be rectangular")
    return outer, middle, inner


def _shape4(tensor: Tensor4D, name: str) -> tuple[int, int, int, int]:
    if not tensor:
        raise ValueError(f"{name} must not be empty")
    pages = len(tensor)
    heads = len(tensor[0])
    if heads == 0:
        raise ValueError(f"{name} head dimension must not be empty")
    page_size = len(tensor[0][0])
    if page_size == 0:
        raise ValueError(f"{name} page dimension must not be empty")
    dim = len(tensor[0][0][0])
    if dim == 0:
        raise ValueError(f"{name} vector dimension must not be empty")
    for page in tensor:
        if len(page) != heads:
            raise ValueError(f"{name} must be rectangular")
        for head in page:
            if len(head) != page_size:
                raise ValueError(f"{name} must be rectangular")
            for vector in head:
                if len(vector) != dim:
                    raise ValueError(f"{name} must be rectangular")
    return pages, heads, page_size, dim


def _seq_len_at(seq_lens: Sequence[int], query_idx: int) -> int:
    if len(seq_lens) == 1:
        return int(seq_lens[0])
    return int(seq_lens[query_idx])


def _cache_vector(
    cache: Tensor4D,
    *,
    position: int,
    kv_head: int,
    page_size: int,
    dim: int,
) -> list[float]:
    page_idx = position // page_size
    offset = position % page_size
    return _as_float_vector(cache[page_idx][kv_head][offset], dim, "cache vector")


def _as_float_vector(vector: Vector, expected_dim: int, name: str) -> list[float]:
    if len(vector) != expected_dim:
        raise ValueError(f"{name} width must match expected dimension")
    return [float(value) for value in vector]


def _dot(left: Vector, right: Vector) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    numerators = [exp(value - max_value) for value in values]
    denominator = sum(numerators)
    return [value / denominator for value in numerators]


def _weighted_sum(weights: Sequence[float], values: Sequence[Vector], dim: int) -> list[float]:
    out = [0.0 for _ in range(dim)]
    for weight, vector in zip(weights, values, strict=True):
        for idx, value in enumerate(vector):
            out[idx] += weight * value
    return out


def _ensure_finite(output: Sequence[Sequence[Vector]]) -> None:
    for query in output:
        for head in query:
            for value in head:
                if not isfinite(value):
                    raise ValueError("fallback output must be finite")
