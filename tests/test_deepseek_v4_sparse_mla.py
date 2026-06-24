from __future__ import annotations

import pytest
from pytest import approx

from hydralisk.admission.deepseek_v4_sparse_mla import reference_sparse_mla_decode


def test_sparse_mla_fallback_produces_finite_nonzero_output() -> None:
    result = reference_sparse_mla_decode(
        query=[[[1.0, 0.0], [0.0, 1.0]]],
        swa_kv_cache=[[
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [2.0, 0.0]],
        ]],
        compressed_kv_cache=[[
            [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, 3.0]],
        ]],
        sparse_indices=[[0, 2]],
        sparse_topk_lens=[2],
        seq_lens=[3],
        sliding_window_tokens=2,
    )

    assert result.stats.query_count == 1
    assert result.stats.head_count == 2
    assert result.stats.dim == 2
    assert result.stats.swa_route_count == 2
    assert result.stats.sparse_route_count == 2
    assert result.stats.masked_sparse_route_count == 0
    assert result.stats.empty_route_count == 0
    assert len(result.output) == 1
    assert len(result.output[0]) == 2
    assert any(value != 0.0 for head in result.output[0] for value in head)


def test_sparse_mla_fallback_honors_sparse_topk_truncation() -> None:
    result = reference_sparse_mla_decode(
        query=[[[1.0, 0.0]]],
        swa_kv_cache=[[[[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]]]],
        compressed_kv_cache=[[[[1.0, 0.0], [0.0, 100.0], [100.0, 0.0]]]],
        sparse_indices=[[0, 1, 2]],
        sparse_topk_lens=[1],
        seq_lens=[3],
        include_swa_window=False,
    )

    assert result.stats.sparse_route_count == 1
    assert result.output == [[[1.0, 0.0]]]


def test_sparse_mla_fallback_masks_sparse_indices_past_sequence_length() -> None:
    result = reference_sparse_mla_decode(
        query=[[[1.0, 0.0]]],
        swa_kv_cache=[[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]],
        compressed_kv_cache=[[[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0], [8.0, 8.0]]]],
        sparse_indices=[[0, 2, 3, -1]],
        sparse_topk_lens=[4],
        seq_lens=[2],
        include_swa_window=False,
    )

    assert result.stats.sparse_route_count == 1
    assert result.stats.masked_sparse_route_count == 3
    assert result.output == [[[1.0, 0.0]]]


def test_sparse_mla_fallback_returns_zero_for_empty_routes() -> None:
    result = reference_sparse_mla_decode(
        query=[[[1.0, 0.0], [0.0, 1.0]]],
        swa_kv_cache=[[[[1.0, 0.0], [0.0, 1.0]]]],
        compressed_kv_cache=[[[[1.0, 0.0], [0.0, 1.0]]]],
        sparse_indices=[[]],
        sparse_topk_lens=[0],
        seq_lens=[0],
        include_swa_window=False,
    )

    assert result.stats.empty_route_count == 2
    assert result.output == [[[0.0, 0.0], [0.0, 0.0]]]


def test_sparse_mla_fallback_respects_sliding_window_truncation() -> None:
    result = reference_sparse_mla_decode(
        query=[[[1.0, 0.0]]],
        swa_kv_cache=[[[[10.0, 0.0], [0.0, 10.0], [1.0, 1.0]]]],
        compressed_kv_cache=[[[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]],
        sparse_indices=[[]],
        sparse_topk_lens=[0],
        seq_lens=[2],
        sliding_window_tokens=1,
    )

    assert result.stats.swa_route_count == 1
    assert result.output == [[[0.0, 10.0]]]


def test_sparse_mla_fallback_broadcasts_single_kv_head() -> None:
    result = reference_sparse_mla_decode(
        query=[[[1.0, 0.0], [0.0, 1.0]]],
        swa_kv_cache=[[[[1.0, 0.0], [0.0, 1.0]]]],
        compressed_kv_cache=[[[[0.0, 3.0], [4.0, 0.0]]]],
        sparse_indices=[[0]],
        sparse_topk_lens=[1],
        seq_lens=[1],
        include_swa_window=False,
    )

    assert result.output == [[[0.0, 3.0], [0.0, 3.0]]]


def test_sparse_mla_fallback_supports_per_head_kv_cache() -> None:
    result = reference_sparse_mla_decode(
        query=[[[1.0, 0.0], [0.0, 1.0]]],
        swa_kv_cache=[[
            [[1.0, 0.0]],
            [[0.0, 1.0]],
        ]],
        compressed_kv_cache=[[
            [[2.0, 0.0]],
            [[0.0, 3.0]],
        ]],
        sparse_indices=[[0]],
        sparse_topk_lens=[1],
        seq_lens=[1],
        include_swa_window=False,
    )

    assert result.output == [[[2.0, 0.0], [0.0, 3.0]]]


def test_sparse_mla_fallback_fails_closed_for_non_hnd_layout() -> None:
    with pytest.raises(ValueError, match="HND"):
        reference_sparse_mla_decode(
            query=[[[1.0, 0.0]]],
            swa_kv_cache=[[[[1.0, 0.0]]]],
            compressed_kv_cache=[[[[1.0, 0.0]]]],
            sparse_indices=[[0]],
            sparse_topk_lens=[1],
            seq_lens=[1],
            kv_layout="NHD",
        )


def test_sparse_mla_fallback_uses_stable_softmax_for_large_logits() -> None:
    result = reference_sparse_mla_decode(
        query=[[[1000.0, 0.0]]],
        swa_kv_cache=[[[[0.0, 0.0], [0.0, 0.0]]]],
        compressed_kv_cache=[[[[1.0, 0.0], [2.0, 0.0]]]],
        sparse_indices=[[0, 1]],
        sparse_topk_lens=[2],
        seq_lens=[2],
        include_swa_window=False,
    )

    assert result.output[0][0] == approx([2.0, 0.0])
