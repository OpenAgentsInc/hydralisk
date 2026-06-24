from __future__ import annotations

import pytest
from pytest import approx

from hydralisk.admission.deepseek_v4_moe import (
    global_to_local_expert,
    reference_b12x_local_shard_shim,
    reference_local_shard_moe,
    remap_routes_for_b12x_local_shard,
    require_b12x_deepseek_clamp_surface,
    silu,
    swiglu_limit,
)


def test_swiglu_limit_matches_deepseek_clamp_semantics() -> None:
    limited = swiglu_limit(gate=[20.0, -20.0, 1.0], up=[20.0, -20.0, 2.0], limit=10.0)
    unlimited = swiglu_limit(
        gate=[20.0, -20.0, 1.0],
        up=[20.0, -20.0, 2.0],
        limit=0.0,
    )

    assert limited == approx(
        [
            silu(10.0) * 10.0,
            silu(-20.0) * -10.0,
            silu(1.0) * 2.0,
        ]
    )
    assert limited[0] != approx(unlimited[0])
    assert limited[1] != approx(unlimited[1])


def test_global_to_local_expert_maps_rank_interval() -> None:
    assert global_to_local_expert(64, local_expert_offset=64, num_local_experts=32) == 0
    assert global_to_local_expert(95, local_expert_offset=64, num_local_experts=32) == 31
    assert global_to_local_expert(63, local_expert_offset=64, num_local_experts=32) is None
    assert global_to_local_expert(96, local_expert_offset=64, num_local_experts=32) is None


def test_reference_local_shard_moe_skips_nonlocal_experts() -> None:
    output = reference_local_shard_moe(
        hidden_states=[[1.0, 0.0]],
        token_selected_experts=[[3, 6]],
        token_final_scales=[[1.0, 1.0]],
        w1_weights=[
            [[20.0, 0.0], [-20.0, 0.0], [20.0, 0.0], [0.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0], [0.0, 0.0], [2.0, 0.0]],
        ],
        w2_weights=[
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        global_num_experts=8,
        local_expert_offset=4,
        num_local_experts=2,
        swiglu_limit_value=10.0,
    )

    assert output == [[0.0, 0.0]]


def test_reference_local_shard_moe_is_deterministic_for_nonzero_fixture() -> None:
    output = reference_local_shard_moe(
        hidden_states=[
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        token_selected_experts=[
            [4, 3],
            [5, 6],
        ],
        token_final_scales=[
            [1.0, 99.0],
            [0.5, 0.5],
        ],
        w1_weights=[
            [[20.0, 0.0], [-20.0, 0.0], [20.0, 0.0], [0.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0], [0.0, 0.0], [2.0, 0.0]],
        ],
        w2_weights=[
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        global_num_experts=8,
        local_expert_offset=4,
        num_local_experts=2,
        swiglu_limit_value=10.0,
    )

    assert output[0] == approx([silu(10.0) * 10.0, 0.0])
    assert output[1] == approx([0.0, 0.5 * silu(2.0) * 3.0])


def test_b12x_local_shard_remap_preserves_shape_and_masks_nonlocal_routes() -> None:
    remapped = remap_routes_for_b12x_local_shard(
        token_selected_experts=[[3, 4, 5, 99, -1]],
        token_final_scales=[[0.1, 0.2, 0.3, 0.4, 0.5]],
        global_num_experts=8,
        local_expert_offset=4,
        num_local_experts=2,
    )

    assert remapped.token_selected_experts == [[0, 0, 1, 0, 0]]
    assert remapped.token_final_scales == [[0.0, 0.2, 0.3, 0.0, 0.0]]
    assert remapped.local_route_count == 2
    assert remapped.masked_route_count == 3
    assert remapped.fill_expert == 0


def test_b12x_local_shard_remap_requires_fixed_top_k_shape() -> None:
    with pytest.raises(ValueError, match="fixed top-k"):
        remap_routes_for_b12x_local_shard(
            token_selected_experts=[[4, 5], [4]],
            token_final_scales=[[1.0, 1.0], [1.0]],
            global_num_experts=8,
            local_expert_offset=4,
            num_local_experts=2,
        )


def test_b12x_local_shard_shim_matches_global_reference_fixture() -> None:
    hidden_states = [
        [1.0, 0.0],
        [1.0, 0.0],
    ]
    token_selected_experts = [
        [4, 3],
        [5, 6],
    ]
    token_final_scales = [
        [1.0, 99.0],
        [0.5, 0.5],
    ]
    w1_weights = [
        [[20.0, 0.0], [-20.0, 0.0], [20.0, 0.0], [0.0, 0.0]],
        [[2.0, 0.0], [3.0, 0.0], [0.0, 0.0], [2.0, 0.0]],
    ]
    w2_weights = [
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
    ]

    expected = reference_local_shard_moe(
        hidden_states=hidden_states,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        w1_weights=w1_weights,
        w2_weights=w2_weights,
        global_num_experts=8,
        local_expert_offset=4,
        num_local_experts=2,
        swiglu_limit_value=10.0,
    )
    output, remapped = reference_b12x_local_shard_shim(
        hidden_states=hidden_states,
        token_selected_experts=token_selected_experts,
        token_final_scales=token_final_scales,
        w1_weights=w1_weights,
        w2_weights=w2_weights,
        global_num_experts=8,
        local_expert_offset=4,
        num_local_experts=2,
        swiglu_limit_value=10.0,
    )

    assert remapped.token_selected_experts == [[0, 0], [1, 0]]
    assert remapped.token_final_scales == [[1.0, 0.0], [0.5, 0.0]]
    assert output[0] == approx(expected[0])
    assert output[1] == approx(expected[1])


def test_b12x_clamp_surface_gate_fails_closed_for_deepseek() -> None:
    with pytest.raises(RuntimeError, match="swiglu_limit"):
        require_b12x_deepseek_clamp_surface(
            supports_swiglu_limit=False,
            swiglu_limit_value=10.0,
        )

    require_b12x_deepseek_clamp_surface(
        supports_swiglu_limit=True,
        swiglu_limit_value=10.0,
    )
