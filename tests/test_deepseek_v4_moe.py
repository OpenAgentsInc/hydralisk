from __future__ import annotations

from pytest import approx

from hydralisk.admission.deepseek_v4_moe import (
    global_to_local_expert,
    reference_local_shard_moe,
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
