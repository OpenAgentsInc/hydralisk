from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp


Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]
ExpertMatrices = Sequence[Matrix]


@dataclass(frozen=True)
class LocalShardB12xInputs:
    token_selected_experts: list[list[int]]
    token_final_scales: list[list[float]]
    local_route_count: int
    masked_route_count: int
    fill_expert: int


def silu(value: float) -> float:
    return value / (1.0 + exp(-value))


def swiglu_limit(gate: Vector, up: Vector, limit: float) -> list[float]:
    """DeepSeek/vLLM SwiGLU clamp: silu(min(gate, limit)) * clamp(up)."""

    if len(gate) != len(up):
        raise ValueError("gate and up branches must have the same length")

    out: list[float] = []
    for gate_value, up_value in zip(gate, up, strict=True):
        if limit > 0:
            gate_value = min(gate_value, limit)
            up_value = min(max(up_value, -limit), limit)
        out.append(silu(gate_value) * up_value)
    return out


def global_to_local_expert(
    expert_id: int,
    *,
    local_expert_offset: int,
    num_local_experts: int,
) -> int | None:
    local_id = expert_id - local_expert_offset
    if local_id < 0 or local_id >= num_local_experts:
        return None
    return local_id


def remap_routes_for_b12x_local_shard(
    token_selected_experts: Sequence[Sequence[int]],
    token_final_scales: Sequence[Vector],
    *,
    global_num_experts: int,
    local_expert_offset: int,
    num_local_experts: int,
    fill_expert: int = 0,
) -> LocalShardB12xInputs:
    """Map global routes into the fixed-shape rank-local B12x domain.

    B12x wants expert IDs in `[0, num_local_experts)` with a fixed top-k shape.
    Nonlocal or invalid global routes are replaced by `fill_expert` and a zero
    route scale, which preserves shape while matching reference skip semantics.
    """

    if global_num_experts <= 0:
        raise ValueError("global_num_experts must be positive")
    if num_local_experts <= 0:
        raise ValueError("num_local_experts must be positive")
    if local_expert_offset < 0:
        raise ValueError("local_expert_offset must be non-negative")
    if fill_expert < 0 or fill_expert >= num_local_experts:
        raise ValueError("fill_expert must be inside the local expert domain")
    if len(token_selected_experts) != len(token_final_scales):
        raise ValueError("selected experts and final scales must have token parity")

    remapped_experts: list[list[int]] = []
    remapped_scales: list[list[float]] = []
    expected_top_k: int | None = None
    local_route_count = 0
    masked_route_count = 0

    for expert_row, scale_row in zip(
        token_selected_experts,
        token_final_scales,
        strict=True,
    ):
        if len(expert_row) != len(scale_row):
            raise ValueError("selected experts and final scales must share top-k length")
        if expected_top_k is None:
            expected_top_k = len(expert_row)
        elif len(expert_row) != expected_top_k:
            raise ValueError("all route rows must share fixed top-k length")

        local_expert_row: list[int] = []
        local_scale_row: list[float] = []
        for expert_id, route_scale in zip(expert_row, scale_row, strict=True):
            local_id = None
            if 0 <= expert_id < global_num_experts:
                local_id = global_to_local_expert(
                    expert_id,
                    local_expert_offset=local_expert_offset,
                    num_local_experts=num_local_experts,
                )

            if local_id is None:
                local_expert_row.append(fill_expert)
                local_scale_row.append(0.0)
                masked_route_count += 1
            else:
                local_expert_row.append(local_id)
                local_scale_row.append(float(route_scale))
                local_route_count += 1

        remapped_experts.append(local_expert_row)
        remapped_scales.append(local_scale_row)

    return LocalShardB12xInputs(
        token_selected_experts=remapped_experts,
        token_final_scales=remapped_scales,
        local_route_count=local_route_count,
        masked_route_count=masked_route_count,
        fill_expert=fill_expert,
    )


def require_b12x_deepseek_clamp_surface(
    *,
    supports_swiglu_limit: bool,
    swiglu_limit_value: float,
) -> None:
    if swiglu_limit_value > 0 and not supports_swiglu_limit:
        raise RuntimeError(
            "DeepSeek-V4 requires swiglu_limit clamp semantics before B12x "
            "can be used for serving"
        )


def reference_b12x_local_shard_shim(
    hidden_states: Sequence[Vector],
    token_selected_experts: Sequence[Sequence[int]],
    token_final_scales: Sequence[Vector],
    w1_weights: ExpertMatrices,
    w2_weights: ExpertMatrices,
    *,
    global_num_experts: int,
    local_expert_offset: int,
    num_local_experts: int,
    swiglu_limit_value: float,
) -> tuple[list[list[float]], LocalShardB12xInputs]:
    remapped = remap_routes_for_b12x_local_shard(
        token_selected_experts,
        token_final_scales,
        global_num_experts=global_num_experts,
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
    )
    output = reference_local_shard_moe(
        hidden_states=hidden_states,
        token_selected_experts=remapped.token_selected_experts,
        token_final_scales=remapped.token_final_scales,
        w1_weights=w1_weights,
        w2_weights=w2_weights,
        global_num_experts=num_local_experts,
        local_expert_offset=0,
        num_local_experts=num_local_experts,
        swiglu_limit_value=swiglu_limit_value,
    )
    return output, remapped


def reference_local_shard_moe(
    hidden_states: Sequence[Vector],
    token_selected_experts: Sequence[Sequence[int]],
    token_final_scales: Sequence[Vector],
    w1_weights: ExpertMatrices,
    w2_weights: ExpertMatrices,
    *,
    global_num_experts: int,
    local_expert_offset: int,
    num_local_experts: int,
    swiglu_limit_value: float,
) -> list[list[float]]:
    """Tiny reference for local-shard DeepSeek MoE routing.

    `w1_weights` use the FlashInfer B12x test convention: first half is the
    linear/up branch, second half is the gate branch.
    """

    if len(w1_weights) != num_local_experts or len(w2_weights) != num_local_experts:
        raise ValueError("local weight count must match num_local_experts")
    if len(hidden_states) != len(token_selected_experts):
        raise ValueError("hidden states and selected experts must have token parity")
    if len(hidden_states) != len(token_final_scales):
        raise ValueError("hidden states and final scales must have token parity")
    if not hidden_states:
        return []

    hidden_size = len(hidden_states[0])
    output: list[list[float]] = [[0.0 for _ in range(hidden_size)] for _ in hidden_states]

    for token_idx, token_input in enumerate(hidden_states):
        if len(token_input) != hidden_size:
            raise ValueError("all hidden states must share hidden size")
        if len(token_selected_experts[token_idx]) != len(token_final_scales[token_idx]):
            raise ValueError("selected experts and final scales must share top-k length")

        for expert_id, route_scale in zip(
            token_selected_experts[token_idx],
            token_final_scales[token_idx],
            strict=True,
        ):
            if expert_id < 0 or expert_id >= global_num_experts:
                continue

            local_id = global_to_local_expert(
                expert_id,
                local_expert_offset=local_expert_offset,
                num_local_experts=num_local_experts,
            )
            if local_id is None:
                continue

            gemm1 = _matvec(w1_weights[local_id], token_input)
            if len(gemm1) % 2 != 0:
                raise ValueError("w1 output must split evenly into up and gate branches")

            intermediate_size = len(gemm1) // 2
            up = gemm1[:intermediate_size]
            gate = gemm1[intermediate_size:]
            activated = swiglu_limit(gate, up, swiglu_limit_value)
            gemm2 = _matvec(w2_weights[local_id], activated)
            if len(gemm2) != hidden_size:
                raise ValueError("w2 output must match hidden size")

            for idx, value in enumerate(gemm2):
                output[token_idx][idx] += route_scale * value

    return output


def _matvec(matrix: Matrix, vector: Vector) -> list[float]:
    out: list[float] = []
    for row in matrix:
        if len(row) != len(vector):
            raise ValueError("matrix row width must match vector length")
        out.append(sum(weight * value for weight, value in zip(row, vector, strict=True)))
    return out
