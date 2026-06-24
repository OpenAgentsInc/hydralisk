from __future__ import annotations

from collections.abc import Sequence
from math import exp


Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]
ExpertMatrices = Sequence[Matrix]


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
