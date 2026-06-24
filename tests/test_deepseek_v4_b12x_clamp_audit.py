from __future__ import annotations

from pathlib import Path

from hydralisk.admission.deepseek_v4_b12x_clamp_audit import (
    build_audit,
    render_markdown,
    write_audit,
)


def test_b12x_clamp_audit_detects_missing_flashinfer_surface(tmp_path: Path) -> None:
    flashinfer_root, vllm_root = _write_fixture_sources(tmp_path)

    audit = build_audit(
        flashinfer_root=flashinfer_root,
        vllm_root=vllm_root,
        generated_at="2026-06-24T00:00:00+00:00",
    )

    assert audit["decision"]["status"] == "b12x_clamp_missing_in_api_launch_and_kernel_terms"
    assert audit["decision"]["b12xLacksClampSurface"] is True
    assert audit["decision"]["b12xKernelLacksClampTerms"] is True
    assert audit["flashinferB12xApi"]["hasNumLocalExpertsParameter"] is True
    assert audit["flashinferB12xApi"]["hasSwiGluLimitParameter"] is False
    assert audit["flashinferB12xDispatch"]["hasGemm1ClampLimitParameter"] is False
    assert audit["flashinferB12xApi"]["hasExpertParallelismRejection"] is True


def test_b12x_clamp_audit_captures_vllm_contract(tmp_path: Path) -> None:
    flashinfer_root, vllm_root = _write_fixture_sources(tmp_path)

    audit = build_audit(
        flashinfer_root=flashinfer_root,
        vllm_root=vllm_root,
        generated_at="2026-06-24T00:00:00+00:00",
    )

    contract = audit["vllmClampContract"]
    assert contract["hasClampContract"] is True
    assert "gate branch is clamped only above +limit" in contract["semantics"]
    assert audit["decision"]["vllmClampContractPresent"] is True


def test_b12x_clamp_audit_markdown_is_public_safe_and_actionable(tmp_path: Path) -> None:
    flashinfer_root, vllm_root = _write_fixture_sources(tmp_path)
    audit = build_audit(
        flashinfer_root=flashinfer_root,
        vllm_root=vllm_root,
        generated_at="2026-06-24T00:00:00+00:00",
    )

    markdown = render_markdown(audit)

    assert "No secrets" in markdown
    assert "No model weights" in markdown
    assert "SM120 Dispatch" in markdown
    assert "nvfp4_activation" in markdown
    assert "swiglu_limit" in markdown
    assert "It does not remove the G4/SM120 B12x clamp blocker." in markdown


def test_b12x_clamp_audit_writes_json_and_markdown(tmp_path: Path) -> None:
    flashinfer_root, vllm_root = _write_fixture_sources(tmp_path)
    audit = build_audit(
        flashinfer_root=flashinfer_root,
        vllm_root=vllm_root,
        generated_at="2026-06-24T00:00:00+00:00",
    )

    json_path, markdown_path = write_audit(audit, tmp_path / "out")

    assert json_path.exists()
    assert markdown_path.exists()
    assert "hydralisk.deepseek-v4.b12x-clamp-audit.v1" in json_path.read_text()
    assert "# DeepSeek B12x clamp patch-point audit" in markdown_path.read_text()


def _write_fixture_sources(tmp_path: Path) -> tuple[Path, Path]:
    flashinfer_root = tmp_path / "flashinfer"
    vllm_root = tmp_path / "vllm"

    _write(
        flashinfer_root / "flashinfer/fused_moe/cute_dsl/b12x_moe.py",
        """
from typing import Optional

def b12x_fused_moe(
    x,
    w1_weight,
    w1_weight_sf,
    w2_weight,
    w2_weight_sf,
    token_selected_experts,
    token_final_scales,
    num_experts: int,
    top_k: int,
    *,
    w1_alpha,
    w2_alpha,
    fc2_input_scale: Optional[object] = None,
    num_local_experts: Optional[int] = None,
    output=None,
    output_dtype=None,
    activation: str = "silu",
    activation_precision: str = "fp4",
    quant_mode: Optional[str] = None,
    source_format: str = "modelopt",
):
    if num_local_experts != num_experts:
        raise NotImplementedError("does not yet support Expert Parallelism")

class B12xMoEWrapper:
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        *,
        num_local_experts: Optional[int] = None,
        activation: str = "silu",
        activation_precision: str = "fp4",
        quant_mode: Optional[str] = None,
    ):
        self.num_local_experts = num_local_experts
""",
    )
    _write(
        flashinfer_root
        / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py",
        """
def launch_sm120_static_moe(**kwargs): ...
def launch_sm120_micro_moe(**kwargs): ...
def launch_sm120_dynamic_moe(**kwargs): ...

def launch_sm120_moe(
    *,
    a,
    topk_ids,
    topk_weights,
    w1_weight,
    w1_weight_sf,
    w1_alpha,
    w2_weight,
    w2_weight_sf,
    w2_alpha,
    num_experts: int,
    top_k: int,
    num_local_experts: int,
    scatter_output,
    activation: str = "silu",
    activation_precision: str = "fp4",
    quant_mode: str | None = None,
):
    is_gated = activation == "silu"
    return launch_sm120_static_moe(activation=activation)
""",
    )
    for kernel_name in (
        "moe_static_kernel.py",
        "moe_micro_kernel.py",
        "moe_dynamic_kernel.py",
        "moe_w4a16_kernel.py",
    ):
        _write(
            flashinfer_root
            / f"flashinfer/fused_moe/cute_dsl/blackwell_sm12x/{kernel_name}",
            """
def kernel():
    # SiLU(gate) * up
    activation = "silu"
    self_is_gated = True
    gate = 1
    up = 2
    silu = gate
    cute_math = "cute.math.exp(-gate"
""",
        )

    _write(
        vllm_root / "vllm/model_executor/layers/activation.py",
        """
class SiluAndMulWithClamp:
    def __init__(self, swiglu_limit: float):
        self.swiglu_limit = float(swiglu_limit)
    def forward_native(self, x):
        gate = torch.clamp(x[..., :d], max=self.swiglu_limit)
        up = torch.clamp(x[..., d:], min=-self.swiglu_limit, max=self.swiglu_limit)
""",
    )
    _write(
        vllm_root / "vllm/model_executor/layers/fused_moe/utils.py",
        """
def swiglu_limit_func(output, input, swiglu_limit: float = 0.0):
    gate = torch.clamp(gate, max=swiglu_limit)
    up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
""",
    )
    _write(
        vllm_root / "vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py",
        """
def activation(self, activation, output, input):
    gemm1_clamp_limit = self.quant_config.gemm1_clamp_limit
    swiglu_limit_func(output, input, float(gemm1_clamp_limit))
""",
    )
    _write(
        vllm_root / "vllm/model_executor/layers/quantization/utils/fp8_utils.py",
        """
def kernel(clamp_limit):
    act_f32 = tl.minimum(act_f32, clamp_limit)
    mul_f32 = tl.clamp(mul_f32, -clamp_limit, clamp_limit)
""",
    )
    return flashinfer_root, vllm_root


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip())
