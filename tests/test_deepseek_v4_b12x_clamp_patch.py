from __future__ import annotations

from pathlib import Path

import pytest

from hydralisk.admission.deepseek_v4_b12x_clamp_patch import (
    B12X_API,
    PATCH_MARKER,
    apply_overlay,
    validate_overlay,
)


def test_b12x_clamp_overlay_fails_validation_before_patch(tmp_path: Path) -> None:
    flashinfer_root = _write_patch_fixture(tmp_path)

    validation = validate_overlay(flashinfer_root)

    assert validation["ok"] is False
    assert validation["b12xFusedMoeHasSwigluLimit"] is False
    assert validation["launchSm120MoeHasSwigluLimit"] is False


def test_b12x_clamp_overlay_dry_run_validates_without_writing(tmp_path: Path) -> None:
    flashinfer_root = _write_patch_fixture(tmp_path)
    before = (flashinfer_root / B12X_API).read_text()

    result = apply_overlay(flashinfer_root, dry_run=True)

    assert result["validation"]["ok"] is True
    assert (flashinfer_root / B12X_API).read_text() == before
    assert all(edit["dryRun"] is True for edit in result["edits"])


def test_b12x_clamp_overlay_applies_and_is_idempotent(tmp_path: Path) -> None:
    flashinfer_root = _write_patch_fixture(tmp_path)

    first = apply_overlay(flashinfer_root)
    second = apply_overlay(flashinfer_root)
    validation = validate_overlay(flashinfer_root)

    assert first["validation"]["ok"] is True
    assert second["validation"]["ok"] is True
    assert validation["ok"] is True
    assert all(edit["status"] == "already_applied" for edit in second["edits"])
    kernel_text = (
        flashinfer_root
        / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py"
    ).read_text()
    assert PATCH_MARKER in kernel_text
    assert "gate=min(gate, limit)" in kernel_text
    assert "up=clamp(up, -limit, limit)" in kernel_text


def test_b12x_clamp_overlay_missing_patch_point_fails(tmp_path: Path) -> None:
    flashinfer_root = _write_patch_fixture(tmp_path)
    api_path = flashinfer_root / B12X_API
    api_path.write_text(api_path.read_text().replace("source_format: str", "format: str"))

    with pytest.raises(RuntimeError, match="Patch point missing"):
        apply_overlay(flashinfer_root)


def _write_patch_fixture(tmp_path: Path) -> Path:
    flashinfer_root = tmp_path / "flashinfer"
    _write(
        flashinfer_root / B12X_API,
        """
from typing import Optional

def b12x_fused_moe(
    *,
    quant_mode: Optional[str] = None,
    source_format: str = "modelopt",
) -> torch.Tensor:
    return launch_sm120_moe(
        quant_mode=quant_mode,
        source_format=source_format,
    )

class B12xMoEWrapper:
    def __init__(
        self,
        *,
        quant_mode: Optional[str] = None,
        source_format: str = "modelopt",
    ):
        self.quant_mode = quant_mode
        self.source_format = source_format

    def run(self):
        return launch_sm120_moe(
            quant_mode=self.quant_mode,
            source_format=self.source_format,
            _workspace=workspace,
        )
""",
    )
    _write(
        flashinfer_root
        / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py",
        """
def launch_sm120_moe(
    *,
    quant_mode: str | None = None,
    source_format: str = "modelopt",
    _workspace=None,
) -> torch.Tensor:
    quant_mode = _normalize_quant_mode(quant_mode, activation_precision)
    source_format = _normalize_source_format_for_quant_mode(source_format, quant_mode)
    activation_precision = _activation_precision_from_quant_mode(quant_mode)
    return None
""",
    )
    for kernel_name in (
        "moe_static_kernel.py",
        "moe_micro_kernel.py",
    ):
        _write(
            flashinfer_root
            / f"flashinfer/fused_moe/cute_dsl/blackwell_sm12x/{kernel_name}",
            """
                                        g = alpha_value * gate_slice[elem_idx]
                                        u = alpha_value * up_slice[elem_idx]
                                        sigmoid_g = cute.arch.rcp_approx(
""",
        )
    _write(
        flashinfer_root
        / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py",
        """
                                            g = alpha_value * gate_slice[elem_idx]
                                            u = alpha_value * up_slice[elem_idx]
                                            sigmoid_g = cute.arch.rcp_approx(
""",
    )
    _write(
        flashinfer_root
        / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py",
        """
                up = fc1_bf16_flat[base + Int32(self.intermediate_size) + col].to(
                    cutlass.Float32
                )
                silu = gate / (

                up = fc1_flat[base + Int32(self.intermediate_size) + col].to(
                    cutlass.Float32
                )
                silu = gate / (
""",
    )
    return flashinfer_root


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip("\n"))
