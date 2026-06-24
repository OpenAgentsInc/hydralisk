from __future__ import annotations

from pathlib import Path

from hydralisk.admission.deepseek_v4_flashmla_sparse_audit import (
    build_audit,
    render_markdown,
    write_audit,
)


def test_flashmla_sparse_audit_finds_existing_flashinfer_probe_path(
    tmp_path: Path,
) -> None:
    vllm_root = _write_fixture_sources(tmp_path)

    audit = build_audit(
        vllm_root=vllm_root,
        generated_at="2026-06-24T00:00:00+00:00",
    )

    assert (
        audit["decision"]["status"]
        == "existing_flashinfer_sparse_backend_is_next_sm120_probe"
    )
    assert audit["flashmlaPath"]["prefillCallsFlashMlaSparseFwd"] is True
    assert audit["flashmlaPath"]["pythonSparseSupportGuardLacksSm120"] is True
    assert audit["flashinferPath"]["avoidsFlashMlaSparseFwd"] is True
    assert audit["selector"]["supportsExplicitFlashinferBackend"] is True
    assert audit["configuration"]["cliCanSelectAttentionBackend"] is True
    assert (
        audit["configuration"]["recommendedWrapperEnv"]
        == "VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4"
    )


def test_flashmla_sparse_audit_markdown_is_public_safe_and_actionable(
    tmp_path: Path,
) -> None:
    vllm_root = _write_fixture_sources(tmp_path)
    audit = build_audit(
        vllm_root=vllm_root,
        generated_at="2026-06-24T00:00:00+00:00",
    )

    markdown = render_markdown(audit)

    assert "No secrets" in markdown
    assert "No model weights" in markdown
    assert "FLASHINFER_MLA_SPARSE_DSV4" in markdown
    assert "VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE_DSV4" in markdown
    assert "flash_mla_sparse_fwd" in markdown
    assert "Existing FlashInfer probe ready: `True`" in markdown


def test_flashmla_sparse_audit_writes_json_and_markdown(tmp_path: Path) -> None:
    vllm_root = _write_fixture_sources(tmp_path)
    audit = build_audit(
        vllm_root=vllm_root,
        generated_at="2026-06-24T00:00:00+00:00",
    )

    json_path, markdown_path = write_audit(audit, tmp_path / "out")

    assert json_path.exists()
    assert markdown_path.exists()
    assert (
        "hydralisk.deepseek-v4.flashmla-sparse-audit.v1"
        in json_path.read_text()
    )
    assert (
        "# DeepSeek FlashMLA sparse-prefill SM120 audit"
        in markdown_path.read_text()
    )


def _write_fixture_sources(tmp_path: Path) -> Path:
    vllm_root = tmp_path / "vllm"
    _write(
        vllm_root / "vllm/models/deepseek_v4/nvidia/model.py",
        """
class AttentionBackendEnum:
    FLASHINFER_MLA_SPARSE_DSV4 = "FLASHINFER_MLA_SPARSE_DSV4"

def _select_dsv4_attn_cls(vllm_config):
    if (
        vllm_config.attention_config.backend
        == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4
    ):
        return DeepseekV4FlashInferMLAAttention
    return DeepseekV4FlashMLAAttention
""",
    )
    _write(
        vllm_root / "vllm/models/deepseek_v4/nvidia/flashmla.py",
        """
class DeepseekV4FlashMLASparseBackend:
    def get_name(self):
        return "FLASHMLA_SPARSE_DSV4"

class DeepseekV4FlashMLAAttention:
    use_flashmla_fp8_layout = True
    def _forward_prefill(self):
        flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=indices,
            out=out,
        )
    def _forward_decode(self):
        flash_mla_with_kvcache(q=q)
""",
    )
    _write(
        vllm_root / "vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py",
        """
class DeepseekV4FlashInferMLASparseBackend:
    def get_name(self):
        return "FLASHINFER_MLA_SPARSE_DSV4"

class DeepseekV4FlashInferMLAAttention:
    use_flashmla_fp8_layout: ClassVar[bool] = False
    def _build_sparse_index_metadata(self):
        return build_flashinfer_mixed_sparse_indices()
    def _forward(self):
        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(
            query=query,
            sparse_indices=sparse_indices,
            out=out,
        )
""",
    )
    _write(
        vllm_root / "vllm/v1/attention/ops/flashmla.py",
        """
def is_flashmla_sparse_supported():
    if not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ):
        return False, "unsupported"
    return True, None

flash_mla_sparse_fwd = object()
""",
    )
    _write(
        vllm_root / "vllm/config/attention.py",
        """
class AttentionConfig:
    backend: AttentionBackendEnum | None = None
    use_fp4_indexer_cache: bool = False
    def validate_backend_before(cls, value): ...
""",
    )
    _write(
        vllm_root / "vllm/engine/arg_utils.py",
        """
vllm_group.add_argument(
    "--attention-config", "-ac", **vllm_kwargs["attention_config"]
)
""",
    )
    _write(
        vllm_root / "vllm/v1/attention/backends/registry.py",
        """
FLASHMLA_SPARSE_DSV4 = (
    "vllm.models.deepseek_v4.nvidia.flashmla.DeepseekV4FlashMLASparseBackend"
)
FLASHINFER_MLA_SPARSE_DSV4 = (
    "vllm.models.deepseek_v4.nvidia.flashinfer_sparse."
    "DeepseekV4FlashInferMLASparseBackend"
)
""",
    )
    return vllm_root


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip())
