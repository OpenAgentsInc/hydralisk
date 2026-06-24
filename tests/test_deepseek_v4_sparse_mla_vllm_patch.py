from __future__ import annotations

from pathlib import Path

import pytest

from hydralisk.admission.deepseek_v4_sparse_mla_vllm_patch import (
    DECODE_CALL,
    PATCH_SENTINEL,
    PATCH_VERSION_SENTINEL,
    PREFILL_CALL,
    build_report,
    patch_source,
)


def test_vllm_sparse_mla_patcher_inserts_fail_closed_fallback_branch() -> None:
    patched, result = patch_source(_fixture_source())

    assert result.patched is True
    assert result.already_patched is False
    assert result.inserted_import is True
    assert result.inserted_helpers is True
    assert result.decode_branch_patched is True
    assert result.prefill_branch_patched is True
    assert f'os.getenv(_HYDRALISK_SPARSE_MLA_FALLBACK_ENV) == "1"' in patched
    assert "_hydralisk_sparse_mla_fallback(" in patched
    assert "_hydralisk_sparse_mla_floatable_dtype" in patched
    assert "unsupported query dtype" in patched
    assert "unsupported KV cache dtypes" in patched
    assert "convertible to fp32" in patched
    assert "_hydralisk_sparse_mla_cache_layout" in patched
    assert "HYDRALISK_DEEPSEEK_SPARSE_MLA_FALLBACK_VECTOR_GATHER_V3" in patched
    assert "_hydralisk_sparse_mla_candidate_keys" in patched
    assert 'torch.einsum("hcd,hd->hc"' in patched
    assert 'torch.einsum(\n            "hc,hcd->hd"' in patched
    assert "[pages, page, dim] or [pages, kv_heads, page, dim]" in patched
    assert "one-token decode" in patched
    assert patched.count("flashinfer_trtllm_batch_decode_sparse_mla_dsv4(") == 2


def test_vllm_sparse_mla_patcher_is_idempotent() -> None:
    patched, first = patch_source(_fixture_source())
    patched_again, second = patch_source(patched)

    assert first.patched is True
    assert second.patched is False
    assert second.already_patched is True
    assert patched_again == patched


def test_vllm_sparse_mla_patcher_upgrades_legacy_helper() -> None:
    patched, first = patch_source(_fixture_source())
    legacy = patched.replace(PATCH_VERSION_SENTINEL, "legacy-cache-layout")

    upgraded, second = patch_source(legacy)

    assert first.patched is True
    assert second.patched is True
    assert second.already_patched is False
    assert second.decode_branch_patched is False
    assert second.prefill_branch_patched is False
    assert PATCH_VERSION_SENTINEL in upgraded
    assert "_hydralisk_sparse_mla_cache_layout" in upgraded


def test_vllm_sparse_mla_patcher_fails_when_call_site_changes() -> None:
    source = _fixture_source().replace(DECODE_CALL, "# changed decode call")

    with pytest.raises(ValueError, match="decode"):
        patch_source(source)


def test_vllm_sparse_mla_patch_report_is_public_safe() -> None:
    _, result = patch_source(_fixture_source())
    report = build_report(result, generated_at="2026-06-24T00:00:00+00:00")

    assert report["envFlag"] == PATCH_SENTINEL
    assert report["defaultEnabled"] is False
    assert report["publicSafety"]["containsSecrets"] is False
    assert report["publicSafety"]["containsPrompts"] is False
    assert report["publicSafety"]["containsWeights"] is False


def test_vllm_sparse_mla_patcher_matches_local_reference_checkout() -> None:
    path = Path(
        "/Users/christopherdavid/work/projects/repos/vllm/"
        "vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py"
    )
    if not path.exists():
        pytest.skip("local vLLM reference checkout is not present")

    _, result = patch_source(path.read_text(), target=str(path))

    assert result.patched is True
    assert result.decode_branch_patched is True
    assert result.prefill_branch_patched is True


def _fixture_source() -> str:
    return f'''from typing import TYPE_CHECKING, ClassVar, cast

import torch

_flashinfer_dsv4_workspace_by_device: dict[torch.device, torch.Tensor] = {{}}


class DeepseekV4FlashInferMLAAttention:
    def _forward(self):
        if num_decode_tokens > 0:
            decode_cu = query_start_loc[: num_decodes + 1]
            decode_cu_cpu = query_start_loc_cpu[: num_decodes + 1]
            decode_lens_cpu = decode_cu_cpu[1:] - decode_cu_cpu[:-1]
{DECODE_CALL}

        if num_prefill_tokens > 0:
            prefill_cu = (
                query_start_loc[num_decodes : num_reqs + 1]
                - query_start_loc[num_decodes]
            )
            prefill_cu_cpu = query_start_loc_cpu[num_decodes : num_reqs + 1]
            prefill_lens_cpu = prefill_cu_cpu[1:] - prefill_cu_cpu[:-1]
{PREFILL_CALL}
'''
