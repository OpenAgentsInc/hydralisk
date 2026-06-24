from __future__ import annotations

from hydralisk.serve.config import HydraliskSettings
from hydralisk.serve.receipts import (
    ReceiptStore,
    build_capabilities,
    build_receipt,
    normalize_usage,
)


def test_normalize_usage_accepts_chat_and_responses_shapes() -> None:
    assert normalize_usage(
        {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    ) == {"promptTokens": 2, "completionTokens": 3, "totalTokens": 5}
    assert normalize_usage({"input_tokens": 7, "output_tokens": 11}) == {
        "promptTokens": 7,
        "completionTokens": 11,
        "totalTokens": 18,
    }


def test_receipt_store_round_trips_public_safe_receipt(tmp_path) -> None:
    store = ReceiptStore(tmp_path)
    receipt = build_receipt(
        run_ref="hydralisk-run-0123456789abcdef0123456789abcdef",
        served_alias="openagents/khala-oss-20b",
        usage={"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        latency={"ttftMs": None, "wallMs": 123},
        config=HydraliskSettings(model_revision="openai/gpt-oss-20b@demo"),
    )

    store.write(receipt)
    stored = store.read("hydralisk-run-0123456789abcdef0123456789abcdef")

    assert stored == receipt
    assert stored["schema"] == "hydralisk.serve.run_receipt.v1"
    assert stored["model"] == "openai/gpt-oss-20b"
    assert stored["servedModel"] == "openai/gpt-oss-20b"
    assert stored["publicSafe"] is True
    assert stored["usage"]["totalTokens"] == 5
    assert "messages" not in str(stored).lower()


def test_glm_profile_evidence_is_public_safe_and_additive() -> None:
    settings = HydraliskSettings(
        served_model="zai-org/GLM-5.2-FP8",
        public_model_aliases=(),
        engine="sglang",
        engine_version="0.5.13.post1",
        gpu_class="g4",
        gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        gpu_count=8,
        model_revision="zai-org/GLM-5.2-FP8@70311cfa0158cce7dd2cf5d2e04f68e3fdc3efc1",
        quantization_weights="FP8",
        model_profile_ref="profiles/glm-5.2-fp8-sglang.json",
        container_image=(
            "lmsysorg/sglang:v0.5.13.post1"
            "@sha256:74084d80c3b7e5649f4b3433b1169db3da26c9b1e31752a43045a34cc26ba5d5"
        ),
        context_window_tokens=1_048_576,
        admitted_context_tokens=4_096,
        tensor_parallel_size=8,
        reasoning_parser="glm45",
        tool_call_parser="glm47",
        cache_policy=(
            "g4-blocked; prefix-cache-disabled-in-minimal-repro; "
            "hicache-planned-after-supported-gpu"
        ),
        kv_cache_dtype="auto",
        dynamo_mode="disabled_preflight",
        speculative_decoding="EAGLE_MTP_planned",
        admission_ref="admission.hydralisk.glm52.g4.rtxpro6000.20260624T034345Z",
        evidence_ref="docs/evidence/2026-06-24-glm-52-sglang-load-smoke.md",
    )

    capabilities = build_capabilities(settings)
    receipt = build_receipt(
        run_ref="hydralisk-run-0123456789abcdef0123456789abcdef",
        served_alias="zai-org/GLM-5.2-FP8",
        usage=None,
        latency={"ttftMs": None, "wallMs": 0},
        config=settings,
        blockers=[
            {
                "code": "rtx_pro_6000_sglang_dsa_kernel_unsupported_architecture",
                "message": "G4 load smoke is blocked before serving.",
            }
        ],
    )

    assert capabilities["profile"]["profileRef"] == "profiles/glm-5.2-fp8-sglang.json"
    assert capabilities["profile"]["context"]["windowTokens"] == 1_048_576
    assert capabilities["profile"]["parallelism"]["tensor"] == 8
    assert capabilities["profile"]["parsers"] == {
        "reasoning": "glm45",
        "toolCalls": "glm47",
    }
    assert receipt["profile"]["evidence"]["admissionRef"].startswith(
        "admission.hydralisk.glm52."
    )
    assert (
        receipt["blockers"][0]["code"]
        == "rtx_pro_6000_sglang_dsa_kernel_unsupported_architecture"
    )
    assert "messages" not in str(receipt).lower()
    assert "hf_token" not in str(receipt).lower()
    assert "bearer" not in str(receipt).lower()
