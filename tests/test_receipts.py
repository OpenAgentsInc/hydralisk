from __future__ import annotations

from hydralisk.serve.config import HydraliskSettings
from hydralisk.serve.receipts import ReceiptStore, build_receipt, normalize_usage


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
