from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import re
from typing import Any

from hydralisk.serve.config import HydraliskSettings


RECEIPT_SCHEMA = "hydralisk.serve.run_receipt.v1"
CAPABILITIES_SCHEMA = "hydralisk.serve.capabilities.v1"
RUN_REF_PATTERN = re.compile(r"^hydralisk-run-[a-f0-9]{32}$")


class ReceiptStore:
    def __init__(self, receipt_dir: Path) -> None:
        self.receipt_dir = receipt_dir

    def write(self, receipt: dict[str, Any]) -> None:
        run_ref = str(receipt.get("runRef", ""))
        if not RUN_REF_PATTERN.match(run_ref):
            raise ValueError("invalid Hydralisk runRef")
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        target = self.receipt_dir / f"{run_ref}.json"
        temp = self.receipt_dir / f".{run_ref}.tmp"
        temp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        temp.replace(target)

    def read(self, run_ref: str) -> dict[str, Any] | None:
        if not RUN_REF_PATTERN.match(run_ref):
            return None
        target = self.receipt_dir / f"{run_ref}.json"
        if not target.exists():
            return None
        return json.loads(target.read_text())


def build_capabilities(config: HydraliskSettings) -> dict[str, Any]:
    blockers = _model_revision_blockers(config)
    return {
        "schema": CAPABILITIES_SCHEMA,
        "servedModel": config.served_model,
        "publicModelAliases": list(config.public_model_aliases),
        "supportedModels": list(config.supported_models),
        "engine": config.engine,
        "engineVersion": config.engine_version,
        "modelRevision": None if blockers else config.model_revision,
        "gpuClass": config.gpu_class,
        "gpu": {
            "name": config.gpu_name,
            "class": config.gpu_class,
            "count": config.gpu_count,
        },
        "quantization": {"weights": config.quantization_weights},
        "chatCompletions": True,
        "responses": True,
        "toolCalls": "disabled_for_gateway_day_zero",
        "reasoningVisibility": "final_answer_only_for_public_receipts",
        "receiptSchema": RECEIPT_SCHEMA,
        "publicSafe": True,
        "blockers": blockers,
    }


def build_receipt(
    *,
    run_ref: str,
    served_alias: str,
    usage: dict[str, Any] | None,
    latency: dict[str, int | None],
    config: HydraliskSettings,
    blockers: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    all_blockers = [*_model_revision_blockers(config), *(blockers or [])]
    return {
        "schema": RECEIPT_SCHEMA,
        "runRef": run_ref,
        "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model": config.served_model,
        "servedModel": config.served_model,
        "servedAlias": served_alias,
        "engine": config.engine,
        "engineVersion": config.engine_version,
        "modelRevision": None if _model_revision_blockers(config) else config.model_revision,
        "gpu": {
            "name": config.gpu_name,
            "class": config.gpu_class,
            "count": config.gpu_count,
        },
        "quantization": {"weights": config.quantization_weights},
        "usage": normalize_usage(usage),
        "latency": latency,
        "publicSafe": True,
        "blockers": all_blockers,
    }


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None
    prompt = _int_value(usage, "prompt_tokens", "input_tokens", "promptTokens")
    completion = _int_value(
        usage,
        "completion_tokens",
        "output_tokens",
        "completionTokens",
    )
    total = _int_value(usage, "total_tokens", "totalTokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    if prompt is None or completion is None or total is None:
        return None
    return {
        "promptTokens": prompt,
        "completionTokens": completion,
        "totalTokens": total,
    }


def _int_value(source: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _model_revision_blockers(config: HydraliskSettings) -> list[dict[str, str]]:
    if config.model_revision != "unknown_model_revision":
        return []
    return [
        {
            "code": "unknown_model_revision",
            "message": "HYDRALISK_MODEL_REVISION has not been pinned for this lane.",
        }
    ]
