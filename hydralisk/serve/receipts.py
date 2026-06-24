from __future__ import annotations

from datetime import datetime, timezone
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
    capabilities = {
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
        "admission": _admission_policy(config),
        "policy": _policy_capabilities(config),
        "requestDefaults": _request_defaults(config),
        "toolCalls": "disabled_for_gateway_day_zero",
        "reasoningVisibility": "final_answer_only_for_public_receipts",
        "receiptSchema": RECEIPT_SCHEMA,
        "publicSafe": True,
        "blockers": blockers,
    }
    profile = _profile_evidence(config)
    if profile:
        capabilities["profile"] = profile
    return capabilities


def build_receipt(
    *,
    run_ref: str,
    served_alias: str,
    usage: dict[str, Any] | None,
    latency: dict[str, int | None],
    config: HydraliskSettings,
    policy_context: dict[str, Any] | None = None,
    blockers: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    all_blockers = [*_model_revision_blockers(config), *(blockers or [])]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "runRef": run_ref,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
        "admission": _admission_policy(config),
        "policy": _policy_receipt(config, policy_context),
        "requestDefaults": _request_defaults(config),
        "usage": normalize_usage(usage),
        "latency": latency,
        "publicSafe": True,
        "blockers": all_blockers,
    }
    profile = _profile_evidence(config)
    if profile:
        receipt["profile"] = profile
    return receipt


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


def _profile_evidence(config: HydraliskSettings) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    if config.model_profile_ref:
        profile["profileRef"] = config.model_profile_ref
    if config.container_image:
        profile["containerImage"] = config.container_image

    context = _compact(
        {
            "windowTokens": config.context_window_tokens,
            "admittedMaxTokens": config.admitted_context_tokens,
        }
    )
    if context:
        profile["context"] = context

    parallelism = _compact(
        {
            "tensor": config.tensor_parallel_size,
            "pipeline": config.pipeline_parallel_size,
            "data": config.data_parallel_size,
            "expert": config.expert_parallel_size,
        }
    )
    if parallelism:
        profile["parallelism"] = parallelism

    parsers = _compact(
        {
            "reasoning": config.reasoning_parser,
            "toolCalls": config.tool_call_parser,
        }
    )
    if parsers:
        profile["parsers"] = parsers

    cache = _compact(
        {
            "policy": config.cache_policy,
            "kvCacheDtype": config.kv_cache_dtype,
        }
    )
    if cache:
        profile["cache"] = cache

    if config.dynamo_mode:
        profile["dynamo"] = {"mode": config.dynamo_mode}
    if config.speculative_decoding:
        profile["speculation"] = {"mode": config.speculative_decoding}

    evidence = _compact(
        {
            "admissionRef": config.admission_ref,
            "evidenceRef": config.evidence_ref,
        }
    )
    if evidence:
        profile["evidence"] = evidence

    return profile


def _admission_policy(config: HydraliskSettings) -> dict[str, Any]:
    return {
        "maxInflightRequests": config.max_inflight_requests,
        "queueTimeoutSeconds": config.inflight_queue_timeout_seconds,
        "singleFlight": config.max_inflight_requests == 1,
    }


def _request_defaults(config: HydraliskSettings) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    sampling = _compact(
        {
            "minP": config.default_min_p,
            "repetitionPenalty": config.default_repetition_penalty,
            "maxTokens": config.default_max_tokens,
        }
    )
    if sampling:
        defaults["sampling"] = sampling
    if config.default_enable_thinking is not None:
        defaults["chatTemplateKwargs"] = {
            "enableThinking": config.default_enable_thinking
        }
    return defaults


def _policy_capabilities(config: HydraliskSettings) -> dict[str, Any]:
    authorized_security = config.model_policy == "authorized_security_lab_only"
    return {
        "mode": config.model_policy,
        "adapterRevision": config.adapter_revision,
        "authorizedSecurity": {
            "required": authorized_security,
            "scopeIdsConfigured": bool(config.authorized_security_scope_ids),
            "toolPoliciesConfigured": bool(config.authorized_security_tool_policies),
            "networkPoliciesConfigured": bool(
                config.authorized_security_network_policies
            ),
        },
    }


def _policy_receipt(
    config: HydraliskSettings,
    policy_context: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "mode": config.model_policy,
        "adapterRevision": config.adapter_revision,
        "authorization": policy_context,
    }


def _compact(source: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in source.items() if value is not None}
