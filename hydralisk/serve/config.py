from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_flag(name: str, default: bool = False) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on", "ready"}


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = _env(name)
    if value is None:
        return default
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or default


def _env_int(name: str, default: int | None = None) -> int | None:
    value = _env(name)
    if value is None:
        return default
    return int(value)


def _env_positive_int(name: str) -> int | None:
    value = _env_int(name)
    if value is None or value <= 0:
        return None
    return value


@dataclass(frozen=True)
class HydraliskSettings:
    served_model: str = "openai/gpt-oss-20b"
    public_model_aliases: tuple[str, ...] = (
        "khala",
        "openagents/khala",
        "openagents/khala-oss-20b",
        "gpt-oss-20b",
    )
    upstream_base_url: str = "http://127.0.0.1:8000"
    bearer_token: str | None = None
    allow_insecure_dev: bool = False
    engine: str = "vllm"
    engine_version: str = "unknown_engine_version"
    gpu_class: str = "l4"
    gpu_name: str = "NVIDIA L4"
    gpu_count: int = 1
    model_revision: str = "unknown_model_revision"
    quantization_weights: str = "MXFP4"
    model_profile_ref: str | None = None
    container_image: str | None = None
    context_window_tokens: int | None = None
    admitted_context_tokens: int | None = None
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    data_parallel_size: int | None = None
    expert_parallel_size: int | None = None
    reasoning_parser: str | None = None
    tool_call_parser: str | None = None
    cache_policy: str | None = None
    kv_cache_dtype: str | None = None
    dynamo_mode: str | None = None
    speculative_decoding: str | None = None
    admission_ref: str | None = None
    evidence_ref: str | None = None
    model_policy: str = "standard"
    adapter_revision: str | None = None
    authorized_security_scope_ids: tuple[str, ...] = ()
    authorized_security_tool_policies: tuple[str, ...] = ()
    authorized_security_network_policies: tuple[str, ...] = ()
    receipt_dir: Path = Path(".hydralisk/receipts")
    request_timeout_seconds: float = 600.0
    max_inflight_requests: int | None = None
    inflight_queue_timeout_seconds: float = 0.0

    @property
    def supported_models(self) -> tuple[str, ...]:
        return (self.served_model, *self.public_model_aliases)

    @property
    def upstream_chat_url(self) -> str:
        return f"{self.upstream_base_url.rstrip('/')}/v1/chat/completions"

    @property
    def upstream_responses_url(self) -> str:
        return f"{self.upstream_base_url.rstrip('/')}/v1/responses"

    def is_supported_model(self, model: Any) -> bool:
        return isinstance(model, str) and model in set(self.supported_models)


def load_settings() -> HydraliskSettings:
    return HydraliskSettings(
        served_model=_env("HYDRALISK_SERVED_MODEL", "openai/gpt-oss-20b")
        or "openai/gpt-oss-20b",
        public_model_aliases=_env_csv(
            "HYDRALISK_PUBLIC_MODEL_ALIASES",
            (
                "khala",
                "openagents/khala",
                "openagents/khala-oss-20b",
                "gpt-oss-20b",
            ),
        ),
        upstream_base_url=_env("HYDRALISK_VLLM_BASE_URL", "http://127.0.0.1:8000")
        or "http://127.0.0.1:8000",
        bearer_token=_env("HYDRALISK_BEARER_TOKEN"),
        allow_insecure_dev=_env_flag("HYDRALISK_ALLOW_INSECURE_DEV"),
        engine=_env("HYDRALISK_ENGINE", "vllm") or "vllm",
        engine_version=_env("HYDRALISK_ENGINE_VERSION", "unknown_engine_version")
        or "unknown_engine_version",
        gpu_class=_env("HYDRALISK_GPU_CLASS", "l4") or "l4",
        gpu_name=_env("HYDRALISK_GPU_NAME", "NVIDIA L4") or "NVIDIA L4",
        gpu_count=int(_env("HYDRALISK_GPU_COUNT", "1") or "1"),
        model_revision=_env("HYDRALISK_MODEL_REVISION", "unknown_model_revision")
        or "unknown_model_revision",
        quantization_weights=_env("HYDRALISK_QUANTIZATION_WEIGHTS", "MXFP4")
        or "MXFP4",
        model_profile_ref=_env("HYDRALISK_MODEL_PROFILE_REF"),
        container_image=_env("HYDRALISK_CONTAINER_IMAGE"),
        context_window_tokens=_env_int("HYDRALISK_CONTEXT_WINDOW_TOKENS"),
        admitted_context_tokens=_env_int("HYDRALISK_ADMITTED_CONTEXT_TOKENS"),
        tensor_parallel_size=_env_int("HYDRALISK_TENSOR_PARALLEL_SIZE"),
        pipeline_parallel_size=_env_int("HYDRALISK_PIPELINE_PARALLEL_SIZE"),
        data_parallel_size=_env_int("HYDRALISK_DATA_PARALLEL_SIZE"),
        expert_parallel_size=_env_int("HYDRALISK_EXPERT_PARALLEL_SIZE"),
        reasoning_parser=_env("HYDRALISK_REASONING_PARSER"),
        tool_call_parser=_env("HYDRALISK_TOOL_CALL_PARSER"),
        cache_policy=_env("HYDRALISK_CACHE_POLICY"),
        kv_cache_dtype=_env("HYDRALISK_KV_CACHE_DTYPE"),
        dynamo_mode=_env("HYDRALISK_DYNAMO_MODE"),
        speculative_decoding=_env("HYDRALISK_SPECULATIVE_DECODING"),
        admission_ref=_env("HYDRALISK_ADMISSION_REF"),
        evidence_ref=_env("HYDRALISK_EVIDENCE_REF"),
        model_policy=_env("HYDRALISK_MODEL_POLICY", "standard") or "standard",
        adapter_revision=_env("HYDRALISK_ADAPTER_REVISION"),
        authorized_security_scope_ids=_env_csv(
            "HYDRALISK_AUTHORIZED_SECURITY_SCOPE_IDS",
            (),
        ),
        authorized_security_tool_policies=_env_csv(
            "HYDRALISK_AUTHORIZED_SECURITY_TOOL_POLICIES",
            (),
        ),
        authorized_security_network_policies=_env_csv(
            "HYDRALISK_AUTHORIZED_SECURITY_NETWORK_POLICIES",
            (),
        ),
        receipt_dir=Path(
            _env("HYDRALISK_RECEIPT_DIR", ".hydralisk/receipts")
            or ".hydralisk/receipts"
        ),
        request_timeout_seconds=float(
            _env("HYDRALISK_REQUEST_TIMEOUT_SECONDS", "600") or "600"
        ),
        max_inflight_requests=_env_positive_int("HYDRALISK_MAX_INFLIGHT_REQUESTS"),
        inflight_queue_timeout_seconds=float(
            _env("HYDRALISK_INFLIGHT_QUEUE_TIMEOUT_SECONDS", "0") or "0"
        ),
    )
