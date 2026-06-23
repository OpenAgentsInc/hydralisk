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


@dataclass(frozen=True)
class HydraliskSettings:
    served_model: str = "openai/gpt-oss-20b"
    public_model_aliases: tuple[str, ...] = (
        "openagents/khala-oss-20b",
        "gpt-oss-20b",
    )
    upstream_base_url: str = "http://127.0.0.1:8000"
    bearer_token: str | None = None
    allow_insecure_dev: bool = False
    engine: str = "vllm"
    engine_version: str = "0.10.1+gptoss"
    gpu_class: str = "l4"
    gpu_name: str = "NVIDIA L4"
    gpu_count: int = 1
    model_revision: str = "unknown_model_revision"
    quantization_weights: str = "MXFP4"
    receipt_dir: Path = Path(".hydralisk/receipts")
    request_timeout_seconds: float = 600.0

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
            ("openagents/khala-oss-20b", "gpt-oss-20b"),
        ),
        upstream_base_url=_env("HYDRALISK_VLLM_BASE_URL", "http://127.0.0.1:8000")
        or "http://127.0.0.1:8000",
        bearer_token=_env("HYDRALISK_BEARER_TOKEN"),
        allow_insecure_dev=_env_flag("HYDRALISK_ALLOW_INSECURE_DEV"),
        engine=_env("HYDRALISK_ENGINE", "vllm") or "vllm",
        engine_version=_env("HYDRALISK_ENGINE_VERSION", "0.10.1+gptoss")
        or "0.10.1+gptoss",
        gpu_class=_env("HYDRALISK_GPU_CLASS", "l4") or "l4",
        gpu_name=_env("HYDRALISK_GPU_NAME", "NVIDIA L4") or "NVIDIA L4",
        gpu_count=int(_env("HYDRALISK_GPU_COUNT", "1") or "1"),
        model_revision=_env("HYDRALISK_MODEL_REVISION", "unknown_model_revision")
        or "unknown_model_revision",
        quantization_weights=_env("HYDRALISK_QUANTIZATION_WEIGHTS", "MXFP4")
        or "MXFP4",
        receipt_dir=Path(
            _env("HYDRALISK_RECEIPT_DIR", ".hydralisk/receipts")
            or ".hydralisk/receipts"
        ),
        request_timeout_seconds=float(
            _env("HYDRALISK_REQUEST_TIMEOUT_SECONDS", "600") or "600"
        ),
    )
