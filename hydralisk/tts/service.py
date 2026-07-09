"""Hydralisk TTS HTTP service (OAV-3).

Follows the Hydralisk proxy conventions: bearer auth on synthesis, public-safe
``/health`` and capabilities, and public-safe receipts. The synthesis route
streams raw seam PCM (16-bit / 24 kHz / mono) so the OAV-2 avatar render
service can lip-sync against chunks as they arrive.

Routes:

- ``GET  /health`` — public-safe status.
- ``GET  /hydralisk/tts/v1/capabilities`` — adapter, voice, PCM contract.
- ``POST /hydralisk/tts/v1/synthesize`` — bearer-authed; JSON
  ``{"text": "...", "voiceId": optional, "languageCode": optional}`` in,
  chunked ``audio/L16;rate=24000;channels=1`` out. Timing headers cannot be
  trailers, so per-run numbers land in the JSONL receipt referenced by the
  ``x-hydralisk-tts-run-ref`` / ``x-hydralisk-tts-receipt-ref`` headers.
- ``GET  /hydralisk/tts/v1/receipts/{run_ref}`` — public-safe receipt lookup.

Configuration (env):

- ``HYDRALISK_TTS_ADAPTER`` — ``chirp3hd`` (default) or ``cosyvoice``.
- ``HYDRALISK_TTS_BEARER_TOKEN`` — required unless
  ``HYDRALISK_TTS_ALLOW_INSECURE_DEV`` is truthy.
- ``HYDRALISK_TTS_RECEIPT_PATH`` — JSONL path
  (default ``.hydralisk/tts-receipts.jsonl``).
- Chirp: ``HYDRALISK_TTS_CHIRP_VOICE``, ``HYDRALISK_TTS_CHIRP_LANGUAGE``,
  plus Application Default Credentials.
- CosyVoice: ``HYDRALISK_TTS_COSYVOICE_PROMPT_WAV``,
  ``HYDRALISK_TTS_COSYVOICE_PROMPT_TEXT``, ``HYDRALISK_TTS_COSYVOICE_MODEL_DIR``,
  ``HYDRALISK_TTS_COSYVOICE_CHECKOUT``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
import uvicorn

from hydralisk.tts.receipts import (
    TTS_RECEIPT_SCHEMA,
    TtsReceiptLog,
    build_tts_receipt,
    tts_run_ref,
)
from hydralisk.tts.seam import (
    PCM_CHANNELS,
    PCM_MEDIA_TYPE,
    PCM_SAMPLE_RATE_HZ,
    PCM_SAMPLE_WIDTH_BYTES,
    TtsAdapter,
    VoiceRef,
    instrument_stream,
)

TTS_CAPABILITIES_SCHEMA = "hydralisk.tts.capabilities.v1"

MAX_TEXT_CHARS = 4_000


@dataclass(frozen=True)
class TtsServiceSettings:
    adapter_name: str = "chirp3hd"
    bearer_token: str | None = None
    allow_insecure_dev: bool = False
    receipt_path: Path = Path(".hydralisk/tts-receipts.jsonl")
    max_text_chars: int = MAX_TEXT_CHARS


def load_tts_settings() -> TtsServiceSettings:
    def _flag(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}

    return TtsServiceSettings(
        adapter_name=os.environ.get("HYDRALISK_TTS_ADAPTER", "").strip() or "chirp3hd",
        bearer_token=os.environ.get("HYDRALISK_TTS_BEARER_TOKEN", "").strip() or None,
        allow_insecure_dev=_flag("HYDRALISK_TTS_ALLOW_INSECURE_DEV"),
        receipt_path=Path(
            os.environ.get("HYDRALISK_TTS_RECEIPT_PATH", "").strip()
            or ".hydralisk/tts-receipts.jsonl"
        ),
        max_text_chars=int(
            os.environ.get("HYDRALISK_TTS_MAX_TEXT_CHARS", "").strip()
            or str(MAX_TEXT_CHARS)
        ),
    )


def _build_adapter(name: str) -> TtsAdapter:
    if name == "chirp3hd":
        from hydralisk.tts.chirp import default_chirp_adapter

        return default_chirp_adapter()
    if name == "cosyvoice":
        from hydralisk.tts.cosyvoice import default_cosyvoice_adapter

        return default_cosyvoice_adapter()
    raise RuntimeError(
        f"unknown HYDRALISK_TTS_ADAPTER {name!r}; expected 'chirp3hd' or 'cosyvoice'"
    )


@dataclass
class _ServiceMetrics:
    synth_requests_total: int = 0
    synth_completed_total: int = 0
    synth_errors_total: int = 0
    last_ms_to_first_chunk: int | None = None
    last_total_ms: int | None = None
    by_status: dict[str, int] = field(default_factory=dict)


def create_tts_app(
    settings: TtsServiceSettings | None = None,
    *,
    adapter: TtsAdapter | None = None,
) -> FastAPI:
    config = settings or load_tts_settings()
    receipts = TtsReceiptLog(config.receipt_path)
    metrics = _ServiceMetrics()

    app = FastAPI(
        title="Hydralisk TTS Service",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.hydralisk_tts_metrics = metrics

    resolved: dict[str, TtsAdapter] = {}
    if adapter is not None:
        resolved["adapter"] = adapter

    def get_adapter() -> TtsAdapter:
        if "adapter" not in resolved:
            resolved["adapter"] = _build_adapter(config.adapter_name)
        return resolved["adapter"]

    async def require_bearer(request: Request) -> None:
        if config.bearer_token is None and not config.allow_insecure_dev:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "hydralisk_tts_unarmed",
                        "message": "Hydralisk TTS bearer auth is not configured.",
                    }
                },
            )
        if config.bearer_token is None and config.allow_insecure_dev:
            return
        expected = f"Bearer {config.bearer_token}"
        if request.headers.get("authorization") != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": {
                        "code": "unauthorized",
                        "message": "Missing or invalid Hydralisk TTS bearer token.",
                    }
                },
            )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": (
                "ready"
                if config.bearer_token or config.allow_insecure_dev
                else "unarmed"
            ),
            "lane": "hydralisk-tts",
            "adapter": config.adapter_name,
            "authRequired": not config.allow_insecure_dev,
            "pcm": {
                "sampleRateHz": PCM_SAMPLE_RATE_HZ,
                "sampleWidthBytes": PCM_SAMPLE_WIDTH_BYTES,
                "channels": PCM_CHANNELS,
            },
        }

    @app.get("/hydralisk/tts/v1/capabilities")
    async def capabilities() -> dict[str, Any]:
        adapter_obj = get_adapter()
        return {
            "schema": TTS_CAPABILITIES_SCHEMA,
            "lane": "hydralisk-tts",
            "adapterRef": adapter_obj.adapter_ref,
            "defaultVoice": adapter_obj.default_voice.public_safe(),
            "streaming": True,
            "pcm": {
                "mediaType": PCM_MEDIA_TYPE,
                "sampleRateHz": PCM_SAMPLE_RATE_HZ,
                "sampleWidthBytes": PCM_SAMPLE_WIDTH_BYTES,
                "channels": PCM_CHANNELS,
            },
            "maxTextChars": config.max_text_chars,
            "receiptSchema": TTS_RECEIPT_SCHEMA,
            "publicSafe": True,
        }

    @app.get("/hydralisk/tts/v1/metrics")
    async def metrics_endpoint() -> dict[str, Any]:
        return {
            "schema": "hydralisk.tts.metrics.v1",
            "publicSafe": True,
            "adapter": config.adapter_name,
            "requests": {
                "total": metrics.synth_requests_total,
                "completed": metrics.synth_completed_total,
                "errors": metrics.synth_errors_total,
                "byStatus": dict(sorted(metrics.by_status.items())),
            },
            "latencyMs": {
                "lastMsToFirstChunk": metrics.last_ms_to_first_chunk,
                "lastTotalMs": metrics.last_total_ms,
            },
        }

    @app.get("/hydralisk/tts/v1/receipts/{run_ref}")
    async def receipt(run_ref: str) -> dict[str, Any]:
        for stored in receipts.read_all():
            if stored.get("runRef") == run_ref:
                return stored
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "receipt_not_found",
                    "message": "Hydralisk TTS receipt was not found.",
                }
            },
        )

    @app.post(
        "/hydralisk/tts/v1/synthesize",
        dependencies=[Depends(require_bearer)],
    )
    async def synthesize(request: Request) -> StreamingResponse:
        payload = await _json_object(request)
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "code": "missing_text",
                        "message": "Request body must include non-empty 'text'.",
                    }
                },
            )
        if len(text) > config.max_text_chars:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": {
                        "code": "text_too_long",
                        "message": (
                            f"Text exceeds maxTextChars={config.max_text_chars}."
                        ),
                    }
                },
            )
        voice_ref = _voice_ref_from_payload(payload)
        adapter_obj = get_adapter()
        run_ref = tts_run_ref()
        synthesis = instrument_stream(adapter_obj, text, voice_ref)
        metrics.synth_requests_total += 1

        async def pcm_chunks() -> AsyncIterator[bytes]:
            blockers: list[dict[str, str]] = []
            try:
                async for chunk in synthesis.stream:
                    yield chunk
            except Exception:
                blockers.append(
                    {
                        "code": "synthesis_stream_failed",
                        "message": "The TTS adapter stream failed mid-response.",
                    }
                )
                raise
            finally:
                if synthesis.metrics.bytes_out == 0 and not blockers:
                    blockers.append(
                        {
                            "code": "empty_synthesis",
                            "message": "The TTS adapter produced no audio.",
                        }
                    )
                receipts.append(
                    build_tts_receipt(
                        run_ref=run_ref,
                        metrics=synthesis.metrics,
                        blockers=blockers,
                    )
                )
                metrics.last_ms_to_first_chunk = synthesis.metrics.ms_to_first_chunk
                metrics.last_total_ms = synthesis.metrics.total_ms
                if blockers:
                    metrics.synth_errors_total += 1
                else:
                    metrics.synth_completed_total += 1

        return StreamingResponse(
            pcm_chunks(),
            media_type=PCM_MEDIA_TYPE,
            headers={
                "cache-control": "no-cache",
                "x-hydralisk-tts-run-ref": run_ref,
                "x-hydralisk-tts-receipt-ref": f"/hydralisk/tts/v1/receipts/{run_ref}",
                "x-hydralisk-tts-adapter": adapter_obj.adapter_ref,
                "x-hydralisk-tts-voice": (voice_ref or adapter_obj.default_voice).voice_id,
                "x-hydralisk-tts-sample-rate": str(PCM_SAMPLE_RATE_HZ),
            },
        )

    return app


def _voice_ref_from_payload(payload: dict[str, Any]) -> VoiceRef | None:
    voice_id = payload.get("voiceId")
    if voice_id is None:
        return None
    if not isinstance(voice_id, str) or not voice_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "invalid_voice_id",
                    "message": "'voiceId' must be a non-empty string when present.",
                }
            },
        )
    language = payload.get("languageCode")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "invalid_language_code",
                    "message": "'languageCode' must be a non-empty string when present.",
                }
            },
        )
    return VoiceRef(
        voice_id=voice_id.strip(),
        language_code=(language or "en-US").strip(),
    )


async def _json_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "invalid_json",
                    "message": "Request body must be valid JSON.",
                }
            },
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "invalid_request_body",
                    "message": "Request body must be a JSON object.",
                }
            },
        )
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Hydralisk TTS service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8022)
    args = parser.parse_args()
    uvicorn.run(create_tts_app(), host=args.host, port=args.port, proxy_headers=True)


if __name__ == "__main__":
    main()
