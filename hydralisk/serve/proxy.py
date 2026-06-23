from __future__ import annotations

from collections.abc import AsyncIterator
import json
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import uvicorn

from hydralisk.serve.config import HydraliskSettings, load_settings
from hydralisk.serve.receipts import (
    ReceiptStore,
    build_capabilities,
    build_receipt,
    normalize_usage,
)


def create_app(settings: HydraliskSettings | None = None) -> FastAPI:
    config = settings or load_settings()
    receipts = ReceiptStore(config.receipt_dir)
    app = FastAPI(
        title="Hydralisk GPT-OSS 20B Proxy",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )

    async def require_bearer(request: Request) -> None:
        if config.bearer_token is None and not config.allow_insecure_dev:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "hydralisk_proxy_unarmed",
                        "message": "Hydralisk bearer auth is not configured.",
                    }
                },
            )
        if config.allow_insecure_dev and config.bearer_token is None:
            return
        expected = f"Bearer {config.bearer_token}"
        if request.headers.get("authorization") != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": {
                        "code": "unauthorized",
                        "message": "Missing or invalid Hydralisk bearer token.",
                    }
                },
            )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ready" if config.bearer_token or config.allow_insecure_dev else "unarmed",
            "servedModel": config.served_model,
            "engine": config.engine,
            "engineVersion": config.engine_version,
            "gpuClass": config.gpu_class,
            "authRequired": not config.allow_insecure_dev,
        }

    @app.get("/hydralisk/v1/capabilities")
    async def capabilities() -> dict[str, Any]:
        return build_capabilities(config)

    @app.get("/hydralisk/v1/receipts/{run_ref}")
    async def receipt(run_ref: str) -> dict[str, Any]:
        stored = receipts.read(run_ref)
        if stored is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "receipt_not_found",
                        "message": "Hydralisk receipt was not found.",
                    }
                },
            )
        return stored

    @app.post("/v1/chat/completions", dependencies=[Depends(require_bearer)])
    async def chat_completions(request: Request) -> Response:
        payload = await _json_object(request)
        admitted_model = _admit_model(payload, config)
        upstream_payload = dict(payload)
        upstream_payload["model"] = config.served_model
        run_ref = _run_ref()

        if upstream_payload.get("stream") is True:
            _request_stream_usage(upstream_payload)
            return await _stream_to_upstream(
                url=config.upstream_chat_url,
                payload=upstream_payload,
                config=config,
                receipts=receipts,
                run_ref=run_ref,
                admitted_model=admitted_model,
                headers={
                    "x-hydralisk-run-ref": run_ref,
                    "x-hydralisk-receipt-ref": f"/hydralisk/v1/receipts/{run_ref}",
                    "x-hydralisk-served-model": config.served_model,
                    "x-hydralisk-served-alias": admitted_model,
                },
            )

        started = perf_counter()
        async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as client:
            upstream = await client.post(config.upstream_chat_url, json=upstream_payload)
        wall_ms = int((perf_counter() - started) * 1000)
        return _json_upstream_response(
            upstream,
            run_ref=run_ref,
            admitted_model=admitted_model,
            config=config,
            receipts=receipts,
            wall_ms=wall_ms,
        )

    @app.post("/v1/responses", dependencies=[Depends(require_bearer)])
    async def responses(request: Request) -> Response:
        payload = await _json_object(request)
        admitted_model = _admit_model(payload, config)
        upstream_payload = dict(payload)
        upstream_payload["model"] = config.served_model
        run_ref = _run_ref()

        if upstream_payload.get("stream") is True:
            _request_stream_usage(upstream_payload)
            return await _stream_to_upstream(
                url=config.upstream_responses_url,
                payload=upstream_payload,
                config=config,
                receipts=receipts,
                run_ref=run_ref,
                admitted_model=admitted_model,
                headers={
                    "x-hydralisk-run-ref": run_ref,
                    "x-hydralisk-receipt-ref": f"/hydralisk/v1/receipts/{run_ref}",
                    "x-hydralisk-served-model": config.served_model,
                    "x-hydralisk-served-alias": admitted_model,
                },
            )

        started = perf_counter()
        async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as client:
            upstream = await client.post(config.upstream_responses_url, json=upstream_payload)
        wall_ms = int((perf_counter() - started) * 1000)
        return _json_upstream_response(
            upstream,
            run_ref=run_ref,
            admitted_model=admitted_model,
            config=config,
            receipts=receipts,
            wall_ms=wall_ms,
        )

    return app


async def _json_object(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
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


def _admit_model(payload: dict[str, Any], config: HydraliskSettings) -> str:
    requested_model = payload.get("model")
    if config.is_supported_model(requested_model):
        return str(requested_model)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={
            "error": {
                "code": "unsupported_model",
                "message": "Hydralisk refuses to proxy unsupported model ids.",
            },
            "supportedModels": list(config.supported_models),
        },
    )


async def _stream_to_upstream(
    *,
    url: str,
    payload: dict[str, Any],
    config: HydraliskSettings,
    receipts: ReceiptStore,
    run_ref: str,
    admitted_model: str,
    headers: dict[str, str],
) -> StreamingResponse:
    client = httpx.AsyncClient(timeout=None)
    started = perf_counter()
    request = client.build_request("POST", url, json=payload)
    upstream = await client.send(request, stream=True)
    if upstream.status_code >= 400:
        body = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        wall_ms = int((perf_counter() - started) * 1000)
        receipts.write(
            build_receipt(
                run_ref=run_ref,
                served_alias=admitted_model,
                usage=None,
                latency={"ttftMs": None, "wallMs": wall_ms},
                config=config,
                blockers=[
                    {
                        "code": "upstream_http_error",
                        "message": f"vLLM returned HTTP {upstream.status_code}.",
                    }
                ],
            )
        )
        return Response(
            content=body,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    stream_state: dict[str, Any] = {
        "buffer": "",
        "usage": None,
        "ttftMs": None,
    }

    async def chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                if stream_state["ttftMs"] is None and chunk:
                    stream_state["ttftMs"] = int((perf_counter() - started) * 1000)
                _capture_stream_usage(chunk, stream_state)
                yield chunk
        finally:
            wall_ms = int((perf_counter() - started) * 1000)
            usage = stream_state["usage"]
            blockers = []
            if normalize_usage(usage) is None:
                blockers.append(
                    {
                        "code": "stream_terminal_usage_unavailable",
                        "message": "The upstream stream ended without terminal usage.",
                    }
                )
            receipts.write(
                build_receipt(
                    run_ref=run_ref,
                    served_alias=admitted_model,
                    usage=usage,
                    latency={"ttftMs": stream_state["ttftMs"], "wallMs": wall_ms},
                    config=config,
                    blockers=blockers,
                )
            )
            await upstream.aclose()
            await client.aclose()

    response_headers = {
        "cache-control": "no-cache",
        **headers,
    }
    return StreamingResponse(
        chunks(),
        media_type=upstream.headers.get("content-type", "text/event-stream"),
        headers=response_headers,
    )


def _json_upstream_response(
    upstream: httpx.Response,
    *,
    run_ref: str,
    admitted_model: str,
    config: HydraliskSettings,
    receipts: ReceiptStore,
    wall_ms: int,
) -> JSONResponse:
    headers = {
        "x-hydralisk-run-ref": run_ref,
        "x-hydralisk-receipt-ref": f"/hydralisk/v1/receipts/{run_ref}",
        "x-hydralisk-served-model": config.served_model,
        "x-hydralisk-served-alias": admitted_model,
    }
    try:
        body = upstream.json()
    except json.JSONDecodeError:
        receipts.write(
            build_receipt(
                run_ref=run_ref,
                served_alias=admitted_model,
                usage=None,
                latency={"ttftMs": None, "wallMs": wall_ms},
                config=config,
                blockers=[
                    {
                        "code": "upstream_non_json_response",
                        "message": "vLLM returned a non-JSON response.",
                    }
                ],
            )
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            headers=headers,
            content={
                "error": {
                    "code": "upstream_non_json_response",
                    "message": "vLLM returned a non-JSON response.",
                },
                "hydralisk": {
                    "runRef": run_ref,
                    "servedModel": config.served_model,
                    "servedAlias": admitted_model,
                    "latency": {"wallMs": wall_ms},
                },
            },
        )

    if upstream.status_code >= 400:
        receipts.write(
            build_receipt(
                run_ref=run_ref,
                served_alias=admitted_model,
                usage=None,
                latency={"ttftMs": None, "wallMs": wall_ms},
                config=config,
                blockers=[
                    {
                        "code": "upstream_http_error",
                        "message": f"vLLM returned HTTP {upstream.status_code}.",
                    }
                ],
            )
        )
        return JSONResponse(status_code=upstream.status_code, headers=headers, content=body)

    usage = body.get("usage") if isinstance(body, dict) else None
    blockers = []
    if normalize_usage(usage) is None:
        blockers.append(
            {
                "code": "upstream_usage_unavailable",
                "message": "vLLM did not return a terminal usage object.",
            }
        )
    receipts.write(
        build_receipt(
            run_ref=run_ref,
            served_alias=admitted_model,
            usage=usage,
            latency={"ttftMs": None, "wallMs": wall_ms},
            config=config,
            blockers=blockers,
        )
    )

    if isinstance(body, dict):
        body.setdefault(
            "hydralisk",
            {
                "runRef": run_ref,
                "servedModel": config.served_model,
                "servedAlias": admitted_model,
                "latency": {"wallMs": wall_ms},
            },
        )
        if normalize_usage(usage) is None:
            body["hydralisk"]["usageBlocker"] = {
                "code": "upstream_usage_unavailable",
                "message": "vLLM did not return a terminal usage object.",
            }

    return JSONResponse(
        status_code=upstream.status_code,
        headers=headers,
        content=body,
    )


def _run_ref() -> str:
    return f"hydralisk-run-{uuid4().hex}"


def _request_stream_usage(payload: dict[str, Any]) -> None:
    stream_options = payload.get("stream_options")
    if not isinstance(stream_options, dict):
        stream_options = {}
    payload["stream_options"] = {**stream_options, "include_usage": True}


def _capture_stream_usage(chunk: bytes, stream_state: dict[str, Any]) -> None:
    text = chunk.decode("utf-8", errors="ignore")
    stream_state["buffer"] += text
    while "\n\n" in stream_state["buffer"]:
        event, stream_state["buffer"] = stream_state["buffer"].split("\n\n", 1)
        data_lines = [
            line.removeprefix("data:").strip()
            for line in event.splitlines()
            if line.startswith("data:")
        ]
        for data in data_lines:
            if not data or data == "[DONE]":
                continue
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = parsed.get("usage") if isinstance(parsed, dict) else None
            if usage is not None:
                stream_state["usage"] = usage


app = create_app()


def main() -> None:
    uvicorn.run(
        "hydralisk.serve.proxy:app",
        host="0.0.0.0",
        port=8080,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
