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


def create_app(settings: HydraliskSettings | None = None) -> FastAPI:
    config = settings or load_settings()
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

    @app.post("/v1/chat/completions", dependencies=[Depends(require_bearer)])
    async def chat_completions(request: Request) -> Response:
        payload = await _json_object(request)
        admitted_model = _admit_model(payload, config)
        upstream_payload = dict(payload)
        upstream_payload["model"] = config.served_model
        run_ref = _run_ref()

        if upstream_payload.get("stream") is True:
            return await _stream_to_upstream(
                url=config.upstream_chat_url,
                payload=upstream_payload,
                config=config,
                headers={
                    "x-hydralisk-run-ref": run_ref,
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
            return await _stream_to_upstream(
                url=config.upstream_responses_url,
                payload=upstream_payload,
                config=config,
                headers={
                    "x-hydralisk-run-ref": run_ref,
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
    headers: dict[str, str],
) -> StreamingResponse:
    client = httpx.AsyncClient(timeout=None)
    request = client.build_request("POST", url, json=payload)
    upstream = await client.send(request, stream=True)
    if upstream.status_code >= 400:
        body = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        return Response(
            content=body,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    async def chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_bytes():
                yield chunk
        finally:
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
    wall_ms: int,
) -> JSONResponse:
    headers = {
        "x-hydralisk-run-ref": run_ref,
        "x-hydralisk-served-model": config.served_model,
        "x-hydralisk-served-alias": admitted_model,
    }
    try:
        body = upstream.json()
    except json.JSONDecodeError:
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
        return JSONResponse(status_code=upstream.status_code, headers=headers, content=body)

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
        if "usage" not in body:
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
