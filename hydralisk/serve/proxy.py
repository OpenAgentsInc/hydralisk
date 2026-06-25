from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
import json
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
import uvicorn

from hydralisk.serve.config import HydraliskSettings, load_settings
from hydralisk.serve.receipts import (
    ReceiptStore,
    build_capabilities,
    build_replica_capabilities,
    build_receipt,
    normalize_usage,
)


class _InflightLease:
    def __init__(self, release_once: Callable[[], None]) -> None:
        self._release_once = release_once
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._release_once()


class _InflightGate:
    def __init__(
        self,
        *,
        limit: int | None,
        queue_timeout_seconds: float,
    ) -> None:
        self.limit = limit
        self.queue_timeout_seconds = max(queue_timeout_seconds, 0.0)
        self._semaphore = asyncio.Semaphore(limit) if limit else None
        self.current = 0
        self.busy_rejections_total = 0
        self.last_busy_at: datetime | None = None

    async def acquire(self) -> _InflightLease:
        if self._semaphore is None:
            self.current += 1
            return _InflightLease(self._release_unbounded)
        if self.queue_timeout_seconds == 0 and self._semaphore.locked():
            self._raise_saturated()
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.queue_timeout_seconds if self.queue_timeout_seconds else None,
            )
        except TimeoutError:
            self._raise_saturated()
        self.current += 1
        return _InflightLease(self._release_bounded)

    def _release_unbounded(self) -> None:
        self.current = max(0, self.current - 1)

    def _release_bounded(self) -> None:
        self.current = max(0, self.current - 1)
        if self._semaphore is not None:
            self._semaphore.release()

    def _raise_saturated(self) -> None:
        self.busy_rejections_total += 1
        self.last_busy_at = datetime.now(timezone.utc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "code": "hydralisk_inflight_saturated",
                    "message": "Hydralisk is at its configured inflight request limit.",
                },
                "admission": {
                    "maxInflightRequests": self.limit,
                    "queueTimeoutSeconds": self.queue_timeout_seconds,
                    "singleFlight": self.limit == 1,
                },
            },
        )


class _ProxyMetrics:
    def __init__(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()
        self.requests_total = 0
        self.responses_total = 0
        self.errors_total = 0
        self.latency_count = 0
        self.latency_total_ms = 0
        self.latency_max_ms = 0
        self.by_route: dict[str, dict[str, int]] = {}
        self.by_status: dict[str, int] = {}

    async def record_started(self, route: str) -> None:
        async with self._lock:
            self.requests_total += 1
            route_metrics = self.by_route.setdefault(
                route,
                {
                    "requests": 0,
                    "responses": 0,
                    "errors": 0,
                },
            )
            route_metrics["requests"] += 1

    async def record_finished(
        self,
        route: str,
        *,
        status_code: int,
        wall_ms: int,
        error: bool = False,
    ) -> None:
        status_key = str(status_code)
        async with self._lock:
            self.responses_total += 1
            self.by_status[status_key] = self.by_status.get(status_key, 0) + 1
            self.latency_count += 1
            self.latency_total_ms += max(wall_ms, 0)
            self.latency_max_ms = max(self.latency_max_ms, max(wall_ms, 0))
            route_metrics = self.by_route.setdefault(
                route,
                {
                    "requests": 0,
                    "responses": 0,
                    "errors": 0,
                },
            )
            route_metrics["responses"] += 1
            if error or status_code >= 400:
                self.errors_total += 1
                route_metrics["errors"] += 1

    async def snapshot(
        self,
        *,
        config: HydraliskSettings,
        inflight_gate: _InflightGate,
    ) -> dict[str, Any]:
        async with self._lock:
            average_latency = (
                self.latency_total_ms / self.latency_count
                if self.latency_count
                else None
            )
            return {
                "schema": "hydralisk.serve.metrics.v1",
                "publicSafe": True,
                "startedAt": self.started_at.isoformat(),
                "checkedAt": datetime.now(timezone.utc).isoformat(),
                "servedModel": config.served_model,
                "engine": config.engine,
                "engineVersion": config.engine_version,
                "modelProfileRef": config.model_profile_ref,
                "evidenceRef": config.evidence_ref,
                "gpu": {
                    "class": config.gpu_class,
                    "name": config.gpu_name,
                    "count": config.gpu_count,
                },
                "inflight": {
                    "current": inflight_gate.current,
                    "limit": inflight_gate.limit,
                    "queueTimeoutSeconds": inflight_gate.queue_timeout_seconds,
                    "singleFlight": inflight_gate.limit == 1,
                    "backpressure": _backpressure_snapshot(inflight_gate),
                },
                "replica": _replica_metrics_snapshot(config, inflight_gate),
                "requests": {
                    "total": self.requests_total,
                    "responses": self.responses_total,
                    "errors": self.errors_total,
                    "byRoute": dict(sorted(self.by_route.items())),
                    "byStatus": dict(sorted(self.by_status.items())),
                },
                "latencyMs": {
                    "count": self.latency_count,
                    "average": round(average_latency, 3)
                    if average_latency is not None
                    else None,
                    "max": self.latency_max_ms,
                },
            }


def _backpressure_snapshot(inflight_gate: _InflightGate) -> dict[str, Any]:
    saturated = (
        inflight_gate.limit is not None and inflight_gate.current >= inflight_gate.limit
    )
    return {
        "busy": saturated,
        "busyRejectsTotal": inflight_gate.busy_rejections_total,
        "lastBusyAt": (
            inflight_gate.last_busy_at.isoformat()
            if inflight_gate.last_busy_at is not None
            else None
        ),
        "lastBusyStatus": (
            status.HTTP_429_TOO_MANY_REQUESTS
            if inflight_gate.last_busy_at is not None
            else None
        ),
    }


def _replica_metrics_snapshot(
    config: HydraliskSettings,
    inflight_gate: _InflightGate,
) -> dict[str, Any]:
    replica = build_replica_capabilities(config)
    replica["capacity"] = {
        "inflight": inflight_gate.current,
        "maxInflight": inflight_gate.limit,
        "queueTimeoutSeconds": inflight_gate.queue_timeout_seconds,
        "singleFlight": inflight_gate.limit == 1,
        "backpressure": _backpressure_snapshot(inflight_gate),
    }
    replica["warmState"] = _keepwarm_state_snapshot(config)
    return replica


def _keepwarm_state_snapshot(config: HydraliskSettings) -> dict[str, Any]:
    path = config.keepwarm_status_path
    if path is None:
        return {
            "configured": False,
            "lastKeepWarmAt": None,
            "lastKeepWarmStatus": None,
            "lastKeepWarmHttpStatus": None,
            "lastKeepWarmWallSeconds": None,
            "lastKeepWarmTokens": None,
        }
    if not path.exists():
        return {
            "configured": True,
            "statusPathRef": "host-local-public-json",
            "lastKeepWarmAt": None,
            "lastKeepWarmStatus": "missing",
            "lastKeepWarmHttpStatus": None,
            "lastKeepWarmWallSeconds": None,
            "lastKeepWarmTokens": None,
        }
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {
            "configured": True,
            "statusPathRef": "host-local-public-json",
            "lastKeepWarmAt": None,
            "lastKeepWarmStatus": "unreadable",
            "lastKeepWarmHttpStatus": None,
            "lastKeepWarmWallSeconds": None,
            "lastKeepWarmTokens": None,
        }
    if not isinstance(payload, dict):
        return {
            "configured": True,
            "statusPathRef": "host-local-public-json",
            "lastKeepWarmAt": None,
            "lastKeepWarmStatus": "unreadable",
            "lastKeepWarmHttpStatus": None,
            "lastKeepWarmWallSeconds": None,
            "lastKeepWarmTokens": None,
        }

    usage = normalize_usage(payload.get("usage"))
    timing = payload.get("timing") if isinstance(payload.get("timing"), dict) else {}
    return {
        "configured": True,
        "statusPathRef": "host-local-public-json",
        "lastKeepWarmAt": payload.get("checkedAt"),
        "lastKeepWarmStatus": payload.get("status"),
        "lastKeepWarmHttpStatus": payload.get("httpStatus"),
        "lastKeepWarmWallSeconds": timing.get("wallSeconds"),
        "lastKeepWarmTokens": usage,
    }


def create_app(settings: HydraliskSettings | None = None) -> FastAPI:
    config = settings or load_settings()
    receipts = ReceiptStore(config.receipt_dir)
    inflight_gate = _InflightGate(
        limit=config.max_inflight_requests,
        queue_timeout_seconds=config.inflight_queue_timeout_seconds,
    )
    metrics = _ProxyMetrics()
    app = FastAPI(
        title="Hydralisk GPT-OSS 20B Proxy",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
    )
    app.state.hydralisk_inflight_gate = inflight_gate
    app.state.hydralisk_metrics = metrics

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

    async def require_ready() -> None:
        blockers = _readiness_blockers(config)
        if blockers:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "hydralisk_profile_evidence_incomplete",
                        "message": "Hydralisk profile evidence is incomplete for this private lane.",
                    },
                    "blockers": blockers,
                },
            )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        blockers = _readiness_blockers(config)
        return {
            "status": (
                "blocked"
                if blockers
                else "ready"
                if config.bearer_token or config.allow_insecure_dev
                else "unarmed"
            ),
            "servedModel": config.served_model,
            "engine": config.engine,
            "engineVersion": config.engine_version,
            "gpuClass": config.gpu_class,
            "authRequired": not config.allow_insecure_dev,
            "blockers": blockers,
        }

    @app.get("/hydralisk/v1/capabilities")
    async def capabilities() -> dict[str, Any]:
        return build_capabilities(config)

    @app.get("/hydralisk/v1/metrics")
    async def metrics_endpoint() -> dict[str, Any]:
        return await metrics.snapshot(config=config, inflight_gate=inflight_gate)

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

    @app.get("/v1/models", dependencies=[Depends(require_bearer), Depends(require_ready)])
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": config.served_model,
                    "object": "model",
                    "created": int(datetime.now(timezone.utc).timestamp()),
                    "owned_by": "hydralisk-private",
                    "root": config.served_model,
                    "parent": None,
                    "permission": [],
                    "hydralisk": {
                        "aliases": list(config.public_model_aliases),
                        "engine": config.engine,
                        "engineVersion": config.engine_version,
                        "profileRef": config.model_profile_ref,
                        "evidenceRef": config.evidence_ref,
                        "containerImage": config.container_image,
                        "requestDefaults": _request_defaults_for_response(config),
                    },
                }
            ],
        }

    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(require_bearer), Depends(require_ready)],
    )
    async def chat_completions(request: Request) -> Response:
        payload = await _json_object(request)
        admitted_model = _admit_model(payload, config)
        policy_context = _admit_policy(payload, config)
        upstream_payload = dict(payload)
        upstream_payload["model"] = config.served_model
        _apply_chat_defaults(upstream_payload, config)
        run_ref = _run_ref()
        inflight_lease = await inflight_gate.acquire()

        if upstream_payload.get("stream") is True:
            _request_stream_usage(upstream_payload)
            await metrics.record_started("chat_completions")
            stream_started = perf_counter()
            try:
                return await _stream_to_upstream(
                    url=config.upstream_chat_url,
                    payload=upstream_payload,
                    config=config,
                    receipts=receipts,
                    run_ref=run_ref,
                    admitted_model=admitted_model,
                    policy_context=policy_context,
                    headers={
                        "x-hydralisk-run-ref": run_ref,
                        "x-hydralisk-receipt-ref": f"/hydralisk/v1/receipts/{run_ref}",
                        "x-hydralisk-served-model": config.served_model,
                        "x-hydralisk-served-alias": admitted_model,
                    },
                    release_inflight=inflight_lease.release,
                    metrics=metrics,
                    metrics_route="chat_completions",
                )
            except Exception:
                await metrics.record_finished(
                    "chat_completions",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    wall_ms=int((perf_counter() - stream_started) * 1000),
                    error=True,
                )
                inflight_lease.release()
                raise

        await metrics.record_started("chat_completions")
        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as client:
                upstream = await client.post(config.upstream_chat_url, json=upstream_payload)
            wall_ms = int((perf_counter() - started) * 1000)
            response = _json_upstream_response(
                upstream,
                run_ref=run_ref,
                admitted_model=admitted_model,
                config=config,
                receipts=receipts,
                wall_ms=wall_ms,
                policy_context=policy_context,
            )
            await metrics.record_finished(
                "chat_completions",
                status_code=response.status_code,
                wall_ms=wall_ms,
            )
            return response
        except Exception:
            await metrics.record_finished(
                "chat_completions",
                status_code=status.HTTP_502_BAD_GATEWAY,
                wall_ms=int((perf_counter() - started) * 1000),
                error=True,
            )
            raise
        finally:
            inflight_lease.release()

    @app.post(
        "/v1/responses",
        dependencies=[Depends(require_bearer), Depends(require_ready)],
    )
    async def responses(request: Request) -> Response:
        payload = await _json_object(request)
        admitted_model = _admit_model(payload, config)
        policy_context = _admit_policy(payload, config)
        upstream_payload = dict(payload)
        upstream_payload["model"] = config.served_model
        run_ref = _run_ref()
        inflight_lease = await inflight_gate.acquire()

        if upstream_payload.get("stream") is True:
            _request_stream_usage(upstream_payload)
            await metrics.record_started("responses")
            stream_started = perf_counter()
            try:
                return await _stream_to_upstream(
                    url=config.upstream_responses_url,
                    payload=upstream_payload,
                    config=config,
                    receipts=receipts,
                    run_ref=run_ref,
                    admitted_model=admitted_model,
                    policy_context=policy_context,
                    headers={
                        "x-hydralisk-run-ref": run_ref,
                        "x-hydralisk-receipt-ref": f"/hydralisk/v1/receipts/{run_ref}",
                        "x-hydralisk-served-model": config.served_model,
                        "x-hydralisk-served-alias": admitted_model,
                    },
                    release_inflight=inflight_lease.release,
                    metrics=metrics,
                    metrics_route="responses",
                )
            except Exception:
                await metrics.record_finished(
                    "responses",
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    wall_ms=int((perf_counter() - stream_started) * 1000),
                    error=True,
                )
                inflight_lease.release()
                raise

        await metrics.record_started("responses")
        started = perf_counter()
        try:
            async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as client:
                upstream = await client.post(config.upstream_responses_url, json=upstream_payload)
            wall_ms = int((perf_counter() - started) * 1000)
            response = _json_upstream_response(
                upstream,
                run_ref=run_ref,
                admitted_model=admitted_model,
                config=config,
                receipts=receipts,
                wall_ms=wall_ms,
                policy_context=policy_context,
            )
            await metrics.record_finished(
                "responses",
                status_code=response.status_code,
                wall_ms=wall_ms,
            )
            return response
        except Exception:
            await metrics.record_finished(
                "responses",
                status_code=status.HTTP_502_BAD_GATEWAY,
                wall_ms=int((perf_counter() - started) * 1000),
                error=True,
            )
            raise
        finally:
            inflight_lease.release()

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


def _apply_chat_defaults(payload: dict[str, Any], config: HydraliskSettings) -> None:
    if config.default_min_p is not None:
        payload.setdefault("min_p", config.default_min_p)
    if config.default_repetition_penalty is not None:
        payload.setdefault("repetition_penalty", config.default_repetition_penalty)
    if config.default_max_tokens is not None:
        payload.setdefault("max_tokens", config.default_max_tokens)
    if config.default_enable_thinking is not None:
        chat_template_kwargs = payload.get("chat_template_kwargs")
        if not isinstance(chat_template_kwargs, dict):
            chat_template_kwargs = {}
        payload["chat_template_kwargs"] = {
            **chat_template_kwargs,
            "enable_thinking": chat_template_kwargs.get(
                "enable_thinking",
                config.default_enable_thinking,
            ),
        }


def _readiness_blockers(config: HydraliskSettings) -> list[dict[str, str]]:
    if not config.require_profile_evidence:
        return []
    blockers: list[dict[str, str]] = []
    if config.model_revision == "unknown_model_revision":
        blockers.append(
            {
                "code": "unknown_model_revision",
                "message": "HYDRALISK_MODEL_REVISION has not been pinned for this lane.",
            }
        )
    if config.engine_version == "unknown_engine_version":
        blockers.append(
            {
                "code": "unknown_engine_version",
                "message": "HYDRALISK_ENGINE_VERSION has not been pinned for this lane.",
            }
        )
    required_refs = {
        "missing_model_profile_ref": config.model_profile_ref,
        "missing_container_image": config.container_image,
        "missing_admission_ref": config.admission_ref,
        "missing_evidence_ref": config.evidence_ref,
    }
    for code, value in required_refs.items():
        if not value:
            blockers.append(
                {
                    "code": code,
                    "message": f"{code.removeprefix('missing_')} is required for this private lane.",
                }
            )
    if not str(config.receipt_dir).strip():
        blockers.append(
            {
                "code": "missing_receipt_dir",
                "message": "HYDRALISK_RECEIPT_DIR is required for this private lane.",
            }
        )
    return blockers


def _request_defaults_for_response(config: HydraliskSettings) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    sampling: dict[str, Any] = {}
    if config.default_min_p is not None:
        sampling["min_p"] = config.default_min_p
    if config.default_repetition_penalty is not None:
        sampling["repetition_penalty"] = config.default_repetition_penalty
    if config.default_max_tokens is not None:
        sampling["max_tokens"] = config.default_max_tokens
    if sampling:
        defaults["sampling"] = sampling
    if config.default_enable_thinking is not None:
        defaults["chat_template_kwargs"] = {
            "enable_thinking": config.default_enable_thinking
        }
    return defaults


def _admit_policy(payload: dict[str, Any], config: HydraliskSettings) -> dict[str, Any] | None:
    if config.model_policy != "authorized_security_lab_only":
        return None

    metadata = payload.get("metadata")
    context = None
    if isinstance(metadata, dict):
        context = metadata.get("hydraliskAuthorizedSecurity")
        if context is None:
            context = metadata.get("hydralisk_authorized_security")
    if not isinstance(context, dict):
        _raise_policy_denied(
            "authorized_security_metadata_required",
            "Authorized-security metadata is required for this model policy.",
        )

    scope_id = _required_policy_string(context, "scopeId", "scope_id")
    authorization_ref = _required_policy_string(
        context,
        "authorizationRef",
        "authorization_ref",
        "labRunId",
        "lab_run_id",
    )
    tool_policy = _required_policy_string(context, "toolPolicy", "tool_policy")
    network_policy = _required_policy_string(context, "networkPolicy", "network_policy")

    if (
        config.authorized_security_scope_ids
        and scope_id not in set(config.authorized_security_scope_ids)
    ):
        _raise_policy_denied(
            "authorized_security_scope_not_allowed",
            "Authorized-security scope is not configured for this Hydralisk lane.",
        )
    if (
        config.authorized_security_tool_policies
        and tool_policy not in set(config.authorized_security_tool_policies)
    ):
        _raise_policy_denied(
            "authorized_security_tool_policy_not_allowed",
            "Authorized-security tool policy is not configured for this Hydralisk lane.",
        )
    if (
        config.authorized_security_network_policies
        and network_policy not in set(config.authorized_security_network_policies)
    ):
        _raise_policy_denied(
            "authorized_security_network_policy_not_allowed",
            "Authorized-security network policy is not configured for this Hydralisk lane.",
        )

    return {
        "admissionResult": "admitted",
        "scopeId": scope_id,
        "authorizationRef": authorization_ref,
        "toolPolicy": tool_policy,
        "networkPolicy": network_policy,
    }


def _required_policy_string(source: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    _raise_policy_denied(
        "authorized_security_metadata_incomplete",
        "Authorized-security metadata is missing a required field.",
    )


def _raise_policy_denied(code: str, message: str) -> None:
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "error": {
                "code": code,
                "message": message,
            },
            "policy": {
                "mode": "authorized_security_lab_only",
                "admissionResult": "rejected",
            },
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
    policy_context: dict[str, Any] | None,
    headers: dict[str, str],
    release_inflight: Callable[[], None] | None = None,
    metrics: _ProxyMetrics | None = None,
    metrics_route: str | None = None,
) -> StreamingResponse:
    client = httpx.AsyncClient(timeout=None)
    started = perf_counter()
    request = client.build_request("POST", url, json=payload)
    upstream = await client.send(request, stream=True)
    if upstream.status_code >= 400:
        body = await upstream.aread()
        await upstream.aclose()
        await client.aclose()
        if release_inflight is not None:
            release_inflight()
        wall_ms = int((perf_counter() - started) * 1000)
        if metrics is not None and metrics_route is not None:
            await metrics.record_finished(
                metrics_route,
                status_code=upstream.status_code,
                wall_ms=wall_ms,
            )
        receipts.write(
            build_receipt(
                run_ref=run_ref,
                served_alias=admitted_model,
                usage=None,
                latency={"ttftMs": None, "wallMs": wall_ms},
                config=config,
                policy_context=policy_context,
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
                    policy_context=policy_context,
                    blockers=blockers,
                )
            )
            await upstream.aclose()
            await client.aclose()
            if release_inflight is not None:
                release_inflight()
            if metrics is not None and metrics_route is not None:
                await metrics.record_finished(
                    metrics_route,
                    status_code=upstream.status_code,
                    wall_ms=wall_ms,
                    error=bool(blockers),
                )

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
    policy_context: dict[str, Any] | None,
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
                policy_context=policy_context,
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
                policy_context=policy_context,
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
            policy_context=policy_context,
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
