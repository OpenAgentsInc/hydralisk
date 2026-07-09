"""HTTP + WebSocket control API for the avatar render service.

Bearer-authed, fail-closed, mirroring the Hydralisk proxy posture:

- `GET  /healthz`                         public liveness + readiness
- `GET  /avatar/capabilities`             public-safe capability manifest
- `POST /avatar/sessions`                 create a session (bearer)
- `GET  /avatar/sessions/{ref}`           session status (bearer)
- `POST /avatar/sessions/{ref}/stop`      stop + summary receipt (bearer)
- `POST /avatar/sessions/{ref}/webrtc`    SDP offer → answer (bearer;
                                          503 when aiortc is not installed)
- `WS   /avatar/sessions/{ref}/control`   LITE-cycle `agent.*` control
                                          (bearer header or ?token=)

The control WebSocket is the contract the apps/sarah OAV-4 seam calls; see
`hydralisk/avatar/protocol.py` for the exact message shapes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import PlainTextResponse
import uvicorn

from hydralisk.avatar.clips import CLIP_CATALOG, STATE_CLIP_CYCLE
from hydralisk.avatar.config import AvatarSettings, load_avatar_settings
from hydralisk.avatar.egress import NullEgress, WebRTCEgress, webrtc_available
from hydralisk.avatar.protocol import (
    PROTOCOL_VERSION,
    ControlType,
    ProtocolError,
    error_event,
    parse_control_message,
    state_event,
)
from hydralisk.avatar.renderer import SharedRenderer, select_renderer
from hydralisk.avatar.session import SessionLimitError, SessionManager

CAPABILITIES_SCHEMA = "hydralisk.avatar.capabilities.v1"


def create_app(
    settings: AvatarSettings | None = None,
    *,
    renderer_factory: Any | None = None,
) -> FastAPI:
    config = settings or load_avatar_settings()
    manager = SessionManager(config)

    if renderer_factory is None:
        # One warm renderer per service: MuseTalk warm-up (weights + every
        # clip reference) is minutes of work and must not run per session.
        shared_holder: dict[str, SharedRenderer] = {}

        def renderer_factory() -> Any:  # noqa: PLW0127 — default factory
            shared = shared_holder.get("renderer")
            if shared is None:
                renderer, _ = select_renderer(config)
                shared = SharedRenderer(renderer)
                shared_holder["renderer"] = shared
            return shared

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        # Warm the shared renderer off the event loop so the first session
        # mints fast and the render loop never blocks on backend warm-up.
        warm_task = asyncio.create_task(
            asyncio.to_thread(lambda: renderer_factory().start())
        )
        yield
        warm_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await warm_task
        await manager.stop_all("service_shutdown")
        renderer = renderer_factory()
        shutdown = getattr(renderer, "shutdown", None)
        if callable(shutdown):
            shutdown()

    app = FastAPI(
        title="Hydralisk Avatar Render Service",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.avatar_settings = config
    app.state.avatar_sessions = manager

    def _token_valid(token: str | None) -> bool:
        if config.bearer_token is None:
            return config.allow_insecure_dev
        return token == config.bearer_token

    async def require_bearer(request: Request) -> None:
        if config.bearer_token is None and not config.allow_insecure_dev:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "hydralisk_avatar_unarmed",
                        "message": "Avatar bearer auth is not configured.",
                    }
                },
            )
        if config.allow_insecure_dev and config.bearer_token is None:
            return
        header = request.headers.get("authorization")
        token = None
        if header and header.startswith("Bearer "):
            token = header.removeprefix("Bearer ")
        if not _token_valid(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error": {
                        "code": "unauthorized",
                        "message": "Missing or invalid avatar bearer token.",
                    }
                },
            )

    def _renderer_blockers() -> list[dict[str, str]]:
        try:
            from hydralisk.avatar.musetalk_backend import musetalk_blockers

            return musetalk_blockers(config)
        except Exception as error:  # pragma: no cover — defensive
            return [{"code": "musetalk_probe_failed", "message": str(error)}]

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "hydralisk-avatar",
            "activeSessions": len(manager.active_sessions),
            "maxSessions": config.max_sessions,
            "webrtcAvailable": webrtc_available(),
            "authConfigured": bool(
                config.bearer_token or config.allow_insecure_dev
            ),
        }

    @app.get("/avatar/capabilities")
    async def capabilities() -> dict[str, Any]:
        return {
            "schema": CAPABILITIES_SCHEMA,
            "protocol": PROTOCOL_VERSION,
            "controlEvents": [item.value for item in ControlType],
            "video": {
                "fps": config.fps,
                "width": config.width,
                "height": config.height,
                "codecPath": "webrtc",
            },
            "audio": {
                "format": "pcm_s16le",
                "sampleRate": config.sample_rate,
                "channels": 1,
                "encoding": "base64",
            },
            "states": {
                state.value: list(clips)
                for state, clips in STATE_CLIP_CYCLE.items()
            },
            "clips": [
                {
                    "index": clip.index,
                    "role": clip.role.value,
                    "note": clip.note,
                }
                for clip in CLIP_CATALOG
            ],
            "webrtcAvailable": webrtc_available(),
            "rendererBackends": {
                "requested": config.renderer_backend,
                "musetalkBlockers": _renderer_blockers(),
            },
            "publicSafe": True,
        }

    def _mint_session() -> Any:
        renderer = renderer_factory()
        egress = None
        if webrtc_available():
            egress = WebRTCEgress(
                fps=config.fps, sample_rate=config.sample_rate
            )
        else:
            egress = NullEgress()
        try:
            return manager.create(renderer=renderer, egress=egress)
        except SessionLimitError as error:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": {
                        "code": "avatar_session_limit",
                        "message": str(error),
                    }
                },
            ) from None

    @app.post(
        "/avatar/sessions",
        dependencies=[Depends(require_bearer)],
        status_code=status.HTTP_201_CREATED,
    )
    async def create_session() -> dict[str, Any]:
        session = _mint_session()
        ref = session.session_ref
        return {
            "sessionRef": ref,
            "state": session.machine.state.value,
            "renderer": session.renderer.backend,
            "controlPath": f"/avatar/sessions/{ref}/control",
            "webrtc": {
                "available": webrtc_available(),
                "offerPath": f"/avatar/sessions/{ref}/webrtc",
            },
            "protocol": PROTOCOL_VERSION,
        }

    def _get_session(session_ref: str) -> Any:
        session = manager.get(session_ref)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "avatar_session_not_found",
                        "message": "No such avatar session.",
                    }
                },
            )
        return session

    @app.get(
        "/avatar/sessions/{session_ref}",
        dependencies=[Depends(require_bearer)],
    )
    async def session_status(session_ref: str) -> dict[str, Any]:
        return _get_session(session_ref).status()

    @app.post(
        "/avatar/sessions/{session_ref}/stop",
        dependencies=[Depends(require_bearer)],
    )
    async def stop_session(session_ref: str) -> dict[str, Any]:
        session = _get_session(session_ref)
        summary = await session.stop("client_stop")
        return {"stopped": True, "summary": summary}

    @app.post(
        "/avatar/sessions/{session_ref}/webrtc",
        dependencies=[Depends(require_bearer)],
    )
    async def webrtc_offer(session_ref: str, request: Request) -> dict[str, Any]:
        session = _get_session(session_ref)
        if not isinstance(session.egress, WebRTCEgress):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "webrtc_unavailable",
                        "message": "aiortc is not installed on this host "
                        "(install the 'avatar' extra).",
                    }
                },
            )
        body = await request.json()
        sdp = body.get("sdp")
        offer_type = body.get("type", "offer")
        if not isinstance(sdp, str) or not sdp:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "code": "invalid_offer",
                        "message": "Body must include an SDP offer.",
                    }
                },
            )
        try:
            return await session.egress.handle_offer(sdp, offer_type)
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "code": "offer_rejected",
                        "message": f"SDP negotiation failed: {error}"[:300],
                    }
                },
            ) from None

    # ------------------------------------------------------------------
    # OAV-4 compat surface — the exact contract apps/sarah
    # services/owned-renderer.ts codes to (openagents#8614):
    #   POST   /sessions                    (bearer) -> {session_id, webrtc:{offer_url}}
    #   POST   /sessions/{id}/control       (bearer) JSON {type:"speak"|...}
    #   DELETE /sessions/{id}               (bearer)
    #   POST   /sessions/{id}/webrtc-offer  (no bearer: capability URL; the
    #          browser posts raw SDP cross-origin, so CORS is answered here)
    # ------------------------------------------------------------------

    _CORS_HEADERS = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "content-type",
        "Access-Control-Max-Age": "86400",
    }

    def _offer_url(ref: str) -> str:
        base = (config.public_base_url or "").rstrip("/")
        return f"{base}/sessions/{ref}/webrtc-offer"

    @app.post(
        "/sessions",
        dependencies=[Depends(require_bearer)],
        status_code=status.HTTP_201_CREATED,
    )
    async def compat_create_session(request: Request) -> dict[str, Any]:
        # Optional JSON body ({conversation_ref}) is accepted and echoed.
        conversation_ref: str | None = None
        with contextlib.suppress(Exception):
            body = await request.json()
            if isinstance(body, dict) and isinstance(
                body.get("conversation_ref"), str
            ):
                conversation_ref = body["conversation_ref"]
        session = _mint_session()
        ref = session.session_ref
        payload: dict[str, Any] = {
            "session_id": ref,
            "state": session.machine.state.value,
            "renderer": session.renderer.backend,
            "webrtc": {
                "available": webrtc_available(),
                "offer_url": _offer_url(ref),
            },
            "protocol": PROTOCOL_VERSION,
        }
        if conversation_ref is not None:
            payload["conversation_ref"] = conversation_ref
        return payload

    @app.post(
        "/sessions/{session_ref}/control",
        dependencies=[Depends(require_bearer)],
    )
    async def compat_control(
        session_ref: str, request: Request
    ) -> dict[str, Any]:
        session = _get_session(session_ref)
        try:
            payload = await request.json()
        except Exception:
            payload = None
        if not isinstance(payload, dict):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "code": "invalid_message",
                        "message": "Body must be a JSON control object.",
                    }
                },
            )
        # The compat contract uses bare event names and `audio_b64`; the WS
        # protocol uses the agent.* prefix and `audio`. Accept both.
        event_type = payload.get("type")
        if isinstance(event_type, str) and not event_type.startswith("agent."):
            payload = {**payload, "type": f"agent.{event_type}"}
        if "audio_b64" in payload and "audio" not in payload:
            payload = {**payload, "audio": payload["audio_b64"]}
            payload.pop("audio_b64", None)
        try:
            message = parse_control_message(json.dumps(payload))
        except ProtocolError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {"code": error.code, "message": error.message}
                },
            ) from None
        events = list(session.handle_control(message))
        return {"ok": True, "events": events}

    @app.delete("/sessions/{session_ref}", dependencies=[Depends(require_bearer)])
    async def compat_delete(session_ref: str) -> dict[str, Any]:
        session = manager.get(session_ref)
        if session is None or session.stopped:
            # Idempotent: reap/stop of an unknown or finished session is fine.
            return {"stopped": True}
        summary = await session.stop("client_stop")
        return {"stopped": True, "summary": summary}

    @app.options("/sessions/{session_ref}/webrtc-offer")
    async def compat_webrtc_preflight(session_ref: str) -> Response:
        return Response(status_code=204, headers=_CORS_HEADERS)

    @app.post("/sessions/{session_ref}/webrtc-offer")
    async def compat_webrtc_offer(
        session_ref: str, request: Request
    ) -> PlainTextResponse:
        # No bearer: the unguessable session ref is the capability; the
        # browser cannot hold the service token.
        session = manager.get(session_ref)
        if session is None or session.stopped:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "code": "avatar_session_not_found",
                        "message": "No such avatar session.",
                    }
                },
            )
        if not isinstance(session.egress, WebRTCEgress):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "code": "webrtc_unavailable",
                        "message": "aiortc is not installed on this host "
                        "(install the 'avatar' extra).",
                    }
                },
            )
        sdp = (await request.body()).decode("utf-8", errors="replace").strip()
        if not sdp:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "code": "invalid_offer",
                        "message": "Body must be a raw SDP offer.",
                    }
                },
            )
        try:
            answer = await session.egress.handle_offer(sdp, "offer")
        except Exception as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": {
                        "code": "offer_rejected",
                        "message": f"SDP negotiation failed: {error}"[:300],
                    }
                },
            ) from None
        return PlainTextResponse(
            answer["sdp"],
            media_type="application/sdp",
            headers=_CORS_HEADERS,
        )

    @app.websocket("/avatar/sessions/{session_ref}/control")
    async def control_socket(websocket: WebSocket, session_ref: str) -> None:
        token: str | None = None
        header = websocket.headers.get("authorization")
        if header and header.startswith("Bearer "):
            token = header.removeprefix("Bearer ")
        if token is None:
            token = websocket.query_params.get("token")
        if not _token_valid(token):
            await websocket.close(code=4401)
            return
        session = manager.get(session_ref)
        if session is None or session.stopped:
            await websocket.close(code=4404)
            return

        await websocket.accept()
        await websocket.send_json(
            state_event(session.machine.state.value, session.session_ref)
        )

        async def pump_outbox() -> None:
            while True:
                event = await session.outbox.get()
                await websocket.send_json(event)

        outbox_task = asyncio.create_task(pump_outbox())
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    message = parse_control_message(raw)
                except ProtocolError as error:
                    await websocket.send_json(
                        error_event(error.code, error.message)
                    )
                    continue
                for event in session.handle_control(message):
                    await websocket.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            outbox_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await outbox_task

    return app


def main() -> None:
    uvicorn.run(
        create_app(),
        host="0.0.0.0",
        port=8020,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
