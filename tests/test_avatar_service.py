from __future__ import annotations

import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

import hydralisk.avatar.service as service_module
from hydralisk.avatar.config import AvatarSettings
from hydralisk.avatar.protocol import encode_pcm16
from hydralisk.avatar.receipts import AvatarReceiptWriter
from hydralisk.avatar.service import create_app


def _settings(tmp_path: Path, **overrides) -> AvatarSettings:
    params = dict(
        bearer_token="secret",
        renderer_backend="cpu",
        width=160,
        height=90,
        receipt_dir=tmp_path / "receipts",
        keepalive_timeout_seconds=300.0,
        jitter_buffer_frames=0,
    )
    params.update(overrides)
    return AvatarSettings(**params)


AUTH = {"Authorization": "Bearer secret"}


def test_unarmed_service_fails_closed(tmp_path: Path) -> None:
    settings = _settings(tmp_path, bearer_token=None)
    with TestClient(create_app(settings)) as client:
        response = client.post("/avatar/sessions")
        assert response.status_code == 503
        assert (
            response.json()["detail"]["error"]["code"]
            == "hydralisk_avatar_unarmed"
        )


def test_bearer_required(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.post("/avatar/sessions").status_code == 401
        assert (
            client.post(
                "/avatar/sessions",
                headers={"Authorization": "Bearer wrong"},
            ).status_code
            == 401
        )


def test_health_and_capabilities_are_public(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["service"] == "hydralisk-avatar"

        caps = client.get("/avatar/capabilities")
        assert caps.status_code == 200
        body = caps.json()
        assert body["schema"] == "hydralisk.avatar.capabilities.v1"
        assert body["protocol"] == "hydralisk.avatar.control.v1"
        assert "agent.speak" in body["controlEvents"]
        assert body["states"]["idle"] == [6, 3]
        assert body["states"]["listening"] == [4, 5]
        assert body["states"]["speaking"] == [8, 0]
        assert body["audio"] == {
            "format": "pcm_s16le",
            "sampleRate": 24000,
            "channels": 1,
            "encoding": "base64",
        }
        assert body["publicSafe"] is True
        # The CPU lane always reports why MuseTalk is inactive.
        assert isinstance(
            body["rendererBackends"]["musetalkBlockers"], list
        )


def test_session_lifecycle_and_receipt(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        created = client.post("/avatar/sessions", headers=AUTH)
        assert created.status_code == 201
        body = created.json()
        ref = body["sessionRef"]
        assert body["state"] == "idle"
        assert body["renderer"] == "cpu-noop"
        assert body["controlPath"] == f"/avatar/sessions/{ref}/control"
        assert body["webrtc"]["offerPath"] == f"/avatar/sessions/{ref}/webrtc"

        # The paced render loop produces frames without any control input.
        time.sleep(0.3)
        status = client.get(f"/avatar/sessions/{ref}", headers=AUTH)
        assert status.status_code == 200
        assert status.json()["framesRendered"] > 0

        stopped = client.post(f"/avatar/sessions/{ref}/stop", headers=AUTH)
        assert stopped.status_code == 200
        summary = stopped.json()["summary"]
        assert summary["sessionRef"] == ref
        assert summary["framesRendered"] > 0
        assert summary["stopReason"] == "client_stop"
        assert summary["publicSafe"] is True

        rows = AvatarReceiptWriter(settings.receipt_dir).read_rows(ref)
        events = [row["event"] for row in rows]
        assert events[0] == "avatar.session_started"
        assert events[-1] == "avatar.session_summary"


def test_session_limit_evicts_peerless_then_blocks_watched(tmp_path: Path) -> None:
    """SQ-4 #8621: a full slot with NO connected peer is evicted by the
    next mint (a wedged slot must never block the next visitor); a slot a
    viewer is actually watching is never evicted — the next mint 429s."""
    settings = _settings(tmp_path, max_sessions=1)
    with TestClient(create_app(settings)) as client:
        first = client.post("/avatar/sessions", headers=AUTH)
        assert first.status_code == 201
        first_ref = first.json()["sessionRef"]

        # Peer-less first session → evicted, second mint succeeds.
        second = client.post("/avatar/sessions", headers=AUTH)
        assert second.status_code == 201
        stopped = client.get(f"/avatar/sessions/{first_ref}", headers=AUTH)
        assert stopped.json()["stopReason"] == "evicted_stale_no_peer"

        # Simulate a connected viewer on the active session: eviction must
        # refuse and the mint must 429.
        manager = client.app.state.avatar_sessions
        active = manager.get(second.json()["sessionRef"])

        class _FakePc:
            connectionState = "connected"
            iceConnectionState = "connected"

            async def close(self) -> None:
                return None

        active.egress.pc = _FakePc()
        third = client.post("/avatar/sessions", headers=AUTH)
        assert third.status_code == 429
        assert (
            third.json()["detail"]["error"]["code"] == "avatar_session_limit"
        )

        # Stopping the watched session frees the slot.
        client.post(
            f"/avatar/sessions/{second.json()['sessionRef']}/stop", headers=AUTH
        )
        fourth = client.post("/avatar/sessions", headers=AUTH)
        assert fourth.status_code == 201


def test_unknown_session_is_404(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        ref = "hydralisk-avatar-" + "0" * 32
        assert (
            client.get(f"/avatar/sessions/{ref}", headers=AUTH).status_code
            == 404
        )


def test_webrtc_offer_503_without_aiortc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service_module, "webrtc_available", lambda: False)
    with TestClient(create_app(_settings(tmp_path))) as client:
        ref = client.post("/avatar/sessions", headers=AUTH).json()["sessionRef"]
        offer = client.post(
            f"/avatar/sessions/{ref}/webrtc",
            headers=AUTH,
            json={"sdp": "v=0...", "type": "offer"},
        )
        assert offer.status_code == 503
        assert (
            offer.json()["detail"]["error"]["code"] == "webrtc_unavailable"
        )


def test_control_socket_requires_token(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        ref = client.post("/avatar/sessions", headers=AUTH).json()["sessionRef"]
        with pytest.raises(Exception):
            with client.websocket_connect(f"/avatar/sessions/{ref}/control"):
                pass


def test_control_socket_speak_interrupt_cycle(tmp_path: Path) -> None:
    settings = _settings(tmp_path, jitter_buffer_frames=0)
    with TestClient(create_app(settings)) as client:
        ref = client.post("/avatar/sessions", headers=AUTH).json()["sessionRef"]

        with client.websocket_connect(
            f"/avatar/sessions/{ref}/control?token=secret"
        ) as socket:
            hello = socket.receive_json()
            assert hello["type"] == "session.state"
            assert hello["state"] == "idle"

            # start_listening → listening
            socket.send_text(json.dumps({"type": "agent.start_listening"}))
            event = socket.receive_json()
            assert event["type"] == "session.state"
            assert event["state"] == "listening"

            # speak chunk → speaking
            pcm = encode_pcm16(np.ones(1000, dtype=np.int16))
            socket.send_text(
                json.dumps(
                    {"type": "agent.speak", "event_id": "utt-1", "audio": pcm}
                )
            )
            started = socket.receive_json()
            assert started["type"] == "agent.speak_started"
            assert started["event_id"] == "utt-1"
            state = socket.receive_json()
            assert state["state"] == "speaking"

            # interrupt → listening crossfade
            socket.send_text(json.dumps({"type": "agent.interrupt"}))
            event = socket.receive_json()
            assert event["type"] == "session.state"
            assert event["state"] == "listening"

            # keepalive is acknowledged
            socket.send_text(json.dumps({"type": "agent.keepalive"}))
            assert socket.receive_json()["type"] == "session.keepalive_ack"

            # malformed frames error without killing the socket
            socket.send_text("not json")
            error = socket.receive_json()
            assert error["type"] == "session.error"
            assert error["code"] == "invalid_message"

        status = client.get(f"/avatar/sessions/{ref}", headers=AUTH).json()
        assert status["interrupts"] == 1
        assert status["utterances"] == 1


def test_speak_end_returns_to_idle_after_drain(tmp_path: Path) -> None:
    settings = _settings(tmp_path, jitter_buffer_frames=0)
    with TestClient(create_app(settings)) as client:
        ref = client.post("/avatar/sessions", headers=AUTH).json()["sessionRef"]
        with client.websocket_connect(
            f"/avatar/sessions/{ref}/control?token=secret"
        ) as socket:
            socket.receive_json()  # hello

            pcm = encode_pcm16(np.ones(500, dtype=np.int16))
            socket.send_text(
                json.dumps(
                    {"type": "agent.speak", "event_id": "utt-9", "audio": pcm}
                )
            )
            socket.receive_json()  # speak_started
            assert socket.receive_json()["state"] == "speaking"

            socket.send_text(
                json.dumps({"type": "agent.speak_end", "event_id": "utt-9"})
            )

            # The render loop drains the audio and pushes speak_ended + idle.
            ended = socket.receive_json()
            assert ended["type"] == "agent.speak_ended"
            assert ended["event_id"] == "utt-9"
            state = socket.receive_json()
            assert state["state"] == "idle"
