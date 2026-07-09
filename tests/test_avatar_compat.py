"""OAV-4 compat surface tests — the exact apps/sarah owned-renderer.ts contract.

POST /sessions, POST /sessions/{id}/control (bare event names, audio_b64),
DELETE /sessions/{id}, and the unauthenticated capability-URL
POST /sessions/{id}/webrtc-offer with CORS.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

import hydralisk.avatar.service as service_module
from hydralisk.avatar.config import AvatarSettings
from hydralisk.avatar.protocol import encode_pcm16
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


def _speak_chunk(ms: int = 100) -> str:
    samples = np.zeros(24_000 * ms // 1000, dtype="<i2")
    return encode_pcm16(samples)


def test_compat_mint_returns_session_id_and_offer_url(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path, public_base_url="https://render.example.test"
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/sessions",
            headers=AUTH,
            json={"conversation_ref": "visitor:abc"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["session_id"]
        assert body["conversation_ref"] == "visitor:abc"
        assert body["webrtc"]["offer_url"] == (
            f"https://render.example.test/sessions/{body['session_id']}/webrtc-offer"
        )


def test_compat_mint_requires_bearer(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.post("/sessions").status_code == 401


def test_compat_mint_accepts_empty_body(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        response = client.post("/sessions", headers=AUTH)
        assert response.status_code == 201
        assert "conversation_ref" not in response.json()


def test_compat_control_speak_cycle(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        session_id = client.post("/sessions", headers=AUTH).json()["session_id"]

        speak = client.post(
            f"/sessions/{session_id}/control",
            headers=AUTH,
            json={
                "type": "speak",
                "event_id": "evt-1",
                "audio_b64": _speak_chunk(),
            },
        )
        assert speak.status_code == 200
        assert speak.json()["ok"] is True

        end = client.post(
            f"/sessions/{session_id}/control",
            headers=AUTH,
            json={"type": "speak_end", "event_id": "evt-1"},
        )
        assert end.status_code == 200

        for control in ("interrupt", "start_listening", "stop_listening", "keepalive"):
            response = client.post(
                f"/sessions/{session_id}/control",
                headers=AUTH,
                json={"type": control},
            )
            assert response.status_code == 200, control


def test_compat_control_rejects_malformed(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        session_id = client.post("/sessions", headers=AUTH).json()["session_id"]
        assert (
            client.post(
                f"/sessions/{session_id}/control",
                headers=AUTH,
                json={"type": "speak", "event_id": "evt-1"},
            ).status_code
            == 422
        )
        assert (
            client.post(
                f"/sessions/{session_id}/control",
                headers=AUTH,
                json={"type": "warp_drive"},
            ).status_code
            == 422
        )


def test_compat_delete_stops_and_is_idempotent(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as client:
        session_id = client.post("/sessions", headers=AUTH).json()["session_id"]
        first = client.delete(f"/sessions/{session_id}", headers=AUTH)
        assert first.status_code == 200
        assert first.json()["stopped"] is True
        again = client.delete(f"/sessions/{session_id}", headers=AUTH)
        assert again.status_code == 200
        assert again.json() == {"stopped": True}
        missing = client.delete("/sessions/never-existed", headers=AUTH)
        assert missing.status_code == 200


def test_compat_webrtc_offer_is_capability_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No bearer on the offer path — but unknown refs 404 and the no-aiortc
    posture (NullEgress) fails closed with 503."""
    monkeypatch.setattr(service_module, "webrtc_available", lambda: False)
    with TestClient(create_app(_settings(tmp_path))) as client:
        session_id = client.post("/sessions", headers=AUTH).json()["session_id"]

        preflight = client.options(f"/sessions/{session_id}/webrtc-offer")
        assert preflight.status_code == 204
        assert preflight.headers["access-control-allow-origin"] == "*"

        missing = client.post(
            "/sessions/nope/webrtc-offer",
            content="v=0",
            headers={"content-type": "application/sdp"},
        )
        assert missing.status_code == 404

        offer = client.post(
            f"/sessions/{session_id}/webrtc-offer",
            content="v=0",
            headers={"content-type": "application/sdp"},
        )
        # aiortc is intentionally absent in CI: capability URL must still be
        # routable (no 401) and fail closed on egress availability.
        assert offer.status_code == 503
        assert (
            offer.json()["detail"]["error"]["code"] == "webrtc_unavailable"
        )
