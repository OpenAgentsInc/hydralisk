from __future__ import annotations

from fastapi.testclient import TestClient

from hydralisk.serve.config import HydraliskSettings
from hydralisk.serve.proxy import create_app


def test_health_is_public_safe_when_unarmed() -> None:
    client = TestClient(create_app(HydraliskSettings()))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unarmed"
    assert body["servedModel"] == "openai/gpt-oss-20b"
    assert "token" not in str(body).lower()
    assert "127.0.0.1" not in str(body)


def test_proxy_requires_bearer_auth() -> None:
    client = TestClient(create_app(HydraliskSettings(bearer_token="secret")))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "openai/gpt-oss-20b", "messages": []},
    )

    assert response.status_code == 401
    assert response.json()["detail"]["error"]["code"] == "unauthorized"


def test_proxy_refuses_unsupported_models_before_upstream() -> None:
    client = TestClient(create_app(HydraliskSettings(allow_insecure_dev=True)))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "not-ours", "messages": []},
    )

    assert response.status_code == 400
    body = response.json()["detail"]
    assert body["error"]["code"] == "unsupported_model"
    assert "khala" in body["supportedModels"]
    assert "openagents/khala" in body["supportedModels"]
    assert "openagents/khala-oss-20b" in body["supportedModels"]


def test_capabilities_are_public_safe() -> None:
    client = TestClient(create_app(HydraliskSettings()))

    response = client.get("/hydralisk/v1/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "hydralisk.serve.capabilities.v1"
    assert body["servedModel"] == "openai/gpt-oss-20b"
    assert body["publicModelAliases"] == [
        "khala",
        "openagents/khala",
        "openagents/khala-oss-20b",
        "gpt-oss-20b",
    ]
    assert body["quantization"]["weights"] == "MXFP4"
    assert "token" not in str(body).lower()
    assert "127.0.0.1" not in str(body)
