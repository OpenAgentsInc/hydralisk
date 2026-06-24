from __future__ import annotations

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from hydralisk.serve.config import HydraliskSettings
import hydralisk.serve.proxy as proxy_module
from hydralisk.serve.proxy import _InflightGate
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
    assert body["admission"] == {
        "maxInflightRequests": None,
        "queueTimeoutSeconds": 0.0,
        "singleFlight": False,
    }
    assert "token" not in str(body).lower()
    assert "127.0.0.1" not in str(body)


def test_capabilities_include_singleflight_admission_policy() -> None:
    client = TestClient(
        create_app(
            HydraliskSettings(
                max_inflight_requests=1,
                inflight_queue_timeout_seconds=0.25,
            )
        )
    )

    response = client.get("/hydralisk/v1/capabilities")

    assert response.status_code == 200
    assert response.json()["admission"] == {
        "maxInflightRequests": 1,
        "queueTimeoutSeconds": 0.25,
        "singleFlight": True,
    }


@pytest.mark.asyncio
async def test_inflight_gate_rejects_when_saturated() -> None:
    gate = _InflightGate(limit=1, queue_timeout_seconds=0)
    lease = await gate.acquire()
    try:
        with pytest.raises(HTTPException) as exc_info:
            await gate.acquire()
    finally:
        lease.release()

    assert exc_info.value.status_code == 429
    assert exc_info.value.detail["error"]["code"] == "hydralisk_inflight_saturated"
    assert exc_info.value.detail["admission"]["singleFlight"] is True


def test_streaming_proxy_releases_singleflight_slot_and_receipts_admission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakeStreamResponse:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
            yield (
                b'data: {"choices":[],"usage":{"prompt_tokens":1,'
                b'"completion_tokens":1,"total_tokens":2}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        async def aclose(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, timeout=None) -> None:
            self.timeout = timeout

        def build_request(self, *args, **kwargs):
            return {"args": args, "kwargs": kwargs}

        async def send(self, request, stream=False):
            return FakeStreamResponse()

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(proxy_module.httpx, "AsyncClient", FakeAsyncClient)
    client = TestClient(
        create_app(
            HydraliskSettings(
                allow_insecure_dev=True,
                max_inflight_requests=1,
                receipt_dir=tmp_path,
            )
        )
    )
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": True,
    }

    first = client.post("/v1/chat/completions", json=payload)
    second = client.post("/v1/chat/completions", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    receipt = client.get(
        f"/hydralisk/v1/receipts/{first.headers['x-hydralisk-run-ref']}"
    ).json()
    assert receipt["admission"] == {
        "maxInflightRequests": 1,
        "queueTimeoutSeconds": 0.0,
        "singleFlight": True,
    }
