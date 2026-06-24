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


def test_authorized_security_policy_is_public_safe_in_capabilities() -> None:
    client = TestClient(
        create_app(
            HydraliskSettings(
                served_model="Chunjiang-Intelligence/DeepSeek-v4-Fable",
                public_model_aliases=(),
                model_revision=(
                    "Chunjiang-Intelligence/DeepSeek-v4-Fable"
                    "@999909137c15e0b5539fee887431824fa7cb5b10"
                ),
                adapter_revision=(
                    "Chunjiang-Intelligence/DeepSeek-v4-Fable"
                    "@999909137c15e0b5539fee887431824fa7cb5b10"
                ),
                model_policy="authorized_security_lab_only",
                authorized_security_scope_ids=("lab-ctf-001",),
                authorized_security_tool_policies=("sandboxed_tools_only",),
                authorized_security_network_policies=("deny_all",),
            )
        )
    )

    response = client.get("/hydralisk/v1/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["publicModelAliases"] == []
    assert body["policy"] == {
        "mode": "authorized_security_lab_only",
        "adapterRevision": (
            "Chunjiang-Intelligence/DeepSeek-v4-Fable"
            "@999909137c15e0b5539fee887431824fa7cb5b10"
        ),
        "authorizedSecurity": {
            "required": True,
            "scopeIdsConfigured": True,
            "toolPoliciesConfigured": True,
            "networkPoliciesConfigured": True,
        },
    }
    assert "lab-ctf-001" not in str(body)


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


def test_authorized_security_policy_rejects_missing_metadata_before_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingAsyncClient:
        def __init__(self, timeout=None) -> None:
            raise AssertionError("upstream should not be constructed")

    monkeypatch.setattr(proxy_module.httpx, "AsyncClient", ExplodingAsyncClient)
    client = TestClient(
        create_app(
            HydraliskSettings(
                served_model="Chunjiang-Intelligence/DeepSeek-v4-Fable",
                public_model_aliases=(),
                allow_insecure_dev=True,
                model_policy="authorized_security_lab_only",
            )
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Chunjiang-Intelligence/DeepSeek-v4-Fable",
            "messages": [],
        },
    )

    assert response.status_code == 403
    assert (
        response.json()["detail"]["error"]["code"]
        == "authorized_security_metadata_required"
    )


def test_authorized_security_policy_rejects_unconfigured_scope() -> None:
    client = TestClient(
        create_app(
            HydraliskSettings(
                served_model="Chunjiang-Intelligence/DeepSeek-v4-Fable",
                public_model_aliases=(),
                allow_insecure_dev=True,
                model_policy="authorized_security_lab_only",
                authorized_security_scope_ids=("lab-ctf-001",),
            )
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Chunjiang-Intelligence/DeepSeek-v4-Fable",
            "messages": [],
            "metadata": {
                "hydraliskAuthorizedSecurity": {
                    "scopeId": "not-this-lab",
                    "authorizationRef": "authz-001",
                    "toolPolicy": "sandboxed_tools_only",
                    "networkPolicy": "deny_all",
                }
            },
        },
    )

    assert response.status_code == 403
    assert (
        response.json()["detail"]["error"]["code"]
        == "authorized_security_scope_not_allowed"
    )


def test_authorized_security_policy_records_receipt_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            }

    class FakeAsyncClient:
        def __init__(self, timeout=None) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            return FakeResponse()

    monkeypatch.setattr(proxy_module.httpx, "AsyncClient", FakeAsyncClient)
    client = TestClient(
        create_app(
            HydraliskSettings(
                served_model="Chunjiang-Intelligence/DeepSeek-v4-Fable",
                public_model_aliases=(),
                allow_insecure_dev=True,
                model_revision=(
                    "Chunjiang-Intelligence/DeepSeek-v4-Fable"
                    "@999909137c15e0b5539fee887431824fa7cb5b10"
                ),
                adapter_revision=(
                    "Chunjiang-Intelligence/DeepSeek-v4-Fable"
                    "@999909137c15e0b5539fee887431824fa7cb5b10"
                ),
                model_policy="authorized_security_lab_only",
                authorized_security_scope_ids=("lab-ctf-001",),
                authorized_security_tool_policies=("sandboxed_tools_only",),
                authorized_security_network_policies=("deny_all",),
                receipt_dir=tmp_path,
            )
        )
    )

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "Chunjiang-Intelligence/DeepSeek-v4-Fable",
            "messages": [],
            "metadata": {
                "hydraliskAuthorizedSecurity": {
                    "scopeId": "lab-ctf-001",
                    "authorizationRef": "authz-001",
                    "toolPolicy": "sandboxed_tools_only",
                    "networkPolicy": "deny_all",
                }
            },
        },
    )

    assert response.status_code == 200
    receipt = client.get(
        f"/hydralisk/v1/receipts/{response.headers['x-hydralisk-run-ref']}"
    ).json()
    assert receipt["policy"] == {
        "mode": "authorized_security_lab_only",
        "adapterRevision": (
            "Chunjiang-Intelligence/DeepSeek-v4-Fable"
            "@999909137c15e0b5539fee887431824fa7cb5b10"
        ),
        "authorization": {
            "admissionResult": "admitted",
            "scopeId": "lab-ctf-001",
            "authorizationRef": "authz-001",
            "toolPolicy": "sandboxed_tools_only",
            "networkPolicy": "deny_all",
        },
    }
