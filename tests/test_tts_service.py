from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from hydralisk.tts.receipts import TtsReceiptLog
from hydralisk.tts.seam import VoiceRef
from hydralisk.tts.service import TtsServiceSettings, create_tts_app


class FakeAdapter:
    adapter_ref = "fake-tts"
    default_voice = VoiceRef(voice_id="fake-voice", language_code="en-US")

    def __init__(self, chunks: list[bytes] | None = None) -> None:
        self._chunks = chunks if chunks is not None else [b"\x00\x01" * 8, b"\x02\x03" * 4]
        self.seen_texts: list[str] = []
        self.seen_voices: list[VoiceRef | None] = []

    async def synthesize_stream(
        self,
        text: str,
        voice_ref: VoiceRef | None = None,
    ) -> AsyncIterator[bytes]:
        self.seen_texts.append(text)
        self.seen_voices.append(voice_ref)
        for chunk in self._chunks:
            yield chunk


def _settings(tmp_path: Path, **overrides: object) -> TtsServiceSettings:
    defaults: dict[str, object] = {
        "adapter_name": "fake",
        "bearer_token": "tts-secret",
        "receipt_path": tmp_path / "tts-receipts.jsonl",
    }
    defaults.update(overrides)
    return TtsServiceSettings(**defaults)  # type: ignore[arg-type]


def test_health_is_public_safe_when_unarmed(tmp_path: Path) -> None:
    client = TestClient(
        create_tts_app(_settings(tmp_path, bearer_token=None), adapter=FakeAdapter())
    )

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unarmed"
    assert body["lane"] == "hydralisk-tts"
    assert body["pcm"]["sampleRateHz"] == 24000
    assert "token" not in str(body).lower()


def test_synthesize_requires_bearer(tmp_path: Path) -> None:
    client = TestClient(create_tts_app(_settings(tmp_path), adapter=FakeAdapter()))

    response = client.post("/hydralisk/tts/v1/synthesize", json={"text": "hello"})

    assert response.status_code == 401
    assert response.json()["detail"]["error"]["code"] == "unauthorized"


def test_synthesize_unarmed_fails_closed(tmp_path: Path) -> None:
    client = TestClient(
        create_tts_app(_settings(tmp_path, bearer_token=None), adapter=FakeAdapter())
    )

    response = client.post("/hydralisk/tts/v1/synthesize", json={"text": "hello"})

    assert response.status_code == 503
    assert response.json()["detail"]["error"]["code"] == "hydralisk_tts_unarmed"


def test_capabilities_expose_pcm_contract(tmp_path: Path) -> None:
    client = TestClient(create_tts_app(_settings(tmp_path), adapter=FakeAdapter()))

    response = client.get("/hydralisk/tts/v1/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "hydralisk.tts.capabilities.v1"
    assert body["adapterRef"] == "fake-tts"
    assert body["defaultVoice"] == {"voiceId": "fake-voice", "languageCode": "en-US"}
    assert body["pcm"]["sampleRateHz"] == 24000
    assert body["pcm"]["sampleWidthBytes"] == 2
    assert body["pcm"]["channels"] == 1
    assert body["streaming"] is True
    assert body["publicSafe"] is True
    assert "token" not in str(body).lower()


def test_synthesize_streams_pcm_and_writes_receipt(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    adapter = FakeAdapter()
    client = TestClient(create_tts_app(settings, adapter=adapter))

    response = client.post(
        "/hydralisk/tts/v1/synthesize",
        json={"text": "hello sarah"},
        headers={"authorization": "Bearer tts-secret"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/L16")
    assert response.content == b"\x00\x01" * 8 + b"\x02\x03" * 4
    run_ref = response.headers["x-hydralisk-tts-run-ref"]
    assert run_ref.startswith("hydralisk-tts-run-")
    assert response.headers["x-hydralisk-tts-adapter"] == "fake-tts"
    assert response.headers["x-hydralisk-tts-voice"] == "fake-voice"
    assert adapter.seen_texts == ["hello sarah"]

    receipts = TtsReceiptLog(settings.receipt_path).read_all()
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["schema"] == "hydralisk.tts.run_receipt.v1"
    assert receipt["runRef"] == run_ref
    assert receipt["charsIn"] == len("hello sarah")
    assert receipt["bytesOut"] == 24
    assert receipt["chunksOut"] == 2
    assert isinstance(receipt["msToFirstChunk"], int)
    assert isinstance(receipt["totalMs"], int)
    assert receipt["publicSafe"] is True
    assert receipt["blockers"] == []
    # The receipt must never carry the synthesized text.
    assert "hello sarah" not in str(receipt)

    lookup = client.get(f"/hydralisk/tts/v1/receipts/{run_ref}")
    assert lookup.status_code == 200
    assert lookup.json()["runRef"] == run_ref


def test_synthesize_honours_requested_voice(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    client = TestClient(create_tts_app(_settings(tmp_path), adapter=adapter))

    response = client.post(
        "/hydralisk/tts/v1/synthesize",
        json={"text": "hi", "voiceId": "en-US-Chirp3-HD-Leda"},
        headers={"authorization": "Bearer tts-secret"},
    )

    assert response.status_code == 200
    assert response.headers["x-hydralisk-tts-voice"] == "en-US-Chirp3-HD-Leda"
    assert adapter.seen_voices[0] is not None
    assert adapter.seen_voices[0].voice_id == "en-US-Chirp3-HD-Leda"


def test_synthesize_rejects_missing_or_oversized_text(tmp_path: Path) -> None:
    client = TestClient(
        create_tts_app(_settings(tmp_path, max_text_chars=10), adapter=FakeAdapter())
    )
    auth = {"authorization": "Bearer tts-secret"}

    missing = client.post("/hydralisk/tts/v1/synthesize", json={}, headers=auth)
    assert missing.status_code == 400
    assert missing.json()["detail"]["error"]["code"] == "missing_text"

    blank = client.post(
        "/hydralisk/tts/v1/synthesize", json={"text": "  "}, headers=auth
    )
    assert blank.status_code == 400

    too_long = client.post(
        "/hydralisk/tts/v1/synthesize",
        json={"text": "x" * 11},
        headers=auth,
    )
    assert too_long.status_code == 413
    assert too_long.json()["detail"]["error"]["code"] == "text_too_long"


def test_empty_synthesis_records_blocker(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    client = TestClient(create_tts_app(settings, adapter=FakeAdapter(chunks=[])))

    response = client.post(
        "/hydralisk/tts/v1/synthesize",
        json={"text": "hello"},
        headers={"authorization": "Bearer tts-secret"},
    )

    assert response.status_code == 200
    assert response.content == b""
    receipts = TtsReceiptLog(settings.receipt_path).read_all()
    assert receipts[0]["blockers"] == [
        {
            "code": "empty_synthesis",
            "message": "The TTS adapter produced no audio.",
        }
    ]


def test_receipt_log_refuses_non_public_safe_keys(tmp_path: Path) -> None:
    log = TtsReceiptLog(tmp_path / "receipts.jsonl")

    with pytest.raises(ValueError):
        log.append({"runRef": "x", "text": "never store this"})


def test_receipt_lookup_missing_returns_404(tmp_path: Path) -> None:
    client = TestClient(create_tts_app(_settings(tmp_path), adapter=FakeAdapter()))

    response = client.get("/hydralisk/tts/v1/receipts/hydralisk-tts-run-missing")

    assert response.status_code == 404


def test_metrics_endpoint_reports_latency(tmp_path: Path) -> None:
    client = TestClient(create_tts_app(_settings(tmp_path), adapter=FakeAdapter()))

    client.post(
        "/hydralisk/tts/v1/synthesize",
        json={"text": "hello"},
        headers={"authorization": "Bearer tts-secret"},
    )
    response = client.get("/hydralisk/tts/v1/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["requests"]["total"] == 1
    assert body["requests"]["completed"] == 1
    assert body["requests"]["errors"] == 0
    assert isinstance(body["latencyMs"]["lastMsToFirstChunk"], int)
    assert isinstance(body["latencyMs"]["lastTotalMs"], int)
