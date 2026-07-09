from __future__ import annotations

from types import SimpleNamespace

import pytest

import hydralisk.tts.chirp as chirp_module
from hydralisk.tts.chirp import (
    SARAH_INTERIM_VOICE,
    ChirpStreamingAdapter,
    default_chirp_adapter,
)
from hydralisk.tts.seam import VoiceRef


class _FakeMessage:
    """Stands in for proto messages: records kwargs."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def __getattr__(self, name: str) -> object:
        try:
            return self.__dict__["kwargs"][name]
        except KeyError as exc:  # pragma: no cover - debug aid
            raise AttributeError(name) from exc


def _fake_texttospeech() -> SimpleNamespace:
    return SimpleNamespace(
        StreamingSynthesizeConfig=_FakeMessage,
        VoiceSelectionParams=_FakeMessage,
        StreamingAudioConfig=_FakeMessage,
        StreamingSynthesizeRequest=_FakeMessage,
        StreamingSynthesisInput=_FakeMessage,
        AudioEncoding=SimpleNamespace(PCM="PCM"),
    )


class FakeChirpClient:
    def __init__(self, audio_chunks: list[bytes]) -> None:
        self._audio_chunks = audio_chunks
        self.seen_requests: list[_FakeMessage] = []

    async def streaming_synthesize(self, requests):
        async for request in requests:
            self.seen_requests.append(request)

        async def responses():
            for chunk in self._audio_chunks:
                yield SimpleNamespace(audio_content=chunk)

        return responses()


@pytest.mark.asyncio
async def test_chirp_adapter_streams_audio_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chirp_module, "_load_texttospeech", _fake_texttospeech)
    client = FakeChirpClient([b"\x01\x02" * 100, b"", b"\x03\x04" * 50])
    adapter = ChirpStreamingAdapter(client=client)

    chunks = [chunk async for chunk in adapter.synthesize_stream("hello sarah")]

    assert chunks == [b"\x01\x02" * 100, b"\x03\x04" * 50]
    # First request carries streaming config; second carries the text input.
    assert len(client.seen_requests) == 2
    config = client.seen_requests[0].kwargs["streaming_config"]
    voice = config.kwargs["voice"]
    assert voice.kwargs["name"] == SARAH_INTERIM_VOICE.voice_id
    assert voice.kwargs["language_code"] == "en-US"
    audio_config = config.kwargs["streaming_audio_config"]
    assert audio_config.kwargs["audio_encoding"] == "PCM"
    assert audio_config.kwargs["sample_rate_hertz"] == 24000
    text_input = client.seen_requests[1].kwargs["input"]
    assert text_input.kwargs["text"] == "hello sarah"


@pytest.mark.asyncio
async def test_chirp_adapter_uses_requested_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chirp_module, "_load_texttospeech", _fake_texttospeech)
    client = FakeChirpClient([b"\x00\x00"])
    adapter = ChirpStreamingAdapter(client=client)

    voice = VoiceRef(voice_id="en-US-Chirp3-HD-Leda", language_code="en-US")
    _ = [chunk async for chunk in adapter.synthesize_stream("hi", voice)]

    config = client.seen_requests[0].kwargs["streaming_config"]
    assert config.kwargs["voice"].kwargs["name"] == "en-US-Chirp3-HD-Leda"


@pytest.mark.asyncio
async def test_chirp_adapter_skips_empty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(chirp_module, "_load_texttospeech", _fake_texttospeech)
    client = FakeChirpClient([b"\x00\x00"])
    adapter = ChirpStreamingAdapter(client=client)

    chunks = [chunk async for chunk in adapter.synthesize_stream("   ")]

    assert chunks == []
    assert client.seen_requests == []


def test_default_voice_is_sarah_interim() -> None:
    adapter = ChirpStreamingAdapter(client=object())

    assert adapter.default_voice == SARAH_INTERIM_VOICE
    assert adapter.default_voice.voice_id == "en-US-Chirp3-HD-Sulafat"
    assert adapter.adapter_ref == "google-chirp3-hd-streaming"


def test_default_chirp_adapter_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYDRALISK_TTS_CHIRP_VOICE", "en-US-Chirp3-HD-Kore")
    monkeypatch.setenv("HYDRALISK_TTS_CHIRP_LANGUAGE", "en-US")

    adapter = default_chirp_adapter()

    assert adapter.default_voice.voice_id == "en-US-Chirp3-HD-Kore"
