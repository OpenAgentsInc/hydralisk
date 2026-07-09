from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from hydralisk.tts.cosyvoice import pcm16_bytes_from_speech
from hydralisk.tts.seam import (
    PCM_SAMPLE_RATE_HZ,
    VoiceRef,
    instrument_stream,
)


class FakeAdapter:
    adapter_ref = "fake-tts"
    default_voice = VoiceRef(voice_id="fake-voice", language_code="en-US")

    def __init__(self, chunks: list[bytes], *, delay_s: float = 0.0) -> None:
        self._chunks = chunks
        self._delay_s = delay_s

    async def synthesize_stream(
        self,
        text: str,
        voice_ref: VoiceRef | None = None,
    ) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            if self._delay_s:
                await asyncio.sleep(self._delay_s)
            yield chunk


class ExplodingAdapter(FakeAdapter):
    async def synthesize_stream(
        self,
        text: str,
        voice_ref: VoiceRef | None = None,
    ) -> AsyncIterator[bytes]:
        yield b"\x00\x01"
        raise RuntimeError("synthesis backend failed")


@pytest.mark.asyncio
async def test_instrument_stream_measures_first_chunk_and_totals() -> None:
    adapter = FakeAdapter([b"\x00\x01" * 10, b"", b"\x02\x03" * 5], delay_s=0.01)

    synthesis = instrument_stream(adapter, "hello sarah")
    collected = [chunk async for chunk in synthesis.stream]

    assert collected == [b"\x00\x01" * 10, b"\x02\x03" * 5]
    metrics = synthesis.metrics
    assert metrics.adapter_ref == "fake-tts"
    assert metrics.voice == {"voiceId": "fake-voice", "languageCode": "en-US"}
    assert metrics.chars_in == len("hello sarah")
    assert metrics.chunks_out == 2
    assert metrics.bytes_out == 30
    assert metrics.ms_to_first_chunk is not None
    assert metrics.ms_to_first_chunk >= 5
    assert metrics.total_ms is not None
    assert metrics.total_ms >= metrics.ms_to_first_chunk
    assert metrics.error_code is None


@pytest.mark.asyncio
async def test_instrument_stream_records_error_and_total_time() -> None:
    synthesis = instrument_stream(ExplodingAdapter([]), "boom")

    with pytest.raises(RuntimeError):
        async for _ in synthesis.stream:
            pass

    assert synthesis.metrics.error_code == "RuntimeError"
    assert synthesis.metrics.total_ms is not None
    assert synthesis.metrics.chunks_out == 1


@pytest.mark.asyncio
async def test_instrument_stream_uses_explicit_voice_ref() -> None:
    adapter = FakeAdapter([b"\x00\x00"])
    voice = VoiceRef(voice_id="other-voice", language_code="en-GB")

    synthesis = instrument_stream(adapter, "hi", voice)
    _ = [chunk async for chunk in synthesis.stream]

    assert synthesis.metrics.voice == {
        "voiceId": "other-voice",
        "languageCode": "en-GB",
    }


def test_metrics_public_safe_never_contains_text() -> None:
    adapter = FakeAdapter([])
    synthesis = instrument_stream(adapter, "the secret utterance")

    payload = synthesis.metrics.public_safe()

    assert payload["charsIn"] == len("the secret utterance")
    assert "secret" not in str(payload)
    assert payload["pcm"]["sampleRateHz"] == PCM_SAMPLE_RATE_HZ


def test_pcm16_bytes_from_speech_scales_and_clips() -> None:
    samples = np.array([0.0, 0.5, 1.5, -1.5], dtype=np.float32)

    pcm = pcm16_bytes_from_speech(samples, source_sample_rate_hz=PCM_SAMPLE_RATE_HZ)

    decoded = np.frombuffer(pcm, dtype="<i2")
    assert decoded[0] == 0
    assert decoded[1] == 16383
    assert decoded[2] == 32767
    assert decoded[3] == -32767


def test_pcm16_bytes_from_speech_resamples_to_24khz() -> None:
    one_second_at_16k = np.zeros(16_000, dtype=np.float32)

    pcm = pcm16_bytes_from_speech(one_second_at_16k, source_sample_rate_hz=16_000)

    assert len(pcm) == PCM_SAMPLE_RATE_HZ * 2


def test_pcm16_bytes_from_speech_accepts_tensor_like() -> None:
    class TensorLike:
        def __init__(self, values: np.ndarray) -> None:
            self._values = values

        def detach(self) -> "TensorLike":
            return self

        def cpu(self) -> "TensorLike":
            return self

        def numpy(self) -> np.ndarray:
            return self._values

    tensor = TensorLike(np.array([[0.0, 0.25]], dtype=np.float32))

    pcm = pcm16_bytes_from_speech(tensor, source_sample_rate_hz=PCM_SAMPLE_RATE_HZ)

    decoded = np.frombuffer(pcm, dtype="<i2")
    assert decoded.tolist() == [0, 8191]
