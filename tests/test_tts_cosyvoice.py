from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pytest

from hydralisk.tts.cosyvoice import (
    CloneReference,
    CosyVoiceCloneAdapter,
    default_cosyvoice_adapter,
)
from hydralisk.tts.seam import PCM_SAMPLE_RATE_HZ, VoiceRef, instrument_stream


class FakeCosyVoiceModel:
    """CPU-safe stand-in for the real CosyVoice AutoModel."""

    def __init__(self, *, sample_rate: int = PCM_SAMPLE_RATE_HZ, delay_s: float = 0.0):
        self.sample_rate = sample_rate
        self._delay_s = delay_s
        self.calls: list[tuple[str, str, str]] = []

    def inference_zero_shot(
        self,
        text: str,
        prompt_text: str,
        prompt_wav: str,
        stream: bool = False,
    ):
        assert stream is True
        self.calls.append((text, prompt_text, prompt_wav))
        for value in (0.25, -0.25):
            if self._delay_s:
                time.sleep(self._delay_s)
            yield {"tts_speech": np.full((1, 480), value, dtype=np.float32)}


def _reference(tmp_path: Path) -> CloneReference:
    wav = tmp_path / "sarah-ref.wav"
    wav.write_bytes(b"RIFF-fake")
    return CloneReference(prompt_wav=wav, prompt_text="an owned sarah reference read")


@pytest.mark.asyncio
async def test_cosyvoice_streams_pcm_from_fake_model(tmp_path: Path) -> None:
    model = FakeCosyVoiceModel()
    adapter = CosyVoiceCloneAdapter(_reference(tmp_path), model=model)

    chunks = [chunk async for chunk in adapter.synthesize_stream("hello sarah")]

    assert len(chunks) == 2
    first = np.frombuffer(chunks[0], dtype="<i2")
    assert len(first) == 480
    assert first[0] == 8191
    second = np.frombuffer(chunks[1], dtype="<i2")
    assert second[0] == -8191
    assert model.calls == [
        ("hello sarah", "an owned sarah reference read", str(adapter.reference.prompt_wav))
    ]


@pytest.mark.asyncio
async def test_cosyvoice_resamples_non_24k_models(tmp_path: Path) -> None:
    model = FakeCosyVoiceModel(sample_rate=22050)
    adapter = CosyVoiceCloneAdapter(_reference(tmp_path), model=model)

    chunks = [chunk async for chunk in adapter.synthesize_stream("hello")]

    expected_samples = round(480 * PCM_SAMPLE_RATE_HZ / 22050)
    assert len(chunks[0]) == expected_samples * 2


@pytest.mark.asyncio
async def test_cosyvoice_rejects_foreign_voice_ids(tmp_path: Path) -> None:
    adapter = CosyVoiceCloneAdapter(_reference(tmp_path), model=FakeCosyVoiceModel())

    with pytest.raises(ValueError):
        async for _ in adapter.synthesize_stream(
            "hi", VoiceRef(voice_id="someone-else")
        ):
            pass


@pytest.mark.asyncio
async def test_cosyvoice_empty_text_yields_nothing(tmp_path: Path) -> None:
    model = FakeCosyVoiceModel()
    adapter = CosyVoiceCloneAdapter(_reference(tmp_path), model=model)

    chunks = [chunk async for chunk in adapter.synthesize_stream("   ")]

    assert chunks == []
    assert model.calls == []


@pytest.mark.asyncio
async def test_cosyvoice_under_instrumentation_measures_first_chunk(
    tmp_path: Path,
) -> None:
    model = FakeCosyVoiceModel(delay_s=0.01)
    adapter = CosyVoiceCloneAdapter(_reference(tmp_path), model=model)

    synthesis = instrument_stream(adapter, "measure me")
    _ = [chunk async for chunk in synthesis.stream]

    assert synthesis.metrics.chunks_out == 2
    assert synthesis.metrics.ms_to_first_chunk is not None
    assert synthesis.metrics.ms_to_first_chunk >= 5


@pytest.mark.asyncio
async def test_cosyvoice_propagates_model_errors(tmp_path: Path) -> None:
    class BrokenModel(FakeCosyVoiceModel):
        def inference_zero_shot(self, *args, **kwargs):
            raise RuntimeError("model exploded")
            yield  # pragma: no cover

    adapter = CosyVoiceCloneAdapter(_reference(tmp_path), model=BrokenModel())

    with pytest.raises(RuntimeError):
        async for _ in adapter.synthesize_stream("hi"):
            pass


def test_default_adapter_fails_closed_without_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HYDRALISK_TTS_COSYVOICE_PROMPT_WAV", raising=False)
    monkeypatch.delenv("HYDRALISK_TTS_COSYVOICE_PROMPT_TEXT", raising=False)

    with pytest.raises(RuntimeError, match="fail-closed"):
        default_cosyvoice_adapter()


def test_default_adapter_reads_reference_from_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HYDRALISK_TTS_COSYVOICE_PROMPT_WAV", str(tmp_path / "ref.wav"))
    monkeypatch.setenv("HYDRALISK_TTS_COSYVOICE_PROMPT_TEXT", "reference transcript")
    monkeypatch.setenv("HYDRALISK_TTS_COSYVOICE_MODEL_DIR", "models/custom")

    adapter = default_cosyvoice_adapter()

    assert adapter.reference.prompt_text == "reference transcript"
    assert adapter.model_dir == "models/custom"
    assert adapter.default_voice.voice_id == "sarah-cosyvoice-clone-v1"
