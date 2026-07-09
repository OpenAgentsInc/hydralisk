"""The OAV-3 TTS seam.

One contract for every TTS backend Hydralisk serves:

    synthesize_stream(text, voice_ref) -> async iterator of PCM chunks

PCM contract (fixed, every adapter must emit exactly this):

- 16-bit signed little-endian samples
- 24,000 Hz sample rate
- mono (1 channel)

Adapters that cannot natively produce this format must convert before
yielding. Consumers (the OAV-2 avatar render service, apps/sarah) may rely
on the format without inspecting adapter identity.

``instrument_stream`` wraps any adapter stream with time-to-first-chunk and
total-wall-time measurement so the HTTP service and receipts report honest
numbers without each adapter re-implementing timing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from time import perf_counter
from typing import Protocol, runtime_checkable

PCM_SAMPLE_RATE_HZ = 24_000
PCM_SAMPLE_WIDTH_BYTES = 2
PCM_CHANNELS = 1
PCM_MEDIA_TYPE = "audio/L16;rate=24000;channels=1"


@dataclass(frozen=True)
class VoiceRef:
    """Public-safe reference to a voice.

    ``voice_id`` is an adapter-scoped identifier: a prebuilt voice name for
    managed adapters (for example ``en-US-Chirp3-HD-Sulafat``) or a cloned
    speaker id for CosyVoice. It must never carry secrets, prompt text, or
    filesystem paths that should not appear in receipts.
    """

    voice_id: str
    language_code: str = "en-US"

    def public_safe(self) -> dict[str, str]:
        return {"voiceId": self.voice_id, "languageCode": self.language_code}


@runtime_checkable
class TtsAdapter(Protocol):
    """The seam every TTS backend implements."""

    adapter_ref: str
    default_voice: VoiceRef

    def synthesize_stream(
        self,
        text: str,
        voice_ref: VoiceRef | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield PCM 16-bit 24 kHz mono chunks for ``text``."""
        ...


@dataclass
class SynthesisMetrics:
    """Public-safe timing for one synthesis run.

    Never stores the synthesized text, only its length.
    """

    adapter_ref: str
    voice: dict[str, str]
    chars_in: int
    ms_to_first_chunk: int | None = None
    total_ms: int | None = None
    chunks_out: int = 0
    bytes_out: int = 0
    error_code: str | None = None

    @property
    def audio_seconds_out(self) -> float:
        bytes_per_second = PCM_SAMPLE_RATE_HZ * PCM_SAMPLE_WIDTH_BYTES * PCM_CHANNELS
        return round(self.bytes_out / bytes_per_second, 3)

    def public_safe(self) -> dict[str, object]:
        return {
            "adapterRef": self.adapter_ref,
            "voice": dict(self.voice),
            "charsIn": self.chars_in,
            "msToFirstChunk": self.ms_to_first_chunk,
            "totalMs": self.total_ms,
            "chunksOut": self.chunks_out,
            "bytesOut": self.bytes_out,
            "audioSecondsOut": self.audio_seconds_out,
            "pcm": {
                "sampleRateHz": PCM_SAMPLE_RATE_HZ,
                "sampleWidthBytes": PCM_SAMPLE_WIDTH_BYTES,
                "channels": PCM_CHANNELS,
            },
            "errorCode": self.error_code,
        }


@dataclass
class InstrumentedSynthesis:
    """A live synthesis stream plus the metrics being filled in.

    ``metrics`` is complete once the stream is exhausted (or has raised).
    """

    stream: AsyncIterator[bytes]
    metrics: SynthesisMetrics = field(kw_only=True)


def instrument_stream(
    adapter: TtsAdapter,
    text: str,
    voice_ref: VoiceRef | None = None,
) -> InstrumentedSynthesis:
    """Wrap an adapter synthesis with first-chunk/total timing."""

    voice = voice_ref or adapter.default_voice
    metrics = SynthesisMetrics(
        adapter_ref=adapter.adapter_ref,
        voice=voice.public_safe(),
        chars_in=len(text),
    )

    async def measured() -> AsyncIterator[bytes]:
        started = perf_counter()
        try:
            async for chunk in adapter.synthesize_stream(text, voice):
                if not chunk:
                    continue
                if metrics.ms_to_first_chunk is None:
                    metrics.ms_to_first_chunk = int((perf_counter() - started) * 1000)
                metrics.chunks_out += 1
                metrics.bytes_out += len(chunk)
                yield chunk
        except Exception as exc:
            metrics.error_code = type(exc).__name__
            raise
        finally:
            metrics.total_ms = int((perf_counter() - started) * 1000)

    return InstrumentedSynthesis(stream=measured(), metrics=metrics)
