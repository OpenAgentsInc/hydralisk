"""Hydralisk TTS lane (OAV-3).

Streaming text-to-speech behind one seam: async PCM 16-bit 24 kHz mono
chunks with time-to-first-chunk instrumentation. Adapters:

- ``chirp3hd``: Google Cloud TTS Chirp 3 HD ``streaming_synthesize``
  (managed interim, no GPU required).
- ``cosyvoice``: self-hosted CosyVoice zero-shot voice clone from an owned
  Sarah reference wav (GPU lane).

Consumers: the OAV-2 avatar render service and future ``apps/sarah``
surfaces. See ``docs/oav3-tts.md``.
"""

from hydralisk.tts.seam import (
    PCM_CHANNELS,
    PCM_SAMPLE_RATE_HZ,
    PCM_SAMPLE_WIDTH_BYTES,
    InstrumentedSynthesis,
    SynthesisMetrics,
    TtsAdapter,
    VoiceRef,
    instrument_stream,
)

__all__ = [
    "PCM_CHANNELS",
    "PCM_SAMPLE_RATE_HZ",
    "PCM_SAMPLE_WIDTH_BYTES",
    "InstrumentedSynthesis",
    "SynthesisMetrics",
    "TtsAdapter",
    "VoiceRef",
    "instrument_stream",
]
