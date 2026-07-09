"""Google Cloud TTS Chirp 3 HD adapter (managed interim for OAV-3).

Uses ``streaming_synthesize`` so first audio arrives before the full
utterance is rendered. Emits the seam PCM contract directly: Chirp 3 HD
streaming supports raw 16-bit PCM at 24 kHz.

Interim-voice decision (see docs/oav3-tts.md): Sarah's interim prebuilt
voice is ``en-US-Chirp3-HD-Sulafat`` — Google's "warm" en-US female Chirp 3
HD voice, matching the warm/professional brief. Instant Custom Voice
(cloning) on Chirp 3 is allow-list gated and intentionally NOT a dependency
of this lane; the owned clone path is CosyVoice (``hydralisk/tts/cosyvoice.py``).

The google-cloud-texttospeech dependency is optional
(``uv sync --extra tts-chirp``); import happens lazily so the rest of the
TTS lane stays importable without it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
import os
from typing import Any

from hydralisk.tts.seam import PCM_SAMPLE_RATE_HZ, VoiceRef

SARAH_INTERIM_VOICE = VoiceRef(
    voice_id="en-US-Chirp3-HD-Sulafat",
    language_code="en-US",
)

ADAPTER_REF = "google-chirp3-hd-streaming"


def _load_texttospeech() -> Any:
    try:
        from google.cloud import texttospeech
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "google-cloud-texttospeech is not installed; "
            "install with `uv sync --extra tts-chirp`."
        ) from exc
    return texttospeech


class ChirpStreamingAdapter:
    """Chirp 3 HD ``streaming_synthesize`` behind the OAV-3 seam.

    Credentials resolve through Application Default Credentials. On the
    Hydralisk operator Mac, point ``GOOGLE_APPLICATION_CREDENTIALS`` at the
    machine-local automation service-account key (never committed).
    """

    adapter_ref = ADAPTER_REF

    def __init__(
        self,
        *,
        default_voice: VoiceRef | None = None,
        client: Any | None = None,
    ) -> None:
        self.default_voice = default_voice or SARAH_INTERIM_VOICE
        self._client = client

    async def _get_client(self) -> Any:
        if self._client is None:
            texttospeech = _load_texttospeech()
            self._client = texttospeech.TextToSpeechAsyncClient()
        return self._client

    async def synthesize_stream(
        self,
        text: str,
        voice_ref: VoiceRef | None = None,
    ) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        texttospeech = _load_texttospeech()
        voice = voice_ref or self.default_voice
        client = await self._get_client()

        streaming_config = texttospeech.StreamingSynthesizeConfig(
            voice=texttospeech.VoiceSelectionParams(
                name=voice.voice_id,
                language_code=voice.language_code,
            ),
            streaming_audio_config=texttospeech.StreamingAudioConfig(
                audio_encoding=texttospeech.AudioEncoding.PCM,
                sample_rate_hertz=PCM_SAMPLE_RATE_HZ,
            ),
        )

        async def requests() -> AsyncIterator[Any]:
            yield texttospeech.StreamingSynthesizeRequest(
                streaming_config=streaming_config
            )
            yield texttospeech.StreamingSynthesizeRequest(
                input=texttospeech.StreamingSynthesisInput(text=text)
            )

        stream = await client.streaming_synthesize(requests=requests())
        async for response in stream:
            audio = bytes(response.audio_content)
            if audio:
                yield audio


def default_chirp_adapter() -> ChirpStreamingAdapter:
    voice_id = os.environ.get("HYDRALISK_TTS_CHIRP_VOICE", "").strip()
    language = os.environ.get("HYDRALISK_TTS_CHIRP_LANGUAGE", "").strip() or "en-US"
    if voice_id:
        return ChirpStreamingAdapter(
            default_voice=VoiceRef(voice_id=voice_id, language_code=language)
        )
    return ChirpStreamingAdapter()
