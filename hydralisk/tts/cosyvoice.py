"""Self-hosted CosyVoice adapter (owned voice clone lane for OAV-3).

Zero-shot Sarah voice clone: a ~10 s owned reference wav plus its transcript
prime CosyVoice's ``inference_zero_shot`` streaming path. Output is converted
to the seam PCM contract (16-bit / 24 kHz / mono) regardless of the model's
native sample rate.

Heavy imports (cosyvoice, torch) are lazy and injectable, so this module and
its tests are CPU-safe: without a GPU or model checkout the adapter is still
constructible and testable against a fake model.

Setup (GPU host) and deploy steps live in docs/oav3-tts.md.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np

from hydralisk.tts.normalize import normalize_spoken
from hydralisk.tts.seam import PCM_SAMPLE_RATE_HZ, VoiceRef

ADAPTER_REF = "cosyvoice-zero-shot-streaming"

SARAH_CLONE_VOICE = VoiceRef(voice_id="sarah-cosyvoice-clone-v1", language_code="en-US")

DEFAULT_MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B-2512"

_STREAM_END = object()


@dataclass(frozen=True)
class CloneReference:
    """The owned voice reference that defines Sarah's cloned voice.

    ``prompt_wav`` must be clean 16 kHz-or-better mono speech we own.
    ``prompt_text`` is its exact transcript. Neither is ever written into
    receipts; only ``voice.voice_id`` is public.
    """

    prompt_wav: Path
    prompt_text: str
    voice: VoiceRef = SARAH_CLONE_VOICE


def pcm16_bytes_from_speech(
    speech: Any,
    *,
    source_sample_rate_hz: int,
) -> bytes:
    """Convert a float speech tensor/array in [-1, 1] to seam PCM bytes."""

    if hasattr(speech, "detach"):
        speech = speech.detach().cpu().numpy()
    samples = np.asarray(speech, dtype=np.float32).reshape(-1)
    if source_sample_rate_hz != PCM_SAMPLE_RATE_HZ:
        if source_sample_rate_hz <= 0:
            raise ValueError("source_sample_rate_hz must be positive")
        target_length = max(
            1, int(round(len(samples) * PCM_SAMPLE_RATE_HZ / source_sample_rate_hz))
        )
        positions = np.linspace(0.0, len(samples) - 1, target_length)
        samples = np.interp(positions, np.arange(len(samples)), samples).astype(
            np.float32
        )
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _load_cosyvoice_model(model_dir: str) -> Any:  # pragma: no cover - GPU path
    """Load the real CosyVoice model. Requires the CosyVoice checkout + GPU."""

    checkout = os.environ.get("HYDRALISK_TTS_COSYVOICE_CHECKOUT", "").strip()
    if checkout:
        for entry in (checkout, str(Path(checkout) / "third_party" / "Matcha-TTS")):
            if entry not in sys.path:
                sys.path.append(entry)
    try:
        from cosyvoice.cli.cosyvoice import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "The cosyvoice package is not importable. Set "
            "HYDRALISK_TTS_COSYVOICE_CHECKOUT to a CosyVoice checkout with its "
            "requirements installed (see docs/oav3-tts.md)."
        ) from exc
    return AutoModel(model_dir=model_dir)


class CosyVoiceCloneAdapter:
    """CosyVoice zero-shot clone behind the OAV-3 seam."""

    adapter_ref = ADAPTER_REF

    def __init__(
        self,
        reference: CloneReference,
        *,
        model_dir: str = DEFAULT_MODEL_DIR,
        model: Any | None = None,
        max_queue_chunks: int = 32,
    ) -> None:
        self.reference = reference
        self.default_voice = reference.voice
        self.model_dir = model_dir
        self._model = model
        self._model_lock = threading.Lock()
        self._max_queue_chunks = max_queue_chunks

    def _get_model(self) -> Any:
        with self._model_lock:
            if self._model is None:
                self._model = _load_cosyvoice_model(self.model_dir)
            return self._model

    async def synthesize_stream(
        self,
        text: str,
        voice_ref: VoiceRef | None = None,
    ) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        text = normalize_spoken(text)
        voice = voice_ref or self.default_voice
        if voice.voice_id != self.reference.voice.voice_id:
            raise ValueError(
                "CosyVoiceCloneAdapter only serves its configured clone voice; "
                f"got voiceId {voice.voice_id!r}."
            )
        model = await asyncio.to_thread(self._get_model)
        sample_rate = int(getattr(model, "sample_rate", PCM_SAMPLE_RATE_HZ))

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[object] = asyncio.Queue(maxsize=self._max_queue_chunks)

        def produce() -> None:
            try:
                chunks: Iterator[dict[str, Any]] = model.inference_zero_shot(
                    text,
                    self.reference.prompt_text,
                    str(self.reference.prompt_wav),
                    stream=True,
                )
                for chunk in chunks:
                    pcm = pcm16_bytes_from_speech(
                        chunk["tts_speech"],
                        source_sample_rate_hz=sample_rate,
                    )
                    if pcm:
                        asyncio.run_coroutine_threadsafe(queue.put(pcm), loop).result()
                asyncio.run_coroutine_threadsafe(queue.put(_STREAM_END), loop).result()
            except BaseException as exc:  # propagate into the async consumer
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()

        worker = threading.Thread(target=produce, name="cosyvoice-synthesis", daemon=True)
        worker.start()
        try:
            while True:
                item = await queue.get()
                if item is _STREAM_END:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item  # type: ignore[misc]
        finally:
            worker.join(timeout=1.0)


def default_cosyvoice_adapter() -> CosyVoiceCloneAdapter:
    prompt_wav = os.environ.get("HYDRALISK_TTS_COSYVOICE_PROMPT_WAV", "").strip()
    prompt_text = os.environ.get("HYDRALISK_TTS_COSYVOICE_PROMPT_TEXT", "").strip()
    if not prompt_wav or not prompt_text:
        raise RuntimeError(
            "CosyVoice clone lane is fail-closed without a voice reference: set "
            "HYDRALISK_TTS_COSYVOICE_PROMPT_WAV and "
            "HYDRALISK_TTS_COSYVOICE_PROMPT_TEXT (see docs/oav3-tts.md)."
        )
    model_dir = (
        os.environ.get("HYDRALISK_TTS_COSYVOICE_MODEL_DIR", "").strip()
        or DEFAULT_MODEL_DIR
    )
    return CosyVoiceCloneAdapter(
        CloneReference(prompt_wav=Path(prompt_wav), prompt_text=prompt_text),
        model_dir=model_dir,
    )
