"""Control-protocol parsing for the avatar WebSocket API.

The wire protocol mirrors the LiveAvatar LITE cycle the OAV spec documents,
so the apps/sarah OAV-4 seam can speak the shape it already knows:

Client → server (JSON text frames):

    {"type": "agent.speak", "event_id": "...", "audio": "<b64 pcm16le mono>"}
    {"type": "agent.speak_end", "event_id": "..."}
    {"type": "agent.interrupt"}
    {"type": "agent.start_listening"}
    {"type": "agent.stop_listening"}
    {"type": "agent.keepalive"}

Audio is 16-bit little-endian mono PCM at 24 kHz ("audio format is king" —
wrong format renders garbled with no error, so parsing here fails closed on
anything malformed). All chunks of one utterance share the same `event_id`.

Server → client:

    {"type": "session.state", "state": "idle|listening|speaking", ...}
    {"type": "agent.speak_started"|"agent.speak_ended", "event_id": "..."}
    {"type": "session.keepalive_ack"}
    {"type": "session.error", "code": "...", "message": "..."}
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from enum import Enum
import json

import numpy as np


PROTOCOL_VERSION = "hydralisk.avatar.control.v1"

MAX_AUDIO_CHUNK_BYTES = 1_048_576  # 1 MiB of PCM per message is plenty.
MAX_EVENT_ID_LENGTH = 128


class ControlType(str, Enum):
    SPEAK = "agent.speak"
    SPEAK_END = "agent.speak_end"
    INTERRUPT = "agent.interrupt"
    START_LISTENING = "agent.start_listening"
    STOP_LISTENING = "agent.stop_listening"
    KEEPALIVE = "agent.keepalive"


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ControlMessage:
    type: ControlType
    event_id: str | None = None
    pcm: np.ndarray | None = None  # int16 mono samples


def decode_pcm16(audio_b64: str) -> np.ndarray:
    """Decode base64 16-bit little-endian mono PCM into int16 samples."""
    if not isinstance(audio_b64, str) or not audio_b64:
        raise ProtocolError("invalid_audio", "audio must be a base64 string")
    try:
        raw = base64.b64decode(audio_b64, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ProtocolError(
            "invalid_audio", "audio is not valid base64"
        ) from error
    if not raw:
        raise ProtocolError("invalid_audio", "audio chunk is empty")
    if len(raw) > MAX_AUDIO_CHUNK_BYTES:
        raise ProtocolError(
            "audio_chunk_too_large",
            f"audio chunk exceeds {MAX_AUDIO_CHUNK_BYTES} bytes",
        )
    if len(raw) % 2 != 0:
        raise ProtocolError(
            "invalid_audio",
            "audio must be 16-bit PCM (even byte length)",
        )
    return np.frombuffer(raw, dtype="<i2")


def encode_pcm16(samples: np.ndarray) -> str:
    return base64.b64encode(
        np.asarray(samples, dtype="<i2").tobytes()
    ).decode("ascii")


def _require_event_id(payload: dict, *, required: bool) -> str | None:
    event_id = payload.get("event_id")
    if event_id is None:
        if required:
            raise ProtocolError("missing_event_id", "event_id is required")
        return None
    if not isinstance(event_id, str) or not event_id:
        raise ProtocolError(
            "invalid_event_id", "event_id must be a non-empty string"
        )
    if len(event_id) > MAX_EVENT_ID_LENGTH:
        raise ProtocolError("invalid_event_id", "event_id is too long")
    return event_id


def parse_control_message(raw: str | bytes) -> ControlMessage:
    """Parse one client control frame; fail closed on anything malformed."""
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProtocolError(
                "invalid_message", "control frames must be UTF-8 JSON"
            ) from error
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ProtocolError(
            "invalid_message", "control frames must be JSON"
        ) from error
    if not isinstance(payload, dict):
        raise ProtocolError(
            "invalid_message", "control frames must be JSON objects"
        )

    type_value = payload.get("type")
    try:
        control_type = ControlType(type_value)
    except ValueError:
        raise ProtocolError(
            "unknown_message_type",
            f"unknown control message type: {type_value!r}",
        ) from None

    if control_type is ControlType.SPEAK:
        event_id = _require_event_id(payload, required=True)
        if "audio" not in payload:
            raise ProtocolError(
                "invalid_audio", "agent.speak requires an audio field"
            )
        pcm = decode_pcm16(payload["audio"])
        return ControlMessage(type=control_type, event_id=event_id, pcm=pcm)

    if control_type is ControlType.SPEAK_END:
        event_id = _require_event_id(payload, required=False)
        return ControlMessage(type=control_type, event_id=event_id)

    return ControlMessage(type=control_type)


def state_event(state: str, session_ref: str) -> dict:
    return {
        "type": "session.state",
        "state": state,
        "sessionRef": session_ref,
        "protocol": PROTOCOL_VERSION,
    }


def error_event(code: str, message: str) -> dict:
    return {"type": "session.error", "code": code, "message": message}
