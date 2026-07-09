from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from hydralisk.avatar.protocol import (
    ControlType,
    ProtocolError,
    decode_pcm16,
    encode_pcm16,
    parse_control_message,
)


def _speak_message(event_id: str = "utt-1", samples: int = 480) -> str:
    pcm = np.arange(samples, dtype=np.int16)
    return json.dumps(
        {
            "type": "agent.speak",
            "event_id": event_id,
            "audio": encode_pcm16(pcm),
        }
    )


def test_parse_speak_decodes_pcm() -> None:
    message = parse_control_message(_speak_message(samples=480))
    assert message.type is ControlType.SPEAK
    assert message.event_id == "utt-1"
    assert message.pcm is not None
    assert message.pcm.dtype == np.int16
    assert message.pcm.shape == (480,)
    assert message.pcm[3] == 3


def test_parse_speak_end_and_simple_types() -> None:
    end = parse_control_message(
        json.dumps({"type": "agent.speak_end", "event_id": "utt-1"})
    )
    assert end.type is ControlType.SPEAK_END
    assert end.event_id == "utt-1"

    for type_name, expected in [
        ("agent.interrupt", ControlType.INTERRUPT),
        ("agent.start_listening", ControlType.START_LISTENING),
        ("agent.stop_listening", ControlType.STOP_LISTENING),
        ("agent.keepalive", ControlType.KEEPALIVE),
    ]:
        message = parse_control_message(json.dumps({"type": type_name}))
        assert message.type is expected
        assert message.pcm is None


def test_parse_accepts_bytes_frames() -> None:
    message = parse_control_message(_speak_message().encode("utf-8"))
    assert message.type is ControlType.SPEAK


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps(["agent.speak"]),
        json.dumps({"type": "agent.unknown"}),
        json.dumps({"type": "avatar.speak_text"}),
    ],
)
def test_parse_fails_closed_on_malformed_frames(raw: str) -> None:
    with pytest.raises(ProtocolError):
        parse_control_message(raw)


def test_speak_requires_event_id_and_audio() -> None:
    with pytest.raises(ProtocolError) as excinfo:
        parse_control_message(
            json.dumps({"type": "agent.speak", "audio": encode_pcm16(np.zeros(4, dtype=np.int16))})
        )
    assert excinfo.value.code == "missing_event_id"

    with pytest.raises(ProtocolError) as excinfo:
        parse_control_message(
            json.dumps({"type": "agent.speak", "event_id": "utt-1"})
        )
    assert excinfo.value.code == "invalid_audio"


def test_decode_pcm16_rejects_bad_audio() -> None:
    with pytest.raises(ProtocolError):
        decode_pcm16("$$$not-base64$$$")
    with pytest.raises(ProtocolError):
        decode_pcm16("")
    # Odd byte count is not 16-bit PCM.
    odd = base64.b64encode(b"\x01\x02\x03").decode("ascii")
    with pytest.raises(ProtocolError) as excinfo:
        decode_pcm16(odd)
    assert excinfo.value.code == "invalid_audio"


def test_pcm_roundtrip() -> None:
    pcm = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    assert np.array_equal(decode_pcm16(encode_pcm16(pcm)), pcm)
