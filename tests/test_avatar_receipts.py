from __future__ import annotations

from pathlib import Path

import pytest

from hydralisk.avatar.receipts import (
    SESSION_RECEIPT_SCHEMA,
    SESSION_REF_PATTERN,
    AvatarReceiptWriter,
    new_session_ref,
)


def test_new_session_ref_shape() -> None:
    ref = new_session_ref()
    assert SESSION_REF_PATTERN.match(ref)
    assert ref != new_session_ref()


def test_writer_rejects_invalid_session_ref(tmp_path: Path) -> None:
    writer = AvatarReceiptWriter(tmp_path)
    with pytest.raises(ValueError):
        writer.session_started(
            "../../etc/passwd", renderer="cpu-noop", meta={}
        )
    with pytest.raises(ValueError):
        writer.session_started(
            "hydralisk-run-" + "a" * 32, renderer="cpu-noop", meta={}
        )


def test_session_jsonl_lifecycle(tmp_path: Path) -> None:
    writer = AvatarReceiptWriter(tmp_path)
    ref = new_session_ref()

    writer.session_started(
        ref,
        renderer="cpu-noop",
        meta={"fps": 24, "width": 1280, "height": 720, "sampleRate": 24000},
    )
    writer.session_event(ref, "interrupt", {"interrupts": 1})
    summary = writer.session_summary(
        ref,
        started_at="2026-07-09T00:00:00Z",
        seconds=93.2,
        frames_rendered=2236,
        speaking_frames=800,
        interrupts=1,
        utterances=3,
        renderer="cpu-noop",
        stop_reason="client_stop",
        gpu={"name": "NVIDIA L4", "class": "l4", "count": 1},
    )

    rows = writer.read_rows(ref)
    assert len(rows) == 3
    assert [row["event"] for row in rows] == [
        "avatar.session_started",
        "avatar.session_event",
        "avatar.session_summary",
    ]
    assert all(row["schema"] == SESSION_RECEIPT_SCHEMA for row in rows)
    assert all(row["publicSafe"] is True for row in rows)
    assert all(row["sessionRef"] == ref for row in rows)

    assert summary["seconds"] == 93.2
    assert summary["minutes"] == 2  # billable minutes round up
    assert summary["framesRendered"] == 2236
    assert summary["interrupts"] == 1
    assert summary["utterances"] == 3
    assert summary["stopReason"] == "client_stop"

    # One JSONL file per session.
    assert (tmp_path / f"{ref}.jsonl").exists()


def test_minutes_rounding(tmp_path: Path) -> None:
    writer = AvatarReceiptWriter(tmp_path)

    def minutes(seconds: float) -> int:
        ref = new_session_ref()
        summary = writer.session_summary(
            ref,
            started_at="2026-07-09T00:00:00Z",
            seconds=seconds,
            frames_rendered=0,
            speaking_frames=0,
            interrupts=0,
            utterances=0,
            renderer="cpu-noop",
            stop_reason="test",
        )
        return summary["minutes"]

    assert minutes(0.0) == 0
    assert minutes(0.5) == 1
    assert minutes(60.0) == 1
    assert minutes(60.1) == 2


def test_read_rows_missing_session(tmp_path: Path) -> None:
    writer = AvatarReceiptWriter(tmp_path)
    assert writer.read_rows(new_session_ref()) == []
