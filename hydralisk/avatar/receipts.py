"""Per-session JSONL receipts for the avatar render service.

Matches Hydralisk's public-safe receipt posture: no prompts, no audio, no
tokens, no endpoint secrets — just session accounting (session ref,
duration, frames, interrupts, utterances, renderer/gpu identity) that can
be published or reconciled later. One JSONL file per session:

    <receipt_dir>/<session_ref>.jsonl

Rows are `avatar.session_started`, optional `avatar.session_event` rows,
and a closing `avatar.session_summary`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any
from uuid import uuid4


SESSION_RECEIPT_SCHEMA = "hydralisk.avatar.session_receipt.v1"
SESSION_REF_PATTERN = re.compile(r"^hydralisk-avatar-[a-f0-9]{32}$")


def new_session_ref() -> str:
    return f"hydralisk-avatar-{uuid4().hex}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class AvatarReceiptWriter:
    def __init__(self, receipt_dir: Path) -> None:
        self.receipt_dir = receipt_dir

    def _path(self, session_ref: str) -> Path:
        if not SESSION_REF_PATTERN.match(session_ref):
            raise ValueError("invalid Hydralisk avatar sessionRef")
        return self.receipt_dir / f"{session_ref}.jsonl"

    def _append(self, session_ref: str, row: dict[str, Any]) -> None:
        path = self._path(session_ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def session_started(
        self, session_ref: str, *, renderer: str, meta: dict[str, Any]
    ) -> None:
        self._append(
            session_ref,
            {
                "schema": SESSION_RECEIPT_SCHEMA,
                "event": "avatar.session_started",
                "sessionRef": session_ref,
                "at": _utc_now(),
                "renderer": renderer,
                "publicSafe": True,
                **meta,
            },
        )

    def session_event(
        self, session_ref: str, event: str, data: dict[str, Any] | None = None
    ) -> None:
        self._append(
            session_ref,
            {
                "schema": SESSION_RECEIPT_SCHEMA,
                "event": "avatar.session_event",
                "sessionRef": session_ref,
                "at": _utc_now(),
                "name": event,
                "publicSafe": True,
                **(data or {}),
            },
        )

    def session_summary(
        self,
        session_ref: str,
        *,
        started_at: str,
        seconds: float,
        frames_rendered: int,
        speaking_frames: int,
        interrupts: int,
        utterances: int,
        renderer: str,
        stop_reason: str,
        gpu: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        summary = {
            "schema": SESSION_RECEIPT_SCHEMA,
            "event": "avatar.session_summary",
            "sessionRef": session_ref,
            "startedAt": started_at,
            "endedAt": _utc_now(),
            "seconds": round(seconds, 3),
            "minutes": _billable_minutes(seconds),
            "framesRendered": frames_rendered,
            "speakingFrames": speaking_frames,
            "interrupts": interrupts,
            "utterances": utterances,
            "renderer": renderer,
            "stopReason": stop_reason,
            "publicSafe": True,
        }
        if gpu is not None:
            summary["gpu"] = gpu
        self._append(session_ref, summary)
        return summary

    def read_rows(self, session_ref: str) -> list[dict[str, Any]]:
        path = self._path(session_ref)
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def _billable_minutes(seconds: float) -> int:
    if seconds <= 0:
        return 0
    return int(math.ceil(seconds / 60.0))
