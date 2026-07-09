"""Public-safe JSONL receipts for the Hydralisk TTS lane.

One line per synthesis run: characters in, ms to first chunk, total ms,
bytes/chunks out, adapter and voice identity. Never the synthesized text,
never prompt wav paths, never credentials.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from hydralisk.tts.seam import SynthesisMetrics

TTS_RECEIPT_SCHEMA = "hydralisk.tts.run_receipt.v1"

_FORBIDDEN_RECEIPT_KEYS = {"text", "promptText", "promptWav", "token", "authorization"}


def tts_run_ref() -> str:
    return f"hydralisk-tts-run-{uuid4().hex}"


class TtsReceiptLog:
    """Append-only JSONL receipt log."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def append(self, receipt: dict[str, Any]) -> None:
        forbidden = _FORBIDDEN_RECEIPT_KEYS.intersection(receipt)
        if forbidden:
            raise ValueError(
                f"refusing to write non-public-safe receipt keys: {sorted(forbidden)}"
            )
        line = json.dumps(receipt, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


def build_tts_receipt(
    *,
    run_ref: str,
    metrics: SynthesisMetrics,
    blockers: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "schema": TTS_RECEIPT_SCHEMA,
        "runRef": run_ref,
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **metrics.public_safe(),
        "publicSafe": True,
        "blockers": blockers or [],
    }
