"""Real-session simulator: the pre-deploy gate for the avatar render service.

Owner-demanded ("can't you simulate a real session"): connect exactly like
the browser does — compat mint, raw-SDP WHEP-style offer, recvonly WebRTC —
then drive the apps/sarah speaking bridge shape against the LIVE service:

- sentence-streamed utterances: PCM chunked like owned-renderer.ts
  ``chunkPcmBase64`` (~600 ms first chunk, 1 s after), pushed in groups with
  realistic synthesis gaps BETWEEN groups under ONE event_id, then speak_end;
- a second turn after a pause;
- a long idle tail.

Per phase it counts actually received video frames off the WebRTC track and fails
loudly if the frame rate collapses or any inter-frame gap exceeds the stall
budget — the exact defect class of the 2026-07-09 mid-utterance freeze
(session ...96b6078c: state=speaking, framesRendered frozen, no traceback).

Usage (bearer via env so it never lands in argv/receipts):

    HYDRALISK_AVATAR_BEARER_TOKEN=... \\
    uv run hydralisk-avatar-sim --base https://<host> --min-fps 18
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import dataclass, field
import json
import math
import os
import sys
import time
from typing import Any
import urllib.request

PCM_SAMPLE_RATE_HZ = 24_000
PCM_BYTES_PER_SAMPLE = 2
FIRST_CHUNK_MS = 600
NEXT_CHUNK_MS = 1_000


def chunk_pcm(pcm: bytes) -> list[bytes]:
    """Replicates apps/sarah owned-renderer.ts chunkPcmBase64 exactly:
    ~600 ms first chunk (fast time-to-first-frame), 1 s chunks after."""
    if not pcm:
        return []
    chunks: list[bytes] = []
    offset = 0
    size = PCM_SAMPLE_RATE_HZ * FIRST_CHUNK_MS // 1000 * PCM_BYTES_PER_SAMPLE
    while offset < len(pcm):
        chunks.append(pcm[offset : offset + size])
        offset += size
        size = PCM_SAMPLE_RATE_HZ * NEXT_CHUNK_MS // 1000 * PCM_BYTES_PER_SAMPLE
    return chunks


def speechlike_pcm(seconds: float, *, base_hz: float = 180.0) -> bytes:
    """Synthetic voiced-ish PCM (AM-modulated glide) — enough to exercise
    the whisper-feature + inpaint path without shipping voice data."""
    n = int(seconds * PCM_SAMPLE_RATE_HZ)
    out = bytearray()
    for i in range(n):
        t = i / PCM_SAMPLE_RATE_HZ
        f = base_hz * (1.0 + 0.2 * math.sin(2 * math.pi * 1.3 * t))
        envelope = 0.55 + 0.45 * math.sin(2 * math.pi * 2.1 * t)
        sample = int(12_000 * envelope * math.sin(2 * math.pi * f * t))
        out += int(sample).to_bytes(2, "little", signed=True)
    return bytes(out)


@dataclass
class PhaseStats:
    name: str
    started: float = 0.0
    ended: float = 0.0
    frames: int = 0
    max_gap_s: float = 0.0

    @property
    def seconds(self) -> float:
        return max(1e-9, self.ended - self.started)

    @property
    def fps(self) -> float:
        return self.frames / self.seconds

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.name,
            "seconds": round(self.seconds, 2),
            "frames": self.frames,
            "fps": round(self.fps, 2),
            "maxInterFrameGapSeconds": round(self.max_gap_s, 3),
        }


def evaluate_phases(
    phases: list[PhaseStats], *, min_fps: float, max_gap_s: float
) -> list[str]:
    """Pure assertion logic (unit-tested): returns failure strings."""
    failures: list[str] = []
    for phase in phases:
        if phase.fps < min_fps:
            failures.append(
                f"{phase.name}: fps {phase.fps:.1f} < required {min_fps}"
            )
        if phase.max_gap_s > max_gap_s:
            failures.append(
                f"{phase.name}: inter-frame gap {phase.max_gap_s:.2f}s "
                f"> allowed {max_gap_s}s (render loop stall)"
            )
    return failures


class FrameMeter:
    """Counts received video frames; tracks the largest inter-frame gap."""

    def __init__(self) -> None:
        self.total = 0
        self.last_at: float | None = None
        self._phase: PhaseStats | None = None

    def start_phase(self, name: str) -> PhaseStats:
        now = time.monotonic()
        if self._phase is not None:
            self._phase.ended = now
        self._phase = PhaseStats(name=name, started=now)
        # A gap spanning a phase boundary counts against the new phase.
        self.last_at = now if self.last_at is None else self.last_at
        return self._phase

    def end_phase(self) -> None:
        if self._phase is not None:
            self._phase.ended = time.monotonic()

    def on_frame(self) -> None:
        now = time.monotonic()
        self.total += 1
        if self._phase is not None:
            self._phase.frames += 1
            if self.last_at is not None:
                self._phase.max_gap_s = max(
                    self._phase.max_gap_s, now - self.last_at
                )
        self.last_at = now


def _http(
    url: str,
    *,
    method: str = "POST",
    token: str | None = None,
    json_body: dict | None = None,
    raw_body: bytes | None = None,
    content_type: str | None = None,
) -> tuple[int, bytes]:
    headers: dict[str, str] = {}
    if token:
        headers["authorization"] = f"Bearer {token}"
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode()
        headers["content-type"] = "application/json"
    elif raw_body is not None:
        data = raw_body
        headers["content-type"] = content_type or "application/octet-stream"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read()


async def run_simulation(
    *,
    base: str,
    token: str,
    min_fps: float = 18.0,
    max_gap_s: float = 1.0,
    idle_seconds: float = 30.0,
    utterance_groups: int = 3,
    group_seconds: float = 2.2,
    group_gap_s: float = 0.8,
) -> dict[str, Any]:
    from aiortc import RTCPeerConnection, RTCSessionDescription

    base = base.rstrip("/")
    status, body = _http(
        f"{base}/sessions", token=token, json_body={"conversation_ref": "sim:gate"}
    )
    if status not in (200, 201):
        raise RuntimeError(f"mint failed: {status} {body[:200]!r}")
    mint = json.loads(body)
    session_id = mint["session_id"]
    offer_url = mint["webrtc"]["offer_url"]

    meter = FrameMeter()
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")

    consumer_tasks: list[asyncio.Task] = []

    @pc.on("track")
    def on_track(track: Any) -> None:
        async def consume() -> None:
            while True:
                try:
                    await track.recv()
                except Exception:
                    return
                if track.kind == "video":
                    meter.on_frame()

        consumer_tasks.append(asyncio.create_task(consume()))

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    status, answer = _http(
        offer_url,
        raw_body=pc.localDescription.sdp.encode(),
        content_type="application/sdp",
    )
    if status != 200:
        raise RuntimeError(f"offer failed: {status} {answer[:200]!r}")
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer.decode(), type="answer")
    )

    async def control(payload: dict[str, Any]) -> None:
        await asyncio.to_thread(
            _http,
            f"{base}/sessions/{session_id}/control",
            token=token,
            json_body=payload,
        )

    async def utterance(event_id: str) -> float:
        """Sentence-streamed utterance: groups of chunks with synthesis gaps
        between groups, all under one event_id — the apps/sarah shape."""
        total_audio = 0.0
        for group_index in range(utterance_groups):
            pcm = speechlike_pcm(group_seconds, base_hz=160.0 + 30 * group_index)
            total_audio += group_seconds
            for chunk in chunk_pcm(pcm):
                await control(
                    {
                        "type": "speak",
                        "event_id": event_id,
                        "audio_b64": base64.b64encode(chunk).decode(),
                    }
                )
            if group_index < utterance_groups - 1:
                await asyncio.sleep(group_gap_s)  # the mid-utterance gap
        await control({"type": "speak_end", "event_id": event_id})
        return total_audio

    phases: list[PhaseStats] = []

    # Phase 0: connect — first frame must arrive at all.
    phase = meter.start_phase("connect")
    phases.append(phase)
    deadline = time.monotonic() + 10
    while meter.total == 0 and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    if meter.total == 0:
        raise RuntimeError("no video frame within 10s of SDP answer")

    await asyncio.sleep(2)

    # Phase 1: first sentence-streamed utterance (the greeting shape).
    phases.append(meter.start_phase("utterance_1"))
    audio_1 = await utterance("sim-evt-1")
    await asyncio.sleep(audio_1 + 2)  # play out + drain

    # Phase 2: between-turns quiet.
    phases.append(meter.start_phase("between_turns"))
    await asyncio.sleep(5)

    # Phase 3: second turn — the exact window where the freeze hit.
    phases.append(meter.start_phase("utterance_2"))
    audio_2 = await utterance("sim-evt-2")
    await asyncio.sleep(audio_2 + 2)

    # Phase 4: long idle tail.
    phases.append(meter.start_phase(f"idle_{int(idle_seconds)}s"))
    await asyncio.sleep(idle_seconds)
    meter.end_phase()

    status, body = _http(
        f"{base}/avatar/sessions/{session_id}", method="GET", token=token
    )
    session_status = json.loads(body) if status == 200 else {"error": status}

    _http(f"{base}/sessions/{session_id}", method="DELETE", token=token)
    for task in consumer_tasks:
        task.cancel()
    await pc.close()

    failures = evaluate_phases(
        [p for p in phases if p.name != "connect"],
        min_fps=min_fps,
        max_gap_s=max_gap_s,
    )
    return {
        "sessionId": session_id,
        "phases": [p.as_dict() for p in phases],
        "totalFramesReceived": meter.total,
        "serverStatus": {
            k: session_status.get(k)
            for k in ("framesRendered", "speakingFrames", "utterances", "state", "stopReason")
        },
        "failures": failures,
        "pass": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="service base URL")
    parser.add_argument("--min-fps", type=float, default=18.0)
    parser.add_argument("--max-gap", type=float, default=1.0)
    parser.add_argument("--idle-seconds", type=float, default=30.0)
    args = parser.parse_args()

    token = os.environ.get("HYDRALISK_AVATAR_BEARER_TOKEN", "").strip()
    if not token:
        print("HYDRALISK_AVATAR_BEARER_TOKEN is required", file=sys.stderr)
        raise SystemExit(2)

    report = asyncio.run(
        run_simulation(
            base=args.base,
            token=token,
            min_fps=args.min_fps,
            max_gap_s=args.max_gap,
            idle_seconds=args.idle_seconds,
        )
    )
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
