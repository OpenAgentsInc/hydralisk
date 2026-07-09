"""Egress sinks: WebRTC (aiortc) when available, a null sink otherwise.

The WebRTC path ports LiveTalking's `PlayerStreamTrack`/`HumanPlayer`
pattern (Apache-2.0, https://github.com/lipku/livetalking,
`server/webrtc.py`): the render loop pushes BGR frames and int16 PCM
chunks into bounded queues; aiortc pulls them out of `recv()` with
monotonic pts pacing so video stays at fps and audio at the chunk cadence.

aiortc/av are optional (`uv sync --extra avatar`); everything here imports
without them and `webrtc_available()` gates the offer endpoint.
"""

from __future__ import annotations

import asyncio
import importlib.util
from typing import Any, Protocol

import numpy as np


def webrtc_available() -> bool:
    try:
        return (
            importlib.util.find_spec("aiortc") is not None
            and importlib.util.find_spec("av") is not None
        )
    except (ImportError, ValueError):
        return False


class EgressSink(Protocol):
    def push_video(self, frame: np.ndarray) -> None: ...

    def push_audio(self, chunk: np.ndarray) -> None: ...

    async def close(self) -> None: ...


class NullEgress:
    """Counts pushed media; used for CPU tests and headless smokes."""

    def __init__(self) -> None:
        self.video_frames = 0
        self.audio_chunks = 0

    def push_video(self, frame: np.ndarray) -> None:
        self.video_frames += 1

    def push_audio(self, chunk: np.ndarray) -> None:
        self.audio_chunks += 1

    async def close(self) -> None:
        return None


_TRACK_CLASS: type | None = None


def _track_class() -> type:
    """Build the aiortc track subclass lazily (aiortc is optional)."""
    global _TRACK_CLASS
    if _TRACK_CLASS is not None:
        return _TRACK_CLASS

    import fractions
    import time

    from aiortc import MediaStreamTrack  # noqa: PLC0415

    class QueueMediaTrack(MediaStreamTrack):
        """LiveTalking PlayerStreamTrack pacing, queue-fed (Apache-2.0)."""

        def __init__(self, *, kind: str, fps: int, sample_rate: int) -> None:
            super().__init__()
            self.kind = kind
            self._fps = fps
            self._sample_rate = sample_rate
            self._queue: asyncio.Queue = asyncio.Queue(maxsize=100)
            self._start: float | None = None
            self._timestamp = 0
            self._count = 0

        def push(self, frame: Any) -> None:
            try:
                self._queue.put_nowait(frame)
            except asyncio.QueueFull:
                # Drop oldest to stay realtime; never block the render loop.
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    self._queue.put_nowait(frame)
                except asyncio.QueueFull:
                    pass

        async def recv(self) -> Any:
            frame = await self._queue.get()

            if self.kind == "video":
                ptime = 1.0 / self._fps
                clock_rate = 90000
            else:
                samples = getattr(frame, "samples", 0) or 1
                ptime = samples / self._sample_rate
                clock_rate = self._sample_rate
            time_base = fractions.Fraction(1, clock_rate)

            if self._start is None:
                self._start = time.time()
            else:
                self._timestamp += int(ptime * clock_rate)
                self._count += 1
                wait = self._start + self._count * ptime - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)

            frame.pts = self._timestamp
            frame.time_base = time_base
            return frame

    _TRACK_CLASS = QueueMediaTrack
    return QueueMediaTrack


class WebRTCEgress:
    """One peer connection carrying the composited video + TTS audio."""

    def __init__(self, *, fps: int, sample_rate: int) -> None:
        if not webrtc_available():
            raise RuntimeError(
                "aiortc/av are not installed; install the 'avatar' extra"
            )
        from aiortc import (  # noqa: PLC0415
            RTCConfiguration,
            RTCIceServer,
            RTCPeerConnection,
        )

        track_cls = _track_class()
        self.fps = fps
        self.sample_rate = sample_rate
        # A public STUN server so the host-side peer discovers its NAT 1:1
        # public address (GCE NICs only carry the internal IP; without srflx
        # candidates arbitrary browsers cannot reach the media path).
        self.pc = RTCPeerConnection(
            configuration=RTCConfiguration(
                iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])]
            )
        )
        self.video_track = track_cls(kind="video", fps=fps, sample_rate=sample_rate)
        self.audio_track = track_cls(kind="audio", fps=fps, sample_rate=sample_rate)
        self.pc.addTrack(self.video_track)
        self.pc.addTrack(self.audio_track)

    def push_video(self, frame: np.ndarray) -> None:
        from av import VideoFrame  # noqa: PLC0415

        self.video_track.push(VideoFrame.from_ndarray(frame, format="bgr24"))

    def push_audio(self, chunk: np.ndarray) -> None:
        from av import AudioFrame  # noqa: PLC0415

        chunk = np.ascontiguousarray(chunk, dtype=np.int16)
        frame = AudioFrame(format="s16", layout="mono", samples=chunk.shape[0])
        frame.planes[0].update(chunk.tobytes())
        frame.sample_rate = self.sample_rate
        self.audio_track.push(frame)

    async def handle_offer(self, sdp: str, offer_type: str) -> dict[str, str]:
        from aiortc import RTCSessionDescription  # noqa: PLC0415

        offer = RTCSessionDescription(sdp=sdp, type=offer_type)
        await self.pc.setRemoteDescription(offer)
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        local = self.pc.localDescription
        return {"sdp": local.sdp, "type": local.type}

    async def close(self) -> None:
        self.video_track.stop()
        self.audio_track.stop()
        await self.pc.close()
