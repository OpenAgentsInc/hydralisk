"""Avatar session runtime: state machine + scheduler + renderer + egress.

One `AvatarSession` is one live avatar stream. Control messages (the
LITE-cycle `agent.*` events) mutate the state machine and feed PCM into the
scheduler; the paced render loop asks the scheduler for a `FrameJob` every
frame, renders it, and pushes media into the egress sink. Session receipts
are written per Hydralisk's public-safe posture.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time
from typing import Any

from hydralisk.avatar.config import AvatarSettings
from hydralisk.avatar.egress import EgressSink, NullEgress
from hydralisk.avatar.protocol import ControlMessage, ControlType, state_event
from hydralisk.avatar.receipts import AvatarReceiptWriter, new_session_ref
from hydralisk.avatar.renderer import Renderer
from hydralisk.avatar.scheduler import FramePacer, FrameScheduler
from hydralisk.avatar.state import AvatarStateMachine, Transition


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class AvatarSession:
    def __init__(
        self,
        *,
        session_ref: str,
        settings: AvatarSettings,
        renderer: Renderer,
        receipts: AvatarReceiptWriter,
        egress: EgressSink | None = None,
    ) -> None:
        self.session_ref = session_ref
        self.settings = settings
        self.renderer = renderer
        self.receipts = receipts
        self.egress: EgressSink = egress if egress is not None else NullEgress()

        self.machine = AvatarStateMachine()
        self.scheduler = FrameScheduler(
            fps=settings.fps,
            sample_rate=settings.sample_rate,
            audio_chunks_per_frame=settings.audio_chunks_per_frame,
            crossfade_frames=settings.crossfade_frames,
            jitter_buffer_frames=settings.jitter_buffer_frames,
        )

        self.created_monotonic = time.monotonic()
        self.started_at = _utc_now()
        self.last_control_monotonic = time.monotonic()
        self.stopped = False
        self.stop_reason: str | None = None
        self.summary: dict[str, Any] | None = None

        # Outbound server events for the control WebSocket.
        self.outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        self._stop_event = asyncio.Event()
        self._loop_task: asyncio.Task | None = None

    def _peer_connected(self) -> bool:
        pc = getattr(self.egress, "pc", None)
        if pc is None:
            return False
        # aiortc can hold connectionState at "connecting" while ICE is fully
        # connected/completed — trust either signal. A watched session was
        # evicted mid-conversation because of this (2026-07-09 live failure).
        if getattr(pc, "connectionState", None) == "connected":
            return True
        return getattr(pc, "iceConnectionState", None) in ("connected", "completed")

    @property
    def peer_connected(self) -> bool:
        return self._peer_connected()

    # ---------------------------------------------------------- control

    def handle_control(self, message: ControlMessage) -> list[dict[str, Any]]:
        """Apply one control message; return events to send to the client."""
        self.last_control_monotonic = time.monotonic()
        events: list[dict[str, Any]] = []

        if message.type is ControlType.KEEPALIVE:
            return [{"type": "session.keepalive_ack"}]

        transition: Transition | None = None
        if message.type is ControlType.SPEAK:
            previous_event = self.machine.active_event_id
            transition = self.machine.on_speak(message.event_id or "")
            if message.event_id != previous_event:
                # New utterance replaces any buffered tail of the old one.
                if previous_event is not None:
                    self.scheduler.flush_audio()
                events.append(
                    {
                        "type": "agent.speak_started",
                        "event_id": message.event_id,
                    }
                )
            if message.pcm is not None:
                self.scheduler.put_pcm(message.pcm)
        elif message.type is ControlType.SPEAK_END:
            self.machine.on_speak_end(message.event_id)
            self.scheduler.end_of_utterance()
        elif message.type is ControlType.INTERRUPT:
            transition = self.machine.on_interrupt()
            if transition is not None:
                self.scheduler.flush_audio()
                self.receipts.session_event(
                    self.session_ref,
                    "interrupt",
                    {"interrupts": self.machine.interrupts},
                )
        elif message.type is ControlType.START_LISTENING:
            transition = self.machine.on_start_listening()
        elif message.type is ControlType.STOP_LISTENING:
            transition = self.machine.on_stop_listening()

        if transition is not None:
            events.append(
                state_event(self.machine.state.value, self.session_ref)
            )
        return events

    # ------------------------------------------------------------ render

    def tick(self) -> None:
        """Render exactly one frame and push it to egress."""
        job = self.scheduler.next_frame(self.machine.state)
        frame = self.renderer.render(job)
        self.egress.push_video(frame)
        for chunk in job.audio_chunks:
            self.egress.push_audio(chunk)

        if self.machine.end_pending and self.scheduler.audio_drained:
            ended_event_id = self.machine.active_event_id
            transition = self.machine.on_audio_drained()
            if transition is not None:
                self._emit(
                    {
                        "type": "agent.speak_ended",
                        "event_id": ended_event_id,
                    }
                )
                self._emit(
                    state_event(self.machine.state.value, self.session_ref)
                )

    def _emit(self, event: dict[str, Any]) -> None:
        try:
            self.outbox.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover — unbounded queue
            pass

    async def render_loop(self) -> None:
        pacer = FramePacer(self.settings.fps, time.monotonic())
        frame_number = 0
        timeout = self.settings.keepalive_timeout_seconds
        # Renderer construction/warmup can consume most of a keepalive
        # window before the first frame; the client's clock starts when the
        # mint returns, so ours starts when the loop does.
        self.last_control_monotonic = time.monotonic()
        try:
            while not self._stop_event.is_set():
                self.tick()
                frame_number += 1
                # A connected WebRTC peer is client liveness: a viewer who
                # only watches sends no control traffic and must not be
                # reaped mid-session (SQ-4 #8621). The timeout clock runs
                # only while no peer is connected.
                if self._peer_connected():
                    self.last_control_monotonic = time.monotonic()
                if (
                    timeout > 0
                    and time.monotonic() - self.last_control_monotonic > timeout
                ):
                    await self.stop("keepalive_timeout")
                    return
                await asyncio.sleep(
                    pacer.wait_seconds(frame_number, time.monotonic())
                )
        except asyncio.CancelledError:  # pragma: no cover — shutdown path
            raise

    def start(self) -> None:
        self.renderer.start()
        self.receipts.session_started(
            self.session_ref,
            renderer=self.renderer.backend,
            meta={
                "fps": self.settings.fps,
                "width": self.settings.width,
                "height": self.settings.height,
                "sampleRate": self.settings.sample_rate,
            },
        )
        self._loop_task = asyncio.create_task(self.render_loop())

    async def stop(self, reason: str) -> dict[str, Any]:
        if self.stopped:
            return self.summary or {}
        self.stopped = True
        self.stop_reason = reason
        self._stop_event.set()
        if self._loop_task is not None and self._loop_task is not asyncio.current_task():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await self.egress.close()
        self.renderer.close()
        seconds = time.monotonic() - self.created_monotonic
        self.summary = self.receipts.session_summary(
            self.session_ref,
            started_at=self.started_at,
            seconds=seconds,
            frames_rendered=self.scheduler.frames_rendered,
            speaking_frames=self.scheduler.speaking_frames,
            interrupts=self.machine.interrupts,
            utterances=self.machine.utterances,
            renderer=self.renderer.backend,
            stop_reason=reason,
            gpu={
                "name": self.settings.gpu_name,
                "class": self.settings.gpu_class,
                "count": self.settings.gpu_count,
            },
        )
        self._emit({"type": "session.stopped", "reason": reason})
        return self.summary

    def status(self) -> dict[str, Any]:
        return {
            "sessionRef": self.session_ref,
            "state": self.machine.state.value,
            "startedAt": self.started_at,
            "uptimeSeconds": round(time.monotonic() - self.created_monotonic, 3),
            "framesRendered": self.scheduler.frames_rendered,
            "speakingFrames": self.scheduler.speaking_frames,
            "interrupts": self.machine.interrupts,
            "utterances": self.machine.utterances,
            "renderer": self.renderer.backend,
            "stopped": self.stopped,
            "stopReason": self.stop_reason,
        }


class SessionLimitError(RuntimeError):
    pass


class SessionManager:
    def __init__(
        self,
        settings: AvatarSettings,
        *,
        receipts: AvatarReceiptWriter | None = None,
    ) -> None:
        self.settings = settings
        self.receipts = receipts or AvatarReceiptWriter(settings.receipt_dir)
        self.sessions: dict[str, AvatarSession] = {}

    @property
    def active_sessions(self) -> list[AvatarSession]:
        return [s for s in self.sessions.values() if not s.stopped]

    def create(
        self,
        *,
        renderer: Renderer,
        egress: EgressSink | None = None,
    ) -> AvatarSession:
        if len(self.active_sessions) >= self.settings.max_sessions:
            raise SessionLimitError(
                f"session limit reached ({self.settings.max_sessions})"
            )
        session = AvatarSession(
            session_ref=new_session_ref(),
            settings=self.settings,
            renderer=renderer,
            receipts=self.receipts,
            egress=egress,
        )
        self.sessions[session.session_ref] = session
        session.start()
        return session

    def get(self, session_ref: str) -> AvatarSession | None:
        return self.sessions.get(session_ref)

    async def evict_one_stale(self) -> str | None:
        """Free a slot by stopping the oldest active session with no
        connected WebRTC peer (abandoned mint, closed tab before connect).
        Never evicts a session a viewer is actually watching (SQ-4 #8621:
        a wedged single slot must not block the next real visitor)."""
        if len(self.active_sessions) < self.settings.max_sessions:
            return None
        for session in sorted(
            self.active_sessions, key=lambda item: item.created_monotonic
        ):
            if not session.peer_connected:
                await session.stop("evicted_stale_no_peer")
                return session.session_ref
        return None

    async def stop(self, session_ref: str, reason: str) -> dict[str, Any] | None:
        session = self.sessions.get(session_ref)
        if session is None:
            return None
        return await session.stop(reason)

    async def stop_all(self, reason: str) -> None:
        for session in list(self.sessions.values()):
            await session.stop(reason)
