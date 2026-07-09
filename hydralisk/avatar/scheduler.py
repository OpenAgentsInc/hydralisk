"""Frame scheduler: turns state + buffered TTS audio into per-frame jobs.

Ported design from LiveTalking's `BaseAvatar.inference`/`process_frames`
loops (Apache-2.0, https://github.com/lipku/livetalking):

- video runs at a fixed fps; every video frame carries exactly
  `audio_chunks_per_frame` PCM chunks (silence chunks when nobody speaks);
- silent frames are raw clip passthrough (no inference); speaking frames
  are re-lipped by the renderer backend against the consumed audio;
- clip cycles loop with `mirror_index` so loops ping-pong instead of
  jump-cutting;
- state changes crossfade for a few frames (LiveTalking's transition blend).

The scheduler itself is clock-free — `FramePacer` computes wall deadlines —
so all of the timing logic unit-tests deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hydralisk.avatar.audio import PcmChunkAssembler
from hydralisk.avatar.clips import ClipCatalog, ClipFrameRef
from hydralisk.avatar.state import AvatarState


@dataclass(frozen=True)
class FrameJob:
    """Everything a renderer backend needs to produce one output frame."""

    frame_number: int
    state: AvatarState
    clip: ClipFrameRef
    speaking: bool
    # Exactly audio_chunks_per_frame int16 chunks (silence when quiet).
    audio_chunks: tuple[np.ndarray, ...]
    # When crossfading out of a previous state: the frame to blend from
    # and the blend weight of the NEW state's frame (0 → all old, 1 → new).
    crossfade_from: ClipFrameRef | None = None
    crossfade_alpha: float = 1.0


class FramePacer:
    """Wall-clock deadlines for a fixed-fps render loop."""

    def __init__(self, fps: int, start_time: float) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.period = 1.0 / fps
        self.start_time = start_time

    def deadline(self, frame_number: int) -> float:
        return self.start_time + frame_number * self.period

    def wait_seconds(self, frame_number: int, now: float) -> float:
        return max(0.0, self.deadline(frame_number) - now)


class FrameScheduler:
    def __init__(
        self,
        *,
        catalog: ClipCatalog | None = None,
        fps: int = 24,
        sample_rate: int = 24000,
        audio_chunks_per_frame: int = 2,
        crossfade_frames: int = 6,
        jitter_buffer_frames: int = 5,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if audio_chunks_per_frame <= 0:
            raise ValueError("audio_chunks_per_frame must be positive")
        self.catalog = catalog or ClipCatalog()
        self.fps = fps
        self.sample_rate = sample_rate
        self.audio_chunks_per_frame = audio_chunks_per_frame
        self.chunk_samples = sample_rate // (fps * audio_chunks_per_frame)
        self.crossfade_frames = max(0, crossfade_frames)
        self.jitter_buffer_chunks = (
            max(0, jitter_buffer_frames) * audio_chunks_per_frame
        )

        self.assembler = PcmChunkAssembler(self.chunk_samples)
        self.frame_number = 0
        self.frames_rendered = 0
        self.speaking_frames = 0

        self._active_state = AvatarState.IDLE
        self._cursor = 0
        self._crossfade_remaining = 0
        self._crossfade_state: AvatarState | None = None
        self._crossfade_cursor = 0
        # The jitter gate holds lips still until ~200 ms of audio is
        # buffered so lips never lead the sound; once open it stays open
        # until the utterance drains or is flushed.
        self._jitter_gate_open = False

    # ------------------------------------------------------------- audio

    def put_pcm(self, samples: np.ndarray) -> None:
        self.assembler.push(samples)

    def end_of_utterance(self) -> None:
        """speak_end arrived: pad the partial tail and drain everything."""
        self.assembler.flush_tail()
        self._jitter_gate_open = True

    def flush_audio(self) -> None:
        """interrupt: drop all buffered speech audio immediately."""
        self.assembler.clear()
        self._jitter_gate_open = False

    @property
    def audio_drained(self) -> bool:
        return self.assembler.empty

    # ------------------------------------------------------------- video

    def _observe_state(self, state: AvatarState) -> None:
        if state == self._active_state:
            return
        if self.crossfade_frames > 0:
            self._crossfade_state = self._active_state
            self._crossfade_cursor = self._cursor
            self._crossfade_remaining = self.crossfade_frames
        self._active_state = state
        self._cursor = 0
        if state is not AvatarState.SPEAKING:
            self._jitter_gate_open = False

    def _silence_chunk(self) -> np.ndarray:
        return np.zeros(self.chunk_samples, dtype=np.int16)

    def _take_audio(self, state: AvatarState) -> tuple[list[np.ndarray], bool]:
        if state is not AvatarState.SPEAKING:
            return (
                [self._silence_chunk() for _ in range(self.audio_chunks_per_frame)],
                False,
            )
        if not self._jitter_gate_open:
            if self.assembler.buffered_chunks >= max(1, self.jitter_buffer_chunks):
                self._jitter_gate_open = True
            else:
                return (
                    [
                        self._silence_chunk()
                        for _ in range(self.audio_chunks_per_frame)
                    ],
                    False,
                )
        chunks: list[np.ndarray] = []
        consumed_any = False
        for _ in range(self.audio_chunks_per_frame):
            chunk = self.assembler.pop()
            if chunk is None:
                chunks.append(self._silence_chunk())
            else:
                chunks.append(chunk)
                consumed_any = True
        if self.assembler.empty and not consumed_any:
            self._jitter_gate_open = False
        return chunks, consumed_any

    def next_frame(self, state: AvatarState) -> FrameJob:
        self._observe_state(state)

        chunks, speaking = self._take_audio(state)

        clip = self.catalog.frame_for(self._active_state, self._cursor)

        crossfade_from: ClipFrameRef | None = None
        alpha = 1.0
        if self._crossfade_remaining > 0 and self._crossfade_state is not None:
            step = self.crossfade_frames - self._crossfade_remaining + 1
            alpha = step / (self.crossfade_frames + 1)
            crossfade_from = self.catalog.frame_for(
                self._crossfade_state, self._crossfade_cursor
            )
            self._crossfade_cursor += 1
            self._crossfade_remaining -= 1
            if self._crossfade_remaining == 0:
                self._crossfade_state = None

        job = FrameJob(
            frame_number=self.frame_number,
            state=self._active_state,
            clip=clip,
            speaking=speaking,
            audio_chunks=tuple(chunks),
            crossfade_from=crossfade_from,
            crossfade_alpha=alpha,
        )

        self.frame_number += 1
        self.frames_rendered += 1
        if speaking:
            self.speaking_frames += 1
        self._cursor += 1
        return job
