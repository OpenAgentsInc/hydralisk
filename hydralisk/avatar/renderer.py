"""Renderer backends behind one seam.

`CpuNoopRenderer` runs anywhere with no GPU, no OpenCV, and no footage —
it produces deterministic synthetic BGR frames tagged by state so the
session loop, egress, and receipts can be exercised in CI. The MuseTalk
backend (see `musetalk_backend.py`) activates only when CUDA, weights, and
preprocessed avatar references are actually present; `select_renderer`
fails toward the CPU backend and reports the blockers instead of crashing.

Frames are HxWx3 uint8 BGR (OpenCV/LiveTalking convention).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from hydralisk.avatar.config import AvatarSettings
from hydralisk.avatar.scheduler import FrameJob
from hydralisk.avatar.state import AvatarState


class Renderer(Protocol):
    backend: str

    def start(self) -> None: ...

    def render(self, job: FrameJob) -> np.ndarray: ...

    def close(self) -> None: ...


# Distinct base colors per state (BGR) so CPU-rendered streams are visually
# debuggable without any footage.
_STATE_COLORS: dict[AvatarState, tuple[int, int, int]] = {
    AvatarState.IDLE: (96, 64, 32),
    AvatarState.LISTENING: (32, 96, 64),
    AvatarState.SPEAKING: (32, 48, 128),
}


class CpuNoopRenderer:
    """GPU-free renderer: synthetic frames, correct geometry and blending."""

    backend = "cpu-noop"

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("frame geometry must be positive")
        self.width = width
        self.height = height
        self._started = False

    def start(self) -> None:
        self._started = True

    def _base_frame(self, state: AvatarState, clip_index: int, frame_index: int) -> np.ndarray:
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frame[:] = _STATE_COLORS[state]
        # Encode clip/frame position as a moving scanline so tests and eye
        # checks can verify the scheduler is actually advancing.
        row = frame_index % self.height
        frame[row, :, :] = 255
        col = (clip_index * 7) % self.width
        frame[:, col, :] = 200
        return frame

    def render(self, job: FrameJob) -> np.ndarray:
        frame = self._base_frame(job.state, job.clip.clip_index, job.clip.frame_index)
        if job.speaking:
            # Mark the "mouth region" band the inpainting backend would own.
            band_top = int(self.height * 0.7)
            frame[band_top:, :, 2] = 255
        if job.crossfade_from is not None and job.crossfade_alpha < 1.0:
            prev = self._base_frame(
                # The previous state is not carried on the job; blending the
                # previous clip's pixels is what matters for the seam.
                job.state,
                job.crossfade_from.clip_index,
                job.crossfade_from.frame_index,
            )
            alpha = float(job.crossfade_alpha)
            frame = (
                frame.astype(np.float32) * alpha
                + prev.astype(np.float32) * (1.0 - alpha)
            ).astype(np.uint8)
        return frame

    def close(self) -> None:
        self._started = False


def select_renderer(
    settings: AvatarSettings,
) -> tuple[Renderer, list[dict[str, str]]]:
    """Pick the renderer backend, fail-closed toward the CPU no-op.

    Returns (renderer, blockers) where blockers explain why the MuseTalk
    backend is not active (empty when it is, or when CPU was explicitly
    requested).
    """
    backend = settings.renderer_backend
    if backend == "cpu":
        return CpuNoopRenderer(settings.width, settings.height), []

    from hydralisk.avatar.musetalk_backend import (
        MuseTalkRenderer,
        musetalk_blockers,
    )

    blockers = musetalk_blockers(settings)
    if backend == "musetalk":
        if blockers:
            raise RuntimeError(
                "musetalk renderer requested but unavailable: "
                + "; ".join(b["message"] for b in blockers)
            )
        return MuseTalkRenderer(settings), []

    # auto
    if blockers:
        return CpuNoopRenderer(settings.width, settings.height), blockers
    return MuseTalkRenderer(settings), []
