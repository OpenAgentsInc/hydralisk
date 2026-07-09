from __future__ import annotations

import numpy as np

from hydralisk.avatar.clips import ClipFrameRef
from hydralisk.avatar.config import AvatarSettings
from hydralisk.avatar.renderer import CpuNoopRenderer, select_renderer
from hydralisk.avatar.scheduler import FrameJob
from hydralisk.avatar.state import AvatarState


def _job(**overrides) -> FrameJob:
    params = dict(
        frame_number=0,
        state=AvatarState.IDLE,
        clip=ClipFrameRef(clip_index=6, frame_index=0),
        speaking=False,
        audio_chunks=(np.zeros(500, dtype=np.int16),) * 2,
        crossfade_from=None,
        crossfade_alpha=1.0,
    )
    params.update(overrides)
    return FrameJob(**params)


def test_cpu_renderer_produces_bgr_frames() -> None:
    renderer = CpuNoopRenderer(width=160, height=90)
    renderer.start()
    frame = renderer.render(_job())
    assert frame.shape == (90, 160, 3)
    assert frame.dtype == np.uint8
    renderer.close()


def test_cpu_renderer_frames_differ_by_state_and_position() -> None:
    renderer = CpuNoopRenderer(width=160, height=90)
    idle = renderer.render(_job())
    speaking = renderer.render(
        _job(
            state=AvatarState.SPEAKING,
            clip=ClipFrameRef(clip_index=8, frame_index=0),
            speaking=True,
        )
    )
    assert not np.array_equal(idle, speaking)

    frame_a = renderer.render(_job(clip=ClipFrameRef(6, 10)))
    frame_b = renderer.render(_job(clip=ClipFrameRef(6, 11)))
    assert not np.array_equal(frame_a, frame_b)


def test_cpu_renderer_crossfade_blends() -> None:
    renderer = CpuNoopRenderer(width=160, height=90)
    pure = renderer.render(_job(clip=ClipFrameRef(4, 0)))
    blended = renderer.render(
        _job(
            clip=ClipFrameRef(4, 0),
            crossfade_from=ClipFrameRef(6, 50),
            crossfade_alpha=0.5,
        )
    )
    assert not np.array_equal(pure, blended)


def test_select_renderer_cpu_explicit() -> None:
    settings = AvatarSettings(renderer_backend="cpu", width=64, height=36)
    renderer, blockers = select_renderer(settings)
    assert renderer.backend == "cpu-noop"
    assert blockers == []


def test_select_renderer_auto_falls_back_with_blockers() -> None:
    # On a machine without GPU/weights, auto must fall back to the CPU
    # backend and explain why MuseTalk is inactive.
    settings = AvatarSettings(renderer_backend="auto", width=64, height=36)
    renderer, blockers = select_renderer(settings)
    assert renderer.backend == "cpu-noop"
    assert blockers, "expected MuseTalk blockers on a CPU-only machine"
    codes = {blocker["code"] for blocker in blockers}
    # The unset config paths are always reported.
    assert "musetalk_repo_unset" in codes
    assert "avatar_data_unset" in codes
