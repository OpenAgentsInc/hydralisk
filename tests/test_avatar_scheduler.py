from __future__ import annotations

import numpy as np
import pytest

from hydralisk.avatar.clips import (
    FRAMES_PER_CLIP,
    STATE_CLIP_CYCLE,
    ClipCatalog,
    mirror_index,
)
from hydralisk.avatar.scheduler import FramePacer, FrameScheduler
from hydralisk.avatar.state import AvatarState


def _scheduler(**overrides) -> FrameScheduler:
    params = dict(
        fps=24,
        sample_rate=24000,
        audio_chunks_per_frame=2,
        crossfade_frames=6,
        jitter_buffer_frames=2,
    )
    params.update(overrides)
    return FrameScheduler(**params)


def test_mirror_index_ping_pongs() -> None:
    assert [mirror_index(3, i) for i in range(8)] == [0, 1, 2, 2, 1, 0, 0, 1]


def test_catalog_routes_states_to_catalogued_clips() -> None:
    catalog = ClipCatalog()
    assert STATE_CLIP_CYCLE[AvatarState.IDLE] == (6, 3)
    assert STATE_CLIP_CYCLE[AvatarState.LISTENING] == (4, 5)
    assert STATE_CLIP_CYCLE[AvatarState.SPEAKING] == (8, 0)

    first = catalog.frame_for(AvatarState.IDLE, 0)
    assert first.clip_index == 6
    assert first.frame_index == 0

    # Cursor walks into the second idle clip after one full clip.
    later = catalog.frame_for(AvatarState.IDLE, FRAMES_PER_CLIP)
    assert later.clip_index == 3

    # Past the end of the cycle it mirrors back instead of jump-cutting.
    cycle = catalog.cycle_length(AvatarState.IDLE)
    mirrored = catalog.frame_for(AvatarState.IDLE, cycle)
    assert mirrored.clip_index == 3
    assert mirrored.frame_index == FRAMES_PER_CLIP - 1


def test_chunk_samples_match_livetalking_convention() -> None:
    scheduler = _scheduler()
    # sample_rate // (fps * 2): two ~20.8 ms chunks per 24 fps frame.
    assert scheduler.chunk_samples == 500


def test_idle_frames_are_silent_passthrough() -> None:
    scheduler = _scheduler()
    job = scheduler.next_frame(AvatarState.IDLE)
    assert job.state is AvatarState.IDLE
    assert not job.speaking
    assert len(job.audio_chunks) == 2
    assert all(not chunk.any() for chunk in job.audio_chunks)
    assert job.clip.clip_index == 6


def test_speaking_waits_for_jitter_buffer_then_consumes_audio() -> None:
    scheduler = _scheduler(jitter_buffer_frames=2)  # gate: 4 chunks
    scheduler.put_pcm(np.ones(500, dtype=np.int16))  # 1 chunk buffered

    job = scheduler.next_frame(AvatarState.SPEAKING)
    assert not job.speaking  # lips must not lead the sound

    scheduler.put_pcm(np.ones(1500, dtype=np.int16))  # now 4 chunks
    job = scheduler.next_frame(AvatarState.SPEAKING)
    assert job.speaking
    assert job.audio_chunks[0].any()
    assert job.clip.clip_index == 8  # strongest talking clip


def test_speak_end_drains_partial_tail_without_jitter_wait() -> None:
    scheduler = _scheduler(jitter_buffer_frames=5)
    scheduler.put_pcm(np.ones(700, dtype=np.int16))  # 1 chunk + 200 tail
    scheduler.end_of_utterance()

    job = scheduler.next_frame(AvatarState.SPEAKING)
    assert job.speaking
    assert not scheduler.audio_drained or scheduler.assembler.empty

    # Both chunks (full + padded tail) fit in one frame.
    assert scheduler.audio_drained


def test_flush_audio_supports_interrupt() -> None:
    scheduler = _scheduler(jitter_buffer_frames=0)
    scheduler.put_pcm(np.ones(5000, dtype=np.int16))
    assert not scheduler.audio_drained
    scheduler.flush_audio()
    assert scheduler.audio_drained
    job = scheduler.next_frame(AvatarState.LISTENING)
    assert not job.speaking


def test_state_change_starts_crossfade_and_resets_cursor() -> None:
    scheduler = _scheduler(crossfade_frames=3)
    for _ in range(10):
        scheduler.next_frame(AvatarState.IDLE)

    job = scheduler.next_frame(AvatarState.LISTENING)
    assert job.state is AvatarState.LISTENING
    assert job.clip.clip_index == 4  # listening cycle starts at clip 4
    assert job.clip.frame_index == 0  # cursor reset
    assert job.crossfade_from is not None
    assert job.crossfade_from.clip_index == 6  # blends from the idle clip
    assert 0.0 < job.crossfade_alpha < 1.0

    alphas = [job.crossfade_alpha]
    for _ in range(3):
        job = scheduler.next_frame(AvatarState.LISTENING)
        alphas.append(job.crossfade_alpha)
    # Alpha ramps up and the fade ends.
    assert alphas[0] < alphas[1] < alphas[2]
    assert job.crossfade_from is None
    assert job.crossfade_alpha == 1.0


def test_frame_counters_track_speaking_frames() -> None:
    scheduler = _scheduler(jitter_buffer_frames=0)
    scheduler.put_pcm(np.ones(1000, dtype=np.int16))
    scheduler.next_frame(AvatarState.SPEAKING)
    scheduler.next_frame(AvatarState.SPEAKING)  # drained → silent
    assert scheduler.frames_rendered == 2
    assert scheduler.speaking_frames == 1


def test_pacer_deadlines_are_fps_spaced() -> None:
    pacer = FramePacer(fps=24, start_time=100.0)
    assert pacer.deadline(0) == pytest.approx(100.0)
    assert pacer.deadline(24) == pytest.approx(101.0)
    assert pacer.wait_seconds(1, now=100.0) == pytest.approx(1 / 24)
    # Late frames never produce negative sleeps.
    assert pacer.wait_seconds(1, now=200.0) == 0.0


def test_audio_video_pairing_is_exact() -> None:
    scheduler = _scheduler(jitter_buffer_frames=0)
    # 3 frames worth of audio: 6 chunks * 500 samples.
    scheduler.put_pcm(np.ones(3000, dtype=np.int16))
    consumed = 0
    for _ in range(3):
        job = scheduler.next_frame(AvatarState.SPEAKING)
        assert len(job.audio_chunks) == 2
        consumed += sum(chunk.any() for chunk in job.audio_chunks)
    assert consumed == 6
    assert scheduler.audio_drained
