"""Unit coverage for the real-session simulator's pure logic."""

from __future__ import annotations

from hydralisk.avatar.simulator import (
    FrameMeter,
    PhaseStats,
    chunk_pcm,
    evaluate_phases,
    speechlike_pcm,
)


def test_chunk_pcm_matches_apps_sarah_shape() -> None:
    # 3.0 s of s16le mono 24 kHz = 144,000 bytes.
    pcm = b"\x00\x01" * (24_000 * 3)
    chunks = chunk_pcm(pcm)
    # 600 ms first chunk, then 1 s chunks: 0.6 + 1 + 1 + 0.4 tail.
    assert len(chunks) == 4
    assert len(chunks[0]) == 24_000 * 600 // 1000 * 2
    assert len(chunks[1]) == 24_000 * 2
    assert len(chunks[2]) == 24_000 * 2
    assert sum(len(c) for c in chunks) == len(pcm)


def test_chunk_pcm_empty() -> None:
    assert chunk_pcm(b"") == []


def test_speechlike_pcm_size_and_range() -> None:
    pcm = speechlike_pcm(0.5)
    assert len(pcm) == 24_000 * 2 // 2  # 0.5 s of 2-byte samples
    sample = int.from_bytes(pcm[:2], "little", signed=True)
    assert -32768 <= sample <= 32767


def test_evaluate_phases_flags_low_fps_and_stall_gap() -> None:
    good = PhaseStats(name="ok", started=0.0, ended=10.0, frames=240, max_gap_s=0.1)
    slow = PhaseStats(name="slow", started=0.0, ended=10.0, frames=100, max_gap_s=0.1)
    stalled = PhaseStats(
        name="stalled", started=0.0, ended=10.0, frames=240, max_gap_s=3.4
    )
    failures = evaluate_phases(
        [good, slow, stalled], min_fps=18.0, max_gap_s=1.0
    )
    assert len(failures) == 2
    assert any("slow" in f for f in failures)
    assert any("stalled" in f and "3.40s" in f for f in failures)


def test_frame_meter_phases_and_gaps(monkeypatch) -> None:
    times = iter([0.0, 0.0, 0.1, 0.2, 1.9, 2.0, 2.0])
    import hydralisk.avatar.simulator as sim

    monkeypatch.setattr(sim.time, "monotonic", lambda: next(times))
    meter = FrameMeter()
    phase = meter.start_phase("p1")  # t=0.0
    meter.on_frame()  # t=0.0
    meter.on_frame()  # t=0.1
    meter.on_frame()  # t=0.2
    meter.on_frame()  # t=1.9 → 1.7s gap
    meter.end_phase()  # t=2.0
    assert phase.frames == 4
    assert abs(phase.max_gap_s - 1.7) < 1e-9
    assert abs(phase.seconds - 2.0) < 1e-9
