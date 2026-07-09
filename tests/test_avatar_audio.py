from __future__ import annotations

import numpy as np
import pytest

from hydralisk.avatar.audio import (
    PcmChunkAssembler,
    pcm_int16_to_float32,
    resample_pcm,
)


def test_assembler_reslices_arbitrary_writes_into_fixed_chunks() -> None:
    assembler = PcmChunkAssembler(chunk_samples=500)
    assembler.push(np.arange(300, dtype=np.int16))
    assert assembler.pop() is None
    assembler.push(np.arange(300, 900, dtype=np.int16))

    first = assembler.pop()
    assert first is not None
    assert first.shape == (500,)
    assert first[0] == 0
    assert first[499] == 499

    # 900 samples written, 500 popped → 400 pending toward the next chunk.
    assert assembler.pop() is None
    assert assembler.buffered_samples == 400


def test_assembler_flush_tail_zero_pads() -> None:
    assembler = PcmChunkAssembler(chunk_samples=500)
    assembler.push(np.ones(120, dtype=np.int16))
    assembler.flush_tail()
    chunk = assembler.pop()
    assert chunk is not None
    assert chunk.shape == (500,)
    assert chunk[:120].tolist() == [1] * 120
    assert chunk[120:].tolist() == [0] * 380
    assert assembler.empty


def test_assembler_clear_drops_everything() -> None:
    assembler = PcmChunkAssembler(chunk_samples=10)
    assembler.push(np.ones(25, dtype=np.int16))
    assert assembler.buffered_chunks == 2
    assembler.clear()
    assert assembler.empty
    assert assembler.pop() is None


def test_assembler_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        PcmChunkAssembler(chunk_samples=0)
    assembler = PcmChunkAssembler(chunk_samples=10)
    with pytest.raises(ValueError):
        assembler.push(np.zeros((2, 5), dtype=np.int16))


def test_resample_24k_to_16k_length() -> None:
    # One 24 kHz frame chunk (1000 samples) → 16 kHz for whisper features.
    src = np.arange(1000, dtype=np.int16)
    out = resample_pcm(src, 24000, 16000)
    assert out.dtype == np.int16
    assert abs(out.size - 667) <= 1
    # Monotone ramp stays monotone under linear resampling.
    assert np.all(np.diff(out.astype(np.int32)) >= 0)


def test_resample_identity_and_empty() -> None:
    src = np.array([1, 2, 3], dtype=np.int16)
    same = resample_pcm(src, 24000, 24000)
    assert np.array_equal(same, src)
    empty = resample_pcm(np.zeros(0, dtype=np.int16), 24000, 16000)
    assert empty.size == 0


def test_pcm_int16_to_float32_range() -> None:
    pcm = np.array([-32768, 0, 32767], dtype=np.int16)
    floats = pcm_int16_to_float32(pcm)
    assert floats.dtype == np.float32
    assert floats[0] == pytest.approx(-1.0)
    assert floats[1] == pytest.approx(0.0)
    assert floats[2] == pytest.approx(32767 / 32768)
