"""PCM assembly and resampling for the avatar audio path.

Incoming `agent.speak` chunks are arbitrary-length 16-bit mono PCM at the
control sample rate (24 kHz). The scheduler consumes fixed-size chunks —
two per video frame, LiveTalking-style — so this module turns the incoming
stream into exactly-sized chunks and provides the 24 kHz → 16 kHz resample
used by the MuseTalk whisper-feature path.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def resample_pcm(
    samples: np.ndarray, src_rate: int, dst_rate: int
) -> np.ndarray:
    """Linear-interpolation resample of int16 mono PCM.

    Good enough for lip-sync feature extraction (the egress audio track
    keeps the original rate and never passes through here).
    """
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rates must be positive")
    samples = np.asarray(samples, dtype=np.int16)
    if src_rate == dst_rate or samples.size == 0:
        return samples.copy()
    dst_size = int(round(samples.size * dst_rate / src_rate))
    if dst_size <= 0:
        return np.zeros(0, dtype=np.int16)
    src_positions = np.arange(samples.size, dtype=np.float64)
    dst_positions = np.linspace(
        0.0, float(samples.size - 1), num=dst_size, dtype=np.float64
    )
    resampled = np.interp(dst_positions, src_positions, samples.astype(np.float64))
    return np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)


def pcm_int16_to_float32(samples: np.ndarray) -> np.ndarray:
    return np.asarray(samples, dtype=np.int16).astype(np.float32) / 32768.0


class PcmChunkAssembler:
    """Accumulates arbitrary-length PCM writes into fixed-size chunks."""

    def __init__(self, chunk_samples: int) -> None:
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        self.chunk_samples = chunk_samples
        self._chunks: deque[np.ndarray] = deque()
        self._pending = np.zeros(0, dtype=np.int16)

    def push(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.int16)
        if samples.ndim != 1:
            raise ValueError("PCM must be a mono 1-D array")
        if samples.size == 0:
            return
        buffered = np.concatenate([self._pending, samples])
        full_chunks = buffered.size // self.chunk_samples
        for i in range(full_chunks):
            start = i * self.chunk_samples
            self._chunks.append(buffered[start : start + self.chunk_samples])
        self._pending = buffered[full_chunks * self.chunk_samples :]

    def pop(self) -> np.ndarray | None:
        if self._chunks:
            return self._chunks.popleft()
        return None

    def flush_tail(self) -> None:
        """Zero-pad any partial tail into a final chunk (utterance end)."""
        if self._pending.size == 0:
            return
        chunk = np.zeros(self.chunk_samples, dtype=np.int16)
        chunk[: self._pending.size] = self._pending
        self._chunks.append(chunk)
        self._pending = np.zeros(0, dtype=np.int16)

    def clear(self) -> None:
        self._chunks.clear()
        self._pending = np.zeros(0, dtype=np.int16)

    @property
    def buffered_chunks(self) -> int:
        return len(self._chunks)

    @property
    def buffered_samples(self) -> int:
        return (
            len(self._chunks) * self.chunk_samples + int(self._pending.size)
        )

    @property
    def empty(self) -> bool:
        return not self._chunks and self._pending.size == 0
