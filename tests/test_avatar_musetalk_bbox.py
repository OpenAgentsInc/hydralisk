"""SQ-4 (#8621): MuseTalk placeholder/invalid bbox must fail closed."""

from __future__ import annotations

import numpy as np

from hydralisk.avatar.clips import ClipFrameRef
from hydralisk.avatar.musetalk_backend import MuseTalkRenderer, _feature2chunks
from hydralisk.avatar.config import AvatarSettings
from hydralisk.avatar.scheduler import FrameJob
from hydralisk.avatar.state import AvatarState


class _FakeCv2:
    def resize(self, frame, size):  # noqa: ANN001
        w, h = size
        if w <= 0 or h <= 0:
            raise ValueError("invalid resize")
        return np.zeros((h, w, 3), dtype=np.uint8)


class _FakeBlending:
    def get_image_blending(self, ori, res, coords, mask, mask_coords):  # noqa: ANN001
        raise AssertionError("blend must not run on invalid bbox")


def _renderer() -> MuseTalkRenderer:
    # Bypass start(); set only fields _paste_back needs.
    r = object.__new__(MuseTalkRenderer)
    r._cv2 = _FakeCv2()
    r._blending = _FakeBlending()
    return r


def _job(*, speaking: bool = True, state: AvatarState = AvatarState.SPEAKING) -> FrameJob:
    return FrameJob(
        frame_number=0,
        state=state,
        clip=ClipFrameRef(clip_index=8, frame_index=0),
        speaking=speaking,
        audio_chunks=(np.zeros(500, dtype=np.int16),),
    )


def test_paste_back_zero_size_bbox_returns_identity() -> None:
    r = _renderer()
    base = np.full((16, 16, 3), 7, dtype=np.uint8)
    refs = {
        "frames": [base],
        "coords": [(0, 0, 0, 0)],  # placeholder
        "masks": [None],
        "mask_coords": [None],
    }
    pred = np.zeros((8, 8, 3), dtype=np.uint8)
    out = r._paste_back(pred, refs, 0)
    assert out is not pred
    assert np.array_equal(out, base)


def test_paste_back_negative_span_returns_identity() -> None:
    r = _renderer()
    base = np.full((16, 16, 3), 3, dtype=np.uint8)
    refs = {
        "frames": [base],
        "coords": [(10, 10, 5, 5)],
        "masks": [None],
        "mask_coords": [None],
    }
    out = r._paste_back(np.zeros((4, 4, 3), dtype=np.uint8), refs, 0)
    assert np.array_equal(out, base)


def test_feature2chunks_passes_batch_size_when_required() -> None:
    class Processor:
        def feature2chunks(self, feature_array, fps, batch_size):  # noqa: ANN001
            return [(feature_array, fps, batch_size)]

    feature = np.zeros((1, 4), dtype=np.float32)
    assert _feature2chunks(Processor(), feature_array=feature, fps=24) == [
        (feature, 24, 1)
    ]


def test_feature2chunks_supports_older_two_argument_signature() -> None:
    class Processor:
        def feature2chunks(self, feature_array, fps):  # noqa: ANN001
            return [(feature_array, fps)]

    feature = np.zeros((1, 4), dtype=np.float32)
    assert _feature2chunks(Processor(), feature_array=feature, fps=24) == [
        (feature, 24)
    ]


def test_musetalk_stride_holds_last_inpainted_frame_between_gpu_passes() -> None:
    r = object.__new__(MuseTalkRenderer)
    r.settings = AvatarSettings(musetalk_frame_stride=2)
    r._speaking_stride_counter = 1
    last = np.full((16, 16, 3), 9, dtype=np.uint8)
    base = np.full((16, 16, 3), 2, dtype=np.uint8)
    r._last_speaking_frame = last
    r._clips = {
        8: {
            "frames": [base],
            "coords": [(1, 1, 4, 4)],
        }
    }

    out = r.render(_job())

    assert np.array_equal(out, last)
    assert out is not last


def test_musetalk_stride_resets_when_not_speaking() -> None:
    r = object.__new__(MuseTalkRenderer)
    r.settings = AvatarSettings(musetalk_frame_stride=2)
    r._speaking_stride_counter = 3
    r._last_speaking_frame = np.full((16, 16, 3), 9, dtype=np.uint8)
    base = np.full((16, 16, 3), 2, dtype=np.uint8)
    r._clips = {
        8: {
            "frames": [base],
        }
    }

    out = r.render(_job(speaking=False, state=AvatarState.IDLE))

    assert np.array_equal(out, base)
    assert r._speaking_stride_counter == 0
    assert r._last_speaking_frame is None
