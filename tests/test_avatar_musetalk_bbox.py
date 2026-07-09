"""SQ-4 (#8621): MuseTalk placeholder/invalid bbox must fail closed."""

from __future__ import annotations

import numpy as np

from hydralisk.avatar.musetalk_backend import MuseTalkRenderer, _feature2chunks
from hydralisk.avatar.config import AvatarSettings


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
