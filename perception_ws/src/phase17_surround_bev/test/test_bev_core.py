from types import SimpleNamespace

import numpy as np

from phase17_surround_bev.bev_core import LatestImageSynchronizer, decode_rgb8


def message(stamp: int, encoding: str = "rgb8"):
    pixel = bytes([1, 2, 3, 99, 99])
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000)),
        height=1,
        width=1,
        step=5,
        encoding=encoding,
        data=pixel,
    )


def test_decode_honors_step_and_rgb():
    image = decode_rgb8(message(0))
    assert image.shape == (1, 1, 3)
    assert image[0, 0].tolist() == [1, 2, 3]


def test_decode_bgr_converts_to_rgb():
    image = decode_rgb8(message(0, "bgr8"))
    assert image[0, 0].tolist() == [3, 2, 1]


def test_synchronizer_emits_each_timestamp_once():
    sync = LatestImageSynchronizer(40.0)
    for index, name in enumerate(("front", "back", "left", "left2", "right", "right2")):
        sync.push(name, message(1_000_000_000 + index * 1_000_000))
    result = sync.ready()
    assert result is not None
    assert result.span_ms == 5.0
    assert sync.ready() is None
