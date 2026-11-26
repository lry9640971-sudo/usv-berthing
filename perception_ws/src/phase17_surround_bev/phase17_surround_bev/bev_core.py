"""六鱼眼图像同步和固定查找表拼接核心。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAMERA_NAMES = ("front", "back", "left", "left2", "right", "right2")


def stamp_ns(message: Any) -> int:
    """读取 ROS Header 时间戳。"""
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def decode_rgb8(message: Any) -> np.ndarray:
    """按 step 安全解码常见 8 位图像并统一为 RGB。"""
    height = int(message.height)
    width = int(message.width)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"不支持的图像编码：{message.encoding}")
    raw = np.frombuffer(bytes(message.data), dtype=np.uint8)
    if raw.size < height * step or step < width * channels:
        raise ValueError("Image data/step 与宽高不一致")
    rows = raw[: height * step].reshape(height, step)
    image = rows[:, : width * channels].reshape(height, width, channels).copy()
    if encoding == "bgr8":
        return image[:, :, ::-1]
    if encoding == "rgba8":
        return image[:, :, :3]
    if encoding == "bgra8":
        return image[:, :, [2, 1, 0]]
    return image


def load_lut(path: str | Path) -> dict[str, np.ndarray]:
    """载入并验证 Phase 10 自船遮挡感知 LUT。"""
    with np.load(Path(path), allow_pickle=False) as data:
        lut = {name: data[name] for name in data.files}
    required = {
        "map_u",
        "map_v",
        "valid_mask",
        "normalized_weight",
        "coverage_count",
        "circle_mask",
        "self_footprint_mask",
    }
    missing = sorted(required - set(lut))
    if missing:
        raise ValueError(f"LUT 缺少字段：{missing}")
    shape = tuple(lut["coverage_count"].shape)
    if shape != (400, 400) or tuple(lut["map_u"].shape) != (6, 400, 400):
        raise ValueError(f"LUT 形状不符合 20m×20m/0.05m 契约：{shape}")
    if not np.isfinite(lut["map_u"]).all() or not np.isfinite(lut["map_v"]).all():
        raise ValueError("LUT 坐标含非有限值")
    weights = lut["normalized_weight"].astype(np.float32, copy=False)
    valid = lut["valid_mask"].astype(bool, copy=False)
    # Phase 10 会保留被船体遮挡的原始 valid 位，最终覆盖由 coverage_count 定义。
    covered = lut["coverage_count"] > 0
    error = np.abs(weights.sum(axis=0)[covered] - 1.0)
    if error.size and float(error.max()) > 1e-5:
        raise ValueError(f"LUT 归一化权重误差过大：{float(error.max())}")
    return lut


def render_rgb(images: dict[str, np.ndarray], lut: dict[str, np.ndarray]) -> np.ndarray:
    """将六路 RGB 图像按遮挡感知权重融合为 400×400 RGB。"""
    missing = [name for name in CAMERA_NAMES if name not in images]
    if missing:
        raise ValueError(f"缺少相机图像：{missing}")
    sampled = []
    for index, name in enumerate(CAMERA_NAMES):
        image = np.asarray(images[name], dtype=np.uint8)
        if image.shape != (720, 1280, 3):
            raise ValueError(f"{name} 图像尺寸错误：{image.shape}")
        remapped = cv2.remap(
            image,
            lut["map_u"][index].astype(np.float32, copy=False),
            lut["map_v"][index].astype(np.float32, copy=False),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        ).astype(np.float32)
        remapped[~lut["valid_mask"][index]] = 0.0
        sampled.append(remapped)
    stack = np.stack(sampled, axis=0)
    fused = np.sum(stack * lut["normalized_weight"][:, :, :, None], axis=0)
    fused[lut["coverage_count"] == 0] = 0.0
    return np.ascontiguousarray(np.clip(fused, 0.0, 255.0).astype(np.uint8))


@dataclass(frozen=True)
class SyncResult:
    messages: dict[str, Any]
    min_stamp_ns: int
    max_stamp_ns: int

    @property
    def span_ms(self) -> float:
        return (self.max_stamp_ns - self.min_stamp_ns) / 1e6


class LatestImageSynchronizer:
    """保留短时队列，从历史帧中选择最近的完整六相机时间组。"""

    def __init__(self, tolerance_ms: float = 40.0, queue_size: int = 20):
        self.tolerance_ns = int(float(tolerance_ms) * 1e6)
        self.buffers = {name: deque(maxlen=int(queue_size)) for name in CAMERA_NAMES}
        self.last_output_max_stamp_ns = -1

    def push(self, camera: str, message: Any) -> None:
        if camera not in CAMERA_NAMES:
            raise KeyError(camera)
        value = stamp_ns(message)
        buffer = self.buffers[camera]
        if not buffer or value > buffer[-1][0]:
            buffer.append((value, message))

    def ready(self) -> SyncResult | None:
        if any(not self.buffers[name] for name in CAMERA_NAMES):
            return None
        candidates = sorted(
            {
                value
                for name in CAMERA_NAMES
                for value, _message in self.buffers[name]
                if value > self.last_output_max_stamp_ns
            },
            reverse=True,
        )
        for anchor in candidates:
            selected = {
                name: min(self.buffers[name], key=lambda item: abs(item[0] - anchor))
                for name in CAMERA_NAMES
            }
            stamps = [item[0] for item in selected.values()]
            lo, hi = min(stamps), max(stamps)
            if hi - lo > self.tolerance_ns or hi <= self.last_output_max_stamp_ns:
                continue
            self.last_output_max_stamp_ns = hi
            for name in CAMERA_NAMES:
                used_stamp = selected[name][0]
                while self.buffers[name] and self.buffers[name][0][0] <= used_stamp:
                    self.buffers[name].popleft()
            return SyncResult(
                messages={name: selected[name][1] for name in CAMERA_NAMES},
                min_stamp_ns=lo,
                max_stamp_ns=hi,
            )
        return None

    def counts(self) -> dict[str, bool]:
        return {name: bool(self.buffers[name]) for name in CAMERA_NAMES}
