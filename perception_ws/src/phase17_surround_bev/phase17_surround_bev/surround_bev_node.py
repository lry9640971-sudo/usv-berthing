"""六鱼眼 10 m 环视鸟瞰 ROS2 节点。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .bev_core import CAMERA_NAMES, LatestImageSynchronizer, decode_rgb8, load_lut, render_rgb


IMAGE_TOPICS = {
    name: f"/wamv/sensors/cameras/{name}_camera_sensor/image_raw" for name in CAMERA_NAMES
}


class SurroundBevNode(Node):
    def __init__(self) -> None:
        super().__init__("phase17_surround_bev")
        share = Path(get_package_share_directory("phase17_surround_bev"))
        self.declare_parameter("lut_path", str(share / "config" / "masked_fixed_geometry_bev_lut.npz"))
        self.declare_parameter("frame_id", "wamv/base_link")
        self.declare_parameter("sync_tolerance_ms", 40.0)
        self.declare_parameter("process_period_sec", 0.02)
        self.declare_parameter("status_period_sec", 1.0)
        self.declare_parameter("output_topic", "/team/perception/surround_bev")
        self.declare_parameter("official_output_topic", "/world/obstacles/bev_check")
        self.declare_parameter("confidence_topic", "/team/perception/surround_bev_confidence")
        self.declare_parameter("status_topic", "/team/perception/surround_bev_status")
        self.declare_parameter("calibration_state", "D0_RELATIVE_EXTRINSICS_RECOVERED_DEV")
        self.declare_parameter("receipt_path", "")

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.calibration_state = str(self.get_parameter("calibration_state").value)
        self.receipt_path = str(self.get_parameter("receipt_path").value)
        self.lut = load_lut(str(self.get_parameter("lut_path").value))
        self.sync = LatestImageSynchronizer(float(self.get_parameter("sync_tolerance_ms").value))
        self.received = {name: 0 for name in CAMERA_NAMES}
        self.decode_errors = 0
        self.output_frames = 0
        self.last_process_ms: float | None = None
        self.last_sync_span_ms: float | None = None
        self.last_output_monotonic: float | None = None
        self.started_monotonic = time.monotonic()

        self.output_pub = self.create_publisher(Image, str(self.get_parameter("output_topic").value), 2)
        self.official_pub = self.create_publisher(Image, str(self.get_parameter("official_output_topic").value), 2)
        self.confidence_pub = self.create_publisher(Image, str(self.get_parameter("confidence_topic").value), 2)
        self.status_pub = self.create_publisher(String, str(self.get_parameter("status_topic").value), 5)
        self._subscriptions_keepalive = []
        input_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        for name, topic in IMAGE_TOPICS.items():
            self._subscriptions_keepalive.append(
                self.create_subscription(
                    Image,
                    topic,
                    lambda message, camera=name: self._image_callback(camera, message),
                    input_qos,
                )
            )
        self.create_timer(float(self.get_parameter("process_period_sec").value), self._process)
        self.create_timer(float(self.get_parameter("status_period_sec").value), self._publish_status)
        coverage = float(np.mean((self.lut["coverage_count"] > 0)[self.lut["circle_mask"]]))
        self.get_logger().info(
            f"六相机 BEV 已启动，10m圆覆盖率={coverage:.4f}，标定状态={self.calibration_state}"
        )

    def _image_callback(self, camera: str, message: Image) -> None:
        self.received[camera] += 1
        self.sync.push(camera, message)

    @staticmethod
    def _rgb_message(image: np.ndarray, stamp, frame_id: str) -> Image:
        message = Image()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.height = int(image.shape[0])
        message.width = int(image.shape[1])
        message.encoding = "rgb8"
        message.is_bigendian = False
        message.step = int(image.shape[1] * 3)
        message.data = image.tobytes()
        return message

    @staticmethod
    def _mono_message(image: np.ndarray, stamp, frame_id: str) -> Image:
        message = Image()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.height = int(image.shape[0])
        message.width = int(image.shape[1])
        message.encoding = "mono8"
        message.is_bigendian = False
        message.step = int(image.shape[1])
        message.data = np.ascontiguousarray(image).tobytes()
        return message

    def _process(self) -> None:
        group = self.sync.ready()
        if group is None:
            return
        started = time.perf_counter()
        try:
            images = {name: decode_rgb8(group.messages[name]) for name in CAMERA_NAMES}
            fused = render_rgb(images, self.lut)
        except Exception as exc:  # 输入损坏时继续等待下一组
            self.decode_errors += 1
            self.get_logger().error(f"BEV 图像组处理失败：{exc}")
            return
        stamp = group.messages[CAMERA_NAMES[0]].header.stamp
        message = self._rgb_message(fused, stamp, self.frame_id)
        self.output_pub.publish(message)
        self.official_pub.publish(message)
        confidence = np.clip(self.lut["weight_sum"] * 255.0, 0.0, 255.0).astype(np.uint8)
        self.confidence_pub.publish(self._mono_message(confidence, stamp, self.frame_id))
        self.output_frames += 1
        self.last_sync_span_ms = group.span_ms
        self.last_process_ms = (time.perf_counter() - started) * 1000.0
        self.last_output_monotonic = time.monotonic()

    def _status(self) -> dict:
        now = time.monotonic()
        age = None if self.last_output_monotonic is None else now - self.last_output_monotonic
        all_seen = all(value > 0 for value in self.received.values())
        state = "READY" if self.output_frames > 0 and age is not None and age <= 0.6 else "WAITING_INPUT"
        if self.decode_errors:
            state = "DEGRADED_INPUT"
        return {
            "schema_version": 1,
            "state": state,
            "calibration_state": self.calibration_state,
            "input_topics": IMAGE_TOPICS,
            "received": self.received,
            "all_cameras_seen": all_seen,
            "output_frames": self.output_frames,
            "decode_errors": self.decode_errors,
            "last_sync_span_ms": self.last_sync_span_ms,
            "last_processing_ms": self.last_process_ms,
            "output_age_sec": age,
            "grid": {
                "frame_id": self.frame_id,
                "shape": [400, 400],
                "resolution_m_per_px": 0.05,
                "radius_m": 10.0,
                "row0": "+X forward",
                "col0": "+Y left",
            },
            "coverage_10m_circle": float(
                np.mean((self.lut["coverage_count"] > 0)[self.lut["circle_mask"]])
            ),
            "self_mask_cells": int(self.lut["self_footprint_mask"].sum()),
            "runtime_sec": now - self.started_monotonic,
            "truth_or_control_topics_consumed": [],
        }

    def _publish_status(self) -> None:
        status = self._status()
        message = String()
        message.data = json.dumps(status, ensure_ascii=False, separators=(",", ":"))
        self.status_pub.publish(message)
        self._write_receipt(status)

    def _write_receipt(self, status: dict) -> None:
        if self.receipt_path:
            path = Path(self.receipt_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SurroundBevNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # SIGINT 可能先关闭 rclpy context，此时只落盘，避免向失效 publisher 发布。
        node._write_receipt(node._status())
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
