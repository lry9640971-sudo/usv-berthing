"""集成消费者示例：只读订阅 public dynamic_tracks，不发布控制。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from phase16z_perception_msgs.msg import DynamicTrackArray


class DynamicTracksConsumer(Node):
    """演示下游如何读取 frame、绝对速度、ID 和生命周期。"""

    def __init__(self, output_path: str) -> None:
        super().__init__("phase16z_dynamic_tracks_consumer_example")
        self.output_path = Path(output_path) if output_path else None
        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.message_count = 0
        self.object_count = 0
        self.subscription = self.create_subscription(
            DynamicTrackArray,
            "/team/perception/dynamic_tracks",
            self.callback,
            10,
        )

    def callback(self, message: DynamicTrackArray) -> None:
        rows = []
        for item in message.tracks:
            row = {
                "stamp_ns": int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec),
                "frame_id": str(message.header.frame_id),
                "track_id": int(item.track_id),
                "x_m": float(item.pose.position.x),
                "y_m": float(item.pose.position.y),
                "vx_mps": float(item.velocity.linear.x),
                "vy_mps": float(item.velocity.linear.y),
                "speed_mps": float(item.speed_mps),
                "classification": int(item.classification),
                "confidence": float(item.confidence),
                "age_sec": float(item.age_sec),
                "hit_count": int(item.hit_count),
                "missed_count": int(item.missed_count),
            }
            rows.append(row)
        self.message_count += 1
        self.object_count += len(rows)
        if self.output_path:
            with self.output_path.open("a", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if rows:
            self.get_logger().info(json.dumps(rows, ensure_ascii=False, sort_keys=True))


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", default="")
    parsed, remaining = parser.parse_known_args(args)
    rclpy.init(args=remaining)
    node = DynamicTracksConsumer(parsed.output_path)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
