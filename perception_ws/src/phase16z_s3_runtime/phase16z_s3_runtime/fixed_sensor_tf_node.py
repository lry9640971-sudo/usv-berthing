"""从公开安装参数补发离线 bag 缺失的固定传感器 TF。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


def load_transforms(path: Path) -> tuple[str, list[tuple[str, np.ndarray]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_frame = str(payload.get("base_frame", "")).strip()
    rows = payload.get("transforms")
    if not base_frame or not isinstance(rows, dict) or len(rows) != 7:
        raise RuntimeError("FIXED_SENSOR_EXTRINSICS_CONTRACT_INVALID")
    transforms: list[tuple[str, np.ndarray]] = []
    for child_frame, item in rows.items():
        matrix = np.asarray(item.get("T_base_sensor"), dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"FIXED_SENSOR_TRANSFORM_INVALID:{child_frame}")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise RuntimeError(f"FIXED_SENSOR_HOMOGENEOUS_ROW_INVALID:{child_frame}")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-6
        ):
            raise RuntimeError(f"FIXED_SENSOR_ROTATION_INVALID:{child_frame}")
        transforms.append((str(child_frame), matrix))
    return base_frame, transforms


class FixedSensorTfNode(Node):
    def __init__(self) -> None:
        super().__init__("phase16z_fixed_sensor_tf")
        default_path = (
            Path(get_package_share_directory("phase16z_s3_runtime"))
            / "config"
            / "fixed_sensor_extrinsics.yaml"
        )
        self.declare_parameter("config_path", str(default_path))
        path = Path(str(self.get_parameter("config_path").value))
        base_frame, transforms = load_transforms(path)
        messages = []
        stamp = self.get_clock().now().to_msg()
        for child_frame, matrix in transforms:
            message = TransformStamped()
            message.header.stamp = stamp
            message.header.frame_id = base_frame
            message.child_frame_id = child_frame
            message.transform.translation.x = float(matrix[0, 3])
            message.transform.translation.y = float(matrix[1, 3])
            message.transform.translation.z = float(matrix[2, 3])
            qx, qy, qz, qw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            message.transform.rotation.x = float(qx)
            message.transform.rotation.y = float(qy)
            message.transform.rotation.z = float(qz)
            message.transform.rotation.w = float(qw)
            messages.append(message)
        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(messages)
        self.get_logger().info(
            f"已从公开安装参数发布 {len(messages)} 条固定 TF，父坐标系={base_frame}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FixedSensorTfNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
