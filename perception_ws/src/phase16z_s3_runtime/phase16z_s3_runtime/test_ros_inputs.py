"""TEST_ONLY ROS 输入发布器：覆盖有效、非法 frame 和过期队列样本。"""

from __future__ import annotations

import math
from pathlib import Path
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, NavSatFix
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from phase16z_perception_msgs.msg import DynamicDetection, DynamicDetectionArray


GPS_TOPIC = "/wamv/sensors/gps/gps/fix"
IMU_TOPIC = "/wamv/sensors/imu/imu/data"
DETECTIONS_TOPIC = "/team/perception/dynamic_detections_base"


class TestRosInputs(Node):
    """不连接 Gazebo 的接口测试数据源，明确标记 TEST_ONLY。"""

    def __init__(self) -> None:
        super().__init__("phase16z_s3_test_only_inputs")
        self.gps_pub = self.create_publisher(NavSatFix, GPS_TOPIC, qos_profile_sensor_data)
        self.imu_pub = self.create_publisher(Imu, IMU_TOPIC, qos_profile_sensor_data)
        self.detection_pub = self.create_publisher(DynamicDetectionArray, DETECTIONS_TOPIC, 10)
        self.static_tf = StaticTransformBroadcaster(self)
        self.started_ns = self.get_clock().now().nanoseconds
        self.count = 0
        self._publish_static_tf()
        self.timer = self.create_timer(0.1, self.publish_once)

    def _publish_static_tf(self) -> None:
        transforms = []
        for child in ("wamv/gps_link", "wamv/imu_link"):
            transform = TransformStamped()
            transform.header.frame_id = "wamv/wamv/base_link"
            transform.child_frame_id = child
            transform.transform.rotation.w = 1.0
            transforms.append(transform)
        self.static_tf.sendTransform(transforms)

    @staticmethod
    def _stamp_message(stamp_ns: int):
        from builtin_interfaces.msg import Time

        stamp = Time()
        stamp.sec = int(stamp_ns // 1_000_000_000)
        stamp.nanosec = int(stamp_ns % 1_000_000_000)
        return stamp

    def publish_once(self) -> None:
        now_ns = int(self.get_clock().now().nanoseconds)
        elapsed_s = max(0.0, (now_ns - self.started_ns) / 1_000_000_000.0)
        gps = NavSatFix()
        gps.header.stamp = self._stamp_message(now_ns)
        gps.header.frame_id = "wamv/gps_link"
        gps.latitude = 30.0
        gps.longitude = 120.0 + (0.5 * elapsed_s) / 96_000.0
        gps.altitude = 0.0
        gps.position_covariance[0] = 0.04
        gps.position_covariance[4] = 0.04
        self.gps_pub.publish(gps)

        imu = Imu()
        imu.header.stamp = self._stamp_message(now_ns)
        imu.header.frame_id = "wamv/imu_link"
        imu.orientation.w = 1.0
        self.imu_pub.publish(imu)

        array = DynamicDetectionArray()
        array.header.stamp = self._stamp_message(now_ns)
        array.header.frame_id = "wamv/wamv/base_link"
        target_x = 10.0 + 1.0 * elapsed_s
        ownship_x = 0.5 * elapsed_s
        detection = DynamicDetection()
        detection.header.stamp = self._stamp_message(now_ns)
        detection.header.frame_id = "wamv/wamv/base_link"
        detection.observation_id = f"test-valid-{self.count}"
        detection.center.x = target_x - ownship_x
        detection.center.y = 0.0
        detection.radius_m = 1.0
        detection.support = 6
        detection.position_std_m = 0.08
        detection.range_m = abs(detection.center.x)
        detection.confidence = 0.85
        detection.source_sensors = ["front"]
        array.detections.append(detection)
        if self.count % 10 == 0:
            invalid = DynamicDetection()
            invalid.header.stamp = self._stamp_message(now_ns)
            invalid.header.frame_id = "wrong_frame"
            invalid.observation_id = f"test-invalid-frame-{self.count}"
            invalid.center.x = detection.center.x
            invalid.center.y = 0.0
            invalid.radius_m = 1.0
            invalid.support = 1
            invalid.position_std_m = 0.2
            invalid.range_m = detection.range_m
            invalid.confidence = 0.5
            invalid.source_sensors = ["front"]
            array.detections.append(invalid)
        if self.count % 15 == 0:
            late = DynamicDetection()
            late.header.stamp = self._stamp_message(max(1, now_ns - 300_000_000))
            late.header.frame_id = "wamv/wamv/base_link"
            late.observation_id = f"test-queue-late-{self.count}"
            late.center.x = detection.center.x
            late.center.y = 2.0
            late.radius_m = 0.8
            late.support = 2
            late.position_std_m = 0.2
            late.range_m = detection.range_m
            late.confidence = 0.4
            late.source_sensors = ["left"]
            array.detections.append(late)
        self.detection_pub.publish(array)
        self.count += 1


def main(args: list[str] | None = None) -> int:
    rclpy.init(args=args)
    node = TestRosInputs()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
