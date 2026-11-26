"""保留原始 NavSat 传感器的速度字段；不订阅世界位姿或真值里程计。"""

import math
import signal

import gz.transport13 as transport
from gz.msgs10.navsat_pb2 import NavSat
from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node


class GPSVelocityBridge(Node):
    def __init__(self):
        super().__init__("s3_gps_velocity_bridge")
        self.declare_parameter("gz_topic", "")
        topic = str(self.get_parameter("gz_topic").value)
        if not topic.endswith("/sensor/navsat/navsat"):
            raise ValueError("必须显式指定原始 NavSat 传感器话题")
        self.publisher = self.create_publisher(TwistStamped, "/wamv/sensors/gps/gps/velocity", 20)
        self.gz = transport.Node()
        if not self.gz.subscribe(NavSat, topic, self.receive):
            raise RuntimeError("GPS 原始速度订阅失败：" + topic)
        self.get_logger().info("保留原始 GPS 速度字段：" + topic)

    def receive(self, message):
        values = (message.velocity_east, message.velocity_north, message.velocity_up)
        if not all(math.isfinite(value) for value in values):
            self.get_logger().error("GPS_VELOCITY_NONFINITE")
            return
        output = TwistStamped()
        output.header.stamp.sec = message.header.stamp.sec
        output.header.stamp.nanosec = message.header.stamp.nsec
        output.header.frame_id = "team_enu"
        output.twist.linear.x, output.twist.linear.y, output.twist.linear.z = values
        self.publisher.publish(output)


def main():
    rclpy.init()
    node = GPSVelocityBridge()
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.2)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
