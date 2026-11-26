"""保留真实 GPS 速度、严格配对和安装偏移校正的契约回归。"""

from types import SimpleNamespace

from geometry_msgs.msg import TransformStamped, TwistStamped
from gz.msgs10.navsat_pb2 import NavSat
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Imu, NavSatFix

from phase16z_s3_runtime.s3_gps_velocity_bridge import GPSVelocityBridge
from phase16z_s3_runtime.s3_localization_math import ConstantVelocityKalman2D
from phase16z_s3_runtime.s3_navigation_node import S3NavigationNode
from phase16z_s3_runtime.s3_runtime_contracts import stamp_to_msg


def test_velocity_bridge_keeps_original_sensor_stamp_and_enu_fields():
    message = NavSat()
    message.header.stamp.sec, message.header.stamp.nsec = 12, 300_000_000
    message.velocity_east, message.velocity_north, message.velocity_up = 1.2, -.4, .03
    outputs = []
    GPSVelocityBridge.receive(SimpleNamespace(publisher=SimpleNamespace(publish=outputs.append)), message)
    assert len(outputs) == 1
    assert outputs[0].header.frame_id == "team_enu"
    assert outputs[0].header.stamp.nanosec == 300_000_000
    assert [outputs[0].twist.linear.x, outputs[0].twist.linear.y, outputs[0].twist.linear.z] == [1.2, -.4, .03]


@pytest.mark.parametrize("velocity_first", [False, True])
def test_same_sensor_measurement_pairs_in_both_arrival_orders(velocity_first):
    rclpy.init()
    node = S3NavigationNode()
    try:
        node.gps_velocity_topic = "/test_velocity"
        published = []
        node.odom_publisher = SimpleNamespace(publish=published.append)
        transform = TransformStamped()
        transform.transform.rotation.w = 1.
        transform.transform.translation.x, transform.transform.translation.y = 1., 2.
        node._lookup_base_sensor_tf = lambda *args: transform
        stamp = stamp_to_msg(1_000_000_000)
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = stamp, "imu"
        imu.orientation.w = 1.
        imu.angular_velocity.z = .2
        node.imu_callback(imu)
        fix = NavSatFix()
        fix.header.stamp, fix.header.frame_id = stamp, "gps"
        datum = node.frame_contract["datum"]
        fix.latitude, fix.longitude, fix.altitude = [datum[key] for key in ("latitude_deg", "longitude_deg", "altitude_m")]
        velocity = TwistStamped()
        velocity.header.stamp, velocity.header.frame_id = stamp, "team_enu"
        # 真实 base 速度 (2,.5) 加上 omega cross lever (-.4,.2)。
        velocity.twist.linear.x, velocity.twist.linear.y = 1.6, .7
        first, second = ((lambda: node.gps_velocity_callback(velocity), lambda: node.gps_pair_callback(fix))
                         if velocity_first else (lambda: node.gps_pair_callback(fix), lambda: node.gps_velocity_callback(velocity)))
        first()
        assert not published
        second()
        assert len(published) == 1
        assert [published[0].twist.twist.linear.x, published[0].twist.twist.linear.y] == pytest.approx([2., .5], abs=1e-4)
        assert node.gps_velocity_pair_count == 1
        assert not node._gps_fix_waiting and not node._gps_velocity_waiting
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_sensor_velocity_removes_acceleration_and_braking_lag():
    position = np.zeros(2)
    raw = ConstantVelocityKalman2D()
    measured = ConstantVelocityKalman2D()
    unmeasured_errors, measured_errors = [], []
    for i in range(200):
        t, dt = i * .05, .05
        velocity = np.asarray((min(.8, max(0., .4 * t)) if t < 5. else max(0., .8 - .6 * (t - 5.)), .05))
        position += velocity * dt
        stamp = int((1. + t) * 1e9)
        raw.update(stamp, position)
        measured.update(stamp, position)
        measured.update_velocity(velocity, 1e-4)
        unmeasured_errors.append(np.linalg.norm(raw.state[2:] - velocity))
        measured_errors.append(np.linalg.norm(measured.state[2:] - velocity))
    assert np.percentile(measured_errors, 95) < .03
    assert np.mean(measured_errors) < .1 * np.mean(unmeasured_errors)
