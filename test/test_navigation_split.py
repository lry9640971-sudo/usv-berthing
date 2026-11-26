"""验证独立导航、标准速度转换和等待真实里程计端点的跟踪契约。"""

import math
import json
from pathlib import Path
from types import SimpleNamespace

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
import pytest
import rclpy
from sensor_msgs.msg import Imu, NavSatFix
from scipy.spatial.transform import Rotation

from phase16z_perception_msgs.msg import DynamicDetectionArray
from phase16z_s3_runtime.s3_localization_dynamic_tracks import S3LocalizationDynamicTracks
from phase16z_s3_runtime.s3_navigation_node import S3NavigationNode
from phase16z_s3_runtime.s3_runtime_contracts import BASE_FRAME, PUBLIC_FRAME, stamp_to_msg
from phase16z_s3_runtime.s3_timestamped_frames import PlanarPose, TimestampedPoseBuffer


@pytest.fixture
def nodes():
    rclpy.init()
    navigation = tracking = None
    try:
        navigation, tracking = S3NavigationNode(), S3LocalizationDynamicTracks()
        yield navigation, tracking
    finally:
        for node in (tracking, navigation):
            if node is not None:
                node.destroy_node()
        rclpy.shutdown()


def odometry(stamp, x=0.0, body_speed=1.0):
    message = Odometry()
    message.header.stamp = stamp_to_msg(stamp)
    message.header.frame_id, message.child_frame_id = PUBLIC_FRAME, BASE_FRAME
    message.pose.pose.position.x = x
    message.pose.pose.orientation.z = math.sqrt(0.5)
    message.pose.pose.orientation.w = math.sqrt(0.5)
    message.twist.twist.linear.x = body_speed
    return message


def detections(stamp):
    message = DynamicDetectionArray()
    message.header.stamp = stamp_to_msg(stamp)
    message.header.frame_id = BASE_FRAME
    return message


def test_navigation_rejects_recorded_late_mount_angular_oscillation(nodes):
    navigation, _ = nodes
    transform = TransformStamped()
    transform.transform.rotation.w = 1.
    navigation._lookup_base_sensor_tf = lambda frame, stamp: transform
    fixture = json.loads((Path(__file__).parent/'fixtures/orientation_mount_124117.json').read_text(encoding='utf-8'))
    errors = []
    begin = fixture['samples'][0]['stamp_ns']
    for sample in fixture['samples']:
        message = Imu()
        message.header.frame_id, message.header.stamp = 'test_imu', stamp_to_msg(sample['stamp_ns'])
        q = sample['quaternion_xyzw']
        message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w = q
        navigation.imu_callback(message)
        if sample['stamp_ns'] >= begin + 300_000_000:
            measured = navigation.imu_history[-1].angular_body_xyz[2]
            errors.append(abs(measured-sample['reference_yaw_rate_rad_s']))
    # 用真实角速度误差断言；不通过剪裁输出掩盖数值关节振动。
    assert np.percentile(errors, 95) < .08
    assert navigation.imu_rejected_count == 0


def test_gps_publishes_navigation_without_running_tracking(nodes):
    navigation, tracking = nodes
    published = []
    navigation.odom_publisher = SimpleNamespace(publish=published.append)
    transform = TransformStamped()
    transform.transform.rotation.w = 1.0
    transform.transform.translation.x, transform.transform.translation.y = 1.0, 2.0
    transform.transform.translation.z = 0.5
    navigation._lookup_base_sensor_tf = lambda frame, stamp: transform

    def tracking_must_not_run():
        raise AssertionError("导航回调不得执行重跟踪")

    navigation._drain_pending_detections = tracking_must_not_run
    imu = Imu()
    imu.header.frame_id, imu.header.stamp = "test_imu", stamp_to_msg(1_000_000_000)
    imu.orientation.z, imu.orientation.w = math.sin(0.2), math.cos(0.2)
    navigation.imu_callback(imu)
    fix = NavSatFix()
    fix.header.frame_id, fix.header.stamp = "test_gps", stamp_to_msg(1_000_000_000)
    datum = navigation.frame_contract["datum"]
    fix.latitude, fix.longitude, fix.altitude = (datum[key] for key in ("latitude_deg", "longitude_deg", "altitude_m"))
    navigation.gps_callback(fix)
    assert len(published) == 1
    expected = -Rotation.from_euler("z", 0.4).apply([1.0, 2.0, 0.5])
    actual = published[0].pose.pose.position
    assert [actual.x, actual.y, actual.z] == pytest.approx(expected, abs=1e-6)
    assert published[0].header.frame_id == PUBLIC_FRAME
    assert published[0].child_frame_id == BASE_FRAME
    assert navigation.localization_output_count == 1
    assert not hasattr(tracking, "gps_subscription")


def test_tracking_interpolates_velocity_at_measurement_time(nodes):
    _, tracking = nodes
    tracking.navigation_callback(odometry(1_000_000_000, x=0.0, body_speed=1.0))
    tracking.navigation_callback(odometry(1_200_000_000, x=2.0, body_speed=3.0))
    pose = tracking.pose_buffer.interpolate(1_100_000_000)
    assert pose.x_enu_m == pytest.approx(1.0)
    assert pose.velocity_enu_xy_mps == pytest.approx((0.0, 2.0), abs=1e-8)
    assert tracking._nearest_pose(1_100_000_000).stamp_ns == 1_100_000_000


def test_detection_waits_for_a_real_navigation_endpoint(nodes):
    _, tracking = nodes
    processed = []
    tracking._process_base_detections = processed.append
    tracking.navigation_callback(odometry(1_000_000_000))
    message = detections(1_150_000_000)
    tracking.base_detections_callback(message)
    assert processed == []
    tracking.navigation_callback(odometry(1_200_000_000))
    assert processed == [message]
    assert tracking.pose_buffer.interpolate(1_150_000_000).stamp_ns == 1_150_000_000


def test_waiting_detections_keep_recent_measurements_and_count_replacement(nodes):
    _, tracking = nodes
    for stamp in (1_000_000_000, 1_100_000_000, 1_200_000_000):
        tracking.base_detections_callback(detections(stamp))
    assert [message.header.stamp.nanosec for message in tracking.pending_base_detections] == [100_000_000, 200_000_000]
    assert tracking.base_detection_drop_reasons["PENDING_NEWER_MEASUREMENT_REPLACED"] == 1


def test_invalid_odometry_does_not_replace_valid_navigation(nodes):
    _, tracking = nodes
    tracking.navigation_callback(odometry(1_000_000_000))
    wrong_frame = odometry(1_200_000_000)
    wrong_frame.header.frame_id = "foreign_world"
    tracking.navigation_callback(wrong_frame)
    tracking.navigation_callback(Odometry())
    assert tracking.last_localization_stamp_ns == 1_000_000_000
    assert tracking.odometry_rejected_count == 2
    assert tracking.callback_metrics["navigation_callback"]["count"] == 3


def test_pruning_retains_the_left_interpolation_endpoint():
    buffer = TimestampedPoseBuffer()
    for index in range(1, 10):
        buffer.add(PlanarPose(index * 100_000_000, float(index), 0.0, 0.0))
    buffer.discard_before(650_000_000)
    assert buffer.interpolate(650_000_000).x_enu_m == pytest.approx(6.5)
    with pytest.raises(ValueError, match="POSE_TIMESTAMP_OUT_OF_RANGE"):
        buffer.interpolate(550_000_000)
