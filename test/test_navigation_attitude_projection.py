"""覆盖实际入口噪声、快速转动与姿态校正导致的假速度。"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, NavSatFix

from phase16z_s3_runtime import s3_navigation_node as module
from phase16z_s3_runtime.s3_orientation_rate import OrientationRate
from phase16z_s3_runtime.s3_runtime_contracts import stamp_to_msg


def test_current_attitude_fit_keeps_real_acceleration_and_yaw_wrap_without_future_samples():
    estimator = OrientationRate(.2)
    for index in range(21):
        t = index*.01
        estimator.observe(round((1.+t)*1e9), Rotation.from_euler('z', 3.+3.6*t+.6*t*t).as_quat())
    current = estimator.history[-1][1]
    # 已收到的未来时戳异常帧也不能被用于更早的测量姿态。
    estimator.observe(1_210_000_000, Rotation.from_euler('xyz', (1.2, -.8, -2.)).as_quat())
    quaternion, covariance = estimator.pose_at(1_200_000_000, current)
    expected = Rotation.from_euler('z', 3.+3.6*.2+.6*.2**2)
    assert (expected.inv()*Rotation.from_quat(quaternion)).magnitude() < 1e-10
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12


def replay_entry(monkeypatch, *, raw_attitude):
    root = Path(__file__).resolve().parents[1]
    package = root/'perception_ws/src/phase16z_s3_runtime'
    fixture = json.loads((root/'test/fixtures/ownship_entry_noise_234038.json').read_text(encoding='utf-8'))
    monkeypatch.setattr(module, 'get_package_share_directory', lambda _: str(package))
    rclpy.init(args=['--ros-args', '--params-file', str(package/'config/sensor_navigation.yaml')])
    node = None
    try:
        node = module.S3NavigationNode()
        if raw_attitude:
            node.orientation_rate.pose_at = lambda *_: None
        outputs = []
        node.odom_publisher = SimpleNamespace(publish=outputs.append)
        node.tf_broadcaster = SimpleNamespace(sendTransform=lambda _: None)
        for row in fixture['inputs']:
            message = Imu() if row['stream'] == 'imu' else NavSatFix()
            message.header.stamp = stamp_to_msg(round(row['stamp']*1e9))
            if row['stream'] == 'imu':
                message.header.frame_id = 'wamv/wamv/imu_wamv_link/imu_wamv_sensor'
                message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w = row['xyzw']
                message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z = row['angular']
                node.imu_callback(message)
            else:
                message.header.frame_id = 'wamv/wamv/gps_wamv_link/navsat'
                message.latitude, message.longitude, message.altitude = row['lla']
                node.gps_pair_callback(message)
        # 以下标签仅在实际业务方法完成输出后读取，不能参与拟合或滤波更新。
        labels = {round(row['stamp']*1e9): row for row in fixture['independent_labels']}
        errors, positions = {}, []
        for output in outputs:
            stamp = output.header.stamp.sec*1_000_000_000+output.header.stamp.nanosec
            if stamp not in labels:
                continue
            q, linear, p = output.pose.pose.orientation, output.twist.twist.linear, output.pose.pose.position
            velocity = Rotation.from_quat((q.x, q.y, q.z, q.w)).apply((linear.x, linear.y, linear.z))[:2]
            errors[stamp] = np.linalg.norm(velocity-labels[stamp]['velocity_xy_mps'])
            positions.append(np.linalg.norm(np.asarray((p.x, p.y))-labels[stamp]['position_xy_m']))
        assert node.gps_rejected_count == 0 and node.imu_rejected_count == 0
        assert node.fuse_gps_velocity is False
        return errors, positions
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_actual_entry_noise_is_not_amplified_into_velocity_after_pose_projection_fix(monkeypatch):
    raw, _ = replay_entry(monkeypatch, raw_attitude=True)
    fitted, positions = replay_entry(monkeypatch, raw_attitude=False)
    matched = sorted(raw.keys() & fitted.keys())
    assert len(matched) >= 100
    before = np.asarray([raw[stamp] for stamp in matched])
    after = np.asarray([fitted[stamp] for stamp in matched])
    assert np.percentile(after, 95) < .16
    assert after.mean() < .65*before.mean()
    assert max(positions) < .025
