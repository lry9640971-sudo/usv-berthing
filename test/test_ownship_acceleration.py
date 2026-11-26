"""覆盖真实制动反向误报、物理加速跟随及原速度融合接口。"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu, NavSatFix

from phase16z_s3_runtime import s3_navigation_node as module
from phase16z_s3_runtime.s3_localization_math import ConstantAccelerationKalman2D, ConstantVelocityKalman2D
from phase16z_s3_runtime.s3_runtime_contracts import stamp_to_msg


def test_acceleration_model_tracks_current_velocity_without_constant_speed_lag():
    accelerated, constant = ConstantAccelerationKalman2D(), ConstantVelocityKalman2D()
    errors = []
    for index in range(90):
        t = index*.05
        position = (.2*t*t+.1*t, -.4*t*t+.2*t)
        velocity = np.asarray((.4*t+.1, -.8*t+.2))
        for model in (accelerated, constant):
            model.update(round((1+t)*1e9), position, (1e-4, 1e-4))
        if t > 1.:
            errors.append(np.linalg.norm(accelerated.state[2:4]-velocity))
    assert max(errors) < .01
    assert np.linalg.norm(constant.state[2:4]-velocity) > .07
    assert accelerated.state[4:] == pytest.approx((.4, -.8), abs=.01)
    assert np.linalg.eigvalsh(accelerated.covariance).min() >= -1e-12


def test_acceleration_model_keeps_position_outlier_gate_and_velocity_fusion():
    model = ConstantAccelerationKalman2D()
    for index in range(40):
        t = index*.05
        model.update(round((1+t)*1e9), (.5*t, .2*t), (1e-4, 1e-4))
    rejected = model.update(3_000_000_000, (100., -100.), (1e-4, 1e-4))
    assert not rejected.accepted
    assert model.rejected_updates == 1
    assert model.update_velocity((.5, .2), 1e-4)
    assert model.state[2:4] == pytest.approx((.5, .2), abs=.01)
    assert np.linalg.eigvalsh(model.covariance).min() >= -1e-12


def replay_recorded_callbacks(monkeypatch, motion_model):
    package = Path(__file__).resolve().parents[1]/"perception_ws/src/phase16z_s3_runtime"
    fixture = json.loads((Path(__file__).parent/"fixtures/ownship_velocity_reversal_201507.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(module, "get_package_share_directory", lambda _: str(package))
    rclpy.init(args=["--ros-args", "--params-file", str(package/"config/sensor_navigation.yaml"),
                    "-p", "motion_model:="+motion_model])
    node = None
    try:
        node = module.S3NavigationNode()
        outputs = []
        node.odom_publisher = SimpleNamespace(publish=outputs.append)
        node.tf_broadcaster = SimpleNamespace(sendTransform=lambda _: None)
        # 生产输入仅取实际IMU与GPS；独立标签在所有回调完成后才读取。
        for row in fixture["inputs"]:
            message = Imu() if row["stream"] == "imu" else NavSatFix()
            message.header.stamp = stamp_to_msg(round(row["stamp"]*1e9))
            if row["stream"] == "imu":
                message.header.frame_id = "wamv/wamv/imu_wamv_link/imu_wamv_sensor"
                message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w = row["xyzw"]
                message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z = row["angular"]
                node.imu_callback(message)
            else:
                message.header.frame_id = "wamv/wamv/gps_wamv_link/navsat"
                message.latitude, message.longitude, message.altitude = row["lla"]
                node.gps_pair_callback(message)
        target = round(fixture["target_measurement_stamp"]*1e9)
        output = next(msg for msg in outputs if msg.header.stamp.sec*1_000_000_000+msg.header.stamp.nanosec == target)
        q, linear = output.pose.pose.orientation, output.twist.twist.linear
        velocity = Rotation.from_quat([q.x, q.y, q.z, q.w]).apply([linear.x, linear.y, linear.z])[:2]
        assert node.gps_rejected_count == 0
        assert node.fuse_gps_velocity is False
        return velocity, fixture, output
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


def test_actual_reversal_cannot_be_reported_as_low_speed_after_acceleration_fix(monkeypatch):
    constant, fixture, _ = replay_recorded_callbacks(monkeypatch, "constant_velocity")
    accelerated, _, output = replay_recorded_callbacks(monkeypatch, "constant_acceleration")
    expected = np.asarray(fixture["expected_velocity_xy_mps"])
    assert np.linalg.norm(constant) < .20
    assert np.linalg.norm(accelerated) > .22
    assert np.linalg.norm(accelerated-expected) < .65*np.linalg.norm(constant-expected)
    assert output.child_frame_id == "wamv/wamv/base_link"
    assert output.header.frame_id == "team_enu"
    covariance = np.asarray(output.twist.covariance).reshape(6, 6)[:3, :3]
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12
